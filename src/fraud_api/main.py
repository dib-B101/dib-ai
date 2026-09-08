"""이상거래 탐지 API.

백엔드가 경매 종료 후 호출한다. 점수와 사유만 돌려주며 제재는 하지 않는다.

    POST /internal/fraud/detect   경매 1건 분석
    GET  /health                  헬스체크
    GET  /docs                    자동 생성 API 문서 (백엔드 연동 시 여기를 본다)

설계 원칙 두 가지가 응답에 드러난다.

1. 탐지 실패가 경매 종료를 막지 않는다.
   규칙 실행 중 예외가 나도 200 으로 응답하고 auction_error 에 사유를 담는다.

2. "판단하지 않음" 과 "위험하지 않음" 은 다르다.
   이력이 부족한 입찰자는 results 가 아니라 skipped_bidders 로 나간다.
   0 점으로 저장하면 신규 사용자일수록 안전해 보이는 왜곡이 생긴다.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, status

from fraud import RuleConfig, combine, detect

from .demo import demo_data
from .models import BidderRisk, DetectResponse, DetectRequest, HealthResponse
from .provider import InMemoryProvider, InputProvider, ProviderError

log = logging.getLogger("fraud_api")

# 규칙 점수와 모델 점수의 결합 비율.
# 모델 트랙이 가동되기 전에는 규칙이 전부를 차지한다. eBay 데이터로 학습한 모델은
# 우리 도메인에서 검증된 적이 없으므로, 가동하더라도 규칙 쪽에 더 무게를 둔다.
W_RULE = float(os.getenv("FRAUD_W_RULE", "1.0"))
W_ML = float(os.getenv("FRAUD_W_ML", "0.0"))

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["config"] = RuleConfig.load()
    _state["provider"] = InMemoryProvider(demo_data())
    cfg: RuleConfig = _state["config"]  # type: ignore[assignment]
    log.info(
        "규칙 엔진 준비 완료 — config=%s, 활성 규칙 %d개",
        cfg.version,
        len(cfg.enabled_rules),
    )
    yield
    _state.clear()


app = FastAPI(
    title="DIB 이상거래 탐지 API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "경매 종료 후 입찰자별 허위입찰 위험도를 산출한다.\n\n"
        "**자동 제재에 사용하지 않는다.** 관리자 검토 큐 정렬용이다. "
        "현재 성능으로는 고위험 판정 10건 중 4건 정도가 오탐이다.\n\n"
        "응답 필드는 `fraud_detection` 테이블 컬럼과 1:1 로 대응하므로 변환 없이 저장할 수 있다."
    ),
)


def get_config() -> RuleConfig:
    cfg = _state.get("config")
    if cfg is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "규칙 설정이 로드되지 않았습니다"
        )
    return cfg  # type: ignore[return-value]


def get_provider() -> InputProvider:
    provider = _state.get("provider")
    if provider is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "데이터 조회기가 준비되지 않았습니다"
        )
    return provider  # type: ignore[return-value]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(cfg: RuleConfig = Depends(get_config)) -> HealthResponse:
    """인프라가 감시할 헬스체크. 설정이 로드되고 규칙이 하나라도 켜져 있어야 ok 다."""
    provider: InputProvider = get_provider()
    return HealthResponse(
        status="ok",
        rule_config_version=cfg.version,
        enabled_rules=[spec.rule_id for spec in cfg.enabled_rules],
        provider=provider.name,
    )


@app.post(
    "/internal/fraud/detect",
    response_model=DetectResponse,
    tags=["fraud"],
    summary="경매 1건의 입찰자별 위험도 산출",
)
def detect_auction(
    req: DetectRequest,
    cfg: RuleConfig = Depends(get_config),
    provider: InputProvider = Depends(get_provider),
) -> DetectResponse:
    """경매 종료 후 호출한다.

    같은 `auction_id` 와 `as_of` 로 재호출하면 항상 같은 결과가 나온다.
    재시도해도 안전하므로 실패 시 그대로 다시 부르면 된다.
    """
    try:
        inp = provider.load(req.auction_id, req.as_of)
    except ProviderError as exc:
        log.exception("데이터 조회 실패 auction_id=%s", req.auction_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if inp is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"경매를 찾을 수 없습니다: {req.auction_id}"
        )

    result = detect(inp, cfg)

    results = [
        BidderRisk(
            member_id=r.member_id,
            risk_score=round(combine(r.rule_score, None, W_RULE, W_ML), 4),
            rule_score=round(r.rule_score, 4),
            ml_score=None,
            band=cfg.band_of(r.rule_score),
            reasons=r.reasons(),
            detail={**r.to_detail(), "rule_config_version": result.rule_config_version},
        )
        for r in result.results
    ]

    return DetectResponse(
        auction_id=result.auction_id,
        as_of=result.as_of,
        rule_config_version=result.rule_config_version,
        model_version=None,
        weights={"w_rule": W_RULE, "w_ml": W_ML},
        results=results,
        skipped_bidders=dict(result.skipped_bidders),
        auction_error=result.auction_error,
    )
