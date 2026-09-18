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

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from fraud import RuleConfig, combine, detect
from fraud import ml_features
from fraud_ml.predict import FraudModel, ModelNotAvailable

from .demo import demo_data
from .models import BidderRisk, DetectResponse, DetectRequest, HealthResponse
from .provider import InMemoryProvider, InputProvider, PostgresProvider, ProviderError

log = logging.getLogger("fraud_api")

# 규칙 점수와 모델 점수의 결합 비율.
#
# **기본값은 규칙 100% 다.** 모델은 eBay 데이터로 학습해 우리 도메인에서 검증된 적이
# 없다. 피처 정의를 논문대로 옮겼지만 분포가 같다는 보장은 없으므로, 우리 데이터로
# 재학습하기 전까지는 낮게 두거나 0 으로 둔다.
W_RULE = float(os.getenv("FRAUD_W_RULE", "1.0"))
W_ML = float(os.getenv("FRAUD_W_ML", "0.0"))


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# 섀도 모드 — **모델 점수를 계산하되 risk_score 에는 섞지 않는다.**
#
# `w_ml=0` 인 채로 모델을 아예 안 부르면, 나중에 가중치를 올릴 때 근거로 삼을 자료가
# 하나도 없다. "검증되면 켠다" 고만 해 두면 검증할 방법이 없어 영영 못 켠다.
#
# 섀도로 돌려 두면 두 트랙이 같은 입찰자를 어떻게 봤는지가 매 경매마다 쌓인다.
#
#     rule 0.535  ml 0.538   두 트랙이 같은 결론
#     rule 0.120  ml 0.810   불일치. 운영자 판정이 붙으면 어느 쪽이 맞았는지 남는다
#
# **risk_score 는 한 자리도 바뀌지 않는다** (`combine` 이 `w_ml<=0` 이면 규칙 점수를
# 그대로 돌려준다). 늘어나는 것은 응답의 `ml_score` 와 그 출처인 `model_version` 뿐이다.
#
# 기본값이 켬인 이유는 점수에 영향이 없고 비용도 경매당 수 밀리초이기 때문이다.
# 끄려면 `FRAUD_ML_SHADOW=false`.
ML_SHADOW = _flag("FRAUD_ML_SHADOW", True)

# DB 가 설정되어 있으면 실제 조회를, 아니면 합성 데이터를 쓴다.
#
# **연결 실패 시 합성 데이터로 넘어가지 않는다.** 그러면 서버는 정상으로 보이는데
# 판정은 데모 경매 3건만 아는 상태가 되어, 실제 경매를 물어보면 "없는 경매" 로
# 답한다. 조용히 틀리느니 뜨지 않는 편이 낫다.
DATABASE_URL = os.getenv("DIB_DATABASE_URL", "").strip()


def _make_provider() -> InputProvider:
    if not DATABASE_URL:
        log.warning("DIB_DATABASE_URL 이 없어 합성 데이터로 동작합니다 (데모 경매 3건)")
        return InMemoryProvider(demo_data())
    log.info("실제 DB 에 연결합니다")
    return PostgresProvider(DATABASE_URL)


# 라우터로 분리해 두면 검수 API 와 한 서버에 합쳐 띄울 수 있다 (src/serve.py).
router = APIRouter()

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["config"] = RuleConfig.load()
    _state["provider"] = _make_provider()
    cfg: RuleConfig = _state["config"]  # type: ignore[assignment]
    log.info(
        "규칙 엔진 준비 완료 — config=%s, 활성 규칙 %d개",
        cfg.version,
        len(cfg.enabled_rules),
    )

    # 모델은 없어도 된다. 탐지는 규칙 트랙만으로 동작하고 모델은 얹히는 구조다.
    # 여기서 죽으면 규칙 탐지까지 같이 멈추므로 경고만 남기고 계속 간다.
    try:
        _state["model"] = FraudModel.load()
        model: FraudModel = _state["model"]  # type: ignore[assignment]
        log.info("모델 준비 완료 — %s, w_ml=%.2f", model.version, W_ML)
    except ModelNotAvailable as exc:
        log.warning("모델 없이 시작합니다 (규칙 트랙만 동작) — %s", exc)
        _state["model"] = None

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


def score_bidders(
    inp, cfg: RuleConfig, model: "FraudModel | None"
) -> tuple[list[BidderRisk], dict[str, float], dict[int, str], str | None, datetime]:
    """규칙 + 모델 점수를 합쳐 입찰자별 위험도를 만든다.

    두 엔드포인트(동기 `/internal/fraud/detect`, 비동기 명세 92번)가 같은 계산을
    쓰도록 함수로 뺀다. **점수가 경로마다 다르면 관리자가 보는 숫자와 저장된 숫자가
    어긋난다.**

    `ml_features` 원값도 함께 돌려준다 — 명세 93번 콜백이 피처를 그대로 요구한다.
    """
    result = detect(inp, cfg)

    # 피처는 **모델 가중치와 무관하게 계산한다.** W_ML 이 0 이어도 백엔드는 이 값을
    # fraud_detection 에 저장하고 관리자 화면에서 근거로 본다. 점수에 안 쓴다는 것과
    # 값을 안 남긴다는 것은 다르다.
    #
    # 코퍼스 기준값이 없으면 피처를 만들 수 없어 None 이 된다. 그 경우 규칙 점수만 쓴다.
    features: dict[int, dict[str, float]] = {
        r.member_id: f
        for r in result.results
        if (f := ml_features.compute(inp, r.member_id)) is not None
    }

    # 모델 점수는 한 번에 계산한다. 입찰자마다 부르면 경매 하나에 수십 번이 된다.
    #
    # **가중치가 0 이어도 섀도 모드면 계산한다.** 점수에는 안 섞이지만(`combine` 이
    # 규칙 점수를 그대로 돌려준다) 두 트랙을 비교할 자료가 그때부터 쌓인다.
    ml_scores: dict[int, float | None] = {r.member_id: None for r in result.results}
    if model is not None and (W_ML > 0 or ML_SHADOW):
        if features:
            try:
                for member_id, score in zip(
                    features, model.score_many(list(features.values()))
                ):
                    ml_scores[member_id] = score
            except Exception:
                # 모델이 터져도 규칙 점수는 나가야 한다.
                log.exception("모델 추론 실패 auction_id=%s", inp.auction.auction_id)

    results = []
    for r in result.results:
        ml = ml_scores[r.member_id]
        risk = combine(r.rule_score, ml, W_RULE, W_ML)
        results.append(
            BidderRisk(
                member_id=r.member_id,
                risk_score=round(risk, 4),
                rule_score=round(r.rule_score, 4),
                ml_score=round(ml, 4) if ml is not None else None,
                band=cfg.band_of(risk),
                reasons=r.reasons(),
                detail={
                    **r.to_detail(),
                    "rule_config_version": result.rule_config_version,
                    # 명세 93번 콜백이 피처 원값을 요구한다. 없으면 빈 dict 다.
                    "ml_features": features.get(r.member_id, {}),
                },
            )
        )

    weights = {"w_rule": W_RULE, "w_ml": W_ML}
    return (
        results,
        weights,
        dict(result.skipped_bidders),
        result.auction_error,
        result.as_of,
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(cfg: RuleConfig = Depends(get_config)) -> HealthResponse:
    """인프라가 감시할 헬스체크. 설정이 로드되고 규칙이 하나라도 켜져 있어야 ok 다."""
    provider: InputProvider = get_provider()
    return HealthResponse(
        status="ok",
        rule_config_version=cfg.version,
        enabled_rules=[spec.rule_id for spec in cfg.enabled_rules],
        provider=provider.name,
    )


@router.post(
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

    model: FraudModel | None = _state.get("model")  # type: ignore[assignment]
    results, weights, skipped, auction_error, as_of = score_bidders(inp, cfg, model)

    # **`model_version` 은 `ml_score` 를 만든 모델을 가리킨다.** 점수에 반영됐는지가
    # 아니라 누가 냈는지다. 섀도 모드에서도 적어야 나중에 "이 점수는 어느 모델이
    # 낸 것인가" 를 답할 수 있고, 그게 섀도로 모으는 이유다.
    #
    # 반영 여부는 `weights.w_ml` 과 `ml_shadow` 가 말한다. 둘을 한 필드로 겸하게
    # 하면 어느 쪽 뜻인지 읽는 사람마다 달라진다.
    scored = any(r.ml_score is not None for r in results)

    return DetectResponse(
        auction_id=inp.auction.auction_id,
        as_of=as_of,
        rule_config_version=cfg.version,
        model_version=model.version if model is not None and scored else None,
        ml_shadow=scored and W_ML <= 0,
        weights=weights,
        results=results,
        skipped_bidders=skipped,
        auction_error=auction_error,
    )


app.include_router(router)
