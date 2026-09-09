"""추천 API.

    GET /internal/reco/home    홈 리스트 노출 순서
    GET /reco/health           헬스체크

**지금은 개인화가 없다.** 인기도와 마감 임박도만으로 순서를 만든다. 이것을 먼저 만드는
이유는 세 곳에서 재사용되기 때문이다.

1. Cold Start 폴백 — 행동 이력이 없는 사용자
2. 성능 평가 baseline — 개인화가 이것보다 나은지 증명해야 한다
3. 장애 폴백 — 임베딩·벡터 검색이 죽어도 추천은 나가야 한다

개인화가 붙어도 이 엔드포인트의 계약은 바뀌지 않는다. `strategy` 필드 값만 달라진다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status

from envfile import load_env
from reco import RecoConfig, rank

from .demo import demo_candidates
from .models import RecoHealthResponse, RecoItem, RecoResponse
from .provider import CandidateProvider, InMemoryProvider, ProviderError

log = logging.getLogger("reco_api")

# 랭킹 전에 조회할 후보 풀. 최종 노출 개수보다 넉넉해야 한다 —
# 여기서 잘리면 랭킹이 볼 수 없는 상품이 생긴다.
CANDIDATE_POOL = 500

router = APIRouter()
_state: dict[str, object] = {}


def startup() -> None:
    load_env()
    cfg = RecoConfig.load()
    _state["config"] = cfg
    _state["provider"] = InMemoryProvider(demo_candidates())
    log.info("추천 준비 완료 — config=%s, 가중치 %s", cfg.version, dict(cfg.weights))


def shutdown() -> None:
    _state.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


def get_config() -> RecoConfig:
    cfg = _state.get("config")
    if cfg is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "추천 설정이 로드되지 않았습니다"
        )
    return cfg  # type: ignore[return-value]


def get_provider() -> CandidateProvider:
    provider = _state.get("provider")
    if provider is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "데이터 조회기가 준비되지 않았습니다"
        )
    return provider  # type: ignore[return-value]


@router.get("/reco/health", response_model=RecoHealthResponse, tags=["ops"])
def health(cfg: RecoConfig = Depends(get_config)) -> RecoHealthResponse:
    provider = get_provider()
    return RecoHealthResponse(
        status="ok",
        config_version=cfg.version,
        weights=dict(cfg.weights),
        provider=provider.name,
    )


@router.get(
    "/internal/reco/home",
    response_model=RecoResponse,
    tags=["reco"],
    summary="홈 리스트에 노출할 경매 순서",
)
def home(
    member_id: int | None = Query(
        None,
        description=(
            "요청한 회원. **현재는 사용하지 않는다** — 개인화 전이라 누가 요청하든 "
            "같은 순서가 나온다. 개인화가 붙으면 이 값으로 프로필을 만든다"
        ),
    ),
    limit: int = Query(20, ge=1, le=100, description="돌려받을 개수"),
    cfg: RecoConfig = Depends(get_config),
    provider: CandidateProvider = Depends(get_provider),
) -> RecoResponse:
    """진행 중인 경매를 마감 임박도·인기도·경쟁도로 정렬해 돌려준다.

    **이미 종료된 경매는 제외된다.** 남은 시간이 0 이하면 마감 임박도가 최대가 되므로,
    걸러내지 않으면 끝난 경매가 목록 맨 위에 올라온다. 제외된 건수는 `excluded` 에 담긴다.
    """
    now = datetime.now(timezone.utc)

    try:
        candidates = provider.load_active(now, CANDIDATE_POOL)
    except ProviderError as exc:
        log.exception("후보 조회 실패")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    result = rank(candidates, cfg, now, limit=limit)

    return RecoResponse(
        as_of=result.as_of,
        strategy=result.strategy,
        config_version=result.config_version,
        items=[
            RecoItem(
                rank=i,
                auction_id=s.auction_id,
                score=round(s.score, 4),
                detail=s.breakdown(),
            )
            for i, s in enumerate(result.items, start=1)
        ],
        excluded=dict(result.excluded),
    )


app = FastAPI(
    title="DIB 추천 API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "홈 리스트에 노출할 경매 순서를 돌려준다.\n\n"
        "**현재는 개인화가 없다.** 마감 임박도·인기도·경쟁도만 쓴다. "
        "개인화가 가동되어도 이 계약은 바뀌지 않고 `strategy` 값만 달라진다."
    ),
)
app.include_router(router)
