"""추천 API.

    GET /internal/reco/home     홈 리스트 노출 순서
    GET /internal/reco/similar  기준 상품과 닮은 경매
    GET /reco/health            헬스체크

**`member_id` 를 주면 개인화한다.** 행동 로그로 관심 프로필을 만들어 후보와의 유사도를
점수에 섞는다. 로그가 없거나 행동한 상품에 임베딩이 없으면 **인기순으로 떨어진다** —
그것이 정상 경로다. 어느 쪽이 적용됐는지는 응답의 `strategy` 로 구분한다.

    personalized   관심 프로필이 반영됨
    similar        기준 상품과의 유사도가 반영됨
    popularity     마감 임박도·인기도·경쟁도만 (폴백 포함)

인기순은 개인화가 붙은 뒤에도 그대로 남는다. 세 곳에서 재사용되기 때문이다.

1. Cold Start 폴백 — 행동 이력이 없는 사용자
2. 성능 평가 baseline — 개인화가 이것보다 나은지 증명해야 한다
3. 장애 폴백 — 임베딩·벡터 검색이 죽어도 추천은 나가야 한다

**엔드포인트 계약은 개인화 전후로 바뀌지 않았다.** `strategy` 필드 값만 달라진다.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status

from envfile import load_env
from reco import RecoConfig, RecoResult, affinity, build_profile, rank, similarities

from .demo import demo_candidates, demo_vectors
from .models import RecoHealthResponse, RecoItem, RecoResponse
from .provider import (
    CandidateProvider,
    InMemoryProvider,
    PostgresProvider,
    ProviderError,
)

log = logging.getLogger("reco_api")

# 랭킹 전에 조회할 후보 풀. 최종 노출 개수보다 넉넉해야 한다 —
# 여기서 잘리면 랭킹이 볼 수 없는 상품이 생긴다.
CANDIDATE_POOL = 500

# DB 가 설정되어 있으면 실제 조회를, 아니면 합성 데이터를 쓴다.
#
# **연결 실패 시 합성 데이터로 넘어가지 않는다.** 그러면 서버는 정상으로 보이는데
# 추천 목록에는 데모 경매 5건만 나온다. 조용히 틀리느니 뜨지 않는 편이 낫다.
def _make_provider() -> CandidateProvider:
    # `.env` 는 startup()에서 로드된다. 모듈 import 시점에 값을 고정하면 uvicorn이
    # 앱을 import한 뒤 `.env`를 읽는 정상 실행 순서에서 영원히 데모 조회기를 쓴다.
    database_url = os.getenv("DIB_DATABASE_URL", "").strip()
    if not database_url:
        log.warning("DIB_DATABASE_URL 이 없어 합성 데이터로 동작합니다 (데모 경매 5건)")
        return InMemoryProvider(demo_candidates(), demo_vectors())
    log.info("실제 DB 에 연결합니다")
    return PostgresProvider(database_url)

router = APIRouter()
_state: dict[str, object] = {}


def startup() -> None:
    load_env()
    cfg = RecoConfig.load()
    _state["config"] = cfg
    _state["provider"] = _make_provider()
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


def _to_response(result: RecoResult) -> RecoResponse:
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


def _load_active(provider: CandidateProvider, now):
    try:
        return provider.load_active(now, CANDIDATE_POOL)
    except ProviderError as exc:
        log.exception("후보 조회 실패")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/reco/health", response_model=RecoHealthResponse, tags=["ops"])
def health(cfg: RecoConfig = Depends(get_config)) -> RecoHealthResponse:
    provider = get_provider()
    return RecoHealthResponse(
        status="ok",
        config_version=cfg.version,
        weights=dict(cfg.weights),
        provider=provider.name,
    )


def _personalize(provider, cfg: RecoConfig, member_id: int, candidates, now):
    """회원 행동으로 후보별 취향 점수를 만든다. 못 만들면 인기순으로 폴백한다."""
    since = now - timedelta(days=cfg.lookback_days)
    try:
        events = provider.load_events(member_id, since, cfg.max_events)
    except ProviderError:
        log.exception("행동 로그 조회 실패 member_id=%s", member_id)
        return None
    if not events:
        return None

    behaved = {e.auction_id for e in events}
    wanted = behaved | {c.auction_id for c in candidates}
    try:
        vectors = provider.load_vectors(sorted(wanted))
    except ProviderError:
        log.exception("임베딩 조회 실패 member_id=%s", member_id)
        return None

    profile = build_profile(
        member_id=member_id,
        events=events,
        vectors=vectors,
        now=now,
        weights=cfg.event_weights or None,
        half_life_days=cfg.half_life_days,
    )
    if profile is None:
        return None

    usable = [vectors[c.auction_id] for c in candidates if c.auction_id in vectors]
    return affinity(profile, usable, cfg.similarity_text_weight) or None


def rank_for_member(
    provider,
    cfg: RecoConfig,
    candidates,
    now,
    member_id: int | None,
    limit: int | None = None,
):
    """동기·비동기 추천이 함께 쓰는 개인화 랭킹 진입점."""
    scores = (
        _personalize(provider, cfg, member_id, candidates, now)
        if member_id is not None
        else None
    )
    if not scores:
        return rank(candidates, cfg, now, limit=limit)

    fresh = [c for c in candidates if c.auction_id in scores]
    result = rank(
        fresh,
        cfg,
        now,
        limit=limit,
        similarity=scores,
        strategy="personalized",
        similarity_weight=cfg.personalization_weight,
    )
    already = len(candidates) - len(fresh)
    if already:
        result.excluded["already_seen"] = already
    return result


def _by_scope(candidates, scope: str):
    """라이브 · 일반 경매를 가른다 (명세 108).

    `auction.live_broadcast_id` 가 채워져 있으면 라이브 방송 중 진행되는 경매다.
    **두 목록의 순위를 따로 매긴다** — 섞어서 매긴 뒤 나누면 한쪽이 상위권을 다
    가져가 다른 쪽 목록이 빈약해진다.
    """
    if scope == "LIVE":
        return [c for c in candidates if c.is_live]
    if scope == "GENERAL":
        return [c for c in candidates if not c.is_live]
    return list(candidates)


@router.get(
    "/internal/reco/home",
    response_model=RecoResponse,
    tags=["reco"],
    summary="홈 리스트에 노출할 경매 순서",
)
def home(
    scope: str = Query(
        "ALL",
        pattern="^(ALL|LIVE|GENERAL)$",
        description=(
            "`LIVE` = 라이브 방송 중 경매, `GENERAL` = 일반 경매, `ALL` = 둘 다. "
            "명세 108 처럼 나눠 보여줄 때는 **두 번 호출해 각각 순위를 받으십시오** — "
            "한 번에 받아 나누면 한쪽이 상위권을 다 가져가 다른 목록이 빈약해집니다"
        ),
    ),
    member_id: int | None = Query(
        None,
        description=(
            "요청한 회원. 주면 행동 로그와 찜을 이용해 개인화한다. "
            "로그나 벡터가 없으면 자동으로 인기순으로 폴백한다"
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
    candidates = _by_scope(_load_active(provider, now), scope)
    return _to_response(rank_for_member(provider, cfg, candidates, now, member_id, limit))


@router.get(
    "/internal/reco/similar",
    response_model=RecoResponse,
    tags=["reco"],
    summary="기준 상품과 닮은 경매",
)
def similar(
    auction_id: int = Query(..., description="기준이 되는 경매. 사용자가 지금 보고 있는 상품"),
    limit: int = Query(10, ge=1, le=100, description="돌려받을 개수"),
    cfg: RecoConfig = Depends(get_config),
    provider: CandidateProvider = Depends(get_provider),
) -> RecoResponse:
    """기준 상품과 닮은 진행 중 경매를 돌려준다.

    유사도만으로 줄 세우지 않는다.

        점수 = w × 유사도 + (1 − w) × (마감 임박도 · 인기도 · 경쟁도)

    닮기만 하고 아무도 안 보는 경매를 위로 올리면 안 되기 때문이다. w 는
    `config/reco.yaml` 의 `similarity.weight` 이며 기본 0.5 다.

    **기준 상품 자신은 결과에서 빠진다.** 유사도 1 이라 반드시 1위가 되는데, 지금
    보고 있는 상품을 "이런 상품은 어때요" 에 다시 띄우는 것은 사고다.

    두 가지 경우에 결과가 달라진다.

    **기준 상품의 임베딩이 없으면 인기순으로 폴백한다.** 응답의 `strategy` 가
    `popularity` 로 온다. 추천을 아예 비워 보내는 것보다 낫다고 판단했다.

    **임베딩이 없는 후보는 목록에서 빠진다.** 건수는 `excluded.no_vector` 에 담긴다.
    유사도를 0 으로 채워 넣으면 "안 닮았다" 로 읽혀, 아직 배치가 안 돌았다는 이유만으로
    순위가 밀린다. 모르는 것과 안 닮은 것은 다르다.
    """
    now = datetime.now(timezone.utc)
    candidates = _load_active(provider, now)

    try:
        vectors = provider.load_vectors([auction_id, *(c.auction_id for c in candidates)])
    except ProviderError as exc:
        log.exception("벡터 조회 실패 auction_id=%s", auction_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    seed = vectors.get(auction_id)
    if seed is None:
        log.warning("기준 상품 임베딩 없음 auction_id=%s — 인기순으로 폴백", auction_id)
        return _to_response(rank(candidates, cfg, now, limit=limit))

    usable = [c for c in candidates if c.auction_id in vectors]
    sims = similarities(seed, [vectors[c.auction_id] for c in usable], cfg.similarity_text_weight)

    # 기준 상품 자신은 similarities() 가 빼므로 후보에서도 빼야 짝이 맞는다.
    usable = [c for c in usable if c.auction_id != auction_id]

    result = rank(usable, cfg, now, limit=limit, similarity=sims, strategy="similar")

    # 종료 제외(`ended`)와 겹치지 않게 **벡터 유무만으로** 센다. 두 사유를 한 건에
    # 이중으로 세면 백엔드가 합계를 맞춰 볼 때 숫자가 안 맞는다.
    no_vector = sum(1 for c in candidates if c.auction_id not in vectors)
    if no_vector:
        result.excluded["no_vector"] = no_vector
    return _to_response(result)


app = FastAPI(
    title="DIB 추천 API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "홈 리스트에 노출할 경매 순서를 돌려준다.\n\n"
        "**`member_id` 를 주면 개인화된다.** 행동 로그로 만든 관심 프로필을 "
        "마감 임박도·인기도·경쟁도와 섞는다. 로그가 없으면 인기순으로 떨어지므로 "
        "신규 사용자에게도 그대로 호출하면 된다.\n\n"
        "적용된 방식은 응답의 `strategy` 로 구분한다 — "
        "`personalized` · `similar` · `popularity`."
    ),
)
app.include_router(router)
