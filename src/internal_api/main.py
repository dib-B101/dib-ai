"""REST API 명세서 92 · 94번 — 백엔드가 AI 를 부르는 내부 엔드포인트.

    POST /internal/v1/ai/bid-anomalies     이상 입찰 분석 요청   (92)
    POST /internal/v1/ai/recommendations   Live 추천 생성 요청   (94)

**둘 다 비동기다.** 요청을 받으면 202 로 접수만 알리고, 분석이 끝나면 요청에 실려
온 `callbackUrl` 로 결과를 POST 한다(93 · 95번). 명세가 정한 구조이며, 경매 종료
직후 수십 건이 몰려도 백엔드가 응답을 기다리며 묶이지 않는다는 실익이 있다.

기존 동기 엔드포인트(`/internal/fraud/detect`, `/internal/reco/home`)는 그대로 둔다.
**계약은 이쪽이고, 그쪽은 개발·시연용이다.** 콜백을 받아 줄 백엔드 없이 결과를
눈으로 보려면 동기 쪽이 필요하다.

## 접수 전에 막는 것과 접수 후에 처리하는 것

202 를 준 뒤에는 호출자에게 실패를 알릴 방법이 없다. 그래서 **미리 알 수 있는 실패는
전부 202 이전에** 잡는다.

    INVALID_PAYLOAD     본문 형식 · 콜백 주소가 잘못됨
    MODEL_UNAVAILABLE   설정 · 조회기 · 경매 데이터가 없어 분석 자체가 불가

접수 후에 나는 실패(모델 추론 예외, 콜백 전송 실패)는 로그로만 남는다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dataclasses import replace as dc_replace

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request, status

from fraud import RuleConfig
from fraud.schema import Bid
from fraud.ml_features import FEATURE_VERSION
from fraud_api import main as fraud_app
from fraud_api import queries as fraud_queries
from fraud_api.provider import ProviderError as FraudProviderError
from reco import RecoConfig, reason_for
from reco_api import main as reco_app
from reco_api.provider import ProviderError as RecoProviderError

from . import callback, hmac_auth
from .models import (
    BidAnomalyCallback,
    BidAnomalyFeatures,
    BidAnomalyRequest,
    JobAccepted,
    RecommendationCallback,
    RecommendationItem,
    RecommendationRequest,
)

log = logging.getLogger("internal_api")

router = APIRouter(prefix="/internal/v1/ai", tags=["ai-internal"])

# 이미 접수한 jobId. 백엔드가 같은 작업을 재시도해도 두 번 분석하지 않는다.
#
# **프로세스 메모리다.** 재시작하면 비므로 완벽한 멱등성은 아니다. 같은 분석을 두 번
# 하는 것은 결과가 같아 해롭지 않고, 명세에도 중복 저장을 막는 장치(fraud_label 의
# UNIQUE 제약)가 있어 이 정도로 둔다. 인스턴스를 여러 개 띄우면 Redis 로 옮긴다.
_seen_jobs: set[str] = set()
MAX_TRACKED_JOBS = 10_000


def _accept(job_id: str) -> bool:
    """처음 보는 작업이면 True. 이미 접수한 것이면 False."""
    if job_id in _seen_jobs:
        return False
    if len(_seen_jobs) >= MAX_TRACKED_JOBS:
        _seen_jobs.clear()  # 무한정 쌓이게 두지 않는다. 최악이라도 재분석일 뿐이다
    _seen_jobs.add(job_id)
    return True


def _release(job_id: str) -> None:
    """결과를 못 보냈으면 접수 기록을 지운다.

    **이걸 안 하면 백엔드의 재시도가 무력해진다.** 백엔드는 같은 jobId 로 다시
    보내는데, 우리가 "이미 접수함" 으로 202 만 주고 아무것도 안 하면 결과가 영영
    가지 않는다. 분석은 같은 입력에 같은 답을 내므로 다시 해도 안전하다.
    """
    _seen_jobs.discard(job_id)


async def _verified_body(request: Request) -> bytes:
    """SERVICE_HMAC 을 검증하고 본문 원문을 돌려준다."""
    body = await request.body()
    try:
        hmac_auth.verify(hmac_auth.service_secret(), body, request.headers)
    except hmac_auth.HmacError as exc:
        # 시크릿 미설정은 우리 설정 문제(503)고, 서명 불일치는 호출자 문제(401)다.
        if hmac_auth.SERVICE_SECRET_ENV in str(exc):
            log.error("HMAC 시크릿 미설정 — 내부 API 를 열 수 없습니다")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        log.warning("HMAC 검증 실패: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    return body


def _parse(model, body: bytes):
    """본문을 명세 스키마로 읽는다. 형식이 틀리면 INVALID_PAYLOAD."""
    try:
        return model.model_validate_json(body)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"INVALID_PAYLOAD: {exc}"
        ) from exc


def _check_callback(url: str) -> None:
    try:
        callback.check_url(url)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"INVALID_PAYLOAD: {exc}"
        ) from exc


def _apply_payload_bids(inp, req: "BidAnomalyRequest"):
    """본문에 실려 온 `bids[]` 로 조회기의 입찰 목록을 대체한다.

    **백엔드가 보낸 쪽이 정답이다.** 경매가 방금 끝난 시점이라 우리 조회기가 마지막
    입찰을 아직 못 봤을 수 있다. 비어 있으면 조회기 값을 그대로 쓴다.

    입찰자 정보(`members`)까지 바꾸지는 않는다. 엔진은 입찰 목록에서 입찰자를
    추려내고 `members` 는 "가입 기록이 있는가" 표시에만 쓰므로, 본문에만 있는
    입찰자는 기록 없음으로 처리되어 조용히 틀리지 않는다.

    시각은 경매 쪽 시간대로 **변환한다.** 백엔드는 DB 의 naive `TIMESTAMP` 를
    `Instant` 로 바꿔 UTC 로 보내는데 우리는 `started_at` 을 naive 로 읽으므로,
    tzinfo 만 떼면 입찰이 경매 시작보다 몇 시간 앞으로 계산된다. 그러면 모든
    입찰자의 `Early_Bidding` 과 `Last_Bidding` 이 1.0(최대 위험)이 되는데,
    예외도 로그도 나지 않는다.
    """
    if not req.bids:
        return inp

    def at(moment):
        return fraud_queries.align_to(moment, inp.auction.started_at)

    bids = tuple(
        Bid(
            bid_id=b.bid_id,
            auction_id=req.auction_id,
            member_id=b.member_id,
            amount=b.amount,
            created_at=at(b.created_at),
        )
        for b in req.bids
    )
    return dc_replace(inp, bids=bids)


def _unavailable(reason: str) -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE, f"MODEL_UNAVAILABLE: {reason}"
    )


# --- 92 · 93 이상 입찰 -------------------------------------------------------


@router.post(
    "/bid-anomalies",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="이상 입찰 분석 요청 (명세 92)",
)
async def request_bid_anomaly(
    request: Request, background: BackgroundTasks
) -> JobAccepted:
    """경매 종료 후 백엔드가 호출한다. 결과는 `callbackUrl` 로 간다.

    `bids[]` 가 오면 **그쪽을 입찰 목록으로 쓴다.** 경매가 방금 끝난 시점이라 우리
    조회기가 마지막 입찰을 아직 못 봤을 수 있다.

    다만 **경매 정보·과거 이력·코퍼스 평균은 본문에 없다.** 그쪽은 조회기가 필요하므로,
    조회기가 해당 경매를 모르면 `MODEL_UNAVAILABLE` 로 거절한다. 202 를 준 뒤에
    알게 되면 알릴 방법이 없다.
    """
    body = await _verified_body(request)
    req: BidAnomalyRequest = _parse(BidAnomalyRequest, body)
    _check_callback(req.callback_url)

    cfg = fraud_app._state.get("config")
    provider = fraud_app._state.get("provider")
    if cfg is None or provider is None:
        raise _unavailable("탐지 엔진이 준비되지 않았습니다")

    try:
        inp = provider.load(req.auction_id, None)
    except FraudProviderError as exc:
        log.exception("데이터 조회 실패 auction_id=%s", req.auction_id)
        raise _unavailable(str(exc)) from exc
    if inp is None:
        raise _unavailable(f"경매를 찾을 수 없습니다: {req.auction_id}")

    if not _accept(req.job_id):
        log.info("이미 접수한 작업입니다 job_id=%s — 재분석하지 않습니다", req.job_id)
        return JobAccepted(job_id=req.job_id)

    background.add_task(_run_bid_anomaly, req, _apply_payload_bids(inp, req), cfg)
    return JobAccepted(job_id=req.job_id)


def _run_bid_anomaly(req: BidAnomalyRequest, inp, cfg: RuleConfig) -> None:
    """분석하고 93번 콜백을 보낸다. 예외를 밖으로 내보내지 않는다."""
    try:
        model = fraud_app._state.get("model")
        results, _, skipped, _, _ = fraud_app.score_bidders(inp, cfg, model)

        mine = next((r for r in results if r.member_id == req.member_id), None)
        if mine is None:
            # 게이트에 걸려 판정하지 않은 입찰자다. **0 점으로 보내지 않는다** —
            # "판단하지 않음" 을 "위험하지 않음" 으로 저장하면 신규 사용자일수록
            # 안전해 보이는 왜곡이 생긴다.
            # 보낼 값이 없는 것은 실패가 아니다. 재시도해도 결과는 같으므로
            # 접수 기록을 유지해 백엔드가 같은 분석을 반복하지 않게 한다.
            log.info(
                "판정 대상이 아닙니다 job_id=%s member_id=%s 사유=%s",
                req.job_id,
                req.member_id,
                skipped.get(req.member_id, "결과 없음"),
            )
            return

        threshold = cfg.bands.get("high", (0.6, 1.01))[0]
        raw = mine.detail.get("ml_features") or {}
        payload = BidAnomalyCallback(
            job_id=req.job_id,
            auction_id=req.auction_id,
            member_id=req.member_id,
            bid_id=req.bid_id,
            features=BidAnomalyFeatures(
                bidder_tendency=raw.get("Bidder_Tendency"),
                bidding_ratio=raw.get("Bidding_Ratio"),
                last_bidding=raw.get("Last_Bidding"),
                auction_bids=raw.get("Auction_Bids"),
                starting_price_average=raw.get("Starting_Price_Average"),
                early_bidding=raw.get("Early_Bidding"),
                winning_ratio=raw.get("Winning_Ratio"),
                auction_duration=None,  # 모듈 문서 참고. 우리 경매 구조에선 의미가 다르다
                rule_score=mine.rule_score,
                ml_score=mine.ml_score,
                risk_score=mine.risk_score,
            ),
            predicted_label=1 if mine.risk_score >= threshold else 0,
            decision_threshold=threshold,
            # **`ml_score` 를 만든 모델**을 적는다. 섀도 모드(`w_ml=0` 인데 모델은
            # 도는 상태)에서도 적는다 — 그 점수가 어느 모델에서 나왔는지 모르면
            # 나중에 두 트랙을 비교할 수 없고, 비교하려고 섀도로 돌리는 것이다.
            #
            # 이 콜백 스키마(명세 93)에는 `weights` 자리가 없다. 대신 `rule_score` ·
            # `ml_score` · `risk_score` 가 따로 담기므로, **`risk_score` 가
            # `rule_score` 와 같으면 모델은 판정에 안 쓰인 것**으로 읽으면 된다.
            model_version=(
                model.version
                if model is not None and mine.ml_score is not None
                else None
            ),
            feature_version=FEATURE_VERSION,
        )
        if not callback.send(
            req.callback_url,
            payload.model_dump(by_alias=True),
            label=f"이상입찰 job_id={req.job_id}",
        ):
            _release(req.job_id)
    except Exception:
        log.exception("이상 입찰 분석 실패 job_id=%s", req.job_id)
        _release(req.job_id)


# --- 94 · 95 추천 -----------------------------------------------------------


@router.post(
    "/recommendations",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Live 추천 결과 생성 요청 (명세 94)",
)
async def request_recommendation(
    request: Request, background: BackgroundTasks
) -> JobAccepted:
    """추천 순서를 만들어 `callbackUrl` 로 보낸다.

    `memberId` 를 주면 **행동 로그로 관심 프로필을 만들어 개인화**한다. 동기
    엔드포인트(`/internal/reco/home`)와 같은 함수를 쓰므로 두 경로의 결과가
    갈라지지 않는다. 로그가 없으면 자동으로 인기순으로 떨어진다.

    `behaviorWindow` 는 아직 쓰지 않는다. 조회 구간은 `config/reco.yaml` 의
    `lookback_days` 가 정한다 — 요청마다 구간이 달라지면 같은 회원의 추천이
    호출자에 따라 바뀐다. 받아만 두고 무시한다.
    """
    body = await _verified_body(request)
    req: RecommendationRequest = _parse(RecommendationRequest, body)
    _check_callback(req.callback_url)

    cfg = reco_app._state.get("config")
    provider = reco_app._state.get("provider")
    if cfg is None or provider is None:
        raise _unavailable("추천 엔진이 준비되지 않았습니다")

    if not _accept(req.job_id):
        log.info("이미 접수한 작업입니다 job_id=%s — 재계산하지 않습니다", req.job_id)
        return JobAccepted(job_id=req.job_id)

    background.add_task(_run_recommendation, req, cfg, provider)
    return JobAccepted(job_id=req.job_id)


def _run_recommendation(req: RecommendationRequest, cfg: RecoConfig, provider) -> None:
    try:
        now = datetime.now(timezone.utc)
        try:
            candidates = provider.load_active(now, reco_app.CANDIDATE_POOL)
        except RecoProviderError:
            log.exception("후보 조회 실패 job_id=%s", req.job_id)
            _release(req.job_id)
            return

        # 후보를 지정해 왔으면 그 안에서만 고른다. 백엔드가 이미 노출 정책으로
        # 걸러 낸 목록일 수 있으므로 우리가 임의로 넓히지 않는다.
        if req.candidate_auction_ids:
            allowed = set(req.candidate_auction_ids)
            candidates = [c for c in candidates if c.auction_id in allowed]

        # 동기 엔드포인트와 **같은 함수**를 쓴다. 두 경로가 갈라지면 같은 회원이
        # 문에 따라 다른 추천을 받는다.
        result = reco_app.rank_for_member(
            provider, cfg, candidates, now, req.member_id
        )
        payload = RecommendationCallback(
            job_id=req.job_id,
            member_id=req.member_id,
            items=[
                RecommendationItem(
                    auction_id=s.auction_id,
                    score=round(s.score, 4),
                    reason=reason_for(s, cfg),
                )
                for s in result.items
            ],
        )
        if not callback.send(
            req.callback_url,
            payload.model_dump(by_alias=True),
            label=f"추천 job_id={req.job_id}",
        ):
            _release(req.job_id)
    except Exception:
        log.exception("추천 생성 실패 job_id=%s", req.job_id)
        _release(req.job_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """단독 실행용. 탐지·추천 상태를 함께 띄운다."""
    async with fraud_app.lifespan(app), reco_app.lifespan(app):
        yield


app = FastAPI(
    title="DIB AI 내부 API",
    version="0.1.0",
    lifespan=lifespan,
    description="REST API 명세서 92 · 94번. 백엔드가 SERVICE_HMAC 으로 호출한다.",
)
app.include_router(router)
