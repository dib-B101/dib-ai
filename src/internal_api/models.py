"""REST API 명세서 92~95번의 요청·응답 스키마.

**필드 이름이 camelCase 다.** 우리 내부 코드는 snake_case 를 쓰지만 명세가 정한
계약이 camelCase 이므로, 경계에서만 이름을 바꾼다. `alias_generator` 가 이 변환을
맡으므로 파이썬 쪽 코드는 평소대로 snake_case 로 읽고 쓴다.

명세에 적히지 않은 것은 여기서 정하지 않고 **받아만 둔다.** `behaviorWindow` 처럼
아직 쓰지 않는 값도 필드로 두는 이유는, 백엔드가 보내는 것을 우리가 거절하면
연동이 깨지기 때문이다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Spec(BaseModel):
    """명세 계약용 기본 모델. camelCase 로 주고받는다."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",  # 백엔드가 필드를 더 보내도 거절하지 않는다
    )


# --- 92 · 93 이상 입찰 -------------------------------------------------------


class BidPayload(Spec):
    """`bids[]` 한 건. `bid` 테이블 컬럼 그대로다.

    **입찰 목록을 요청 본문으로 받는 것이 중요하다.** 이 값이 오면 입찰 조회를 위해
    우리가 DB 를 붙을 필요가 없다. 다만 경매 정보·과거 이력·코퍼스 평균은 본문에
    없으므로 그쪽은 여전히 조회기가 필요하다.
    """

    bid_id: int
    member_id: int
    amount: int
    created_at: datetime


class BidAnomalyRequest(Spec):
    job_id: str = Field(..., description="백엔드가 만든 작업 ID. 재시도 시 같은 값")
    auction_id: int
    member_id: int = Field(..., description="분석 대상 입찰자")
    bid_id: int | None = Field(None, description="분석을 촉발한 입찰. 결과에 그대로 돌려준다")
    bids: list[BidPayload] = Field(default_factory=list, description="이 경매의 전체 입찰")
    callback_url: str = Field(..., description="결과를 POST 할 주소")


class JobAccepted(Spec):
    job_id: str
    status: Literal["ACCEPTED"] = "ACCEPTED"


class BidAnomalyFeatures(Spec):
    """93번 콜백의 `features`. **명세의 11개 키를 그대로 채운다.**

    `auction_duration` 만 항상 `null` 이다. eBay 는 경매 기간을 1~10 '일' 로
    기록하는데 우리는 라이브(분)와 일반 경매(판매자 자유 설정)가 섞여, 이 값이
    '긴 경매인가' 가 아니라 '경매 유형' 을 가리키게 된다. 학습 때 없던 의미라
    모델에 넣지 않았고, 없는 값을 0 으로 채우면 '가장 짧은 경매' 라는 뜻이 되므로
    `null` 로 보낸다.
    """

    bidder_tendency: float | None = None
    bidding_ratio: float | None = None
    last_bidding: float | None = None
    auction_bids: float | None = None
    starting_price_average: float | None = None
    early_bidding: float | None = None
    winning_ratio: float | None = None
    auction_duration: float | None = Field(
        None, description="**항상 null.** 우리 경매 구조에서는 학습 때와 의미가 달라진다"
    )
    rule_score: float | None = None
    ml_score: float | None = None
    risk_score: float | None = None


class BidAnomalyCallback(Spec):
    job_id: str
    auction_id: int
    member_id: int
    bid_id: int | None = None
    features: BidAnomalyFeatures
    predicted_label: int = Field(..., description="0 정상 · 1 의심. decisionThreshold 기준")
    decision_threshold: float
    model_version: str | None = None
    feature_version: str


# --- 94 · 95 추천 -----------------------------------------------------------


class RecommendationRequest(Spec):
    job_id: str
    member_id: int
    scope: Literal["ALL", "LIVE", "GENERAL"] = Field(
        "ALL",
        description=(
            "추천 후보 범위. `GENERAL`은 일반 경매, `LIVE`는 라이브에 편성된 경매, "
            "`ALL`은 전체 진행 중 경매다"
        ),
    )
    behavior_window: Any | None = Field(
        None,
        description=(
            "행동 로그 조회 구간. **아직 쓰지 않는다** — 개인화(STEP 5) 전이라 "
            "누가 요청하든 같은 순서가 나온다. 받아만 두고 무시한다"
        ),
    )
    candidate_auction_ids: list[int] = Field(
        default_factory=list, description="이 안에서만 고른다. 비면 진행 중 경매 전체"
    )
    callback_url: str


class RecommendationItem(Spec):
    auction_id: int
    score: float
    reason: str = Field(..., description="왜 위로 올라왔는지 한 문장")


class RecommendationCallback(Spec):
    job_id: str
    member_id: int
    items: list[RecommendationItem] = Field(default_factory=list)
