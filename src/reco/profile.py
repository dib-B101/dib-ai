"""사용자 관심 프로필 (추천 STEP 4·5).

행동 로그를 모아 **그 사람이 좋아할 만한 상품의 벡터**를 만든다. 그리고 후보
상품과의 유사도를 재서 개인화 점수로 쓴다.

    행동 로그          입찰 · 찜 · 조회 ...
        ↓ 행동 가중치   입찰이 조회보다 강한 신호다
        ↓ 시간 감쇠     지난달 관심보다 어제 관심이 크다
    가중 평균한 벡터    = 관심 프로필
        ↓
    후보 상품과 유사도  → rank() 의 similarity 로 들어간다

**DB 를 모른다.** 이벤트와 벡터를 받아 계산만 한다. 조회는 호출자가 맡는다.

## 로그가 없어도 깨지지 않는다

지금 우리 서비스에는 행동 로그가 거의 없다. 그래서 **프로필을 만들 수 없는 경우를
정상 경로로** 다룬다 — `None` 을 돌려주면 호출자가 인기순으로 떨어진다.

프로필을 못 만드는 경우는 세 가지다.

    이벤트가 없다                  신규 사용자
    이벤트의 상품에 임베딩이 없다    배치가 아직 안 돔
    가중치 합이 0 이하다            음의 신호만 남았다

셋 다 **0 벡터를 만들지 않는다.** 0 벡터는 모든 상품과 유사도 0 이 되어
"아무것도 안 좋아함" 으로 읽히는데, 그건 "모름" 과 다르다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .schema import ProductVector
from .similarity import compare

# 행동별 관심 가중치 기본값. `config/reco.yaml` 에서 덮어쓴다.
#
# 구매 의사에 가까울수록 크게 준다. 입찰은 돈을 거는 행동이라 가장 강하고,
# 조회는 지나가다 누른 것일 수 있어 가장 약하다.
#
# **음의 신호는 조심해서 쓴다.** 찜 해제(UNWATCH)는 이전 찜을 되돌리는 명시적
# 행동이라 빼는 것이 맞다. 반면 스크롤로 지나친 것(SCROLL_PASS)은 관심이 없어서일
# 수도, 이미 아는 상품이어서일 수도 있다. 한 번 지나쳤다고 싫어한다고 보면
# 오탐이 쌓이므로 아주 작게만 준다.
DEFAULT_EVENT_WEIGHTS: Mapping[str, float] = {
    "BID": 6.0,
    "PURCHASE": 6.0,
    "WATCH": 4.0,       # 찜
    "SEARCH": 2.0,
    "CLICK": 2.0,
    "SHARE": 2.0,
    "VIEW": 1.0,
    "IMPRESSION": 0.0,  # 노출됐을 뿐. 관심의 증거가 아니다
    "SCROLL_PASS": -0.5,
    "UNWATCH": -4.0,    # 찜 해제. 같은 크기로 되돌린다
}

# 관심이 절반으로 줄어드는 기간(일).
#
# 중고 거래는 필요가 생겼다 사라지는 주기가 짧다. 한 달 전에 냉장고를 찾아본
# 사람에게 지금도 냉장고를 보여주면 이미 산 물건을 계속 권하는 셈이 된다.
DEFAULT_HALF_LIFE_DAYS = 14.0


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """`member_event` 테이블 1행 중 추천에 쓰는 것만.

    `auction_id` 를 키로 쓴다. 벡터 저장소가 경매 단위라 그래야 바로 맞물린다.
    이벤트에 경매가 없으면(검색 등) 프로필에 넣을 상품을 특정할 수 없어 버린다.
    """

    event_type: str
    auction_id: int
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class UserProfile:
    """관심 벡터. 텍스트와 이미지를 따로 들고 있다.

    하나로 합치지 않는 이유는 상품 벡터와 같다 — 차원도 축의 의미도 다르다.
    이미지 프로필은 `None` 일 수 있다. 행동한 상품에 사진이 하나도 없던 경우다.
    """

    member_id: int
    text: tuple[float, ...]
    image: tuple[float, ...] | None = None

    # 프로필을 만든 근거. 관리자 화면과 튜닝에 쓴다.
    event_count: int = 0
    total_weight: float = 0.0

    # 이미 관심을 보인 경매. **추천 후보에서 빼야 한다.**
    # 입찰한 경매를 "이런 상품은 어때요" 로 다시 보여주는 것은 추천이 아니다.
    seen: frozenset[int] = frozenset()


def decay(age_days: float, half_life_days: float) -> float:
    """시간 감쇠 계수 (0, 1]. 반감기마다 절반이 된다.

    선형 감쇠를 쓰지 않는 이유는 **오래된 행동이 어느 순간 0 이 되어 버리기**
    때문이다. 지수 감쇠는 작아지되 사라지지 않아, 로그가 적은 초기에도 쓸 수 있다.
    """
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(age_days, 0.0) / half_life_days)


def _weighted(
    events: Sequence[BehaviorEvent],
    now: datetime,
    weights: Mapping[str, float],
    half_life_days: float,
) -> dict[int, float]:
    """경매별 최종 가중치. 같은 상품에 여러 행동이 있으면 더한다.

    더하는 것이 맞다 — 보고(VIEW) 찜하고(WATCH) 입찰한(BID) 상품은 셋 중 하나만
    한 상품보다 관심이 크다.
    """
    out: dict[int, float] = {}
    for e in events:
        base = weights.get(e.event_type)
        if not base:
            continue  # 모르는 이벤트이거나 가중치 0. 조용히 건너뛴다
        age_days = (now - e.occurred_at).total_seconds() / 86400.0
        out[e.auction_id] = out.get(e.auction_id, 0.0) + base * decay(age_days, half_life_days)
    return out


def _blend_vectors(
    weights: Mapping[int, float], vectors: Mapping[int, ProductVector], attr: str
) -> tuple[float, ...] | None:
    """가중 평균한 뒤 L2 정규화. 쓸 벡터가 없으면 None."""
    dim = 0
    acc: list[float] = []
    used = 0.0

    for auction_id, w in weights.items():
        vec = vectors.get(auction_id)
        values = getattr(vec, attr, None) if vec else None
        if not values:
            continue
        if not acc:
            dim = len(values)
            acc = [0.0] * dim
        elif len(values) != dim:
            continue  # 차원이 다른 벡터는 섞지 않는다. 모델이 바뀐 흔적이다
        for i, v in enumerate(values):
            acc[i] += w * v
        used += w

    if not acc or used <= 0:
        return None

    norm = math.sqrt(sum(v * v for v in acc))
    if norm == 0:
        return None  # 양·음 신호가 정확히 상쇄됐다. 방향이 없으므로 모른다
    return tuple(v / norm for v in acc)


def build_profile(
    member_id: int,
    events: Sequence[BehaviorEvent],
    vectors: Mapping[int, ProductVector],
    now: datetime,
    weights: Mapping[str, float] | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> UserProfile | None:
    """관심 프로필. 만들 수 없으면 None 이고, 그것이 정상 경로다.

    `None` 을 받은 호출자는 인기순으로 떨어진다. **0 벡터를 만들지 않는다** —
    모든 상품과 유사도 0 이 되어 "아무것도 안 좋아함" 으로 읽히는데, 그건 "모름"
    과 다르다.
    """
    if not events:
        return None

    weighted = _weighted(events, now, weights or DEFAULT_EVENT_WEIGHTS, half_life_days)
    positive = {a: w for a, w in weighted.items() if w > 0}
    if not positive:
        return None  # 음의 신호만 남았다. 무엇을 좋아하는지 알 수 없다

    text = _blend_vectors(positive, vectors, "text")
    if text is None:
        return None  # 행동한 상품에 임베딩이 없다. 배치가 아직 안 돌았다

    return UserProfile(
        member_id=member_id,
        text=text,
        image=_blend_vectors(positive, vectors, "image"),
        event_count=len(events),
        total_weight=sum(positive.values()),
        seen=frozenset(weighted),
    )


def affinity(
    profile: UserProfile,
    candidates: Sequence[ProductVector],
    w_text: float,
) -> dict[int, float]:
    """프로필과 후보 상품의 유사도 [0, 1].

    **이미 관심을 보인 경매는 뺀다.** 입찰하거나 찜한 상품은 그 사람이 이미 아는
    상품이고, 마이페이지에서 따로 본다. 추천 목록에 다시 올리면 자리만 차지한다.
    프로필이 그 상품들의 평균이라 유사도도 당연히 최상위로 나와, 안 빼면 **추천
    목록이 이미 본 상품으로 채워진다.**

    유사 상품 추천과 **같은 결합 방식**을 쓴다 — 각각 표준화한 뒤 가중합하고
    마지막에 백분위로 편다. 두 경로가 갈라지면 한쪽만 고쳐지는 사고가 난다.
    """
    fresh = [v for v in candidates if v.auction_id not in profile.seen]
    return compare(profile.text, profile.image, fresh, w_text)
