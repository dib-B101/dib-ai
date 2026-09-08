"""규칙 구현.

각 규칙은 순수 함수이며 (점수, 사유) 또는 (판단 불가, 사유) 를 돌려준다.
점수는 항상 0.0 ~ 1.0 이고, 사유에는 실제 수치가 들어간다 — 관리자 검토와
이의제기 대응에서 "왜 그 점수인지" 를 설명할 수 있어야 하기 때문이다.

R5(동일 환경 다계정)는 device fingerprint / IP 를 수집하지 않아 구현하지 않았다.
설정에서 비활성 상태이며, 수집이 결정되면 함수를 추가하고 enabled 만 켜면 된다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from .config import RuleSpec
from .features import (
    account_age_days,
    bidder_ids,
    bidding_ratio,
    clamp,
    dominance_ceiling,
    late_surge,
    max_bid_multiple,
    reclaim_latencies,
    robust_dispersion,
    seller_concentration,
)
from .schema import Bid, DetectionInput, HistoryEntry, Member


@dataclass(frozen=True, slots=True)
class RuleContext:
    inp: DetectionInput
    member_id: int
    ordered: tuple[Bid, ...]
    history: tuple[HistoryEntry, ...]
    member: Member | None
    spec: RuleSpec


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    score: float | None   # None 이면 데이터 부족으로 판단하지 않음 (0점과 다르다)
    reason: str


# ---------------------------------------------------------------- 시간 스케일

def scaled_seconds(auction_time: int, ratio: float, floor: float, cap: float) -> float:
    """경매 길이에 비례하는 시간 임계값을 초 단위로 계산한다.

    라이브 경매는 몇 분 단위인데 eBay 데이터는 며칠 단위였다. 절대 초로 임계값을
    박아두면 경매 길이가 바뀔 때마다 의미가 달라진다 — 3분 경매에서 "60초 안에
    재탈환" 은 거의 모든 입찰이 해당되어 변별력이 사라지고, "마지막 5분" 은
    경매 전체를 가리키게 된다.

    비율만 쓰면 반대로 긴 경매에서 무의미해지므로 (5일 경매의 5% 는 6시간이다)
    floor 와 cap 으로 양쪽을 막는다.
    """
    return min(max(auction_time * ratio, floor), cap)


def fmt_seconds(sec: float) -> str:
    """사유 문구용. 60초 미만은 초로, 그 이상은 분으로 적는다."""
    return f"{sec:.0f}초" if sec < 60 else f"{sec / 60:.0f}분"


# ---------------------------------------------------------------- R1

def r1_reclaim_speed(ctx: RuleContext) -> RuleOutcome:
    """최고가를 빼앗긴 뒤 되찾기까지의 속도와 균일성.

    원 티켓의 "짧은 시간의 반복 입찰" 을 우리 정책에 맞게 재정의한 것이다.
    최고 입찰자는 추가 입찰이 불가하므로 A→A→A 가 구조적으로 나올 수 없고,
    실제로 관측 가능한 것은 재탈환 속도뿐이다.
    """
    p = ctx.spec
    min_reclaims = int(p.p("min_reclaims", 2))
    latencies = reclaim_latencies(ctx.ordered, ctx.member_id)

    if len(latencies) < min_reclaims:
        return RuleOutcome(None, f"재탈환 {len(latencies)}회 — 최소 {min_reclaims}회 미만")

    fast_seconds = scaled_seconds(
        ctx.inp.auction.auction_time,
        float(p.p("fast_ratio", 0.05)),
        float(p.p("min_fast_seconds", 5)),
        float(p.p("max_fast_seconds", 120)),
    )
    median_lat = statistics.median(latencies)
    speed = clamp(1.0 - median_lat / fast_seconds) if fast_seconds > 0 else 0.0

    dispersion = robust_dispersion(latencies)
    threshold = float(p.p("dispersion_threshold", 1.0))
    uniformity = clamp(1.0 - dispersion / threshold) if threshold > 0 else 0.0

    # 균일성은 가산항이 아니라 속도의 배수다.
    # 느리게 재탈환하면 아무리 규칙적이어도 위험하지 않다 — 그냥 성실한 구매자다.
    influence = clamp(float(p.p("uniformity_influence", 0.3)))
    score = speed * (1.0 - influence + influence * uniformity)

    return RuleOutcome(
        clamp(score),
        f"최고가를 {len(latencies)}회 빼앗긴 뒤 중앙값 {median_lat:.1f}초 만에 재입찰 "
        f"(기준 {fmt_seconds(fast_seconds)}, 간격 산포도 {dispersion:.2f})",
    )


# ---------------------------------------------------------------- R2

def r2_bid_dominance(ctx: RuleContext) -> RuleOutcome:
    """한 경매에서 입찰을 얼마나 독식했는가.

    원시 비율을 그대로 쓰지 않는다. 우리 정책상 상한이 (n+1)/2n 으로 막혀 있어
    eBay 처럼 1.0 에 근접할 수 없기 때문이다. 균등 참여값과 상한 사이 어디에
    위치하는지로 정규화한다.
    """
    bids = ctx.inp.bids
    n_bids = len(bids)
    n_bidders = len(bidder_ids(bids))
    if n_bids == 0 or n_bidders == 0:
        return RuleOutcome(None, "입찰 없음")

    ratio = bidding_ratio(bids, ctx.member_id)
    fair = 1.0 / n_bidders
    ceiling = dominance_ceiling(n_bids)

    if ceiling <= fair:
        return RuleOutcome(
            None, f"입찰 {n_bids}건 / 입찰자 {n_bidders}명 — 상한과 균등값이 같아 판단 불가"
        )

    score = clamp((ratio - fair) / (ceiling - fair))
    mine = sum(1 for b in bids if b.member_id == ctx.member_id)
    return RuleOutcome(
        score,
        f"전체 {n_bids}건 중 {mine}건({ratio * 100:.1f}%)을 차지 — "
        f"균등 참여 시 {fair * 100:.1f}%, 정책 상한 {ceiling * 100:.1f}%",
    )


# ---------------------------------------------------------------- R3

def r3_new_account_high_bid(ctx: RuleContext) -> RuleOutcome:
    """신규 계정이 시작가 대비 얼마나 공격적으로 올렸는가.

    신규성과 공격성을 곱하므로 둘 다 커야 점수가 남는다.
    신규지만 소액이거나, 고액이지만 오래된 계정이면 0 에 가깝다.
    """
    if ctx.member is None:
        return RuleOutcome(None, "회원 정보 없음")

    p = ctx.spec
    new_days = float(p.p("new_account_days", 7))
    high_multiple = float(p.p("high_multiple", 3.0))

    days = account_age_days(ctx.member.created_at, ctx.inp.as_of)
    multiple = max_bid_multiple(ctx.inp.bids, ctx.member_id, ctx.inp.auction.start_price)

    newness = clamp(1.0 - days / new_days) if new_days > 0 else 0.0
    aggressiveness = clamp((multiple - 1.0) / (high_multiple - 1.0)) if high_multiple > 1 else 0.0
    score = clamp(newness * aggressiveness)

    return RuleOutcome(
        score,
        f"가입 {days:.1f}일차 계정이 시작가의 {multiple:.2f}배까지 입찰",
    )


# ---------------------------------------------------------------- R4

def r4_seller_concentration(ctx: RuleContext) -> RuleOutcome:
    """구독하지 않은 판매자에게 반복해서 몰리는가.

    원래는 "특정 판매자 편중" 만 봤는데, 라이브 방송 구독 모델에서는 그것만으로
    판단할 수 없다. 구독자가 좋아하는 방송자의 경매에만 참여하는 것은 **정상 행동**
    이며, 편중도만 보면 충성 고객이 그대로 고위험으로 찍힌다.

    그래서 구독 관계가 있으면 이 규칙은 판단하지 않는다. 0 점이 아니라 skip 이다 —
    "구독했으니 안전하다" 가 아니라 "이 신호로는 판단할 수 없다" 이기 때문이다.
    가중 평균의 분모에서도 빠져 다른 규칙이 온전한 무게를 갖는다.

    한계: 공모자가 위장하려고 구독할 수 있다. 다만 구독은 공개 관계라 관리자가
    확인할 수 있고, 핑퐁·재탈환 같은 다른 규칙은 그대로 작동한다.
    detail 의 flags 에 구독 여부를 남겨 관리자가 판단할 수 있게 한다.
    """
    p = ctx.spec
    min_auctions = int(p.p("min_auctions", 3))
    history = ctx.history
    seller_id = ctx.inp.auction.seller_id

    if p.p("skip_if_subscribed", True) and ctx.inp.is_subscribed(ctx.member_id, seller_id):
        return RuleOutcome(None, "구독 중인 판매자 — 편중은 정상 행동이므로 판단하지 않음")

    if len(history) < min_auctions:
        return RuleOutcome(
            None, f"과거 참여 {len(history)}건 — 최소 {min_auctions}건 미만"
        )

    concentration = seller_concentration(history, seller_id)
    reliable = int(p.p("reliable_auctions", 8))
    confidence = clamp(len(history) / reliable) if reliable > 0 else 1.0
    score = clamp(concentration * confidence)

    same = sum(1 for h in history if h.seller_id == seller_id)
    return RuleOutcome(
        score,
        f"최근 {len(history)}개 참여 경매 중 {same}개({concentration * 100:.1f}%)가 동일 판매자",
    )


# ---------------------------------------------------------------- R6

def r6_late_surge(ctx: RuleContext) -> RuleOutcome:
    """최초 예정 종료 직전의 급격한 가격 상승과 그 기여분.

    end_at 이 아니라 original_end_at 기준이다. 30초 연장 정책 때문에 end_at 은
    입찰이 들어올 때마다 움직여서, 같은 입력에 같은 결과가 나오지 않는다.
    """
    p = ctx.spec
    window = scaled_seconds(
        ctx.inp.auction.auction_time,
        float(p.p("window_ratio", 0.15)),
        float(p.p("min_window_seconds", 10)),
        float(p.p("max_window_seconds", 600)),
    )
    threshold = float(p.p("surge_threshold", 0.30))

    stat = late_surge(ctx.ordered, ctx.inp.auction, window, ctx.member_id)
    if stat["window_bids"] == 0:
        return RuleOutcome(None, f"최초 마감 {fmt_seconds(window)} 이내 입찰 없음")

    surge_norm = clamp(stat["surge_ratio"] / threshold) if threshold > 0 else 0.0
    score = clamp(surge_norm * stat["contribution"])

    return RuleOutcome(
        score,
        f"최초 마감 {fmt_seconds(window)} 전 가격이 {stat['surge_ratio'] * 100:.1f}% 상승했고 "
        f"그중 {stat['contribution'] * 100:.1f}%를 이 입찰자가 올림",
    )


# ---------------------------------------------------------------- 레지스트리

RULE_FUNCS: dict[str, Callable[[RuleContext], RuleOutcome]] = {
    "R1_RECLAIM_SPEED": r1_reclaim_speed,
    "R2_BID_DOMINANCE": r2_bid_dominance,
    "R3_NEW_ACCOUNT_HIGH_BID": r3_new_account_high_bid,
    "R4_SELLER_CONCENTRATION": r4_seller_concentration,
    # "R5_MULTI_ACCOUNT": 미구현 — device fingerprint / IP 미수집
    "R6_LATE_SURGE": r6_late_surge,
}
