"""합성 시나리오.

실제 DB 없이 규칙을 개발·검증하기 위한 데이터다.
각 시나리오는 "어떤 규칙이 반응해야 하는가" 를 하나씩 겨냥한다.

주의: 우리 정책상 최고 입찰자는 연속으로 입찰할 수 없으므로,
      입찰 시퀀스에 같은 회원이 연달아 나오면 안 된다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fraud import Auction, Bid, DetectionInput, HistoryEntry, Member

T0 = datetime(2026, 8, 20, 10, 0, 0)
SELLER = 1000
OTHER_SELLER = 2000
AUCTION_ID = 500
START_PRICE = 10_000
DURATION_HOURS = 6


def make_auction(
    auction_id: int = AUCTION_ID,
    seller_id: int = SELLER,
    start_price: int = START_PRICE,
    duration_hours: int = DURATION_HOURS,
) -> Auction:
    return Auction(
        auction_id=auction_id,
        seller_id=seller_id,
        category_id=7,
        start_price=start_price,
        started_at=T0,
        auction_time=duration_hours * 3600,
        ended_at=T0 + timedelta(hours=duration_hours),
    )


def make_bids(
    sequence: list[int],
    offsets_sec: list[int],
    *,
    auction_id: int = AUCTION_ID,
    start_price: int = START_PRICE,
    step: int = 1_000,
) -> tuple[Bid, ...]:
    """입찰 시퀀스를 만든다.

    sequence     : 입찰한 회원 id 를 순서대로
    offsets_sec  : 경매 시작 이후 몇 초에 입찰했는가 (오름차순이어야 한다)
    """
    if len(sequence) != len(offsets_sec):
        raise ValueError("sequence 와 offsets_sec 의 길이가 달라야 하지 않습니다")
    for a, b in zip(sequence, sequence[1:]):
        if a == b:
            raise ValueError("정책상 같은 회원이 연속으로 입찰할 수 없습니다")

    bids = []
    amount = start_price
    for i, (member_id, off) in enumerate(zip(sequence, offsets_sec), start=1):
        amount += step
        bids.append(
            Bid(
                bid_id=auction_id * 100 + i,
                auction_id=auction_id,
                member_id=member_id,
                amount=amount,
                created_at=T0 + timedelta(seconds=off),
            )
        )
    return tuple(bids)


def make_members(ids: list[int], *, age_days: float = 365.0) -> dict[int, Member]:
    return {
        m: Member(member_id=m, created_at=T0 - timedelta(days=age_days)) for m in ids
    }


def make_history(
    member_id: int, sellers: list[int], *, days_ago_start: int = 60
) -> tuple[HistoryEntry, ...]:
    """과거 참여 이력. sellers 순서대로 한 건씩 만든다."""
    return tuple(
        HistoryEntry(
            auction_id=9000 + i,
            seller_id=s,
            participated_at=T0 - timedelta(days=days_ago_start - i),
            won=False,
        )
        for i, s in enumerate(sellers)
    )


# ---------------------------------------------------------------- 시나리오

def normal_auction() -> DetectionInput:
    """정상. 5명이 불규칙한 간격으로 고르게 경쟁한다.

    각자 3번씩 입찰해 재탈환이 2회 생긴다. R1 이 게이트에 걸리지 않고
    실제로 "느린 재탈환" 을 채점하도록 하기 위함이다.
    """
    seq = [11, 12, 13, 14, 15] * 3
    offs = [60, 400, 900, 1500, 2200,
            3000, 4100, 4600, 6000, 7200,
            9000, 11000, 12500, 15000, 18000]
    members = make_members(seq)
    histories = {
        m: make_history(m, [OTHER_SELLER, OTHER_SELLER + 1, OTHER_SELLER + 2, SELLER])
        for m in set(seq)
    }
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


def pingpong_auction(duration_hours: int = DURATION_HOURS) -> DetectionInput:
    """의심. 21·22 두 계정이 3초 간격으로 균일하게 주고받는다 → R1."""
    seq = [21, 22, 21, 22, 21, 22, 21, 22, 23, 21]
    offs = [60, 63, 66, 69, 72, 75, 78, 81, 2000, 2003]
    members = make_members(seq)
    histories = {m: make_history(m, [OTHER_SELLER] * 4) for m in set(seq)}
    return DetectionInput(
        auction=make_auction(duration_hours=duration_hours),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=duration_hours),
    )


def dominant_bidder_auction() -> DetectionInput:
    """의심. 31 이 전체 입찰의 절반을 차지한다 → R2."""
    seq = [31, 32, 31, 33, 31, 34, 31, 35, 31, 32]
    offs = [60, 500, 1000, 1600, 2200, 2900, 3600, 4300, 5000, 5700]
    members = make_members(seq)
    histories = {m: make_history(m, [OTHER_SELLER] * 4) for m in set(seq)}
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


def new_account_auction() -> DetectionInput:
    """의심. 41 은 가입 0.5일차인데 시작가의 4배까지 올린다 → R3."""
    seq = [41, 42, 41, 43, 41, 44, 41, 42]
    offs = [60, 500, 1000, 1600, 2200, 2900, 3600, 4300]
    members = make_members(seq, age_days=365.0)
    members[41] = Member(member_id=41, created_at=T0 - timedelta(hours=12))
    histories = {m: make_history(m, [OTHER_SELLER] * 4) for m in set(seq)}
    return DetectionInput(
        auction=make_auction(start_price=2_000),  # step 1000 이라 8번째면 10배
        bids=make_bids(seq, offs, start_price=2_000),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


def subscriber_auction() -> DetectionInput:
    """정상. 51 은 판매자를 구독한 충성 고객이다 → R4 는 판단하지 않아야 한다.

    라이브 구독 모델에서 구독자가 그 방송자 경매에만 참여하는 것은 정상 행동이다.
    편중도만 보면 고위험으로 찍히므로 구독 관계를 확인해야 한다.
    """
    seq = [51, 52, 51, 53, 51, 54, 52, 51]
    offs = [30, 55, 90, 130, 175, 220, 260, 300]
    members = make_members(seq)
    histories = {m: make_history(m, [SELLER] * 8) for m in set(seq)}
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
        subscriptions={m: frozenset({SELLER}) for m in set(seq)},
    )


def loyal_bidder_auction() -> DetectionInput:
    """의심. 51 은 최근 10경매가 전부 같은 판매자다 → R4."""
    seq = [51, 52, 53, 51, 54, 52, 51, 55]
    offs = [60, 500, 1000, 1600, 2200, 2900, 3600, 4300]
    members = make_members(seq)
    histories = {m: make_history(m, [OTHER_SELLER] * 5) for m in set(seq)}
    histories[51] = make_history(51, [SELLER] * 10)
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


def late_surge_auction() -> DetectionInput:
    """의심. 61 이 최초 마감 5분 전에 가격을 크게 밀어올린다 → R6."""
    end_sec = DURATION_HOURS * 3600
    seq = [61, 62, 63, 61, 62, 61, 62, 61]
    offs = [
        60, 500, 1000,                       # 초반
        end_sec - 240, end_sec - 200,        # 마감 4분 전부터
        end_sec - 150, end_sec - 100, end_sec - 40,
    ]
    members = make_members(seq)
    histories = {m: make_history(m, [OTHER_SELLER] * 4) for m in set(seq)}
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs, step=8_000),   # 큰 폭으로 올린다
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


def tiny_auction() -> DetectionInput:
    """최소 분석 조건 미달. 입찰자 2명 / 입찰 3건."""
    seq = [71, 72, 71]
    offs = [60, 500, 1000]
    members = make_members(seq)
    histories = {m: () for m in set(seq)}
    return DetectionInput(
        auction=make_auction(),
        bids=make_bids(seq, offs),
        members=members,
        histories=histories,
        as_of=T0 + timedelta(hours=DURATION_HOURS),
    )


ALL_SCENARIOS = {
    "normal": normal_auction,
    "pingpong": pingpong_auction,
    "dominant": dominant_bidder_auction,
    "new_account": new_account_auction,
    "loyal": loyal_bidder_auction,
    "late_surge": late_surge_auction,
    "tiny": tiny_auction,
}
