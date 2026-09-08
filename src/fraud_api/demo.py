"""데모용 합성 경매.

bid 테이블이 생기기 전까지 API 를 실제로 호출해 보기 위한 데이터다.
백엔드가 연동 코드를 짜면서 응답 형태를 눈으로 확인하는 용도이기도 하다.

tests/fixtures.py 와 목적이 겹치지만 그쪽은 테스트 전용이라 배포에 포함되지 않는다.
여기는 앱과 함께 나가므로 최소한만 둔다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fraud import Auction, Bid, DetectionInput, HistoryEntry, Member

T0 = datetime(2026, 9, 8, 14, 0, 0)
SELLER = 1000
OTHER_SELLER = 2000
DURATION_SEC = 180          # 3분. 라이브 경매를 가정한다
START_PRICE = 10_000

NORMAL_AUCTION_ID = 10001
PINGPONG_AUCTION_ID = 10002


def _auction(auction_id: int) -> Auction:
    return Auction(
        auction_id=auction_id,
        seller_id=SELLER,
        category_id=7,
        start_price=START_PRICE,
        started_at=T0,
        auction_time=DURATION_SEC,
        ended_at=T0 + timedelta(seconds=DURATION_SEC),
    )


def _bids(auction_id: int, sequence: list[int], offsets: list[int]) -> tuple[Bid, ...]:
    amount = START_PRICE
    out = []
    for i, (member_id, off) in enumerate(zip(sequence, offsets), start=1):
        amount += 1_000
        out.append(
            Bid(
                bid_id=auction_id * 100 + i,
                auction_id=auction_id,
                member_id=member_id,
                amount=amount,
                created_at=T0 + timedelta(seconds=off),
            )
        )
    return tuple(out)


def _members(ids: list[int], age_days: float = 365.0) -> dict[int, Member]:
    return {m: Member(member_id=m, created_at=T0 - timedelta(days=age_days)) for m in ids}


def _history(sellers: list[int]) -> tuple[HistoryEntry, ...]:
    return tuple(
        HistoryEntry(
            auction_id=9000 + i,
            seller_id=s,
            participated_at=T0 - timedelta(days=30 - i),
            won=False,
        )
        for i, s in enumerate(sellers)
    )


def _normal() -> DetectionInput:
    """정상. 5명이 불규칙하게 경쟁한다."""
    seq = [11, 12, 13, 11, 14, 15, 12, 13]
    offs = [12, 31, 47, 68, 95, 121, 140, 166]
    return DetectionInput(
        auction=_auction(NORMAL_AUCTION_ID),
        bids=_bids(NORMAL_AUCTION_ID, seq, offs),
        members=_members(seq),
        histories={m: _history([OTHER_SELLER] * 4) for m in set(seq)},
        as_of=T0 + timedelta(seconds=DURATION_SEC),
    )


def _pingpong() -> DetectionInput:
    """의심. 21·22 두 계정이 3초 간격으로 균일하게 주고받는다.

    23 은 우연히 끼어든 일반 입찰자다. 다양한 판매자 이력을 줘서
    "이 경매에 참여했다" 는 사실만으로는 점수가 오르지 않는 것을 보인다.
    """
    seq = [21, 22, 21, 22, 21, 22, 21, 22, 23, 21]
    offs = [20, 23, 26, 29, 32, 35, 38, 41, 150, 153]
    concentrated = _history([SELLER] * 6)
    diverse = _history([OTHER_SELLER, 3000, OTHER_SELLER, 4000, 3000, SELLER])
    return DetectionInput(
        auction=_auction(PINGPONG_AUCTION_ID),
        bids=_bids(PINGPONG_AUCTION_ID, seq, offs),
        members=_members(seq),
        histories={21: concentrated, 22: concentrated, 23: diverse},
        as_of=T0 + timedelta(seconds=DURATION_SEC),
    )


def demo_data() -> dict[int, DetectionInput]:
    return {
        NORMAL_AUCTION_ID: _normal(),
        PINGPONG_AUCTION_ID: _pingpong(),
    }
