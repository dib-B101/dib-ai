"""탐지 입력을 만드는 SQL 과 행 변환.

`PostgresProvider` 에서 SQL 을 떼어냈다. **DB 없이 검증할 수 있게 하기 위해서다.**
조회 결과를 자료구조로 옮기는 부분이 실제로 틀리기 쉬운 곳인데, 연결 없이는
테스트가 안 되면 그 부분을 영영 못 본다. 여기 함수들은 전부 dict 를 받아 dict 를
돌려주므로 그냥 부를 수 있다.

## 시점 누수를 막는 규칙

모든 이력 조회에 **`as_of` 이전** 조건을 건다. 빠뜨리면 미래 정보가 섞여, 평가할
때는 잘 맞는 것처럼 보이지만 실제 운영에서는 성능이 나오지 않는다.

    입찰          created_at <= as_of      이 경매의 입찰
    과거 참여     created_at <  as_of      다른 경매의 입찰
    낙찰 여부     ended_at   <  as_of      아직 안 끝난 경매는 낙찰자를 모른다
    구독          created_at <  as_of      경매 뒤에 구독했으면 그때는 남이었다
    코퍼스 평균   ended_at   <  as_of      미래 경매의 평균을 쓸 수 없다

## 판매자와 카테고리는 `product` 에 있다

`auction` 에는 `member_id` 도 `category_id` 도 없다. 둘 다 `product` 를 조인해야
나온다. 조인을 빠뜨리면 판매자 편중도(R4)와 카테고리별 코퍼스가 통째로 틀린다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from fraud import Auction, Bid, CorpusStats, HistoryEntry, Member

# 코퍼스 평균을 믿기 위한 최소 표본 수.
#
# 두세 건의 평균은 값이 아니라 잡음이다. 그 잡음으로 "평균보다 입찰이 많다" 를
# 판정하면 근거 없는 점수가 나간다. 모자라면 **코퍼스를 없는 것으로 친다** —
# 그러면 ML 점수를 내지 않고 규칙 점수만 쓴다. 서비스 초기에는 이쪽이 정상이다.
MIN_CORPUS_AUCTIONS = 20


AUCTION_SQL = """
SELECT a.auction_id,
       p.member_id   AS seller_id,
       p.category_id AS category_id,
       a.start_price,
       a.started_at,
       a.auction_time,
       a.ended_at
FROM auction a
JOIN product p ON p.product_id = a.product_id
WHERE a.auction_id = %(auction_id)s
  AND a.deleted_at IS NULL
"""

BIDS_SQL = """
SELECT bid_id, auction_id, member_id, amount, created_at
FROM bid
WHERE auction_id = %(auction_id)s
  AND created_at <= %(as_of)s
ORDER BY created_at, bid_id
"""

MEMBERS_SQL = """
SELECT member_id, created_at
FROM member
WHERE member_id = ANY(%(member_ids)s)
"""

# 과거 참여 이력.
#
# `mine` 이 각 입찰자가 참여한 다른 경매와 그 경매에서의 입찰 수,
# `totals` 가 그 경매들의 전체 입찰 수다. 둘을 나눠 Winning_Ratio 가 쓰는
# "적극 참여였는가" 를 판단한다.
#
# 낙찰 여부는 **as_of 이전에 끝난 경매에서만** 인정한다. 아직 진행 중인 경매의
# top_bid_id 는 그 시점에 알 수 없는 정보다.
HISTORIES_SQL = """
WITH mine AS (
    SELECT b.member_id,
           b.auction_id,
           MIN(b.created_at) AS participated_at,
           COUNT(*)          AS my_bids
    FROM bid b
    WHERE b.member_id = ANY(%(member_ids)s)
      AND b.auction_id <> %(auction_id)s
      AND b.created_at < %(as_of)s
    GROUP BY b.member_id, b.auction_id
),
totals AS (
    SELECT auction_id, COUNT(*) AS total_bids
    FROM bid
    WHERE auction_id IN (SELECT auction_id FROM mine)
      AND created_at < %(as_of)s
    GROUP BY auction_id
)
SELECT m.member_id,
       m.auction_id,
       p.member_id AS seller_id,
       m.participated_at,
       (a.ended_at IS NOT NULL
        AND a.ended_at < %(as_of)s
        AND w.member_id = m.member_id) AS won,
       m.my_bids::float / NULLIF(t.total_bids, 0) AS bidding_ratio
FROM mine m
JOIN auction a  ON a.auction_id = m.auction_id
JOIN product p  ON p.product_id = a.product_id
JOIN totals  t  ON t.auction_id = m.auction_id
LEFT JOIN bid w ON w.bid_id = a.top_bid_id
ORDER BY m.member_id, m.participated_at
"""

SUBSCRIPTIONS_SQL = """
SELECT subscriber_id, broadcaster_id
FROM subscription
WHERE subscriber_id = ANY(%(member_ids)s)
  AND created_at < %(as_of)s
"""

# 카테고리별 기준값.
#
# 이 경매 자신을 뺀다. 넣으면 자기 값이 자기 기준선을 끌어올려, 붐빈 경매일수록
# "평균과 비슷하다" 로 보이는 역전이 생긴다.
CORPUS_SQL = """
SELECT COUNT(*)                  AS n,
       AVG(a.bid_count)::float   AS mean_bids,
       AVG(a.start_price)::float AS mean_start_price
FROM auction a
JOIN product p ON p.product_id = a.product_id
WHERE p.category_id = %(category_id)s
  AND a.status = 'ENDED'
  AND a.ended_at < %(as_of)s
  AND a.auction_id <> %(auction_id)s
  AND a.deleted_at IS NULL
"""


def to_auction(row: Mapping[str, Any]) -> Auction:
    return Auction(
        auction_id=row["auction_id"],
        seller_id=row["seller_id"],
        category_id=row["category_id"],
        start_price=int(row["start_price"]),
        started_at=row["started_at"],
        auction_time=int(row["auction_time"]),
        ended_at=row["ended_at"],
    )


def to_bids(rows: Sequence[Mapping[str, Any]]) -> tuple[Bid, ...]:
    return tuple(
        Bid(
            bid_id=r["bid_id"],
            auction_id=r["auction_id"],
            member_id=r["member_id"],
            amount=int(r["amount"]),
            created_at=r["created_at"],
        )
        for r in rows
    )


def to_members(rows: Sequence[Mapping[str, Any]]) -> dict[int, Member]:
    return {
        r["member_id"]: Member(member_id=r["member_id"], created_at=r["created_at"])
        for r in rows
    }


def to_histories(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[HistoryEntry, ...]]:
    """입찰자별로 묶는다. 참여한 적이 없으면 키 자체가 없다.

    빈 튜플을 넣어 두는 것과 키가 없는 것은 엔진 입장에서 같지만, 없는 쪽이
    "조회했는데 없었다" 를 그대로 드러낸다.
    """
    out: dict[int, list[HistoryEntry]] = {}
    for r in rows:
        out.setdefault(r["member_id"], []).append(
            HistoryEntry(
                auction_id=r["auction_id"],
                seller_id=r["seller_id"],
                participated_at=r["participated_at"],
                won=bool(r["won"]),
                # NULLIF 로 0 을 걸렀으므로 NULL 이 올 수 있다. 그 경매의 입찰을
                # 한 건도 못 셌다는 뜻이라 비중을 0 으로 둔다 (소극 참여).
                bidding_ratio=float(r["bidding_ratio"] or 0.0),
            )
        )
    return {m: tuple(v) for m, v in out.items()}


def to_subscriptions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, frozenset[int]]:
    out: dict[int, set[int]] = {}
    for r in rows:
        out.setdefault(r["subscriber_id"], set()).add(r["broadcaster_id"])
    return {m: frozenset(v) for m, v in out.items()}


def to_corpus(row: Mapping[str, Any] | None, category_id: int) -> CorpusStats | None:
    """표본이 모자라거나 평균이 없으면 None.

    **0 으로 채우지 않는다.** 0 은 "평균이 0 이다" 라는 주장이고, 그러면
    `Auction_Bids` 가 모든 경매에서 최대값이 된다.
    """
    if row is None or (row.get("n") or 0) < MIN_CORPUS_AUCTIONS:
        return None

    mean_bids = row.get("mean_bids")
    mean_start_price = row.get("mean_start_price")
    if not mean_bids or not mean_start_price:
        return None

    return CorpusStats(
        category_id=category_id,
        mean_bids=float(mean_bids),
        mean_start_price=float(mean_start_price),
    )


def resolve_as_of(requested: datetime | None, auction: Auction) -> datetime:
    """기준 시각을 정하고 경매 시각과 tz 종류를 맞춘다.

    ERD 의 `TIMESTAMP` 에는 시간대가 없어 드라이버가 naive 로 준다. 요청에 실려 온
    `as_of` 는 보통 tz 를 달고 오므로, 섞으면 비교할 때 TypeError 가 난다.
    """
    base = requested or auction.ended_at or datetime.now()
    aware = auction.started_at.tzinfo is not None

    if aware and base.tzinfo is None:
        return base.replace(tzinfo=auction.started_at.tzinfo)
    if not aware and base.tzinfo is not None:
        return base.replace(tzinfo=None)
    return base
