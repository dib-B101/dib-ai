"""추천 후보를 만드는 SQL 과 행 변환.

탐지 쪽(`fraud_api/queries.py`)과 같은 이유로 SQL 을 떼어냈다. **DB 없이 검증할 수
있게 하기 위해서다.** 행을 자료구조로 옮기는 부분이 실제로 틀리기 쉬운데, 연결이
없으면 그 부분을 영영 못 본다.

## 노출하면 안 되는 상품을 여기서 거른다

랭킹은 "무엇을 보여줘도 되는가" 를 모른다. 점수만 매긴다. 그래서 **노출 정책은
조회에서 끝내야 한다.**

    a.status = 'ACTIVE'            진행 중인 경매만
    a.started_at IS NOT NULL       시작 시각이 없으면 남은 시간을 계산할 수 없다
    p.status 검수 통과              PENDING · REJECTED 는 검수를 통과하지 못한 상품이다
    deleted_at IS NULL             삭제된 경매 · 상품

**검수 필터가 특히 중요하다.** 빠뜨리면 거래 제한 품목이 추천 목록에 올라온다.
`REGISTERED` 만 남기면 안 된다 — 경매가 걸린 상품은 `ON_AUCTION` 으로 바뀌므로
그렇게 거르면 후보가 통째로 사라진다. 통과하지 못한 상태를 빼는 방식으로 쓴다.

## 시각은 경계에서 UTC 로 올린다

ERD 의 `TIMESTAMP` 에는 시간대가 없어 드라이버가 naive 로 준다. 랭킹은
`datetime.now(timezone.utc)` 와 비교하므로, 변환하지 않으면 그대로 터진다.
어느 지역 시각으로 해석할지는 `DIB_DB_TIMEZONE` 이 정한다 (`src/dbtime.py`).

## 판매자와 카테고리는 `product` 에 있다

`auction` 에는 `member_id` 도 `category_id` 도 없다. 둘 다 `product` 를 조인해야
나온다. 탐지 쪽과 같은 제약이다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dbtime import to_utc
from reco import BehaviorEvent, Candidate, ProductVector

# 검수를 통과하지 못한 상품 상태. 이 목록에 있으면 추천에 노출하지 않는다.
#
# 화이트리스트(`REGISTERED` 만)로 쓰지 않는 이유는, 경매가 시작되면 상품 상태가
# `ON_AUCTION` 으로 바뀌어 **진행 중인 경매가 전부 빠지기 때문이다.**
HIDDEN_PRODUCT_STATUS = ("PENDING", "REJECTED")

ACTIVE_CANDIDATES_SQL = """
SELECT a.auction_id,
       a.product_id,
       p.member_id   AS seller_id,
       p.category_id AS category_id,
       a.started_at,
       a.auction_time,
       a.ended_at,
       a.live_broadcast_id,
       a.view_count,
       a.bookmark_count,
       a.bid_count,
       a.bidder_count
FROM auction a
JOIN product p ON p.product_id = a.product_id
WHERE a.status = 'ACTIVE'
  AND a.started_at IS NOT NULL
  AND a.deleted_at IS NULL
  AND p.deleted_at IS NULL
  AND p.status <> ALL(%(hidden_status)s)
ORDER BY a.auction_id
LIMIT %(limit)s
"""

# 후보들의 임베딩.
#
# **LEFT JOIN 이 아니라 JOIN 이다.** 벡터가 없는 상품은 행 자체가 빠져야 호출자가
# "아직 임베딩 안 됨" 으로 처리할 수 있다. LEFT JOIN 으로 NULL 을 받으면 0 벡터로
# 채우고 싶은 유혹이 생기는데, 0 은 "안 닮음" 으로 읽혀 순위가 밀린다.
#
# `text_embedding` 이 NULL 인 행도 뺀다. 텍스트 없이 이미지만으로는 비교하지 않는다.
VECTORS_SQL = """
SELECT a.auction_id,
       p.text_embedding,
       p.image_embedding
FROM auction a
JOIN product p ON p.product_id = a.product_id
WHERE a.auction_id = ANY(%(auction_ids)s)
  AND p.text_embedding IS NOT NULL
"""


# 사용자 행동 로그 (개인화용).
#
# **컬럼명이 `occured_at` 이다.** `occurred_at` 이 아니다 — 백엔드 스키마의 오타이고
# 엔티티에도 `@Column(name = "occured_at")` 로 그대로 매핑돼 있다. 맞춤법을 고치면
# 조회가 깨지므로 스키마를 따른다.
#
# `auction_id` 가 없는 이벤트(검색어 입력 등)는 뺀다. 어떤 상품에 대한 관심인지
# 특정할 수 없으면 프로필에 넣을 벡터가 없다.
EVENTS_SQL = """
SELECT event_type, auction_id, occured_at
FROM member_event
WHERE member_id = %(member_id)s
  AND auction_id IS NOT NULL
  AND occured_at >= %(since)s
ORDER BY occured_at DESC
LIMIT %(limit)s
"""


def to_events(rows: Sequence[Mapping[str, Any]]) -> tuple[BehaviorEvent, ...]:
    """행동 로그를 프로필 계산용 자료구조로.

    시각을 UTC 로 올린다. 감쇠 계산이 `datetime.now(timezone.utc)` 와 빼기를 하므로
    맞추지 않으면 그대로 터진다 — 후보 조회에서 이미 한 번 겪은 일이다.
    """
    return tuple(
        BehaviorEvent(
            event_type=str(r["event_type"]),
            auction_id=r["auction_id"],
            occurred_at=to_utc(r["occured_at"]),
        )
        for r in rows
        if r["auction_id"] is not None and r["occured_at"] is not None
    )


def to_candidates(rows: Sequence[Mapping[str, Any]]) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            auction_id=r["auction_id"],
            product_id=r["product_id"],
            seller_id=r["seller_id"],
            category_id=r["category_id"],
            # **DB 시각을 UTC 로 올린다.** 랭킹은 `datetime.now(timezone.utc)` 와
            # 비교하는데, ERD 의 TIMESTAMP 에는 시간대가 없어 그대로 두면
            # "can't subtract offset-naive and offset-aware datetimes" 로 터진다.
            started_at=to_utc(r["started_at"]),
            auction_time=int(r["auction_time"]),
            ended_at=to_utc(r["ended_at"]),
            live_broadcast_id=r["live_broadcast_id"],
            view_count=int(r["view_count"] or 0),
            bookmark_count=int(r["bookmark_count"] or 0),
            bid_count=int(r["bid_count"] or 0),
            bidder_count=int(r["bidder_count"] or 0),
        )
        for r in rows
    )


def _to_vector(value) -> tuple[float, ...] | None:
    """pgvector 값을 파이썬 튜플로.

    드라이버 설정에 따라 리스트로 올 수도 있고 `'[0.1,0.2]'` 문자열로 올 수도 있다.
    **둘 다 받아 둔다** — 한쪽만 처리하면 드라이버를 바꾸는 순간 조용히 빈 벡터가 되고,
    그러면 유사도가 전부 0 이 되어 추천이 인기순과 구별되지 않는다.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return None
        return tuple(float(x) for x in stripped.split(","))
    return tuple(float(x) for x in value) or None


def to_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[int, ProductVector]:
    out: dict[int, ProductVector] = {}
    for r in rows:
        text = _to_vector(r["text_embedding"])
        if text is None:
            continue  # 쿼리에서 걸렀지만 방어적으로 한 번 더 본다
        out[r["auction_id"]] = ProductVector(
            auction_id=r["auction_id"],
            text=text,
            image=_to_vector(r["image_embedding"]),
        )
    return out
