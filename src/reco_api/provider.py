"""추천 후보 조회기.

랭킹 로직은 DB 를 모른다. 조회를 여기로 몰아두면 실제 DB 가 붙을 때
`PostgresProvider` 만 채우면 되고, **랭킹 코드는 한 줄도 바뀌지 않는다.**
이상거래 탐지에서 이미 쓴 구조와 같다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from reco import Candidate


class ProviderError(RuntimeError):
    """데이터 조회 실패. 추천이 안 나가도 서비스는 살아 있어야 한다."""


class CandidateProvider(Protocol):
    name: str

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]: ...


class InMemoryProvider:
    """합성 데이터. `bid` 테이블 없이도 API 를 호출해 볼 수 있다."""

    name = "in-memory(demo)"

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._candidates = tuple(candidates)

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]:
        return self._candidates


class PostgresProvider:
    """실제 DB 조회. 아래 쿼리로 구현한다.

        SELECT a.auction_id, a.product_id, p.member_id AS seller_id, p.category_id,
               a.started_at, a.auction_time, a.ended_at,
               a.view_count, a.bookmark_count, a.bid_count, a.bidder_count
        FROM auction a
        JOIN product p USING (product_id)
        WHERE a.status = 'ACTIVE'
          AND a.deleted_at IS NULL
        LIMIT :pool

    두 가지를 주의한다.

    **판매자는 `auction` 이 아니라 `product` 에 있다.** `auction` 에는 `member_id` 가
    없으므로 `product` 를 조인해야 한다.

    **`LIMIT` 은 랭킹 전에 자르는 후보 풀이다.** 최종 노출 개수보다 넉넉히 잡아야
    한다 — 여기서 잘리면 랭킹이 볼 수 없는 상품이 생긴다.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]:
        raise ProviderError(
            "PostgresProvider 는 아직 구현되지 않았습니다. DB 연결 후 채우세요."
        )
