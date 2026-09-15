"""추천 후보 조회기.

랭킹 로직은 DB 를 모른다. 조회를 여기로 몰아두면 실제 DB 가 붙을 때
`PostgresProvider` 만 채우면 되고, **랭킹 코드는 한 줄도 바뀌지 않는다.**
이상거래 탐지에서 이미 쓴 구조와 같다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, Sequence

from reco import Candidate, ProductVector


class ProviderError(RuntimeError):
    """데이터 조회 실패. 추천이 안 나가도 서비스는 살아 있어야 한다."""


class CandidateProvider(Protocol):
    name: str

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]: ...

    def load_vectors(
        self, auction_ids: Sequence[int]
    ) -> Mapping[int, ProductVector]: ...


class InMemoryProvider:
    """합성 데이터. `bid` 테이블 없이도 API 를 호출해 볼 수 있다."""

    name = "in-memory(demo)"

    def __init__(
        self,
        candidates: Sequence[Candidate],
        vectors: Sequence[ProductVector] = (),
    ) -> None:
        self._candidates = tuple(candidates)
        self._vectors = {v.auction_id: v for v in vectors}

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]:
        return self._candidates

    def load_vectors(self, auction_ids: Sequence[int]) -> Mapping[int, ProductVector]:
        """가진 것만 돌려준다. **없는 것을 0 벡터로 채우지 않는다.**

        아직 임베딩 배치가 돌지 않은 상품은 "유사도를 모르는" 상태지, "안 닮은"
        상태가 아니다. 호출자가 그 차이를 구분할 수 있어야 한다.
        """
        return {i: self._vectors[i] for i in auction_ids if i in self._vectors}


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

    def load_vectors(self, auction_ids: Sequence[int]) -> Mapping[int, ProductVector]:
        """`product_embedding` 에서 벡터를 읽는다.

            SELECT a.auction_id, e.text_embedding, e.image_embedding
            FROM auction a
            JOIN product_embedding e USING (product_id)
            WHERE a.auction_id = ANY(:auction_ids)

        **`JOIN` 이지 `LEFT JOIN` 이 아니다.** 벡터가 없는 상품은 행 자체가 빠져야
        호출자가 "아직 임베딩 안 됨" 으로 처리할 수 있다. `LEFT JOIN` 으로 NULL 을
        받으면 0 벡터로 채우고 싶은 유혹이 생기는데, 0 은 "안 닮음" 으로 읽힌다.

        `image_embedding` 은 컬럼 자체가 NULL 일 수 있다 — 사진을 못 읽었거나
        이미지 배치가 아직 안 돈 상품이다. 그대로 `ProductVector.image=None` 이다.

        후보가 수만 건으로 늘면 이 방식(전체 로드 후 파이썬 계산)이 한계에 온다.
        그때는 pgvector 인덱스로 DB 에서 상위 N 만 받아 오면 된다.

            SELECT a.auction_id, e.text_embedding <=> :seed_vec AS distance
            FROM auction a JOIN product_embedding e USING (product_id)
            WHERE a.status = 'ACTIVE' AND a.auction_id <> :seed_id
            ORDER BY e.text_embedding <=> :seed_vec
            LIMIT :n

        `<=>` 는 코사인 거리라 **작을수록 가깝다.** 유사도로 쓰려면 `1 - distance`
        로 뒤집어야 한다. 부호를 그대로 두면 가장 안 닮은 상품이 1위가 된다.
        """
        raise ProviderError(
            "PostgresProvider 는 아직 구현되지 않았습니다. DB 연결 후 채우세요."
        )
