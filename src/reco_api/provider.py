"""추천 후보 조회기.

랭킹 로직은 DB 를 모른다. 조회를 여기로 몰아두면 실제 DB 가 붙을 때
`PostgresProvider` 만 채우면 되고, **랭킹 코드는 한 줄도 바뀌지 않는다.**
이상거래 탐지에서 이미 쓴 구조와 같다.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Mapping, Protocol, Sequence

from reco import BehaviorEvent, Candidate, ProductVector

from . import queries

log = logging.getLogger("reco_api.provider")


class ProviderError(RuntimeError):
    """데이터 조회 실패. 추천이 안 나가도 서비스는 살아 있어야 한다."""


class CandidateProvider(Protocol):
    name: str

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]: ...

    def load_vectors(
        self, auction_ids: Sequence[int]
    ) -> Mapping[int, ProductVector]: ...

    def load_events(
        self, member_id: int, since: datetime, limit: int
    ) -> Sequence[BehaviorEvent]: ...


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

    def load_events(
        self, member_id: int, since: datetime, limit: int
    ) -> Sequence[BehaviorEvent]:
        return ()


class PostgresProvider:
    """실제 DB 조회. 읽기 전용 계정으로 접속한다.

        auction JOIN product   진행 중인 경매 · 판매자 · 카테고리 · 라이브 여부
        product (벡터)         text_embedding · image_embedding

    SQL 과 행 변환은 `queries.py` 에 있다 — DB 없이 검증할 수 있게 하기 위해서다.

    노출 정책(검수 통과 · 삭제 · 진행 상태)은 **조회에서 끝낸다.** 랭킹은 무엇을
    보여줘도 되는지 모르고 점수만 매기므로, 여기서 안 거르면 검수에 걸린 상품이
    추천 목록에 올라온다.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - 설치 여부에 달린 경로
            raise ProviderError(
                "psycopg 가 설치되어 있지 않습니다. pip install 'psycopg[binary]'"
            ) from exc

        try:
            return psycopg.connect(self._dsn, row_factory=dict_row)
        except Exception as exc:
            raise ProviderError(f"DB 연결 실패: {exc}") from exc

    def load_active(self, now: datetime, limit: int) -> Sequence[Candidate]:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    queries.ACTIVE_CANDIDATES_SQL,
                    {
                        "hidden_status": list(queries.HIDDEN_PRODUCT_STATUS),
                        "limit": limit,
                    },
                )
                return queries.to_candidates(cur.fetchall())
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"추천 후보 조회 실패: {exc}") from exc
        finally:
            conn.close()

    def load_vectors(self, auction_ids: Sequence[int]) -> Mapping[int, ProductVector]:
        """`product` 의 임베딩 두 컬럼을 읽는다.

        **가진 것만 돌려준다.** 아직 배치가 돌지 않은 상품은 행 자체가 빠지므로
        호출자가 "유사도를 모름" 과 "안 닮음" 을 구분할 수 있다.

        후보가 수만 건으로 늘면 이 방식(전부 읽어 파이썬에서 계산)이 한계에 온다.
        그때는 pgvector 인덱스로 DB 에서 상위 N 만 받아 오면 된다.

            SELECT a.auction_id, p.text_embedding <=> %(seed)s AS distance
            FROM auction a JOIN product p ON p.product_id = a.product_id
            WHERE a.status = 'ACTIVE' AND a.auction_id <> %(seed_id)s
            ORDER BY p.text_embedding <=> %(seed)s
            LIMIT %(n)s

        `<=>` 는 코사인 **거리**라 작을수록 가깝다. 유사도로 쓰려면 `1 - distance`
        로 뒤집어야 한다. 부호를 그대로 두면 가장 안 닮은 상품이 1위가 된다.
        """
        if not auction_ids:
            return {}

        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(queries.VECTORS_SQL, {"auction_ids": list(auction_ids)})
                return queries.to_vectors(cur.fetchall())
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"임베딩 조회 실패: {exc}") from exc
        finally:
            conn.close()

    def load_events(
        self, member_id: int, since: datetime, limit: int
    ) -> Sequence[BehaviorEvent]:
        """회원 행동과 찜을 합쳐 읽는다. 실패 시 인기순 폴백을 위해 빈 값 반환."""
        conn = self._connect()
        try:
            params = {"member_id": member_id, "since": since, "limit": limit}
            with conn, conn.cursor() as cur:
                cur.execute(queries.EVENTS_SQL, params)
                logged = queries.to_events(cur.fetchall())
                cur.execute(queries.BOOKMARKS_SQL, params)
                bookmarks = queries.to_bookmark_events(cur.fetchall())
            return queries.merge_events(logged, bookmarks)
        except Exception:
            log.warning("행동 로그 조회 실패 member_id=%s — 인기순으로 폴백", member_id)
            return ()
        finally:
            conn.close()
