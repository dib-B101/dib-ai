"""탐지 입력 조회기.

규칙 엔진은 DB 를 모른다. 어디선가 DetectionInput 을 만들어 넘겨줘야 하는데,
그 "어디선가" 를 갈아끼울 수 있게 분리한 것이 이 모듈이다.

    InMemoryProvider   합성 데이터. DB 없이 API 를 검증한다
    PostgresProvider   실제 조회. 엔진과 엔드포인트는 건드리지 않는다

어느 쪽을 쓸지는 `DIB_DATABASE_URL` 이 정한다 (src/fraud_api/main.py).

이 경계를 두는 이유는 피처 계산식이 AI 쪽에서 가장 자주 바뀌는 부분이기 때문이다.
백엔드가 데이터를 모아 보내는 구조였다면 식을 하나 고칠 때마다 백엔드 배포가 필요하다.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from fraud import DetectionInput

from . import queries

log = logging.getLogger("fraud_api.provider")


class ProviderError(RuntimeError):
    """조회 자체가 실패했다. 경매가 없는 것과 구분한다."""


class InputProvider(Protocol):
    """auction_id 로 탐지 입력을 만들어 온다."""

    name: str

    def load(self, auction_id: int, as_of: datetime | None) -> DetectionInput | None:
        """경매를 찾지 못하면 None 을 돌려준다. 조회 실패는 ProviderError 를 던진다."""
        ...


class InMemoryProvider:
    """미리 만들어 둔 DetectionInput 을 돌려준다.

    데모와 테스트에 쓴다. bid 테이블이 생기기 전까지 API 를 실제로 호출해 볼 수 있다.
    """

    name = "in-memory"

    def __init__(self, data: dict[int, DetectionInput]) -> None:
        self._data = dict(data)

    def load(self, auction_id: int, as_of: datetime | None) -> DetectionInput | None:
        inp = self._data.get(auction_id)
        if inp is None:
            return None
        if as_of is None or as_of == inp.as_of:
            return inp
        # as_of 를 바꿔 호출하면 그 시점 기준으로 다시 계산한다.
        return DetectionInput(
            auction=inp.auction,
            bids=tuple(b for b in inp.bids if b.created_at <= as_of),
            members=inp.members,
            histories={
                m: tuple(h for h in hs if h.participated_at < as_of)
                for m, hs in inp.histories.items()
            },
            as_of=as_of,
        )

    def auction_ids(self) -> list[int]:
        return sorted(self._data)


class PostgresProvider:
    """실제 DB 조회. 읽기 전용 계정으로 접속한다.

        auction JOIN product   경매 · 판매자 · 카테고리
        bid                    이 경매의 입찰, 그리고 입찰자들의 과거 참여 이력
        member                 가입 시각
        subscription           구독 관계 (R4 가 판단을 건너뛸지 결정한다)
        auction (집계)         카테고리별 평균 입찰 수 · 평균 시작가

    SQL 과 행 변환은 `queries.py` 에 있다. **DB 없이 검증할 수 있게 하기 위해서다.**

    요청 1건마다 연결을 새로 연다. 풀을 두지 않은 것은 이 조회가 **경매가 끝날 때만**
    일어나기 때문이다. 초당 수백 건이 아니라면 풀이 주는 이득보다 연결이 죽었을 때의
    복구 경로가 늘어나는 비용이 크다. 호출이 잦아지면 그때 psycopg_pool 로 바꾼다.
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

    def load(self, auction_id: int, as_of: datetime | None) -> DetectionInput | None:
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(queries.AUCTION_SQL, {"auction_id": auction_id})
                row = cur.fetchone()
                if row is None:
                    return None

                # 시작 전 경매는 분석 대상이 아니다. 입찰이 있을 수 없고,
                # started_at 이 NULL 이라 모든 시점 비율이 계산되지 않는다.
                if row["started_at"] is None:
                    log.info("아직 시작하지 않은 경매입니다 auction_id=%s", auction_id)
                    return None

                auction = queries.to_auction(row)
                moment = queries.resolve_as_of(as_of, auction)

                cur.execute(
                    queries.BIDS_SQL, {"auction_id": auction_id, "as_of": moment}
                )
                bids = queries.to_bids(cur.fetchall())

                member_ids = sorted({b.member_id for b in bids})
                params = {
                    "member_ids": member_ids,
                    "auction_id": auction_id,
                    "as_of": moment,
                }

                # 입찰이 없으면 나머지를 조회할 이유가 없다. 빈 조회를 돌리면
                # ANY(빈 배열) 로 전 테이블을 훑는 계획이 나올 수 있다.
                if member_ids:
                    cur.execute(queries.MEMBERS_SQL, params)
                    members = queries.to_members(cur.fetchall())

                    cur.execute(queries.HISTORIES_SQL, params)
                    histories = queries.to_histories(cur.fetchall())

                    cur.execute(queries.SUBSCRIPTIONS_SQL, params)
                    subscriptions = queries.to_subscriptions(cur.fetchall())
                else:
                    members, histories, subscriptions = {}, {}, {}

                cur.execute(
                    queries.CORPUS_SQL,
                    {
                        "category_id": auction.category_id,
                        "auction_id": auction_id,
                        "as_of": moment,
                    },
                )
                corpus = queries.to_corpus(cur.fetchone(), auction.category_id)
                if corpus is None:
                    log.info(
                        "카테고리 %s 의 코퍼스 표본이 부족합니다 — 규칙 점수만 냅니다",
                        auction.category_id,
                    )

            return DetectionInput(
                auction=auction,
                bids=bids,
                members=members,
                histories=histories,
                as_of=moment,
                subscriptions=subscriptions,
                corpus=corpus,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"탐지 입력 조회 실패 auction_id={auction_id}: {exc}") from exc
        finally:
            conn.close()
