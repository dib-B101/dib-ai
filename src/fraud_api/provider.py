"""탐지 입력 조회기.

규칙 엔진은 DB 를 모른다. 어디선가 DetectionInput 을 만들어 넘겨줘야 하는데,
그 "어디선가" 를 갈아끼울 수 있게 분리한 것이 이 모듈이다.

    지금   InMemoryProvider   합성 데이터. bid 테이블 없이 API 를 검증한다
    나중   PostgresProvider   실제 조회. 엔진과 엔드포인트는 건드리지 않는다

이 경계를 두는 이유는 피처 계산식이 AI 쪽에서 가장 자주 바뀌는 부분이기 때문이다.
백엔드가 데이터를 모아 보내는 구조였다면 식을 하나 고칠 때마다 백엔드 배포가 필요하다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fraud import DetectionInput


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
    """실제 DB 조회. bid 테이블이 생기면 구현한다.

    읽기 전용 계정으로 접속하며, 아래 표의 컬럼만 조회한다.

        auction   auction_id, member_id(판매자), category_id, start_price,
                  started_at, auction_time, ended_at
        bid       bid_id, auction_id, member_id, amount, created_at
        member    member_id, created_at
        history   과거 참여 이력 (bid JOIN auction, created_at < as_of)

    주의: 모든 이력 조회에 created_at < as_of 조건을 걸어야 한다.
          빠뜨리면 미래 데이터가 섞여 학습·평가에 누수가 생긴다.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def load(self, auction_id: int, as_of: datetime | None) -> DetectionInput | None:
        raise ProviderError(
            "PostgresProvider 는 아직 구현되지 않았습니다. "
            "bid 테이블 생성 후 연결하세요."
        )
