"""추천 입출력 자료구조.

이 모듈은 DB 도 HTTP 도 모른다. 순수한 값 객체만 정의하며, 조회는 호출자가 담당한다.
덕분에 랭킹 로직을 실제 DB 없이 합성 데이터로 개발·테스트할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """추천 후보 경매 1건. `auction` 조인 결과에서 온다.

    벡터는 `product` 에 있지만 추천 대상은 **진행 중인 경매**다. 유찰 후 재등록되면
    같은 `product_id` 에 여러 `auction` 이 붙으므로 둘을 분리해서 다룬다.
    """

    auction_id: int
    product_id: int
    seller_id: int          # product.member_id — auction 에는 판매자가 없다
    category_id: int
    started_at: datetime
    auction_time: int       # 초. 생성 시 확정되며 연장해도 바뀌지 않는다
    ended_at: datetime | None = None   # 연장을 반영한 현재 예정 종료시각

    view_count: int = 0
    bookmark_count: int = 0
    bid_count: int = 0
    bidder_count: int = 0

    def remaining_seconds(self, now: datetime) -> float:
        """남은 시간(초). 이미 끝났으면 0 이하가 나온다.

        `ended_at` 은 연장이 반영된 현재 예정 종료시각이다. 아직 채워지지 않았다면
        `started_at + auction_time` 으로 유도한다 — 연장 전이라면 같은 값이다.
        """
        end = self.ended_at or (self.started_at + timedelta(seconds=self.auction_time))
        return (end - now).total_seconds()


@dataclass(frozen=True, slots=True)
class Scored:
    """랭킹 결과 1건."""

    auction_id: int
    score: float
    urgency: float
    popularity: float
    competition: float
    remaining_seconds: float

    def breakdown(self) -> dict[str, Any]:
        """왜 이 순위인지 설명하는 값들.

        추천은 이상거래 탐지와 달리 사용자에게 사유를 보여주지 않지만, **왜 이게 위로
        올라왔는지 우리가 설명할 수 없으면 튜닝도 할 수 없다.** 관리자 화면과 평가에 쓴다.
        """
        return {
            "urgency": round(self.urgency, 4),
            "popularity": round(self.popularity, 4),
            "competition": round(self.competition, 4),
            "remaining_seconds": round(self.remaining_seconds, 1),
        }


@dataclass(frozen=True, slots=True)
class RecoResult:
    as_of: datetime
    config_version: str
    strategy: str                     # popularity | personalized ...
    items: tuple[Scored, ...] = ()
    excluded: dict[str, int] = field(default_factory=dict)   # 사유 -> 건수
