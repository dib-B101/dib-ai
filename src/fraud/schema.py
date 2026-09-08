"""탐지 엔진의 입출력 자료구조.

이 모듈은 DB 를 모른다. 순수한 값 객체만 정의하며, 조회는 호출자가 담당한다.
덕분에 규칙 엔진을 실제 DB 없이 합성 데이터로 개발·테스트할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping

# ---------------------------------------------------------------- 입력

@dataclass(frozen=True, slots=True)
class Bid:
    """bid 테이블 1행."""
    bid_id: int
    auction_id: int
    member_id: int
    amount: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Auction:
    """auction 테이블 1행 중 탐지에 쓰는 컬럼만.

    시간 모델은 ERD 를 그대로 따른다.

        started_at    판매자가 라이브 중 경매를 시작한 시각
        auction_time  시작 기준으로 주어진 진행 시간(초). 생성 시 확정된다
        ended_at      실제 종료 시각. 진행 중에는 NULL

    ERD 에 최초 예정 종료시각 컬럼이 없으므로 ``original_end_at`` 으로 유도한다.
    시점 계산의 기준은 ``ended_at`` 이 아니라 항상 이 유도값이다 — 연장이 발생하면
    ``ended_at`` 이 뒤로 밀려 같은 입찰이 경매마다 다른 비율로 계산되기 때문이다.
    """
    auction_id: int
    seller_id: int          # auction.member_id (비정규화된 판매자)
    category_id: int
    start_price: int
    started_at: datetime    # auction.started_at
    auction_time: int       # auction.auction_time (초)
    ended_at: datetime | None = None   # auction.ended_at. 진행 중이면 None

    @property
    def original_end_at(self) -> datetime:
        """연장을 반영하지 않은 최초 예정 종료시각."""
        return self.started_at + timedelta(seconds=self.auction_time)


@dataclass(frozen=True, slots=True)
class Member:
    """member 테이블 1행 중 탐지에 쓰는 컬럼만."""
    member_id: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """이 입찰자가 과거에 참여한 경매 1건.

    as_of 이전 데이터만 담아야 한다. 미래 정보가 섞이면 누수가 된다.
    """
    auction_id: int
    seller_id: int
    participated_at: datetime
    won: bool = False


@dataclass(frozen=True, slots=True)
class DetectionInput:
    auction: Auction
    bids: tuple[Bid, ...]                               # 이 경매의 전체 입찰 (순서 무관)
    members: Mapping[int, Member]                       # member_id -> Member
    histories: Mapping[int, tuple[HistoryEntry, ...]]   # member_id -> 과거 참여 이력
    as_of: datetime                                     # 기준 시각. 보통 경매 종료 시각


# ---------------------------------------------------------------- 출력

@dataclass(frozen=True, slots=True)
class RuleHit:
    """규칙 1개의 판정 결과."""
    rule_id: str
    score: float        # 0.0 ~ 1.0
    weight: float
    reason: str         # 실제 수치가 들어간 사람이 읽을 수 있는 근거


@dataclass(frozen=True, slots=True)
class BidderResult:
    member_id: int
    rule_score: float                       # 가중 평균. 0.0 ~ 1.0
    hits: tuple[RuleHit, ...]               # 점수를 낸 규칙만 (게이트에 걸린 것은 제외)
    features: Mapping[str, float]
    flags: Mapping[str, bool]
    skipped_rules: Mapping[str, str]        # rule_id -> 건너뛴 사유
    errors: tuple[str, ...] = ()            # 규칙 실행 중 발생한 예외

    def to_detail(self) -> dict:
        """fraud_detection.detail JSONB 에 그대로 넣을 수 있는 형태."""
        return {
            "rules": {h.rule_id: round(h.score, 4) for h in self.hits},
            "features": {k: round(v, 4) for k, v in sorted(self.features.items())},
            "flags": dict(sorted(self.flags.items())),
            "skipped_rules": dict(sorted(self.skipped_rules.items())),
            "errors": list(self.errors),
        }

    def reasons(self) -> list[str]:
        """점수가 높은 순으로 정렬한 탐지 사유."""
        ranked = sorted(self.hits, key=lambda h: h.score * h.weight, reverse=True)
        return [h.reason for h in ranked if h.score > 0]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    auction_id: int
    as_of: datetime
    rule_config_version: str
    results: tuple[BidderResult, ...] = ()
    skipped_bidders: Mapping[int, str] = field(default_factory=dict)
    auction_error: str | None = None     # 경매 단위 실패. 이 경우 results 는 비어 있다
