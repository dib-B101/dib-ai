"""인기순 추천 — 개인화 이전의 기준선.

**가장 먼저 만든다.** 난이도는 제일 낮은데 세 곳에서 쓰인다.

1. Cold Start 폴백 — 행동 이력이 없는 사용자
2. 성능 평가 baseline — 개인화가 이것보다 나은지 증명해야 한다
3. 장애 폴백 — 임베딩·벡터 검색이 죽어도 추천은 나가야 한다

한 번 만들어 세 곳에서 재사용한다.

    score = w_urgency     × urgency
          + w_popularity  × pct(조회수, 찜수)
          + w_competition × pct(입찰수, 입찰자수)

두 가지 결정이 이 모듈의 전부다.

**백분위로 정규화한다.** 조회수는 수만, 입찰수는 수백이라 원값을 그대로 더하면
스케일이 큰 항이 가중치와 무관하게 순위를 지배한다.

**마감 임박도를 지수로 정의한다.** 남은 시간의 역수로 두면 1초 남은 경매가 무한대가
되어 다른 항을 전부 눌러버린다.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Callable, Sequence

from .config import RecoConfig
from .schema import Candidate, RecoResult, Scored


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """후보 집합 안에서의 백분위 [0, 1].

    절대값이 아니라 **지금 후보들 사이에서 몇 등인지**를 쓴다. 조회수 1000이 많은
    편인지 적은 편인지는 함께 놓인 후보에 따라 달라지기 때문이다.

    같은 값은 같은 백분위를 받는다. 서비스 초기처럼 **모든 값이 0 이면 전원 0 이 되어
    이 항이 순위에 영향을 주지 않는다** — 정보가 없을 때 억지로 순서를 만들지 않는다.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    ordered = sorted(values)
    # 자기보다 작은 값의 개수 / (n - 1). 동점은 같은 값을 받는다.
    less_than: dict[float, int] = {}
    for i, v in enumerate(ordered):
        less_than.setdefault(v, i)

    return [less_than[v] / (n - 1) for v in values]


def urgency(remaining_seconds: float, tau: float) -> float:
    """마감 임박도 [0, 1]. 이미 끝났으면 1 이다.

        exp(-남은시간 / tau)

    tau = 3600 이면 1시간 남았을 때 0.37, 10분 남았을 때 0.85 다.
    """
    if remaining_seconds <= 0:
        return 1.0
    return math.exp(-remaining_seconds / tau)


def _weighted_percentile(
    candidates: Sequence[Candidate], spec: dict[str, float]
) -> list[float]:
    """각 필드를 백분위로 만든 뒤 가중 평균한다."""
    per_field = {
        field: percentile_ranks([float(getattr(c, field)) for c in candidates])
        for field in spec
    }
    return [
        sum(weight * per_field[field][i] for field, weight in spec.items())
        for i in range(len(candidates))
    ]


def rank(
    candidates: Sequence[Candidate],
    cfg: RecoConfig,
    now: datetime,
    limit: int | None = None,
    boost: Callable[[Candidate], float] | None = None,
) -> RecoResult:
    """인기순 추천 목록을 만든다.

    `boost` 는 개인화 점수를 더할 자리다. 지금은 쓰지 않지만, STEP 5 에서 유사도를
    여기에 끼워 넣으면 **랭킹 코드를 다시 쓰지 않아도 된다.**

    이미 끝난 경매는 제외한다. 남은 시간이 0 이하면 마감 임박도가 최대가 되어
    **종료된 경매가 목록 맨 위에 올라오는 사고**가 난다.
    """
    excluded: dict[str, int] = {}
    alive = []
    for c in candidates:
        if c.remaining_seconds(now) <= 0:
            excluded["ended"] = excluded.get("ended", 0) + 1
            continue
        alive.append(c)

    if not alive:
        return RecoResult(now, cfg.version, "popularity", (), excluded)

    pops = _weighted_percentile(alive, dict(cfg.popularity))
    comps = _weighted_percentile(alive, dict(cfg.competition))

    w = cfg.weights
    scored: list[Scored] = []
    for i, c in enumerate(alive):
        remaining = c.remaining_seconds(now)
        u = urgency(remaining, cfg.tau_seconds)
        score = (
            w.get("urgency", 0.0) * u
            + w.get("popularity", 0.0) * pops[i]
            + w.get("competition", 0.0) * comps[i]
        )
        if boost is not None:
            score += boost(c)
        scored.append(Scored(c.auction_id, score, u, pops[i], comps[i], remaining))

    # 점수 내림차순. 동점은 auction_id 오름차순으로 고정해 결과를 재현 가능하게 둔다.
    scored.sort(key=lambda s: (-s.score, s.auction_id))
    return RecoResult(
        now,
        cfg.version,
        "popularity",
        tuple(scored[: limit or cfg.limit]),
        excluded,
    )
