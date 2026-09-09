"""개인화 추천.

STEP 1 (인기순) 까지 구현되어 있다. 개인화는 popularity.rank() 의 boost 인자로
끼워 넣는다 — 랭킹 코드를 다시 쓰지 않기 위한 자리다.
"""

from .config import RecoConfig
from .popularity import percentile_ranks, rank, urgency
from .schema import Candidate, RecoResult, Scored

__all__ = [
    "RecoConfig",
    "Candidate",
    "RecoResult",
    "Scored",
    "rank",
    "urgency",
    "percentile_ranks",
]
