"""개인화 추천.

STEP 1 (인기순) · STEP 3 (유사 상품) · STEP 5 (개인화) 가 구현되어 있다.
행동 로그가 없으면 개인화는 자동으로 인기순으로 폴백한다.
"""

from .config import RecoConfig
from .explain import reason_for
from .popularity import percentile_ranks, rank, urgency
from .schema import Candidate, ProductVector, RecoResult, Scored
from .profile import BehaviorEvent, UserProfile, affinity, build_profile
from .similarity import blend, compare, cosine, similarities

__all__ = [
    "RecoConfig",
    "reason_for",
    "Candidate",
    "ProductVector",
    "RecoResult",
    "Scored",
    "rank",
    "urgency",
    "percentile_ranks",
    "cosine",
    "blend",
    "similarities",
    "compare",
    "BehaviorEvent",
    "UserProfile",
    "build_profile",
    "affinity",
]
