"""개인화 추천.

STEP 1 (인기순) · STEP 3 (유사 상품) · STEP 5 (개인화) 가 구현되어 있다.

셋은 같은 랭킹 함수를 쓴다. 무엇을 `similarity` 로 넘기느냐만 다르다.

    인기순      아무것도 안 넘김
    유사 상품   지금 보고 있는 상품과의 유사도
    개인화      이 사람의 관심 프로필과의 유사도

개인화는 **행동 로그가 없으면 자동으로 인기순으로 떨어진다.** 로그가 쌓이는
만큼만 개인화된다.
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
