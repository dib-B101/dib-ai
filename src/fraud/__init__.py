"""규칙 기반 이상거래 탐지.

사용 예
    from fraud import RuleConfig, detect

    cfg = RuleConfig.load()                 # config/rules.yaml
    result = detect(detection_input, cfg)   # 예외를 던지지 않는다

    for r in result.results:
        print(r.member_id, r.rule_score, r.reasons())
        save(detail=r.to_detail())          # fraud_detection.detail 에 그대로
"""

from .config import RuleConfig, RuleSpec
from .engine import combine, detect
from .schema import (
    Auction,
    Bid,
    BidderResult,
    DetectionInput,
    DetectionResult,
    HistoryEntry,
    Member,
    RuleHit,
)

__all__ = [
    "Auction",
    "Bid",
    "BidderResult",
    "DetectionInput",
    "DetectionResult",
    "HistoryEntry",
    "Member",
    "RuleConfig",
    "RuleHit",
    "RuleSpec",
    "combine",
    "detect",
]
