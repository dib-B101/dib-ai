"""탐지 엔진 오케스트레이션.

설계 원칙 세 가지
    1. 재현성 — 기준 시각(as_of)은 주입받고 내부에서 now() 를 부르지 않는다.
       규칙 실행 순서도 rule_id 정렬로 고정한다.
    2. 실패 격리 — 규칙 하나가 예외를 던져도 나머지는 계속 계산한다.
       detect() 는 어떤 경우에도 예외를 밖으로 던지지 않는다.
       탐지 실패가 경매 종료나 낙찰 처리를 막아서는 안 되기 때문이다.
    3. 자동 차단 없음 — 점수와 사유만 돌려준다. 제재 판단은 관리자 몫이다.
"""

from __future__ import annotations

from typing import Mapping

from .config import RuleConfig
from .features import bidder_ids, ordered_bids, pingpong_ratio
from .rules import RULE_FUNCS, RuleContext, RuleOutcome
from .schema import BidderResult, DetectionInput, DetectionResult, RuleHit


def detect(inp: DetectionInput, cfg: RuleConfig) -> DetectionResult:
    """경매 1건의 모든 입찰자를 판정한다. 예외를 던지지 않는다."""
    try:
        return _detect(inp, cfg)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패도 호출자에게 전파하지 않는다
        return DetectionResult(
            auction_id=getattr(inp.auction, "auction_id", -1),
            as_of=inp.as_of,
            rule_config_version=cfg.version,
            auction_error=f"{type(exc).__name__}: {exc}",
        )


def _detect(inp: DetectionInput, cfg: RuleConfig) -> DetectionResult:
    ordered = tuple(ordered_bids(inp.bids))
    members = bidder_ids(inp.bids)

    # 경매 단위 최소 분석 조건. 미달이면 아무도 판정하지 않는다.
    # 입찰자가 적으면 입찰 비중이 기계적으로 커져 오탐만 늘어난다.
    if len(ordered) < cfg.min_bids or len(members) < cfg.min_bidders:
        reason = (
            f"최소 분석 조건 미충족 (입찰 {len(ordered)}/{cfg.min_bids}건, "
            f"입찰자 {len(members)}/{cfg.min_bidders}명)"
        )
        return DetectionResult(
            auction_id=inp.auction.auction_id,
            as_of=inp.as_of,
            rule_config_version=cfg.version,
            skipped_bidders={m: reason for m in members},
        )

    results: list[BidderResult] = []
    skipped: dict[int, str] = {}

    for member_id in members:
        history = tuple(inp.histories.get(member_id, ()))
        # 미래 정보 차단. 호출자가 걸러 왔더라도 한 번 더 막는다.
        history = tuple(h for h in history if h.participated_at < inp.as_of)

        flags = {
            "has_history": len(history) >= 1,
            "has_sufficient_history": len(history) >= 3,
            "has_member_record": member_id in inp.members,
        }

        hits: list[RuleHit] = []
        skipped_rules: dict[str, str] = {}
        errors: list[str] = []

        for spec in cfg.enabled_rules:  # rule_id 정렬 순 — 재현성
            fn = RULE_FUNCS.get(spec.rule_id)
            if fn is None:
                skipped_rules[spec.rule_id] = "구현되지 않은 규칙"
                continue

            ctx = RuleContext(
                inp=inp,
                member_id=member_id,
                ordered=ordered,
                history=history,
                member=inp.members.get(member_id),
                spec=spec,
            )
            try:
                outcome: RuleOutcome = fn(ctx)
            except Exception as exc:  # noqa: BLE001 - 규칙 하나의 실패가 전체를 막지 않는다
                errors.append(f"{spec.rule_id}: {type(exc).__name__}: {exc}")
                continue

            if outcome.score is None:
                skipped_rules[spec.rule_id] = outcome.reason
            else:
                hits.append(
                    RuleHit(
                        rule_id=spec.rule_id,
                        score=outcome.score,
                        weight=spec.weight,
                        reason=outcome.reason,
                    )
                )

        rule_score = _weighted_average(hits)
        pp_ratio, partner = pingpong_ratio(ordered, member_id)

        features: dict[str, float] = {
            "bidding_ratio": sum(1 for b in ordered if b.member_id == member_id) / len(ordered),
            "pingpong_ratio": pp_ratio,
            "history_count": float(len(history)),
        }
        if partner is not None:
            features["pingpong_partner_id"] = float(partner)

        results.append(
            BidderResult(
                member_id=member_id,
                rule_score=rule_score,
                hits=tuple(hits),
                features=features,
                flags=flags,
                skipped_rules=skipped_rules,
                errors=tuple(errors),
            )
        )

    return DetectionResult(
        auction_id=inp.auction.auction_id,
        as_of=inp.as_of,
        rule_config_version=cfg.version,
        results=tuple(results),
        skipped_bidders=skipped,
    )


def _weighted_average(hits: list[RuleHit]) -> float:
    """점수를 낸 규칙만으로 가중 평균한다.

    게이트에 걸린 규칙을 0점으로 세면 데이터가 부족할수록 점수가 낮아진다.
    "판단하지 않음" 과 "위험하지 않음" 은 다르므로 분모에서 제외한다.
    """
    total_weight = sum(h.weight for h in hits)
    if total_weight <= 0:
        return 0.0
    return sum(h.score * h.weight for h in hits) / total_weight


def combine(rule_score: float, ml_score: float | None, w_rule: float, w_ml: float) -> float:
    """규칙 점수와 모델 점수를 합쳐 최종 risk_score 를 만든다.

    피처를 합치는 것이 아니라 각각 점수를 낸 뒤 합친다. eBay 로 학습한 모델은
    입력 피처 8개가 고정이라 우리 신규 피처를 넣을 수 없기 때문이다.
    두 점수는 각각 저장해야 나중에 어느 쪽이 정확했는지 비교할 수 있다.
    """
    if ml_score is None:
        return rule_score
    total = w_rule + w_ml
    if total <= 0:
        return rule_score
    return (w_rule * rule_score + w_ml * ml_score) / total
