"""검수 파이프라인 — 1차 규칙 필터 → 2차 AI → 판정.

    상품 등록
        ↓
    1차 규칙 필터
        ├─ BLOCK     제목에 명확한 금칙어 → AI 호출 없이 즉시 차단
        ├─ ESCALATE  설명에만 있거나 우회 의심 → 신호를 붙여 AI 로
        └─ PASS      아무것도 안 걸림 → 일반 AI 검수
        ↓
    2차 AI 검수 (Claude 멀티모달)
        ↓
    confidence 로 등급 조정
        ↓
    정상 / 검토 필요 / 금지

설계 원칙 두 가지.

**오탐이 미탐보다 비싸다.** 정상 상품을 막으면 판매자가 이탈하지만, 금지 품목이
한 번 통과해도 신고·사후 탐지로 잡을 수 있다. 그래서 확신이 없으면 차단이 아니라
검토 필요로 둔다.

**AI 호출 실패가 상품 등록을 막지 않는다.** 외부 API 는 언제든 죽는다. 실패하면
등록을 거부하는 대신 검토 필요로 보류하고 관리자·재시도로 넘긴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .keywords import KeywordFilter, KeywordHit
from .llm import ModerationLLM
from .schema import ModerationResult, ProductInput, RuleOutcome, Stage, Verdict

log = logging.getLogger("moderation")


@dataclass(frozen=True, slots=True)
class Thresholds:
    """confidence 로 판정을 조정하는 경계.

    초기에는 보수적으로 잡는다 — 검토 필요를 넓게 두면 관리자 일이 늘지만
    정상 상품이 막히는 일은 줄어든다. 관리자 판정 로그가 쌓이면 조정한다.
    """

    block_min_confidence: float = 0.85   # 이보다 낮으면 금지 대신 검토 필요
    normal_min_confidence: float = 0.70  # 이보다 낮으면 정상 대신 검토 필요


class ModerationPipeline:
    def __init__(
        self,
        keyword_filter: KeywordFilter,
        llm: ModerationLLM | None = None,
        thresholds: Thresholds | None = None,
    ) -> None:
        self._keywords = keyword_filter
        self._llm = llm
        self._thresholds = thresholds or Thresholds()

    def review(self, product: ProductInput) -> ModerationResult:
        """상품 1건을 검수한다. 예외를 밖으로 던지지 않는다."""
        content_hash = product.content_hash()
        outcome, hits = self._keywords.decide(product.title, product.description)

        if outcome is RuleOutcome.BLOCK:
            return self._blocked_by_rule(product, hits, content_hash)

        hint = _hint(hits) if outcome is RuleOutcome.ESCALATE else None

        if self._llm is None:
            return ModerationResult(
                product_id=product.product_id,
                verdict=Verdict.NEEDS_REVIEW,
                stage=Stage.FALLBACK,
                reason="AI 검수가 구성되지 않아 관리자 확인이 필요합니다.",
                content_hash=content_hash,
                detail={"rule_outcome": outcome.value, "keyword_hits": _dump(hits)},
            )

        try:
            judged = self._llm.judge(product, hint)
        except Exception as exc:
            # 등록을 거부하지 않는다. 보류 후 재시도·관리자 확인으로 넘긴다.
            log.exception("AI 검수 실패 product_id=%s", product.product_id)
            return ModerationResult(
                product_id=product.product_id,
                verdict=Verdict.NEEDS_REVIEW,
                stage=Stage.FALLBACK,
                reason="AI 검수를 완료하지 못해 관리자 확인이 필요합니다.",
                content_hash=content_hash,
                detail={
                    "rule_outcome": outcome.value,
                    "keyword_hits": _dump(hits),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        verdict = self._apply_thresholds(judged.verdict, judged.confidence)
        return ModerationResult(
            product_id=product.product_id,
            verdict=verdict,
            stage=Stage.AI,
            category=judged.category,
            confidence=judged.confidence,
            reason=judged.reason,
            content_hash=content_hash,
            detail={
                "rule_outcome": outcome.value,
                "keyword_hits": _dump(hits),
                "ai_verdict": judged.verdict.value,
                "model_version": judged.model_version,
                "downgraded": verdict is not judged.verdict,
            },
        )

    def _blocked_by_rule(
        self, product: ProductInput, hits: list[KeywordHit], content_hash: str
    ) -> ModerationResult:
        hit = next(h for h in hits if h.matched_in == "title" and not h.evasion)
        return ModerationResult(
            product_id=product.product_id,
            verdict=Verdict.BLOCKED,
            stage=Stage.RULE,
            category=hit.category,
            confidence=1.0,
            reason=f"상품명에 거래 제한 품목({hit.category})이 포함되어 등록할 수 없습니다.",
            content_hash=content_hash,
            detail={"rule_outcome": RuleOutcome.BLOCK.value, "keyword_hits": _dump(hits)},
        )

    def _apply_thresholds(self, verdict: Verdict, confidence: float) -> Verdict:
        """확신이 부족하면 양쪽 모두 검토 필요로 내린다.

        금지 쪽만 낮추면 오탐은 줄지만 미탐이 그대로 통과한다. 정상 쪽도 같이
        내려야 "애매한 건 사람이 본다" 는 원칙이 성립한다.
        """
        t = self._thresholds
        if verdict is Verdict.BLOCKED and confidence < t.block_min_confidence:
            return Verdict.NEEDS_REVIEW
        if verdict is Verdict.NORMAL and confidence < t.normal_min_confidence:
            return Verdict.NEEDS_REVIEW
        return verdict


def _hint(hits: list[KeywordHit]) -> str:
    parts = [f"{h.category}/{h.keyword}({h.matched_in}" + (", 우회 의심)" if h.evasion else ")")
             for h in hits]
    return ", ".join(parts)


def _dump(hits: list[KeywordHit]) -> list[dict]:
    return [
        {
            "keyword": h.keyword,
            "category": h.category,
            "matched_in": h.matched_in,
            "evasion": h.evasion,
        }
        for h in hits
    ]
