"""검수 입출력 자료구조.

이 모듈은 DB 도 HTTP 도 모른다. 순수한 값 객체만 정의한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """최종 판정. product.status 와 대응한다."""

    NORMAL = "정상"          # → REGISTERED
    NEEDS_REVIEW = "검토 필요"  # → PENDING 유지, 관리자 큐로
    BLOCKED = "금지"          # → REJECTED


class Stage(str, Enum):
    RULE = "rule"       # 1차 규칙 필터에서 결정
    AI = "ai"           # 2차 LLM 판정
    FALLBACK = "fallback"  # LLM 호출 실패 시 보수적 처리


class RuleOutcome(str, Enum):
    """1차 필터의 결론. 통과/차단 둘로 나누면 오탐이 늘어난다.

    제목에 금칙어가 그대로 있으면 문맥을 볼 것도 없이 위반이다. 하지만 설명에만
    있으면 "레플리카 아닙니다" 처럼 부정문일 수 있고, 유사 문자 치환까지 해야
    걸린 것도 오탐 가능성이 있다. 그런 건 차단하지 말고 LLM 에 넘겨 문맥을 보게 한다.
    """

    BLOCK = "block"        # 즉시 차단. AI 를 부르지 않는다
    ESCALATE = "escalate"  # 의심 신호와 함께 AI 로 넘긴다
    PASS = "pass"          # 아무것도 안 걸림. 일반 AI 검수로


@dataclass(frozen=True, slots=True)
class ProductInput:
    """검수 대상. product 테이블 + product_image 에서 온다."""

    product_id: int
    title: str
    description: str | None = None
    image_urls: tuple[str, ...] = ()
    category_name: str | None = None

    def content_hash(self) -> str:
        """상품명·설명·이미지의 해시.

        같은 내용을 다시 검수하지 않기 위한 캐시 키이자, 수정 시 재검수가 필요한지
        판단하는 기준이다. 가격만 바꾼 경우까지 LLM 을 부르면 비용과 지연이 낭비된다.
        """
        import hashlib

        payload = "\x1f".join(
            [self.title, self.description or "", *sorted(self.image_urls)]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModerationResult:
    product_id: int
    verdict: Verdict
    stage: Stage
    category: str | None = None       # 담배 · 주류 · 의약품 ...
    confidence: float | None = None   # AI 판정일 때만
    reason: str = ""                  # 사용자 안내와 관리자 검토에 그대로 쓴다
    content_hash: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.NORMAL
