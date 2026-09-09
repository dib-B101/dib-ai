"""1차 규칙 기반 금칙어 필터.

목적은 **AI 호출을 줄이는 것**이다. 명확한 위반은 여기서 0.001초에 걸러내고,
애매한 것만 LLM 으로 넘긴다. 비용과 응답 지연이 함께 줄어든다.

Aho-Corasick 을 쓰는 이유는 금칙어가 수천 개로 늘어나도 텍스트를 **한 번만** 훑기
때문이다. 정규식 반복문은 금칙어 수에 비례해 느려진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import ahocorasick
import yaml

from .normalize import variants
from .schema import RuleOutcome

DEFAULT_KEYWORDS_PATH = Path(__file__).resolve().parents[2] / "config" / "banned_keywords.yaml"


@dataclass(frozen=True, slots=True)
class KeywordHit:
    keyword: str        # 사전에 등록된 원래 표기
    category: str       # 담배 · 주류 · 의약품 ...
    matched_in: str     # title | description
    evasion: bool       # 유사 문자 치환까지 해야 걸린 경우


class KeywordFilter:
    """금칙어 사전을 Aho-Corasick 오토마타로 올려두고 재사용한다."""

    def __init__(self, categories: Mapping[str, Iterable[str]]) -> None:
        self._automaton = ahocorasick.Automaton()
        self._count = 0
        for category, words in categories.items():
            for word in words:
                for key in set(variants(word)):
                    if not key:
                        continue
                    self._automaton.add_word(key, (word, category))
                    self._count += 1
        if self._count:
            self._automaton.make_automaton()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "KeywordFilter":
        p = Path(path) if path else DEFAULT_KEYWORDS_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(raw.get("categories", {}))

    @property
    def size(self) -> int:
        return self._count

    def _scan(self, text: str, field: str) -> list[KeywordHit]:
        if not text or not self._count:
            return []
        plain, aggressive = variants(text)
        hits: dict[tuple[str, str], KeywordHit] = {}

        for _, (word, category) in self._automaton.iter(plain):
            hits[(word, field)] = KeywordHit(word, category, field, evasion=False)

        # 유사 문자까지 치환해야 걸리는 것은 우회 시도로 본다.
        # 오탐 가능성이 있어 별도 표시해 두고 관리자가 판단할 수 있게 한다.
        if aggressive != plain:
            for _, (word, category) in self._automaton.iter(aggressive):
                hits.setdefault((word, field), KeywordHit(word, category, field, evasion=True))

        return list(hits.values())

    def scan(self, title: str, description: str | None = None) -> list[KeywordHit]:
        """제목과 설명에서 걸린 금칙어를 돌려준다. 없으면 빈 리스트다."""
        found = self._scan(title, "title")
        found += self._scan(description or "", "description")
        return found

    def decide(self, title: str, description: str | None = None) -> tuple[RuleOutcome, list[KeywordHit]]:
        """1차 필터의 결론을 낸다.

        통과/차단 둘로만 나누면 오탐이 난다. 실제로 이런 문장들이 걸린다.

            "명품 가방 판매"  /  "레플리카 아닙니다"
            "캠핑용 나이프"   /  "사시미칼 아니고 캠핑용입니다"

        정상 판매자가 흔히 쓰는 문구인데 키워드만 보면 차단된다. 규칙은 부정문을
        읽지 못하므로, **문맥이 필요한 경우는 판단하지 말고 LLM 에 넘긴다.**

            제목에 그대로 있음        → BLOCK     문맥을 볼 것도 없다
            설명에만 있음             → ESCALATE  부정문일 수 있다
            유사 문자 치환해야 걸림    → ESCALATE  오탐 가능성이 있다
        """
        hits = self.scan(title, description)
        if not hits:
            return RuleOutcome.PASS, []

        certain = [h for h in hits if h.matched_in == "title" and not h.evasion]
        if certain:
            return RuleOutcome.BLOCK, hits
        return RuleOutcome.ESCALATE, hits
