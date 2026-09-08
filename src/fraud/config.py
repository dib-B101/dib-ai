"""규칙 설정 로더.

임계값을 코드에 박지 않고 YAML 로 분리한다.
version 은 판정 결과에 함께 기록되어, 임계값을 바꿔도 과거 판정을 해석할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"


@dataclass(frozen=True, slots=True)
class RuleSpec:
    rule_id: str
    enabled: bool
    weight: float
    params: Mapping[str, Any]

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass(frozen=True, slots=True)
class RuleConfig:
    version: str
    min_bids: int
    min_bidders: int
    rules: Mapping[str, RuleSpec]
    bands: Mapping[str, tuple[float, float]]

    @property
    def enabled_rules(self) -> list[RuleSpec]:
        """rule_id 오름차순. 순서를 고정해야 결과가 재현된다."""
        return [self.rules[k] for k in sorted(self.rules) if self.rules[k].enabled]

    def band_of(self, score: float) -> str:
        for name, (lo, hi) in self.bands.items():
            if lo <= score < hi:
                return name
        return "high"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RuleConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuleConfig":
        gates = raw.get("gates", {})
        rules = {
            rid: RuleSpec(
                rule_id=rid,
                enabled=bool(spec.get("enabled", False)),
                weight=float(spec.get("weight", 0.0)),
                params=dict(spec.get("params") or {}),
            )
            for rid, spec in (raw.get("rules") or {}).items()
        }
        bands = {
            name: (float(lo), float(hi))
            for name, (lo, hi) in (raw.get("bands") or {}).items()
        }
        return cls(
            version=str(raw.get("version", "unversioned")),
            min_bids=int(gates.get("min_bids", 1)),
            min_bidders=int(gates.get("min_bidders", 1)),
            rules=rules,
            bands=bands,
        )
