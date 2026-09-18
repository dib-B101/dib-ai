"""추천 랭킹 설정 로더.

가중치를 코드에 박지 않고 YAML 로 분리한다. version 이 추천 응답에 함께 나가므로,
값을 바꿔도 과거 결과가 어떤 설정으로 나온 것인지 해석할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "reco.yaml"


@dataclass(frozen=True, slots=True)
class RecoConfig:
    version: str
    tau_seconds: float
    popularity: Mapping[str, float]
    competition: Mapping[str, float]
    weights: Mapping[str, float]
    similarity_weight: float
    similarity_text_weight: float

    # 개인화 (STEP 5). 행동 로그가 없으면 쓰이지 않는다.
    personalization_weight: float
    half_life_days: float
    lookback_days: int
    max_events: int
    event_weights: Mapping[str, float]

    limit: int

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RecoConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        cfg = cls(
            version=raw.get("version", "unknown"),
            tau_seconds=float(raw.get("urgency", {}).get("tau_seconds", 3600)),
            popularity=dict(raw.get("popularity", {})),
            competition=dict(raw.get("competition", {})),
            weights=dict(raw.get("weights", {})),
            similarity_weight=float(raw.get("similarity", {}).get("weight", 0.5)),
            similarity_text_weight=float(raw.get("similarity", {}).get("text", 0.6)),
            personalization_weight=float(
                raw.get("personalization", {}).get("weight", 0.4)
            ),
            half_life_days=float(
                raw.get("personalization", {}).get("half_life_days", 14)
            ),
            lookback_days=int(raw.get("personalization", {}).get("lookback_days", 90)),
            max_events=int(raw.get("personalization", {}).get("max_events", 500)),
            event_weights=dict(raw.get("personalization", {}).get("events", {})),
            limit=int(raw.get("limit", 50)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """설정이 틀렸으면 서버가 뜰 때 죽는 편이 낫다.

        가중치 오타 하나로 추천 순서가 조용히 망가지면 원인을 찾는 데 훨씬 오래 걸린다.
        """
        for name, value in (
            ("similarity.weight", self.similarity_weight),
            ("similarity.text", self.similarity_text_weight),
            ("personalization.weight", self.personalization_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 는 0~1 이어야 합니다: {value}")
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days 는 양수여야 합니다: {self.half_life_days}")
        if self.tau_seconds <= 0:
            raise ValueError(f"tau_seconds 는 양수여야 합니다: {self.tau_seconds}")

        for name, group in (
            ("popularity", self.popularity),
            ("competition", self.competition),
            ("weights", self.weights),
        ):
            if not group:
                raise ValueError(f"{name} 설정이 비어 있습니다")
            total = sum(group.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"{name} 가중치 합이 1 이 아닙니다: {total}")
