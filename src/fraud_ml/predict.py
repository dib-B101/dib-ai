"""학습된 모델을 불러와 점수를 낸다.

`artifacts/` 의 joblib 파일은 **확률 보정까지 포함한 파이프라인**이다. 보정 없이 나온
확률은 0.8 이 실제 80% 를 뜻하지 않아 등급 경계(0.3 / 0.6)가 의미를 잃는다.

모델이 없어도 서버는 뜬다. 이상거래 탐지는 규칙 트랙만으로도 동작하고, 모델은
가중치 `w_ml` 로 얹히는 구조다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fraud.ml_features import FEATURE_NAMES, to_vector

log = logging.getLogger("fraud_ml.predict")

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_FILE = ARTIFACTS / "shill_lgbm_v0.joblib"
META_FILE = ARTIFACTS / "model_meta.json"


class ModelNotAvailable(RuntimeError):
    """모델 파일이 없거나 계약이 어긋난다. 규칙 트랙만으로 계속 간다."""


class FraudModel:
    """점수 계산기. 서버 시작 시 한 번 만들어 재사용한다."""

    def __init__(self, model, version: str, features: tuple[str, ...]) -> None:
        self._model = model
        self.version = version
        self.features = features

    @classmethod
    def load(cls, model_path: Path | None = None, meta_path: Path | None = None) -> "FraudModel":
        model_path = model_path or MODEL_FILE
        meta_path = meta_path or META_FILE

        if not model_path.exists():
            raise ModelNotAvailable(
                f"모델 파일이 없습니다: {model_path}. "
                "scripts/train_bootstrap_model.py 로 생성하십시오."
            )

        import joblib

        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        features = tuple(meta.get("features", FEATURE_NAMES))
        version = meta.get("model_version", "unknown")

        # 우리가 계산하는 피처와 모델이 학습한 피처가 같은 집합인지 본다.
        # **순서는 맞추지 않고 모델 쪽을 따른다** — 하드코딩한 순서가 모델과
        # 어긋나면 값이 뒤섞여 조용히 틀린 점수가 나온다.
        missing = set(features) - set(FEATURE_NAMES)
        extra = set(FEATURE_NAMES) - set(features)
        if missing or extra:
            raise ModelNotAvailable(
                "학습 피처와 서빙 피처가 다릅니다.\n"
                f"  모델이 요구하는데 계산하지 않음: {sorted(missing) or '없음'}\n"
                f"  계산하는데 모델이 안 씀:        {sorted(extra) or '없음'}"
            )

        log.info("모델 로드 완료 — %s, 피처 %d개", version, len(features))
        return cls(model, version, features)

    def score(self, features: dict[str, float]) -> float:
        """허위입찰 확률 [0, 1]. 보정된 값이다."""
        return self.score_many([features])[0]

    def score_many(self, rows: list[dict[str, float]]) -> list[float]:
        """여러 명을 한 번에. 입찰자마다 모델을 부르면 경매 하나에 수십 번이 된다."""
        if not rows:
            return []

        # 컬럼 이름을 붙여 넘긴다. 이름 없는 배열로 주면 sklearn 이 경고만 내고
        # 순서를 그대로 믿는데, 순서가 어긋나도 알려주지 않는다.
        import pandas as pd

        frame = pd.DataFrame(
            [to_vector(r, self.features) for r in rows], columns=list(self.features)
        )
        return [float(p[1]) for p in self._model.predict_proba(frame)]
