"""eBay 데이터로 부트스트랩 모델을 학습한다.

이 모델은 **배포용이 아니다.** eBay 데이터셋은 우리 서비스와 도메인이 달라
성능 숫자를 그대로 인용할 수 없다. 목적은 두 가지다.

1. 학습·평가·서빙 파이프라인이 실제로 도는지 확인한다
2. 서빙 입력 스키마를 확정해 백엔드가 준비할 데이터를 명확히 한다

핵심 제약이 하나 있다. 이 데이터셋에서 가장 강한 피처인 ``Successive_Outbidding``
(자기 자신을 연속 추월한 정도)은 우리 정책상 **항상 0** 이다. 최고 입찰자는 추가
입찰이 막혀 있기 때문이다. 그래서 그 피처를 제외한 조건이 우리의 실제 상한이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42

TARGET = "Class"
GROUP = "Auction_ID"
ID_COLS = ("Record_ID", "Auction_ID", "Bidder_ID")

# 우리 정책상 최고 입찰자는 추가 입찰이 불가능해 A→A→A 가 나올 수 없다.
# 따라서 이 값은 모든 행에서 항상 0 이며, 학습에 넣을 수 없다.
LEAKED = "Successive_Outbidding"

# 라이브 구독 모델에서는 구독자가 좋아하는 방송자의 경매에만 참여하는 것이 정상이다.
# eBay 모델은 "판매자 편중이 높으면 위험" 으로 배우므로 충성 고객이 고위험으로 찍힌다.
# 실험 결과 제외하는 편이 성능·오탐 양쪽에서 낫기도 해서 서빙 피처에서 뺀다.
SUBSCRIPTION_CONFLICT = "Bidder_Tendency"

# 서빙 입력 스키마. 백엔드가 준비해야 할 값이자 model_meta.json 의 계약이다.
SERVING_FEATURES = (
    "Bidding_Ratio",
    "Last_Bidding",
    "Auction_Bids",
    "Starting_Price_Average",
    "Early_Bidding",
    "Winning_Ratio",
    "Auction_Duration",
)


@dataclass
class Report:
    name: str
    features: tuple[str, ...]
    roc_auc: float
    pr_auc: float
    baseline_pr: float
    best_f1: float
    best_threshold: float
    precision_at_best: float
    recall_at_best: float
    brier_raw: float
    brier_calibrated: float
    importance: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.name:<34} PR-AUC {self.pr_auc:.4f}  ROC {self.roc_auc:.4f}  "
            f"F1 {self.best_f1:.3f}  P {self.precision_at_best:.3f} R {self.recall_at_best:.3f}"
        )


def load(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """경매 단위 분할.

    같은 경매의 행들은 Auction_Bids · Starting_Price_Average · Auction_Duration 값을
    공유한다. 행 단위로 나누면 같은 경매가 학습과 평가 양쪽에 걸쳐 성능이 부풀려진다.
    실서비스에서도 '처음 보는 경매'를 판정하므로 이 분할이 운영 조건과 맞는다.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, df[TARGET], groups=df[GROUP]))


def _make_model() -> LGBMClassifier:
    """LightGBM 을 쓰는 이유는 성능보다 ``monotone_constraints`` 때문이다.

    '입찰 비중이 높을수록 위험도가 낮아질 수는 없다' 같은 도메인 규칙을 모델에
    강제할 수 있어야, 관리자가 판정 근거를 물었을 때 설명이 무너지지 않는다.
    오탐 이의제기가 정책에 있는 이상 설명 가능성은 정확도만큼 중요하다.
    """
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def train_one(
    df: pd.DataFrame,
    features: Sequence[str],
    name: str,
    *,
    seed: int = RANDOM_STATE,
) -> tuple[Report, CalibratedClassifierCV, LGBMClassifier]:
    """학습 → 보정 → 평가.

    보정 데이터를 학습 데이터에서 다시 떼어낸다(prefit 방식). 같은 데이터로
    학습과 보정을 하면 보정이 무의미해지기 때문이다.
    """
    feats = list(features)
    trainval_idx, test_idx = _split(df, 0.30, seed)
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    fit_idx, calib_idx = _split(trainval, 0.25, seed + 1)
    fit_df, calib_df = trainval.iloc[fit_idx], trainval.iloc[calib_idx]

    base = _make_model().fit(fit_df[feats], fit_df[TARGET])

    # FrozenEstimator 로 감싸면 이미 학습된 모델을 다시 학습하지 않고 보정만 한다.
    # (sklearn 1.6 에서 cv="prefit" 을 대체했다)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(calib_df[feats], calib_df[TARGET])

    y_test = test[TARGET].to_numpy()
    p_raw = base.predict_proba(test[feats])[:, 1]
    p_cal = calibrated.predict_proba(test[feats])[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, p_cal)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    k = int(np.nanargmax(f1[:-1]))

    importance = dict(
        sorted(
            zip(feats, base.feature_importances_ / max(base.feature_importances_.sum(), 1)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    report = Report(
        name=name,
        features=tuple(feats),
        roc_auc=roc_auc_score(y_test, p_cal),
        pr_auc=average_precision_score(y_test, p_cal),
        baseline_pr=float(y_test.mean()),
        best_f1=float(f1[k]),
        best_threshold=float(thresholds[k]),
        precision_at_best=float(precision[k]),
        recall_at_best=float(recall[k]),
        brier_raw=brier_score_loss(y_test, p_raw),
        brier_calibrated=brier_score_loss(y_test, p_cal),
        importance={k_: round(v, 4) for k_, v in importance.items()},
    )
    return report, calibrated, base


ADOPTED = "C"   # 서빙에 쓸 구성. 아래 experiments() 의 라벨 첫 글자와 맞춘다.


def experiments(df: pd.DataFrame) -> list[tuple[str, tuple[str, ...]]]:
    """절제 실험 — 어떤 피처를 넣고 빼느냐의 차이다.

    A  9개 전부           이 데이터셋의 상한. 우리는 쓸 수 없으므로 참고용이다
    B  A − 누수 피처       Successive_Outbidding 은 정책상 항상 0 이라 계산 불가
    C  B − 판매자 편중도   구독 모델과 충돌한다. **서빙 채택 구성**
    D  상위 3개만          최소 구성으로 어디까지 되는지

    C 가 B 보다 성능이 좋다는 점이 중요하다. 구독 충돌을 피하려고 성능을 포기하는
    트레이드오프가 아니라, 빼는 편이 오탐까지 줄어드는 순이득이다.
    """
    all_feats = tuple(c for c in df.columns if c not in ID_COLS + (TARGET,))
    without_leak = tuple(f for f in all_feats if f != LEAKED)
    without_both = tuple(f for f in without_leak if f != SUBSCRIPTION_CONFLICT)
    return [
        ("A. 전체 9개 — 참고용, 누수 포함", all_feats),
        ("B. Successive_Outbidding 제외 — 정책상 항상 0", without_leak),
        ("C. B + Bidder_Tendency 제외 — 구독 충돌", without_both),
        ("D. 상위 3개만 — 최소 구성", ("Bidding_Ratio", "Winning_Ratio", "Bidder_Tendency")),
    ]
