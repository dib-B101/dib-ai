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

# eBay 는 경매 기간을 1·3·5·7·10 "일" 로 기록한다. 우리 경매는 몇 분에서 한 시간이라
# 모든 값이 학습 범위 왼쪽 바깥으로 떨어진다. 트리는 우리 데이터를 전부 같은 쪽으로
# 보내므로 이 피처는 서빙에서 상수가 된다 — eBay 평가 점수와 무관하게 죽은 피처다.
#
# 다른 코퍼스 상대 피처(Auction_Bids · Starting_Price_Average)는 eBay 값 자체가
# 0~1 정규화라 우리 코퍼스로 같은 변환을 하면 의미가 대응된다. 이것만 절대 단위라
# 대응점이 없다.
OUT_OF_RANGE = "Auction_Duration"

# 원식과 방향을 특정할 수 없다. 데이터를 열어보면 정상 중앙값 0.000 / 허위 중앙값 0.961 의
# 이봉 분포인데 Class 와의 상관은 +0.043 로 거의 0 이다. 정규화 기준(판매자별·카테고리별·
# 전체)도, 값을 뒤집었는지도 알 수 없다.
#
# 학습은 eBay 원본 값으로 하고 서빙은 우리가 추정한 식으로 계산하므로, **방향이 반대면
# 모델이 정반대로 읽는다.** 아예 빼는 것보다 나쁘다. 5시드 짝비교에서 빼도 성능 차이가
# 없었으므로(PR-AUC +0.0016 ± 0.0090) 위험만 없앴다.
UNSPECIFIABLE = "Starting_Price_Average"

# 판매자 편중도는 **재정의해서 쓴다.**
#
# eBay 정의 그대로 쓰면 구독 모델에서 단골 구매자가 고위험으로 찍힌다. 그래서
# 서빙에서는 구독 중인 판매자와의 참여를 분자·분모에서 빼고 계산한다.
#
#     편중도 = (구독하지 않은 판매자 경매 참여 수) / (구독하지 않은 전체 참여 수)
#
# eBay 데이터에는 구독 개념이 없어 뺄 것이 없으므로, **학습은 원본 값 그대로** 하고
# 서빙에서만 구독 건을 제외한다. "높으면 의심" 이라는 의미가 양쪽에서 유지된다.
REDEFINED = "Bidder_Tendency"

# 서빙 입력 스키마. 백엔드가 준비해야 할 값이자 model_meta.json 의 계약이다.
#
# 성능만 보면 어느 조합도 유의미하게 낫지 않다(5시드 기준 표준편차가 조합 간 차이보다
# 크다). 그래서 **우리 데이터로 재현 가능한가**를 기준으로 골랐다.
SERVING_FEATURES = (
    "Bidding_Ratio",
    "Last_Bidding",
    "Auction_Bids",
    "Early_Bidding",
    "Winning_Ratio",
    "Bidder_Tendency",
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

    # 임계값을 바꾸면 관리자 검토량과 정밀도가 함께 움직인다. 모델 성능이 아니라
    # **운영 설계**를 정하는 표다. 인력이 하루에 볼 수 있는 건수에서 역산한다.
    operating_points: list[dict] = field(default_factory=list)

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


# 하이퍼파라미터는 탐색으로 정했다.
#
#   ① 평가셋 25% 를 경매 단위로 먼저 떼어낸다 (탐색이 절대 보지 못한다)
#   ② 나머지 75% 에서 GroupKFold(5) · 120회 무작위 탐색
#   ③ 떼어둔 평가셋으로 한 번만 측정
#
# 탐색 CV 0.7665 → 최종 평가 0.7539 로 0.013 만 떨어졌다. 탐색이 평가셋에
# 과적합하지 않았다는 뜻이다.
#
# 손으로 고른 기존 설정 대비 PR-AUC +0.039, 재현율 0.713 → 0.812 였다.
# **트리를 얕게(15→7) 하고 학습률을 낮춰(0.05→0.011) 천천히 학습**하는 쪽이 나았다.
# 6천 행짜리 데이터에 복잡한 모델이 과적합하고 있었다.
TUNED_PARAMS = {
    "n_estimators": 324,
    "learning_rate": 0.011041,
    "num_leaves": 7,
    "min_child_samples": 36,
    "subsample": 0.79578,
    "colsample_bytree": 0.76441,
    "reg_alpha": 0.0090834,
    "reg_lambda": 0.0037982,
}


def compute_operating_points(y_true, prob, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> list[dict]:
    """임계값별 검토량·정밀도·놓친 건수.

    "정밀도 0.63" 만으로는 운영 판단이 안 된다. **몇 건을 검토해야 하고 몇 건을
    놓치는지**가 실제로 정해야 할 값이다.
    """
    y_true = np.asarray(y_true)
    total_positive = int(y_true.sum())
    rows = []
    for t in thresholds:
        flagged = prob >= t
        n = int(flagged.sum())
        hit = int((flagged & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "flagged": n,                                   # 관리자가 볼 건수
                "precision": hit / n if n else 0.0,
                "recall": hit / total_positive if total_positive else 0.0,
                "missed": total_positive - hit,                 # 놓친 허위입찰
            }
        )
    return rows


def _make_model() -> LGBMClassifier:
    """LightGBM 을 쓰는 이유는 성능보다 ``monotone_constraints`` 때문이다.

    '입찰 비중이 높을수록 위험도가 낮아질 수는 없다' 같은 도메인 규칙을 모델에
    강제할 수 있어야, 관리자가 판정 근거를 물었을 때 설명이 무너지지 않는다.
    오탐 이의제기가 정책에 있는 이상 설명 가능성은 정확도만큼 중요하다.
    """
    return LGBMClassifier(
        class_weight="balanced",
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **TUNED_PARAMS,
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
        operating_points=compute_operating_points(y_test, p_cal),
    )
    return report, calibrated, base


ADOPTED = "C"   # 서빙에 쓸 구성. 아래 experiments() 의 라벨 첫 글자와 맞춘다.


def experiments(df: pd.DataFrame) -> list[tuple[str, tuple[str, ...]]]:
    """절제 실험 — 어떤 피처를 넣고 빼느냐의 차이다.

    A  9개 전부           이 데이터셋의 상한. 우리는 쓸 수 없으므로 참고용이다
    B  A − 누수 피처       Successive_Outbidding 은 정책상 항상 0 이라 계산 불가
    C  B − 경매 기간 − 시작가   재현 불가한 둘을 뺀 것. **서빙 채택 구성**
    D  상위 3개만          최소 구성으로 어디까지 되는지

    **단일 시드 결과로 조합을 고르지 말 것.** 5시드로 재보면 표준편차가 0.02~0.06 이라
    조합 간 차이가 대부분 그 안에 들어간다. 성능으로는 우열을 가릴 수 없으므로
    "우리 데이터로 재현 가능한가" 를 기준으로 C 를 골랐다.
    """
    all_feats = tuple(c for c in df.columns if c not in ID_COLS + (TARGET,))
    without_leak = tuple(f for f in all_feats if f != LEAKED)
    without_both = tuple(
        f for f in without_leak if f not in (OUT_OF_RANGE, UNSPECIFIABLE)
    )
    return [
        ("A. 전체 9개 — 참고용, 누수 포함", all_feats),
        ("B. Successive_Outbidding 제외 — 정책상 항상 0", without_leak),
        ("C. B + 경매기간·시작가 제외 — 재현 불가", without_both),
        ("D. 상위 3개만 — 최소 구성", ("Bidding_Ratio", "Winning_Ratio", "Bidder_Tendency")),
    ]
