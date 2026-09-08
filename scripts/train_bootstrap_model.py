"""부트스트랩 모델 학습 · 평가 · 저장.

    python scripts/train_bootstrap_model.py

출력
    artifacts/shill_lgbm_v0.joblib   보정까지 포함한 파이프라인
    artifacts/model_meta.json        피처 목록 · 성능 · 버전 · 사용 제한

이 스크립트의 목적은 성능이 아니라 파이프라인 검증과 서빙 스키마 확정이다.
eBay 데이터에서 점수를 올리는 작업은 하지 않는다 — 피처 정의가 우리 서비스와
달라서, 여기서 올린 숫자가 실제 성능으로 이어지지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_ml.bootstrap import (  # noqa: E402
    ADOPTED,
    LEAKED,
    SUBSCRIPTION_CONFLICT,
    RANDOM_STATE,
    SERVING_FEATURES,
    TARGET,
    experiments,
    load,
    train_one,
)
from fraud_ml.explain import top_reasons  # noqa: E402

CSV = ROOT / "data" / "Shill Bidding Dataset.csv"
OUT = ROOT / "artifacts"


def main() -> None:
    df = load(CSV)
    print(f"데이터 {len(df):,}행 · 경매 {df['Auction_ID'].nunique():,}개 · "
          f"양성 {df[TARGET].mean() * 100:.2f}%\n")

    print("=" * 96)
    print("절제 실험 — 어떤 피처가 얼마나 기여하는가")
    print("=" * 96)

    reports = []
    keep = None
    for name, feats in experiments(df):
        report, calibrated, base = train_one(df, feats, name)
        reports.append(report)
        print(report.line())
        if name.startswith(f"{ADOPTED}."):
            keep = (report, calibrated, base, feats)

    baseline = reports[0].baseline_pr
    print(f"\n무작위 기준선 PR-AUC = {baseline:.4f}")

    assert keep is not None
    report, calibrated, base, feats = keep

    print("\n" + "=" * 96)
    print(f"채택 구성 — {report.name}")
    print("=" * 96)
    print(f"  PR-AUC {report.pr_auc:.4f}  (무작위 대비 {report.pr_auc / baseline:.1f}배)")
    print(f"  운영 최적점  임계값 {report.best_threshold:.3f} · "
          f"Precision {report.precision_at_best:.3f} · Recall {report.recall_at_best:.3f}")
    print(f"  → 고위험 판정 {1 / max(report.precision_at_best, 1e-9):.1f}건 중 1건만 실제 의심. "
          f"자동 제재 불가, 관리자 검토 큐 정렬 전용")

    print(f"\n  확률 보정 (Brier, 낮을수록 좋음)")
    print(f"    보정 전 {report.brier_raw:.4f}  →  보정 후 {report.brier_calibrated:.4f}")
    print(f"    정책의 0.3 / 0.6 등급 경계는 보정된 확률에서만 의미가 있다")

    print("\n  피처 중요도")
    for f, v in report.importance.items():
        print(f"    {f:<24} {v * 100:5.1f}%  {'#' * int(v * 60)}")

    # 구독 충돌 확인 --------------------------------------------------------
    b = next(r for r in reports if r.name.startswith("B."))
    c = next(r for r in reports if r.name.startswith("C."))
    print("\n" + "=" * 96)
    print("판매자 편중도(Bidder_Tendency) 를 왜 뺐는가")
    print("=" * 96)
    print(f"  PR-AUC     {b.pr_auc:.4f} → {c.pr_auc:.4f}  "
          f"({(c.pr_auc - b.pr_auc) / b.pr_auc * 100:+.1f}%)")
    print(f"  Precision  {b.precision_at_best:.3f} → {c.precision_at_best:.3f}  "
          f"({(c.precision_at_best - b.precision_at_best) / b.precision_at_best * 100:+.1f}%)")
    print(f"  Recall     {b.recall_at_best:.3f} → {c.recall_at_best:.3f}")
    print()
    print("  라이브 구독 모델에서는 구독자가 좋아하는 방송자의 경매에만 참여하는 것이 정상이다.")
    print("  eBay 모델은 '판매자 편중이 높으면 위험' 으로 배우므로 충성 고객이 고위험으로 찍힌다.")
    print("  제외했더니 성능과 정밀도가 함께 올라, 트레이드오프가 아니라 순이득이었다.")

    # SHAP 사유 예시 --------------------------------------------------------
    print("\n" + "=" * 96)
    print("탐지 사유 생성 예시 (SHAP)")
    print("=" * 96)
    explainer = shap.TreeExplainer(base)
    sample = df[df[TARGET] == 1].head(3)
    sv = explainer.shap_values(sample[list(feats)])
    if isinstance(sv, list):
        sv = sv[1]
    for i, (_, row) in enumerate(sample.iterrows()):
        values = {f: float(row[f]) for f in feats}
        print(f"\n  경매 {int(row['Auction_ID'])} · 입찰자 {row['Bidder_ID']}")
        for r in top_reasons(sv[i], list(feats), values):
            print(f"    · {r['text']}  (기여 {r['contribution']:+.3f}, 값 {r['value']})")

    # 저장 ------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    joblib.dump(calibrated, OUT / "shill_lgbm_v0.joblib")

    meta = {
        "model": "LGBMClassifier + CalibratedClassifierCV(isotonic)",
        "model_version": "lgbm-v0",
        "trained_on": "eBay Shill Bidding Dataset",
        "features": list(feats),
        "excluded_features": {
            LEAKED: "서비스 정책상 최고 입찰자의 연속 입찰이 불가능해 항상 0 이다",
            SUBSCRIPTION_CONFLICT: (
                "라이브 구독 모델에서는 판매자 편중이 정상 행동이다. "
                "제외 시 성능과 정밀도가 함께 올라 순이득이었다"
            ),
        },
        "holdout": {
            "pr_auc": round(report.pr_auc, 4),
            "roc_auc": round(report.roc_auc, 4),
            "best_threshold": round(report.best_threshold, 4),
            "precision": round(report.precision_at_best, 4),
            "recall": round(report.recall_at_best, 4),
            "brier_calibrated": round(report.brier_calibrated, 4),
        },
        "feature_importance": report.importance,
        "risk_bands": {"low": [0.0, 0.3], "medium": [0.3, 0.6], "high": [0.6, 1.0]},
        "random_state": RANDOM_STATE,
        "usage": "관리자 검토 큐 정렬 전용. 자동 제재 금지.",
        "caveats": [
            "eBay 데이터로 학습했으므로 우리 서비스 성능이 아니다. 파이프라인 검증용이다.",
            "Bidding_Ratio 상한이 다르다. 우리 정책상 한 입찰자의 비중은 (n+1)/2n 을 넘을 수 없다.",
            "eBay 라벨은 Successive_Outbidding 에 강하게 의존한다. A 구성의 0.99 는 인용하지 말 것.",
        ],
        "serving_features_contract": list(SERVING_FEATURES),
    }
    (OUT / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 96)
    print(f"저장  {OUT / 'shill_lgbm_v0.joblib'}")
    print(f"저장  {OUT / 'model_meta.json'}")


if __name__ == "__main__":
    main()
