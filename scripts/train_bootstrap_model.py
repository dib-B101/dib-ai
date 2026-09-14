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
from statistics import mean, stdev
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
    OUT_OF_RANGE,
    REDEFINED,
    UNSPECIFIABLE,
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


# 피처 조합을 단일 시드로 판단하지 않기 위한 재측정용 시드.
SEEDS = (42, 7, 123, 2024, 31337)


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

    # 시드 분산 --------------------------------------------------------------
    # 단일 시드 숫자로 피처를 고르면 안 된다. 분할이 달라지면 순위가 뒤집힌다.
    print("\n" + "=" * 96)
    print(f"시드 분산 — 채택 구성을 {len(SEEDS)}개 시드로 재측정")
    print("=" * 96)
    pr, prec, rec = [], [], []
    for seed in SEEDS:
        r, _, _ = train_one(df, feats, report.name, seed=seed)
        pr.append(r.pr_auc); prec.append(r.precision_at_best); rec.append(r.recall_at_best)

    for label, values in [("PR-AUC", pr), ("Precision", prec), ("Recall", rec)]:
        print(f"  {label:<10} {mean(values):.3f} +- {stdev(values):.3f}   "
              f"[{min(values):.3f} ~ {max(values):.3f}]")
    print()
    print("  표준편차가 조합 간 차이보다 크다. 성능만으로는 피처 조합의 우열을 가릴 수 없다.")
    print("  그래서 '우리 데이터로 재현 가능한가' 를 기준으로 골랐다.")

    # 피처 선택 근거 ----------------------------------------------------------
    print("\n" + "=" * 96)
    print("서빙 피처를 이렇게 고른 이유")
    print("=" * 96)
    print(f"  제외   {LEAKED:<24} 정책상 최고 입찰자의 연속 입찰이 불가능해 항상 0")
    print(f"  제외   {OUT_OF_RANGE:<24} 라이브(분)와 일반 경매(판매자 자유)가 섞여")
    print(f"  {'':8}{'':<24} '긴 경매인가' 가 아니라 '경매 유형' 을 가리키게 된다")
    print(f"  제외   {UNSPECIFIABLE:<24} 원식·방향을 특정할 수 없다 (Class 상관 +0.043).")
    print(f"  {'':8}{'':<24} 방향이 반대면 모델이 정반대로 읽는다")
    print(f"  재정의 {REDEFINED:<24} 구독하지 않은 판매자에 대한 편중도로 바꿔 쓴다.")
    print(f"  {'':8}{'':<24} eBay 에는 구독이 없어 학습은 원본 그대로 하고,")
    print(f"  {'':8}{'':<24} 서빙에서만 구독 건을 빼면 '높으면 의심' 이 유지된다")
    print()
    print("  Auction_Bids 는 '입찰 수를 코퍼스 내 min-max 정규화' 로 식이 자명하다.")
    print("  우리 코퍼스로 같은 변환을 하면 의미가 대응되므로 그대로 쓴다.")

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
            OUT_OF_RANGE: (
                "eBay 는 경매 기간을 1~10 '일' 로 기록한다. 우리는 라이브(분 단위)와 "
                "일반 경매(판매자 자유 설정)가 섞여 이 값이 '긴 경매인가' 가 아니라 "
                "'라이브냐 일반이냐' 를 가리키게 된다. 학습 때 없던 의미다"
            ),
            UNSPECIFIABLE: (
                "원식과 방향을 특정할 수 없다. Class 와의 상관이 +0.043 이고 정규화 기준도 "
                "알 수 없어, 우리가 추정한 식의 방향이 반대면 모델이 정반대로 읽는다. "
                "5시드 짝비교에서 빼도 차이가 없어(PR-AUC +0.0016 ± 0.0090) 위험만 없앴다"
            ),
        },
        "redefined_features": {
            REDEFINED: (
                "구독하지 않은 판매자에 대한 편중도로 재정의한다. eBay 에는 구독 개념이 "
                "없어 학습은 원본 값 그대로 하고, 서빙에서만 구독 건을 분자·분모에서 뺀다"
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
            "holdout 은 단일 시드 결과다. 시드 간 표준편차가 0.02~0.06 이므로 0.01 수준의 "
            "차이로 피처 조합을 판단하지 말 것. seed_variance 를 함께 볼 것.",
        ],
        "seed_variance": {
            "seeds": list(SEEDS),
            "pr_auc": {"mean": round(mean(pr), 4), "std": round(stdev(pr), 4)},
            "precision": {"mean": round(mean(prec), 4), "std": round(stdev(prec), 4)},
            "recall": {"mean": round(mean(rec), 4), "std": round(stdev(rec), 4)},
        },
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
