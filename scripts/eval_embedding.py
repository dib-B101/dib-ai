"""임베딩이 우리 상품을 잘 나누는지 정량 평가한다.

`check_embedding.py` 는 유사도 행렬을 눈으로 보는 용도다. 이 스크립트는 **숫자를 낸다.**

    각 상품의 가장 가까운 K 개를 뽑는다
        ↓
    그중 같은 카테고리인 비율        ← Precision@K

**무작위 기준선을 함께 낸다.** 카테고리가 5개면 아무렇게나 뽑아도 0.2 는 나온다.
기준선 없이 "Precision@5 = 0.4" 만 보면 좋은 건지 나쁜 건지 알 수 없다.

세 가지를 비교한다.

    텍스트만          BGE-M3 유사도
    이미지만          SigLIP 유사도
    가중합            0.6 × 텍스트 + 0.4 × 이미지

가중합은 **원값과 백분위 두 방식**을 모두 낸다. 두 유사도의 분포가 다르면 원값 가중합은
가중치가 이름뿐인 값이 되는데, 그게 실제로 일어나는지 데이터로 확인하기 위해서다.

입력 CSV::

    title,description,image,category
    아이폰 15 프로 256GB,정품 미개봉입니다,img/iphone.jpg,디지털기기
    나이키 에어포스1 270,미착용 새제품,img/af1.jpg,의류

`description` 과 `image` 는 비어 있어도 된다. 이미지가 없는 상품은 이미지 평가에서만
빠지고 텍스트 평가에는 들어간다.

사용법::

    python scripts/eval_embedding.py products.csv
    python scripts/eval_embedding.py products.csv --k 3 --text-weight 0.7
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding import ImageEncoder, TextEncoder, build_text, resolve_device  # noqa: E402
from reco import percentile_ranks  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("title") or "").strip()]

    missing = {"title", "category"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"CSV 에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
    return rows


def random_baseline(categories: list[str]) -> float:
    """아무렇게나 뽑았을 때의 기대 Precision.

    상품마다 "나를 뺀 전체 중 같은 카테고리 비율" 을 구해 평균낸다.
    이 값을 못 넘으면 임베딩이 아무 일도 하지 않은 것이다.
    """
    counts = Counter(categories)
    n = len(categories)
    return sum((counts[c] - 1) / (n - 1) for c in categories) / n


def perfect_score(categories: list[str], k: int) -> float:
    """완벽한 모델이 낼 수 있는 최대 Precision.

    카테고리에 5건뿐인데 K=5 로 재면 자기 자신을 뺀 4건이 한계라 0.8 이 최대다.
    이걸 모르면 0.475 를 보고 "절반도 못 맞췄다" 고 잘못 읽는다.
    """
    counts = Counter(categories)
    return sum(min(k, counts[c] - 1) / k for c in categories) / len(categories)


def precision_at_k(sim, categories: list[str], k: int) -> tuple[float, list[float]]:
    """각 상품의 최근접 K 개 중 같은 카테고리 비율."""
    import numpy as np

    n = len(categories)
    scores = []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # 자기 자신 제외
        top = np.argsort(-row)[:k]
        scores.append(sum(categories[j] == categories[i] for j in top) / k)
    return float(np.mean(scores)), scores


def blend(text_sim, image_sim, w_text: float, has_image, use_percentile: bool):
    """두 유사도를 합친다. 이미지가 없는 상품은 텍스트만 쓴다."""
    import numpy as np

    n = text_sim.shape[0]
    out = np.zeros((n, n), dtype="float64")

    for i in range(n):
        t_row = text_sim[i]
        if use_percentile:
            t_row = np.array(percentile_ranks(list(t_row)))

        if image_sim is None or not has_image[i]:
            out[i] = t_row
            continue

        # 이미지가 있는 상품끼리만 이미지 유사도를 쓴다.
        # 없는 상품을 0 으로 채우면 "닮지 않음" 으로 오해되어 순위가 밀린다.
        i_row = image_sim[i].copy()
        i_row[~has_image] = np.nan
        if use_percentile:
            valid = ~np.isnan(i_row)
            ranked = np.full(n, np.nan)
            ranked[valid] = percentile_ranks(list(i_row[valid]))
            i_row = ranked

        combined = w_text * t_row + (1 - w_text) * i_row
        out[i] = np.where(np.isnan(combined), t_row, combined)

    return out


def report(name: str, sim, categories, k: int, baseline: float, ceiling: float) -> float:
    score, _ = precision_at_k(sim, categories, k)
    lift = score / baseline if baseline else float("inf")
    pct = score / ceiling * 100 if ceiling else 0.0
    bar = "#" * int(round(score * 40))
    print(f"  {name:<22} {score:.3f}  ({lift:4.1f}배)  {bar:<40} 최대 대비 {pct:.0f}%")
    return score


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--text-weight", type=float, default=0.6)
    args = ap.parse_args()

    import numpy as np

    rows = load_rows(args.csv)
    categories = [r["category"].strip() for r in rows]
    counts = Counter(categories)

    print(f"상품        {len(rows)}건")
    print(f"카테고리     {len(counts)}개 — " + ", ".join(f"{c}({n})" for c, n in counts.most_common()))
    print(f"device      {resolve_device()}\n")

    if len(rows) <= args.k:
        raise SystemExit(f"상품이 {len(rows)}건이라 K={args.k} 평가가 무의미합니다. 더 모으십시오.")
    if len(counts) < 2:
        raise SystemExit("카테고리가 1개뿐이라 평가할 수 없습니다.")

    # ---- 텍스트
    text_vectors = TextEncoder().encode(
        [build_text(r["title"], r.get("description")) for r in rows]
    )
    text_sim = text_vectors @ text_vectors.T

    # ---- 이미지 (있는 것만)
    paths = [(r.get("image") or "").strip() for r in rows]
    image_sim, has_image = None, np.zeros(len(rows), dtype=bool)

    if any(paths):
        encoder = ImageEncoder()
        kept, vectors = encoder.encode_paths(paths)
        if kept:
            full = np.zeros((len(rows), encoder.dim), dtype="float32")
            for slot, idx in enumerate(kept):
                full[idx] = vectors[slot]
                has_image[idx] = True
            image_sim = full @ full.T
        missing = len(rows) - int(has_image.sum())
        if missing:
            print(f"이미지 없음·읽기 실패 {missing}건 — 텍스트 평가에는 포함됩니다\n")

    # ---- 평가
    baseline = random_baseline(categories)
    ceiling = perfect_score(categories, args.k)
    print("=" * 78)
    print(f"Precision@{args.k}  —  최근접 {args.k}개 중 같은 카테고리 비율")
    print("=" * 78)
    print(f"  {'무작위 기준선':<22} {baseline:.3f}  ( 1.0배)  {'#' * int(round(baseline * 40))}")
    print(f"  {'이론상 최대':<22} {ceiling:.3f}  ({ceiling / baseline:4.1f}배)  "
          f"{'#' * int(round(ceiling * 40))}   ← 카테고리당 상품 수의 한계")
    print()

    text_score = report("텍스트만", text_sim, categories, args.k, baseline, ceiling)

    if image_sim is not None and has_image.sum() >= 2:
        report("이미지만", image_sim, categories, args.k, baseline, ceiling)
        w = args.text_weight
        raw = report(
            f"가중합 (원값 {w}/{1 - w:.1f})",
            blend(text_sim, image_sim, w, has_image, False), categories, args.k, baseline, ceiling,
        )
        pct = report(
            f"가중합 (백분위 {w}/{1 - w:.1f})",
            blend(text_sim, image_sim, w, has_image, True), categories, args.k, baseline, ceiling,
        )

        print()
        best = max(text_score, raw, pct)
        if pct >= best:
            print("  → 백분위 가중합이 가장 낫습니다. 현재 설계대로 가면 됩니다.")
        elif text_score >= best:
            print("  → 텍스트 단독이 가장 낫습니다. 이미지 비중을 낮추는 것을 검토하십시오.")
        else:
            print("  → 원값 가중합이 더 낫습니다. 두 유사도의 분포가 비슷하다는 뜻입니다.")

    print()
    print("=" * 78)
    print("읽는 법")
    print("=" * 78)
    print("  무작위 기준선을 못 넘으면 임베딩이 아무 일도 하지 않은 것입니다.")
    print("  이론상 최대 대비 70% 를 넘으면 추천에 쓸 만합니다.")
    print("  무작위 대비 1.2배 아래면 모델이나 텍스트 구성을 다시 봐야 합니다.")
    print("  상품이 30건 미만이면 숫자가 크게 흔들리니 참고로만 보십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
