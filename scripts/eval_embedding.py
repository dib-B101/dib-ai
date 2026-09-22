"""임베딩이 우리 상품을 잘 나누는지 정량 평가한다.

`check_embedding.py` 는 유사도 행렬을 눈으로 보는 용도다. 이 스크립트는 **숫자를 낸다.**

    각 상품의 가장 가까운 K 개를 뽑는다
        ↓
    그중 같은 카테고리인 비율        ← Precision@K

**무작위 기준선을 함께 낸다.** 카테고리가 5개면 아무렇게나 뽑아도 0.2 는 나온다.
기준선 없이 "Precision@5 = 0.4" 만 보면 좋은 건지 나쁜 건지 알 수 없다.

다섯 가지를 비교한다.

    텍스트만          BGE-M3 유사도
    이미지만          SigLIP 유사도
    가중합            0.6 × 텍스트 + 0.4 × 이미지 — 원값 · 백분위 · 표준화 세 방식

두 유사도는 평균도 폭도 다르다. 그래서 합치기 전에 눈금을 맞춰야 하는데, **어떻게
맞추느냐가 결과를 바꾼다.** 세 방식을 모두 내는 이유다.

    원값      눈금을 안 맞춘다. 평균이 높은 쪽이 유리해진다
    백분위    순서만 남긴다. "얼마나 더 닮았는가" 를 버린다
    표준화    (값 − 평균) / 표준편차. 평균을 맞추면서 간격은 남긴다   ← 서빙에서 쓰는 방식

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


def rescale(values, mode: str):
    """NaN 을 건드리지 않고 눈금만 바꾼다."""
    import numpy as np

    out = np.array(values, dtype="float64")
    valid = ~np.isnan(out)
    if mode == "raw" or valid.sum() < 2:
        return out

    if mode == "pct":
        out[valid] = percentile_ranks(list(out[valid]))
    elif mode == "z":
        sd = out[valid].std()
        out[valid] = (out[valid] - out[valid].mean()) / (sd if sd else 1.0)
    return out


def blend(text_sim, image_sim, w_text: float, has_image, mode: str):
    """두 유사도를 합친다. 이미지가 없는 상품은 텍스트만 쓴다."""
    import numpy as np

    n = text_sim.shape[0]
    out = np.zeros((n, n), dtype="float64")

    for i in range(n):
        t_row = rescale(text_sim[i], mode)

        if image_sim is None or not has_image[i]:
            out[i] = t_row
            continue

        # 이미지가 있는 상품끼리만 이미지 유사도를 쓴다.
        # 없는 상품을 0 으로 채우면 "닮지 않음" 으로 오해되어 순위가 밀린다.
        i_row = image_sim[i].astype("float64").copy()
        i_row[~has_image] = np.nan
        i_row = rescale(i_row, mode)

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
        scores = {
            name: report(
                f"가중합 ({name} {w}/{1 - w:.1f})",
                blend(text_sim, image_sim, w, has_image, mode),
                categories, args.k, baseline, ceiling,
            )
            for mode, name in (("raw", "원값"), ("pct", "백분위"), ("z", "표준화"))
        }

        print()
        best_name = max(scores, key=lambda k: scores[k])
        if max(scores.values()) <= text_score:
            print("  → 텍스트 단독이 가장 낫습니다. 이미지 비중을 낮추는 것을 검토하십시오.")
        else:
            print(f"  → 이미지를 섞는 것이 텍스트 단독보다 낫습니다 "
                  f"({text_score:.3f} → {max(scores.values()):.3f}).")
            print(f"    눈금 맞추는 방식은 '{best_name}' 이 가장 좋습니다. "
                  f"서빙은 '표준화' 를 씁니다({scores['표준화']:.3f}).")
            if best_name != "표준화":
                print(f"    차이가 0.01 을 넘으면 src/reco/similarity.py 의 blend() 를 재검토하십시오.")

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
