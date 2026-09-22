"""임베딩이 쓸 만한지 확인한다.

모델이 로드되고 벡터가 나오는 것만으로는 아무것도 증명하지 못한다. **비슷한 상품끼리
더 가깝게 나오는지**를 봐야 한다. 그래서 서로 다른 카테고리의 상품을 넣고 유사도
행렬을 출력한다.

확인하는 것은 네 가지다.

1. 모델이 로드되는가 (첫 실행은 모델 다운로드로 몇 분 걸린다)
2. 출력 차원이 DB 컬럼과 맞는가
3. 벡터가 L2 정규화되어 있는가 — 안 되어 있으면 pgvector 내적 쿼리가 틀린 값을 낸다
4. 같은 카테고리끼리 유사도가 높은가

사용법::

    python scripts/check_embedding.py
    python scripts/check_embedding.py --image 사진1.jpg 사진2.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding import IMAGE_MODEL, TEXT_MODEL, ImageEncoder, TextEncoder, resolve_device  # noqa: E402

# 같은 카테고리 2개씩. 대각선 바깥에서 같은 카테고리 쌍이 높게 나와야 한다.
SAMPLES = [
    ("빈티지 필름 카메라", "수동 필름 카메라입니다. 셔터 정상 동작 확인했습니다."),
    ("니콘 FM2 필름카메라", "오래된 수동 카메라, 노출계 작동합니다."),
    ("나이키 에어포스1 270", "새 운동화입니다. 미착용 정품."),
    ("아디다스 삼바 265", "운동화 거의 새것, 몇 번만 신었습니다."),
    ("아이폰 15 프로 256GB", "정품 미개봉 공기계입니다."),
]


def show_matrix(labels, vectors) -> None:
    import numpy as np

    sim = vectors @ vectors.T  # 정규화되어 있으므로 내적 = 코사인
    width = max(len(l) for l in labels) + 2

    print(" " * width + "".join(f"{i + 1:>7}" for i in range(len(labels))))
    for i, label in enumerate(labels):
        row = "".join(f"{sim[i][j]:7.3f}" for j in range(len(labels)))
        print(f"{i + 1}. {label:<{width - 3}}{row}")

    # 대각선을 뺀 최고 유사도 쌍
    masked = sim - np.eye(len(labels)) * 2
    i, j = np.unravel_index(masked.argmax(), masked.shape)
    print(f"\n가장 가까운 쌍  {labels[i]}  ↔  {labels[j]}   ({sim[i][j]:.3f})")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="*", default=[], help="이미지 임베딩도 확인할 경로")
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()

    print(f"device     {resolve_device()}")
    print(f"텍스트      {TEXT_MODEL}")
    print(f"이미지      {IMAGE_MODEL}\n")

    if not args.skip_text:
        print("=" * 78)
        print("텍스트 임베딩")
        print("=" * 78)

        encoder = TextEncoder()
        started = time.perf_counter()
        try:
            vectors = encoder.encode([f"{t}\n{d}" for t, d in SAMPLES])
        except Exception as exc:
            print(f"실패 — {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - started

        import numpy as np

        norms = np.linalg.norm(vectors, axis=1)
        print(f"모양       {vectors.shape}   (기대 (5, {encoder.dim}))")
        print(f"정규화      길이 {norms.min():.4f} ~ {norms.max():.4f}   (기대 1.0)")
        print(f"소요        {elapsed:.1f}초 (모델 로드 포함)\n")

        show_matrix([t for t, _ in SAMPLES], vectors)

    if args.image:
        print("\n" + "=" * 78)
        print("이미지 임베딩")
        print("=" * 78)

        encoder = ImageEncoder()
        started = time.perf_counter()
        try:
            kept, vectors = encoder.encode_paths(args.image)
        except Exception as exc:
            print(f"실패 — {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - started

        if not kept:
            print("이미지를 하나도 읽지 못했습니다. 경로를 확인하십시오.")
            return 1

        import numpy as np

        norms = np.linalg.norm(vectors, axis=1)
        print(f"모양       {vectors.shape}   (기대 ({len(kept)}, {encoder.dim}))")
        print(f"정규화      길이 {norms.min():.4f} ~ {norms.max():.4f}   (기대 1.0)")
        print(f"소요        {elapsed:.1f}초 (모델 로드 포함)")
        if len(kept) < len(args.image):
            skipped = [args.image[i] for i in range(len(args.image)) if i not in kept]
            print(f"건너뜀      {skipped}")
        print()

        show_matrix([Path(args.image[i]).stem for i in kept], vectors)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
