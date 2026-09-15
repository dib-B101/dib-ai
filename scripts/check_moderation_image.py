"""이미지 검수가 실제로 동작하는지 확인한다.

텍스트 판정은 `check_moderation_llm.py` 로 확인했다. 이 스크립트는 **이미지가 판정에
실제로 영향을 주는지**를 본다.

방법은 같은 상품을 두 번 부르는 것이다.

    ① 텍스트만          →  판정 A
    ② 텍스트 + 이미지    →  판정 B

A 와 B 가 같으면 이미지가 무시된 것이다. 이미지 인코딩이 실패했거나, 게이트웨이가
이미지를 전달하지 않았거나, 모델이 보지 않은 것이다. **응답이 오는 것만으로는 이미지가
쓰였다고 말할 수 없다.**

그래서 상품명을 일부러 중립적으로 둔다. "전자담배 팝니다" 라고 쓰면 이미지를 안 봐도
차단되므로 아무것도 검증하지 못한다.

사용법::

    python scripts/check_moderation_image.py 이미지.jpg [이미지2.jpg ...]
    python scripts/check_moderation_image.py 이미지.jpg --title "미개봉 새제품"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envfile import load_env  # noqa: E402
from moderation.keywords import KeywordFilter  # noqa: E402
from moderation.llm import create_llm, encode_image  # noqa: E402
from moderation.pipeline import ModerationPipeline  # noqa: E402
from moderation.schema import ProductInput  # noqa: E402

# 이미지를 봐야만 알 수 있도록 일부러 정보를 뺀 상품 설명.
NEUTRAL_TITLE = "미개봉 새제품 팝니다"
NEUTRAL_DESC = "사진 참고 부탁드립니다. 직거래만 합니다."


def review(pipeline: ModerationPipeline, title, desc, images) -> tuple:
    started = time.perf_counter()
    result = pipeline.review(
        ProductInput(product_id=1, title=title, description=desc, image_urls=tuple(images))
    )
    return result, time.perf_counter() - started


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="검사할 이미지 경로")
    ap.add_argument("--title", default=NEUTRAL_TITLE)
    ap.add_argument("--desc", default=NEUTRAL_DESC)
    args = ap.parse_args()

    load_env()
    try:
        llm = create_llm()
    except Exception as exc:
        print(f"모델을 만들지 못했습니다 — {type(exc).__name__}: {exc}")
        return 1
    if llm is None:
        print("API 키가 없습니다. .env 를 확인하십시오.")
        return 1

    print(f"model      {getattr(llm, '_model', '?')}")
    print(f"상품명      {args.title}")
    print(f"설명        {args.desc}\n")

    pipeline = ModerationPipeline(KeywordFilter.load(), llm)

    # 텍스트만으로 한 번. 이미지가 붙었을 때와 비교할 기준선이다.
    base, base_sec = review(pipeline, args.title, args.desc, [])
    print(f"[텍스트만]  {base.verdict.value}  (conf {base.confidence or 0:.2f}, {base_sec:.1f}초)")
    print(f"            {base.reason}\n")
    print("-" * 78)

    changed = 0
    for path in args.images:
        # 인코딩이 실패하면 파이프라인이 조용히 이미지를 건너뛴다.
        # 그 상태로 "판정이 같다" 는 결과를 보면 원인을 잘못 짚게 된다.
        if encode_image(path) is None:
            print(f"\n{Path(path).name}\n  이미지를 읽지 못했습니다 — 경로·형식을 확인하십시오")
            continue

        result, sec = review(pipeline, args.title, args.desc, [path])
        moved = result.verdict is not base.verdict
        changed += moved

        print(f"\n{Path(path).name}   ({sec:.1f}초)")
        print(f"  {base.verdict.value}  →  {result.verdict.value}   {'변함' if moved else '그대로'}")
        if result.category:
            print(f"  분류      {result.category}")
        print(f"  확신도    {result.confidence if result.confidence is not None else '—'}")
        print(f"  사유      {result.reason}")
        if err := result.detail.get("error"):
            print(f"  실패      {err}")

    print("\n" + "-" * 78)
    print(f"{changed}/{len(args.images)} 건에서 이미지가 판정을 바꿨습니다.")
    if changed == 0:
        print("\n판정이 하나도 안 바뀌었다면 이미지가 모델에 닿지 않았을 수 있습니다.")
        print("이미지 없이도 같은 결론이 나오는 사진이었을 가능성도 있으니 함께 확인하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
