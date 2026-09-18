"""otpensource 데이터셋을 임베딩 평가용 CSV 로 바꾼다.

    python scripts/fetch_otpensource.py
    python scripts/fetch_otpensource.py --per-category 10 --with-attrs

`hateslopacademy/otpensource_dataset` (CC-BY-4.0) 은 한국어 상품명과 실제 상품
사진 URL 을 가진 공개 데이터다. **우리 서비스 데이터가 0건인 상태에서 임베딩이
실제로 동작하는지 재 볼 수 있는 유일한 수단**이라 가져다 쓴다.

두 가지 한계를 알고 쓴다.

**패션 한정이다.** 상의·아우터·하의·원피스 4개 대분류뿐이다. 우리는 디지털기기·
가전·도서·카메라까지 다루므로, 여기서 나온 점수가 **"아이폰과 갤럭시를 구분한다"
는 근거는 되지 않는다.** 이미지 임베딩이 켜져서 돌아가는지, 가중합·백분위가 도움이
되는지를 확인하는 용도다.

**판매자가 쓴 설명이 없다.** 우리 실제 입력은 "니콘 FM2 필름카메라 바디 / 수동
필름 카메라입니다. 셔터 정상 동작 확인했습니다." 같은 자유 문장인데, 이 데이터는
상품명 한 줄뿐이다. `--with-attrs` 로 color·material·feature 를 이어 붙여 설명을
흉내 낼 수는 있지만 **실제와 다르다.** 두 방식의 점수를 비교하는 용도로만 쓴다.

평가는 **소분류**로 한다. 대분류 4개는 너무 쉬워 변별이 안 된다. "긴소매 티셔츠 vs
반팔 티셔츠" 를 구분하는지 보는 것이, 우리 서비스의 "아이폰 14 vs 아이폰 15" 에
그나마 가깝다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPO = "hateslopacademy/otpensource_dataset"
DEFAULT_OUT = ROOT / "data" / "otpensource.csv"
DEFAULT_IMAGE_DIR = ROOT / "data" / "otpensource_images"


def load_records() -> list[dict]:
    """소분류별 JSON 을 모두 받아 한 리스트로 편다."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        s.rfilename
        for s in HfApi().dataset_info(REPO).siblings
        if s.rfilename.endswith(".json")
    ]
    records = []
    for name in sorted(files):
        path = hf_hub_download(REPO, name, repo_type="dataset")
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(rows if isinstance(rows, list) else [rows])
    return records


def sample(records: list[dict], per_category: int, seed: int) -> list[dict]:
    """소분류마다 같은 수만 뽑는다.

    고르게 뽑지 않으면 **많이 뽑힌 소분류가 무작위 기준선을 끌어올려** 점수를
    해석할 수 없게 된다. 카테고리별 개수가 같으면 기준선이 1/카테고리수 로 딱 떨어진다.
    """
    buckets: dict[str, list[dict]] = {}
    for r in records:
        key = (r.get("sub_category") or "").strip()
        if key and (r.get("product_name") or "").strip():
            buckets.setdefault(key, []).append(r)

    rng = random.Random(seed)
    out = []
    for key in sorted(buckets):
        rows = buckets[key]
        if len(rows) < per_category:
            continue  # 표본이 모자란 소분류는 통째로 뺀다. 섞이면 기준선이 흔들린다
        out.extend(rng.sample(rows, per_category))
    return out


def describe(record: dict) -> str:
    """color · material · feature 를 이어 붙여 설명을 흉내 낸다.

    **판매자가 쓴 문장이 아니다.** 속성 나열이라 실제 설명보다 정형화되어 있어,
    여기서 나온 개선폭이 실서비스에서 그대로 나온다고 볼 수 없다.
    """
    parts = [(record.get(k) or "").strip() for k in ("color", "material", "feature")]
    return " ".join(p for p in parts if p)


def download_images(records: list[dict], image_dir: Path) -> int:
    """이미지를 로컬에 받아 둔다. 재실행할 때 다시 받지 않기 위해서다.

    받기에 실패한 상품도 **버리지 않는다.** 이미지 칸만 비우면 텍스트 평가에는
    그대로 들어간다 — 사진 한 장 못 받았다고 표본을 줄일 이유가 없다.
    """
    from media import read_image_bytes

    image_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, r in enumerate(records):
        url = (r.get("image_url") or "").strip()
        if not url:
            r["_image_path"] = ""
            continue

        dest = image_dir / f"{i:05d}.jpg"
        if dest.exists():
            r["_image_path"] = str(dest)
            ok += 1
            continue

        data = read_image_bytes(url)
        if data is None:
            r["_image_path"] = ""
        else:
            dest.write_bytes(data)
            r["_image_path"] = str(dest)
            ok += 1

        if (i + 1) % 25 == 0:
            print(f"  이미지 {i + 1}/{len(records)} ... 성공 {ok}", flush=True)
    return ok


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=6, help="소분류당 표본 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument(
        "--with-attrs",
        action="store_true",
        help="color·material·feature 를 description 으로 넣는다 (실제 설명은 아니다)",
    )
    ap.add_argument("--no-images", action="store_true", help="텍스트만 평가할 때")
    ap.add_argument(
        "--level",
        choices=("sub", "big"),
        default="sub",
        help="평가 라벨. sub=소분류 48개(가혹), big=대분류 4개(변별 약함)",
    )
    args = ap.parse_args()

    print(f"데이터셋    {REPO}")
    records = load_records()
    print(f"전체        {len(records):,}건")

    # 표본 추출은 **항상 소분류 기준**이다. 라벨만 바꿔서 같은 표본을 두 잣대로
    # 재야 두 숫자를 비교할 수 있다. 표본이 달라지면 비교가 무의미해진다.
    rows = sample(records, args.per_category, args.seed)
    label = "sub_category" if args.level == "sub" else "big_category"
    categories = sorted({(r.get(label) or "").strip() for r in rows})
    print(f"표본        {len(rows)}건 · 평가 라벨 {args.level} {len(categories)}개")
    print(f"무작위 기준선 {1 / len(categories):.3f} — 이 값을 못 넘으면 임베딩이 놀고 있는 것")

    if args.no_images:
        for r in rows:
            r["_image_path"] = ""
    else:
        print(f"\n이미지 받는 중 → {args.image_dir}")
        ok = download_images(rows, args.image_dir)
        print(f"  완료 — {ok}/{len(rows)}건")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title", "description", "image", "category"])
        for r in rows:
            w.writerow([
                r["product_name"].strip(),
                describe(r) if args.with_attrs else "",
                r.get("_image_path", ""),
                (r.get(label) or "").strip(),
            ])

    print(f"\n저장        {args.out}")
    print(f"\n다음:\n  python scripts/eval_embedding.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
