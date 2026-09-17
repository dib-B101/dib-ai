"""상품 임베딩 배치 — `product` 의 벡터 컬럼을 채운다.

    python scripts/embed_products.py                 아직 안 된 것만
    python scripts/embed_products.py --limit 50      50건만 (처음 돌려볼 때)
    python scripts/embed_products.py --dry-run       계산만 하고 쓰지 않는다
    python scripts/embed_products.py --all           전부 다시 (모델을 바꿨을 때)

**AI 코드 중 유일하게 DB 에 쓰는 곳이다.** 백엔드는 이 두 컬럼을 건드리지 않는다
(`@Column(insertable=false, updatable=false)`) — 채우는 것은 우리 몫이다.

    DIB_EMBED_DATABASE_URL   쓰기 가능한 접속 문자열. 없으면 DIB_DATABASE_URL 을 쓴다
    GRANT UPDATE (text_embedding, image_embedding) ON product TO ai_user;

## 다시 돌려도 안전하다

이미 채워진 상품은 건너뛴다. 중간에 죽어도 그 다음에 이어서 하면 된다. 한 묶음씩
커밋하므로 **앞서 끝낸 작업이 날아가지 않는다.**

## 수정된 상품을 어떻게 알아내나

`embedded_at` 컬럼이 없어도 알 수 있다. 근거는 백엔드 구현에 있다 — **상품을
수정하면 반드시 `status` 가 `PENDING` 으로 돌아간다.** 재검수를 거쳐야 하기 때문이다.

    수정 → PENDING → 배치가 벡터를 비움 → 재검수 통과 → 배치가 새로 계산

그래서 실행할 때마다 먼저 **`PENDING` · `REJECTED` · 삭제된 상품의 벡터를 비운다.**
`NULL` 이 곧 "다시 계산해야 함" 이 되므로 따로 기록할 것이 없다. 거부·삭제된 상품의
벡터를 남겨 두지 않는 효과도 같이 얻는다 — 남아 있으면 노출 정책이 바뀔 때 조용히
추천에 섞여 들어간다.

비운 상품을 같은 실행에서 다시 계산하지는 않는다. `PENDING` 은 계산 대상이 아니다.

## 남은 한계 — 모델을 바꿨을 때

어느 행이 어떤 모델로 계산됐는지는 DB 에 없다. 모델이나 텍스트 구성을 바꾸면
`--all` 로 전부 다시 돌리는 수밖에 없다. 드물고 의도적인 작업이라 지금은 이걸로
충분하지만, 상품이 수천 건을 넘으면 `embedding_model` 컬럼을 요청하는 편이 낫다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embedding import ImageEncoder, TextEncoder, build_text, resolve_device  # noqa: E402
from embedding import store  # noqa: E402
from envfile import load_env  # noqa: E402

log = logging.getLogger("embed_products")


def connect(dsn: str):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, row_factory=dict_row)


def fetch_pending(conn, limit: int, force: bool):
    with conn.cursor() as cur:
        cur.execute(
            store.PENDING_SQL,
            {
                "skip_status": list(store.SKIP_PRODUCT_STATUS),
                "force": force,
                "limit": limit,
            },
        )
        return store.to_pending(cur.fetchall())


def reclaim_stale(conn) -> int:
    """낡았거나 남아 있으면 안 되는 벡터를 비운다. 비운 건수를 돌려준다.

    상품을 수정하면 백엔드가 `status` 를 `PENDING` 으로 되돌린다. 그 사실이
    **`embedded_at` 컬럼 없이 재계산 시점을 알아내는 근거**다.
    """
    with conn.cursor() as cur:
        cur.execute(store.RECLAIM_SQL, {"skip_status": list(store.SKIP_PRODUCT_STATUS)})
        wiped = cur.rowcount
    conn.commit()
    return max(wiped, 0)


def report_counts(conn) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(store.COUNT_SQL, {"skip_status": list(store.SKIP_PRODUCT_STATUS)})
        row = cur.fetchone()
    return int(row["pending"]), int(row["total"])


def encode_batch(products, text_encoder: TextEncoder, image_encoder: ImageEncoder):
    """한 묶음을 벡터로. 이미지가 없거나 못 읽은 상품은 텍스트만 돌려준다.

    **사진 한 장 때문에 배치가 멈추면 안 된다.** 읽기 실패는 그 상품의 이미지 벡터를
    비우는 것으로 끝내고, 텍스트는 그대로 저장한다.
    """
    texts = text_encoder.encode([build_text(p.title, p.description) for p in products])

    images: dict[int, list[float]] = {}
    sources = [p.image_url or "" for p in products]
    if any(sources):
        kept, vectors = image_encoder.encode_paths(sources)
        for slot, index in enumerate(kept):
            images[index] = vectors[slot].tolist()

    return [
        {
            "product_id": p.product_id,
            "text": store.to_pgvector(texts[i].tolist()),
            "image": store.to_pgvector(images.get(i)),
        }
        for i, p in enumerate(products)
    ]


def write_batch(conn, rows) -> int:
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(store.UPDATE_SQL, row)
    conn.commit()
    return len(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32, help="한 번에 처리할 상품 수")
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 최대 건수 (0=전부)")
    ap.add_argument("--all", action="store_true", help="이미 채워진 것도 다시 계산한다")
    ap.add_argument("--dry-run", action="store_true", help="계산만 하고 DB 에 쓰지 않는다")
    ap.add_argument(
        "--no-reclaim",
        action="store_true",
        help="수정·거부·삭제된 상품의 낡은 벡터를 비우는 단계를 건너뛴다",
    )
    args = ap.parse_args()

    load_env()

    import os

    dsn = (os.getenv("DIB_EMBED_DATABASE_URL") or os.getenv("DIB_DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit(
            "DIB_EMBED_DATABASE_URL (또는 DIB_DATABASE_URL) 이 필요합니다.\n"
            "이 배치는 product 의 벡터 컬럼에 UPDATE 를 하므로 쓰기 권한이 있어야 합니다."
        )

    conn = connect(dsn)
    try:
        if not (args.dry_run or args.no_reclaim):
            wiped = reclaim_stale(conn)
            if wiped:
                print(f"낡은 벡터    {wiped:,}건 비움 (수정·거부·삭제된 상품)")

        pending, total = report_counts(conn)
        print(f"device      {resolve_device()}")
        print(f"대상 상품    {total:,}건 중 임베딩 없음 {pending:,}건")
        if args.all:
            print("--all 지정 — 이미 채워진 것도 다시 계산합니다")
        if args.dry_run:
            print("--dry-run 지정 — DB 에 쓰지 않습니다")
        print()

        text_encoder, image_encoder = TextEncoder(), ImageEncoder()
        done = 0
        remaining = args.limit or None

        while True:
            size = args.batch_size if remaining is None else min(args.batch_size, remaining)
            if size <= 0:
                break

            products = fetch_pending(conn, size, args.all)
            if not products:
                break

            rows = encode_batch(products, text_encoder, image_encoder)
            with_image = sum(1 for r in rows if r["image"] is not None)

            if args.dry_run:
                # 커밋하지 않으면 다음 조회가 같은 상품을 또 집어 무한 반복이 된다.
                print(f"  {len(rows)}건 계산 (이미지 {with_image}건) — 쓰지 않음")
                done += len(rows)
                break

            done += write_batch(conn, rows)
            print(f"  {done:,}건 완료 (이번 묶음 이미지 {with_image}/{len(rows)})", flush=True)

            if remaining is not None:
                remaining -= len(products)

        print()
        if args.dry_run:
            print(f"계산만 했습니다 — {done:,}건")
        else:
            left, _ = report_counts(conn)
            print(f"저장 완료 {done:,}건 · 남은 상품 {left:,}건")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
