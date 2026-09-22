"""실제 DB 에 붙여 조회기들이 도는지 확인한다.

    python scripts/check_db.py

가짜 커넥션 테스트로는 못 잡는 것들을 본다.

    SQL 문법 · 컬럼명 오타
    psycopg 파라미터 바인딩 (`= ANY(%(x)s)` 같은 배열)
    pgvector `::vector` 캐스팅
    드라이버가 벡터를 리스트로 주는지 문자열로 주는지

**읽기만 한다.** 쓰기는 `embed_products.py` 쪽이고, 여기서는 건드리지 않는다.
붙을 DB 가 비어 있어도 실패가 아니다 — 쿼리가 도는지가 확인 대상이다.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envfile import load_env  # noqa: E402

OK, FAIL, SKIP = "[ OK ]", "[FAIL]", "[SKIP]"


def line(mark: str, name: str, detail: str = "") -> None:
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""), flush=True)


def check(name: str, fn) -> bool:
    """예외를 사람이 읽을 수 있는 한 줄로 바꾼다."""
    try:
        detail = fn()
    except Exception as exc:
        line(FAIL, name, f"{type(exc).__name__}: {exc}")
        return False
    line(OK, name, detail or "")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()

    dsn = (os.getenv("DIB_DATABASE_URL") or "").strip()
    if not dsn:
        raise SystemExit("DIB_DATABASE_URL 이 필요합니다.")

    import psycopg
    from psycopg.rows import dict_row

    from embedding import store
    from fraud_api.provider import PostgresProvider as FraudProvider
    from reco_api.provider import PostgresProvider as RecoProvider

    print(f"접속  {dsn.rsplit('@', 1)[-1]}\n")

    failures = 0

    # --- 기본 ---------------------------------------------------------------
    print("기본")
    conn = psycopg.connect(dsn, row_factory=dict_row)
    try:
        with conn.cursor() as cur:
            def version():
                cur.execute("SELECT version() AS v")
                return cur.fetchone()["v"].split(",")[0]

            def extension():
                cur.execute("SELECT extversion AS v FROM pg_extension WHERE extname='vector'")
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("pgvector 확장이 없습니다")
                return f"pgvector {row['v']}"

            def columns():
                cur.execute(
                    "SELECT column_name, udt_name FROM information_schema.columns "
                    "WHERE table_name='product' AND column_name LIKE '%embedding%' "
                    "ORDER BY column_name"
                )
                found = {r["column_name"] for r in cur.fetchall()}
                if {"text_embedding", "image_embedding"} <= found:
                    return "준비됨"
                raise RuntimeError(
                    "text_embedding · image_embedding 이 없습니다 "
                    f"(현재: {', '.join(sorted(found)) or '없음'}) — "
                    "scripts/split_embedding_columns.sql 을 실행하십시오"
                )

            failures += not check("연결", version)
            failures += not check("pgvector", extension)

            split_done = check("임베딩 컬럼", columns)
            failures += not split_done

            def counts():
                cur.execute(
                    "SELECT (SELECT COUNT(*) FROM product)  AS p,"
                    "       (SELECT COUNT(*) FROM auction)  AS a,"
                    "       (SELECT COUNT(*) FROM bid)      AS b"
                )
                r = cur.fetchone()
                return f"product {r['p']} · auction {r['a']} · bid {r['b']}"

            failures += not check("행 수", counts)
    finally:
        conn.close()

    # --- 탐지 조회기 --------------------------------------------------------
    print("\n이상탐지 조회기")
    fraud = FraudProvider(dsn)

    def fraud_unknown():
        if fraud.load(-1, None) is not None:
            raise RuntimeError("없는 경매인데 None 이 아닙니다")
        return "없는 경매 → None"

    def fraud_real():
        conn2 = psycopg.connect(dsn, row_factory=dict_row)
        try:
            with conn2.cursor() as cur:
                cur.execute(
                    "SELECT auction_id FROM auction "
                    "WHERE started_at IS NOT NULL ORDER BY auction_id LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn2.close()
        if row is None:
            return "시작된 경매가 없어 건너뜀"
        inp = fraud.load(row["auction_id"], None)
        if inp is None:
            return f"경매 {row['auction_id']} → None (시작 전이거나 삭제됨)"
        return (
            f"경매 {inp.auction.auction_id} · 입찰 {len(inp.bids)}건 · "
            f"입찰자 {len(inp.members)}명 · 코퍼스 {'있음' if inp.corpus else '없음'}"
        )

    failures += not check("없는 경매", fraud_unknown)
    failures += not check("실제 경매 조회", fraud_real)

    # --- 추천 조회기 --------------------------------------------------------
    print("\n추천 조회기")
    reco = RecoProvider(dsn)
    now = datetime.now(timezone.utc)
    loaded: list = []

    def reco_active():
        loaded.extend(reco.load_active(now, 500))
        live = sum(1 for c in loaded if c.is_live)
        return f"진행 중 {len(loaded)}건 (라이브 {live} · 일반 {len(loaded) - live})"

    def reco_vectors():
        if not split_done:
            raise RuntimeError("임베딩 컬럼이 아직 분리되지 않았습니다")
        ids = [c.auction_id for c in loaded]
        vectors = reco.load_vectors(ids)
        if not vectors:
            return f"후보 {len(ids)}건 중 임베딩 0건 (배치를 아직 안 돌림)"
        sample = next(iter(vectors.values()))
        return (
            f"{len(vectors)}건 · 텍스트 {len(sample.text)}차원 · "
            f"이미지 {len(sample.image) if sample.image else '없음'}"
        )

    failures += not check("진행 중 경매", reco_active)
    if split_done:
        failures += not check("임베딩 조회", reco_vectors)
    else:
        line(SKIP, "임베딩 조회", "컬럼 분리 대기")

    # --- 임베딩 배치 SQL ----------------------------------------------------
    print("\n임베딩 배치 (읽기만)")
    if not split_done:
        line(SKIP, "대상 조회", "컬럼 분리 대기")
    else:
        conn3 = psycopg.connect(dsn, row_factory=dict_row)
        try:
            with conn3.cursor() as cur:
                def pending():
                    cur.execute(
                        store.PENDING_SQL,
                        {
                            "skip_status": list(store.SKIP_PRODUCT_STATUS),
                            "force": False,
                            "limit": 5,
                        },
                    )
                    rows = store.to_pending(cur.fetchall())
                    with_image = sum(1 for r in rows if r.image_url)
                    return f"{len(rows)}건 (이미지 있음 {with_image})"

                def counting():
                    cur.execute(
                        store.COUNT_SQL,
                        {"skip_status": list(store.SKIP_PRODUCT_STATUS)},
                    )
                    r = cur.fetchone()
                    return f"대상 {r['total']}건 중 임베딩 없음 {r['pending']}건"

                def cast_roundtrip():
                    """`::vector` 캐스팅과 값 왕복을 확인한다. 테이블은 안 건드린다."""
                    literal = store.to_pgvector([0.1, -0.25, 0.5])
                    cur.execute("SELECT %(v)s::vector AS v", {"v": literal})
                    from reco_api.queries import _to_vector

                    back = _to_vector(cur.fetchone()["v"])
                    if back is None or len(back) != 3:
                        raise RuntimeError(f"왕복 실패: {back}")
                    return f"{literal} → {back}"

                failures += not check("대상 조회", pending)
                failures += not check("진행 집계", counting)
                failures += not check("벡터 캐스팅 왕복", cast_roundtrip)
        finally:
            conn3.close()

    print()
    if failures:
        print(f"실패 {failures}건 — 위 FAIL 을 보십시오")
        return 1
    if not split_done:
        print("확인한 범위는 모두 통과했습니다.")
        print("임베딩 관련은 **확인하지 못했습니다** — 백엔드가 컬럼을 나눠야 합니다.")
        return 0
    print("모두 통과. 실제 DB 에서 쿼리가 돕니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
