"""임베딩을 DB 에서 읽고 쓰는 부분.

인코더는 DB 를 모른다. 조회와 저장을 여기로 몰아 두면 **모델을 바꿔도 이 파일은
안 바뀌고, 스키마가 바뀌어도 인코더는 안 바뀐다.**

SQL 과 행 변환을 분리한 이유는 조회기들과 같다 — 연결 없이 검증하기 위해서다.
벡터를 문자열로 만드는 부분이 특히 조용히 틀리기 쉬운데, DB 가 있어야만 확인할 수
있으면 영영 못 본다.

## 쓰기 권한이 필요한 유일한 곳

나머지 AI 코드는 전부 읽기 전용이다. 여기만 `UPDATE` 를 한다.

    GRANT UPDATE (text_embedding, image_embedding) ON product TO ai_user;

백엔드는 이 두 컬럼을 쓰지 않는다 (`@Column(insertable=false, updatable=false)`).
채우는 것은 우리 몫이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# 검수를 통과하지 못한 상품은 임베딩하지 않는다.
#
# 추천에 노출되지 않을 상품을 계산하는 것은 GPU 시간 낭비이고, 거부된 상품의 벡터가
# DB 에 남아 있으면 나중에 정책이 바뀔 때 조용히 추천에 섞여 들어갈 수 있다.
SKIP_PRODUCT_STATUS = ("PENDING", "REJECTED")

# 아직 임베딩되지 않은 상품.
#
# **대표 이미지는 `product_image` 의 `sequence` 가 가장 앞선 것이다.** 그 테이블이
# 비어 있으면 `thumbnail_url` 로 떨어진다 — 상품 등록 경로에 따라 둘 중 하나만
# 채워질 수 있어서다.
PENDING_SQL = """
SELECT p.product_id,
       p.title,
       p.description,
       COALESCE(
           (SELECT i.image_url
            FROM product_image i
            WHERE i.product_id = p.product_id
            ORDER BY i.sequence
            LIMIT 1),
           p.thumbnail_url
       ) AS image_url
FROM product p
WHERE p.deleted_at IS NULL
  AND p.status <> ALL(%(skip_status)s)
  AND (%(force)s OR p.text_embedding IS NULL)
ORDER BY p.product_id
LIMIT %(limit)s
"""

# `::vector` 캐스팅이 필요하다. psycopg 는 파이썬 리스트를 pgvector 타입으로
# 자동 변환하지 못하므로 `'[0.1,0.2]'` 문자열로 넘기고 DB 에서 캐스팅한다.
UPDATE_SQL = """
UPDATE product
SET text_embedding  = %(text)s::vector,
    image_embedding = %(image)s::vector
WHERE product_id = %(product_id)s
"""

# 낡았거나 남아 있으면 안 되는 벡터를 비운다.
#
# **`NULL` 이 곧 "다시 계산해야 함" 이다.** 별도의 `embedded_at` 컬럼 없이 재계산
# 시점을 알아내는 방법이며, 근거는 백엔드 구현에 있다 — 상품을 수정하면 반드시
# `status` 가 `PENDING` 으로 돌아간다 (재검수를 거쳐야 하므로).
#
#     수정 → PENDING → 여기서 벡터를 비움 → 재검수 통과 → 배치가 새로 계산
#
# 거부(`REJECTED`)와 삭제된 상품의 벡터도 함께 지운다. 남겨 둘 이유가 없고, 남아
# 있으면 노출 정책이 바뀔 때 조용히 추천에 섞여 들어간다.
#
# 비운 상품을 같은 실행에서 다시 계산하지는 않는다 — `PENDING` · `REJECTED` 는
# 계산 대상에서 빠지기 때문이다.
RECLAIM_SQL = """
UPDATE product
SET text_embedding  = NULL,
    image_embedding = NULL
WHERE text_embedding IS NOT NULL
  AND (status = ANY(%(skip_status)s) OR deleted_at IS NOT NULL)
"""

COUNT_SQL = """
SELECT COUNT(*) FILTER (WHERE text_embedding IS NULL) AS pending,
       COUNT(*)                                       AS total
FROM product
WHERE deleted_at IS NULL
  AND status <> ALL(%(skip_status)s)
"""

REQUIRED_COLUMNS = ("text_embedding", "image_embedding")

COLUMNS_SQL = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'product'
  AND column_name = ANY(%(names)s)
"""


def missing_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    found = {r["column_name"] for r in rows}
    return tuple(name for name in REQUIRED_COLUMNS if name not in found)


@dataclass(frozen=True, slots=True)
class PendingProduct:
    """임베딩해야 할 상품 1건."""

    product_id: int
    title: str
    description: str | None = None
    image_url: str | None = None


def to_pending(rows: Sequence[Mapping[str, Any]]) -> tuple[PendingProduct, ...]:
    return tuple(
        PendingProduct(
            product_id=r["product_id"],
            title=(r["title"] or "").strip(),
            description=(r["description"] or None),
            image_url=(r["image_url"] or "").strip() or None,
        )
        for r in rows
    )


def to_pgvector(values: Sequence[float] | None) -> str | None:
    """pgvector 리터럴. 값이 없으면 None 이고, **0 벡터로 채우지 않는다.**

    0 은 "닮은 것이 없다" 로 읽혀, 이미지가 없다는 이유만으로 추천 순위가 밀린다.
    NULL 이어야 "아직 모른다" 가 된다.
    """
    if values is None:
        return None
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"
