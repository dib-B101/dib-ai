"""임베딩 배치의 DB 경계 검증 — 실제 DB 없이.

벡터를 문자열로 만드는 부분이 조용히 틀리기 쉽다. 형식이 어긋나면 `::vector` 캐스팅이
실패하거나, 더 나쁘게는 **값이 잘려 들어가 유사도만 이상해진다.**
"""

from __future__ import annotations

import pytest

from embedding import store


def row(**over):
    return {
        "product_id": 1,
        "title": "아이폰 15 프로 256GB",
        "description": "정품 미개봉입니다",
        "image_url": "https://cdn.example.com/a.jpg",
        **over,
    }


# --- 행 변환 ----------------------------------------------------------------


def test_pending_product_is_mapped():
    p = store.to_pending([row()])[0]

    assert p.product_id == 1
    assert p.title == "아이폰 15 프로 256GB"
    assert p.image_url == "https://cdn.example.com/a.jpg"


def test_blank_fields_become_none():
    """빈 문자열과 없음을 구분한다. `""` 를 URL 로 넘기면 읽기 실패 로그만 쌓인다."""
    p = store.to_pending([row(description=None, image_url="   ")])[0]

    assert p.description is None
    assert p.image_url is None


def test_title_is_stripped():
    assert store.to_pending([row(title="  아이폰  ")])[0].title == "아이폰"


# --- pgvector 리터럴 --------------------------------------------------------


def test_vector_literal_format():
    assert store.to_pgvector([0.1, -0.25, 0.0]) == "[0.1,-0.25,0]"


def test_missing_vector_is_none_not_zeros():
    """**0 벡터로 채우지 않는다.**

    0 은 "닮은 것이 없다" 로 읽혀, 사진이 없다는 이유만으로 추천 순위가 밀린다.
    NULL 이어야 "아직 모른다" 가 된다.
    """
    assert store.to_pgvector(None) is None


def test_vector_literal_survives_a_round_trip():
    """우리가 쓴 형식을 우리 조회기가 그대로 읽어야 한다.

    쓰기와 읽기가 다른 모듈이라 형식이 갈라지면 아무도 모르게 어긋난다.
    """
    from reco_api.queries import _to_vector

    original = [0.1234567, -0.5, 0.0, 0.9999999]
    literal = store.to_pgvector(original)

    assert _to_vector(literal) == pytest.approx(original)


def test_large_vector_is_not_truncated():
    """1024차원이 통째로 나가야 한다. 잘리면 차원 불일치로 캐스팅이 실패한다."""
    literal = store.to_pgvector([0.001] * 1024)

    assert literal.count(",") == 1023


# --- 조회 대상 --------------------------------------------------------------


def test_moderation_failures_are_not_embedded():
    """추천에 안 나갈 상품을 계산하는 것은 GPU 시간 낭비다.

    거부된 상품의 벡터가 남아 있으면 정책이 바뀔 때 조용히 추천에 섞인다.
    """
    assert set(store.SKIP_PRODUCT_STATUS) == {"PENDING", "REJECTED"}


def test_pending_query_only_picks_empty_rows_by_default():
    """다시 돌려도 안전해야 한다. 중간에 죽으면 이어서 하면 된다."""
    assert "p.text_embedding IS NULL" in store.PENDING_SQL
    assert "%(force)s OR" in store.PENDING_SQL, "--all 로 전부 다시 돌릴 수 있어야 한다"
    assert "p.deleted_at IS NULL" in store.PENDING_SQL


def test_representative_image_comes_from_sequence():
    """대표 이미지는 `product_image.sequence` 가 가장 앞선 것.

    그 테이블이 비어 있으면 `thumbnail_url` 로 떨어진다 — 등록 경로에 따라 둘 중
    하나만 채워질 수 있다.
    """
    assert "ORDER BY i.sequence" in store.PENDING_SQL
    assert "p.thumbnail_url" in store.PENDING_SQL
    assert "COALESCE(" in store.PENDING_SQL


def test_update_casts_to_vector():
    """psycopg 는 파이썬 값을 pgvector 타입으로 자동 변환하지 못한다."""
    assert "%(text)s::vector" in store.UPDATE_SQL
    assert "%(image)s::vector" in store.UPDATE_SQL
    assert "WHERE product_id = %(product_id)s" in store.UPDATE_SQL


def test_update_touches_only_the_vector_columns():
    """이 배치는 벡터 말고 아무것도 건드리지 않는다.

    쓰기 권한도 두 컬럼에만 받았다 —
    `GRANT UPDATE (text_embedding, image_embedding) ON product`.
    """
    body = store.UPDATE_SQL[store.UPDATE_SQL.index("SET") : store.UPDATE_SQL.index("WHERE")]

    assert set(c.strip() for c in body.replace("SET", "").split(",") if "=" in c) == {
        "text_embedding  = %(text)s::vector",
        "image_embedding = %(image)s::vector",
    }


# --- 낡은 벡터 회수 ---------------------------------------------------------


def test_reclaim_targets_products_that_went_back_to_pending():
    """`embedded_at` 컬럼 없이 재계산 시점을 알아내는 근거.

    백엔드는 상품을 수정하면 반드시 `status` 를 `PENDING` 으로 되돌린다
    (`ProductCommandServiceImpl:129`). 재검수를 거쳐야 하기 때문이다. 그래서
    **`PENDING` 이면 내용이 바뀌었다는 뜻**이고, 벡터를 비우면 재검수 통과 후
    배치가 알아서 다시 계산한다.
    """
    sql = store.RECLAIM_SQL

    assert "status = ANY(%(skip_status)s)" in sql
    assert "text_embedding IS NOT NULL" in sql, "이미 비어 있으면 건드릴 필요가 없다"


def test_reclaim_also_clears_deleted_and_rejected():
    """남겨 두면 노출 정책이 바뀔 때 조용히 추천에 섞여 들어간다."""
    assert "deleted_at IS NOT NULL" in store.RECLAIM_SQL
    assert "REJECTED" in store.SKIP_PRODUCT_STATUS


def test_reclaim_wipes_both_columns():
    """텍스트만 비우면 이미지 벡터가 낡은 채 남아 유사도를 오염시킨다."""
    body = store.RECLAIM_SQL[
        store.RECLAIM_SQL.index("SET") : store.RECLAIM_SQL.index("WHERE")
    ]

    assert "text_embedding  = NULL" in body
    assert "image_embedding = NULL" in body


def test_reclaimed_rows_are_not_recomputed_in_the_same_run():
    """비운 상품을 곧바로 다시 계산하면 무한 반복이 된다.

    `PENDING` · `REJECTED` 가 계산 대상에서 빠지므로 그런 일이 없다.
    두 SQL 이 같은 목록을 쓰는지 확인한다.
    """
    assert "%(skip_status)s" in store.RECLAIM_SQL
    assert "%(skip_status)s" in store.PENDING_SQL
