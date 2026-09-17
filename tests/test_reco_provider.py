"""추천 조회기 검증 — 실제 DB 없이.

여기서 틀리면 **노출하면 안 되는 상품이 추천에 올라온다.** 랭킹은 무엇을 보여줘도
되는지 모르고 점수만 매기므로, 노출 정책은 전부 조회에서 끝나야 한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from reco_api import queries
from reco_api.provider import PostgresProvider, ProviderError
from serve import app

T0 = datetime(2026, 9, 9, 12, 0, 0)
HOME = "/internal/reco/home"


def row(**over):
    return {
        "auction_id": 10,
        "product_id": 1,
        "seller_id": 500,
        "category_id": 7,
        "started_at": T0,
        "auction_time": 3600,
        "ended_at": T0 + timedelta(hours=1),
        "live_broadcast_id": None,
        "view_count": 100,
        "bookmark_count": 5,
        "bid_count": 3,
        "bidder_count": 2,
        **over,
    }


# --- 행 변환 ----------------------------------------------------------------


def test_candidate_takes_seller_and_category_from_product():
    """`auction` 에는 판매자도 카테고리도 없다. product 조인 결과를 읽어야 한다."""
    c = queries.to_candidates([row()])[0]

    assert (c.seller_id, c.category_id) == (500, 7)


def test_db_times_are_lifted_to_utc(monkeypatch):
    """**엔드포인트가 500 으로 죽던 버그.**

    ERD 의 TIMESTAMP 에는 시간대가 없어 드라이버가 naive 로 준다. 랭킹은
    `datetime.now(timezone.utc)` 와 비교하므로, 변환하지 않으면
    `TypeError: can't subtract offset-naive and offset-aware datetimes` 가 난다.

    가짜 커넥션 테스트로는 못 잡았다 — 행 변환만 보고 랭킹까지 이어 보지 않았다.
    """
    monkeypatch.setenv("DIB_DB_TIMEZONE", "Asia/Seoul")
    c = queries.to_candidates([row()])[0]

    assert c.started_at.tzinfo is not None
    assert c.ended_at.tzinfo is not None
    # KST 12:00 == UTC 03:00
    assert c.started_at == datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc)


def test_ranking_runs_on_provider_output(monkeypatch):
    """조회 결과가 랭킹까지 흘러가는지 끝까지 본다.

    행 변환만 검증하면 시간대 불일치처럼 **그다음 단계에서 터지는 것**을 놓친다.
    실제로 놓쳤고, 실DB 에 붙여서야 발견했다.
    """
    from reco import RecoConfig, rank

    monkeypatch.setenv("DIB_DB_TIMEZONE", "Asia/Seoul")

    # DB 가 주는 것과 같은 형태 — 시간대 없는 지역 시각. 진행 중이어야 하므로
    # 지금을 기준으로 잡는다.
    started = datetime.now()
    rows = [
        row(started_at=started, ended_at=started + timedelta(hours=1)),
        row(auction_id=11, started_at=started, ended_at=started + timedelta(hours=1)),
    ]
    candidates = queries.to_candidates(rows)

    result = rank(candidates, RecoConfig.load(), datetime.now(timezone.utc))

    assert len(result.items) == 2, "시간대가 어긋나면 종료된 것으로 걸러지거나 터진다"


def test_live_broadcast_id_marks_a_live_auction():
    general, live = queries.to_candidates([row(), row(auction_id=11, live_broadcast_id=9)])

    assert general.is_live is False
    assert live.is_live is True


def test_null_counters_become_zero():
    """비정규화 컬럼이 NULL 이어도 랭킹이 죽으면 안 된다."""
    c = queries.to_candidates([row(view_count=None, bid_count=None)])[0]

    assert (c.view_count, c.bid_count) == (0, 0)


# --- 벡터 변환 --------------------------------------------------------------


def test_vector_accepts_a_list():
    out = queries.to_vectors(
        [{"auction_id": 10, "text_embedding": [0.1, 0.2], "image_embedding": [0.3]}]
    )

    assert out[10].text == (0.1, 0.2)
    assert out[10].image == (0.3,)


def test_vector_accepts_a_pgvector_string():
    """드라이버 설정에 따라 `'[0.1,0.2]'` 문자열로 온다.

    한쪽만 처리하면 드라이버를 바꾸는 순간 조용히 빈 벡터가 되고, 그러면 유사도가
    전부 0 이 되어 유사 상품 추천이 인기순과 구별되지 않는다.
    """
    out = queries.to_vectors(
        [{"auction_id": 10, "text_embedding": "[0.1,0.2]", "image_embedding": None}]
    )

    assert out[10].text == (0.1, 0.2)
    assert out[10].image is None, "이미지가 없으면 None 이지 0 벡터가 아니다"


def test_row_without_text_vector_is_dropped():
    """텍스트 없이 이미지만으로는 비교하지 않는다."""
    out = queries.to_vectors(
        [{"auction_id": 10, "text_embedding": None, "image_embedding": [0.3]}]
    )

    assert out == {}


# --- 노출 정책 --------------------------------------------------------------


def test_query_hides_products_that_failed_moderation():
    """**빠뜨리면 거래 제한 품목이 추천 목록에 올라온다.**"""
    assert "p.status <> ALL(%(hidden_status)s)" in queries.ACTIVE_CANDIDATES_SQL
    assert set(queries.HIDDEN_PRODUCT_STATUS) == {"PENDING", "REJECTED"}


def test_moderation_filter_is_a_blacklist_not_a_whitelist():
    """`REGISTERED` 만 남기면 후보가 통째로 사라진다.

    경매가 시작되면 상품 상태가 `ON_AUCTION` 으로 바뀌기 때문이다.
    """
    assert "ON_AUCTION" not in queries.HIDDEN_PRODUCT_STATUS
    assert "SOLD" not in queries.HIDDEN_PRODUCT_STATUS


def test_query_excludes_deleted_and_unstarted():
    sql = queries.ACTIVE_CANDIDATES_SQL

    assert "a.status = 'ACTIVE'" in sql
    assert "a.started_at IS NOT NULL" in sql, "남은 시간을 계산할 수 없다"
    assert "a.deleted_at IS NULL" in sql and "p.deleted_at IS NULL" in sql


def test_vector_query_is_an_inner_join():
    """LEFT JOIN 으로 NULL 을 받으면 0 벡터로 채우고 싶어진다. 0 은 "안 닮음" 이다."""
    assert "LEFT JOIN" not in queries.VECTORS_SQL
    assert "p.text_embedding IS NOT NULL" in queries.VECTORS_SQL


# --- provider ---------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows, calls):
        self._rows, self._calls = rows, calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._calls.append((sql, dict(params or {})))

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows):
        self.calls, self.closed, self._rows = [], False, rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self._rows, self.calls)

    def close(self):
        self.closed = True


def provider_with(monkeypatch, rows):
    conn = FakeConnection(rows)
    p = PostgresProvider("postgresql://fake")
    monkeypatch.setattr(p, "_connect", lambda: conn)
    return p, conn


def test_load_active_passes_the_pool_limit(monkeypatch):
    """`LIMIT` 은 랭킹 전에 자르는 후보 풀이다. 여기서 잘리면 랭킹이 못 본다."""
    p, conn = provider_with(monkeypatch, [row()])
    candidates = p.load_active(T0, 500)

    assert [c.auction_id for c in candidates] == [10]
    assert conn.calls[0][1]["limit"] == 500
    assert conn.calls[0][1]["hidden_status"] == ["PENDING", "REJECTED"]


def test_load_vectors_skips_the_query_when_nothing_is_asked(monkeypatch):
    """ANY(빈 배열) 로 전 테이블을 훑는 계획이 나올 수 있다."""
    p, conn = provider_with(monkeypatch, [])

    assert p.load_vectors([]) == {}
    assert conn.calls == []


def test_failure_becomes_provider_error(monkeypatch):
    """조회 실패가 "후보 없음" 으로 둔갑하면 장애가 빈 추천으로 나간다."""
    conn = FakeConnection([])
    p = PostgresProvider("postgresql://fake")
    monkeypatch.setattr(p, "_connect", lambda: conn)
    monkeypatch.setattr(conn, "cursor", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    with pytest.raises(ProviderError):
        p.load_active(T0, 10)
    assert conn.closed


# --- scope (명세 108) -------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_scope_splits_live_and_general(client):
    live = client.get(HOME, params={"scope": "LIVE"}).json()["items"]
    general = client.get(HOME, params={"scope": "GENERAL"}).json()["items"]

    assert {i["auction_id"] for i in live} == {20002, 20003}
    assert {i["auction_id"] for i in general} == {20001, 20004}
    assert not ({i["auction_id"] for i in live} & {i["auction_id"] for i in general})


def test_each_scope_is_ranked_on_its_own(client):
    """섞어서 매긴 뒤 나누면 한쪽이 상위권을 다 가져가 다른 목록이 빈약해진다.

    일반 경매만 놓고 보면 20001 이 1위인데, 전체로 보면 라이브인 20003 에 밀린다.
    """
    everything = client.get(HOME, params={"scope": "ALL"}).json()["items"]
    general = client.get(HOME, params={"scope": "GENERAL"}).json()["items"]

    assert everything[0]["auction_id"] == 20003
    assert general[0]["auction_id"] == 20001
    assert [i["rank"] for i in general] == list(range(1, len(general) + 1))


def test_live_broadcast_id_is_exposed_in_detail(client):
    """백엔드가 다시 조회하지 않고 가를 수 있어야 한다 (명세 108)."""
    items = client.get(HOME, params={"scope": "ALL"}).json()["items"]
    by_id = {i["auction_id"]: i["detail"]["live_broadcast_id"] for i in items}

    assert by_id[20003] == 3001
    assert by_id[20001] is None


def test_unknown_scope_is_rejected(client):
    assert client.get(HOME, params={"scope": "SHORTS"}).status_code == 422
