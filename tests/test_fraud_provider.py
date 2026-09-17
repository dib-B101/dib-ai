"""PostgresProvider 검증 — 실제 DB 없이.

조회 결과를 자료구조로 옮기는 부분이 실제로 틀리기 쉬운 곳인데, 연결이 없으면
영영 못 본다. 그래서 SQL 과 행 변환을 `queries.py` 로 떼어 두고 여기서 직접 부른다.

**시점 누수를 특히 본다.** 과거 이력 조회에 `as_of` 조건을 빠뜨리면 미래 정보가
섞이는데, 평가할 때는 오히려 잘 맞는 것처럼 보여서 눈으로는 안 잡힌다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fraud import Auction
from fraud_api import queries
from fraud_api.provider import PostgresProvider, ProviderError

T0 = datetime(2026, 9, 8, 14, 0, 0)
AS_OF = T0 + timedelta(minutes=3)


def auction_row(**over):
    return {
        "auction_id": 10,
        "seller_id": 500,
        "category_id": 7,
        "start_price": 10_000,
        "started_at": T0,
        "auction_time": 180,
        "ended_at": AS_OF,
        **over,
    }


# --- 행 변환 ----------------------------------------------------------------


def test_auction_takes_seller_and_category_from_product():
    """`auction` 에는 판매자도 카테고리도 없다. product 조인 결과를 읽어야 한다."""
    a = queries.to_auction(auction_row())

    assert (a.seller_id, a.category_id) == (500, 7)
    assert a.original_end_at == T0 + timedelta(seconds=180)


def test_bids_are_mapped_in_order():
    rows = [
        {"bid_id": 1, "auction_id": 10, "member_id": 21, "amount": 11_000, "created_at": T0},
        {"bid_id": 2, "auction_id": 10, "member_id": 22, "amount": 12_000, "created_at": AS_OF},
    ]
    bids = queries.to_bids(rows)

    assert [b.bid_id for b in bids] == [1, 2]
    assert bids[0].member_id == 21


def test_histories_group_by_member():
    rows = [
        {"member_id": 21, "auction_id": 1, "seller_id": 500,
         "participated_at": T0, "won": False, "bidding_ratio": 0.5},
        {"member_id": 21, "auction_id": 2, "seller_id": 600,
         "participated_at": T0, "won": True, "bidding_ratio": 0.3},
        {"member_id": 22, "auction_id": 1, "seller_id": 500,
         "participated_at": T0, "won": False, "bidding_ratio": 0.2},
    ]
    out = queries.to_histories(rows)

    assert set(out) == {21, 22}
    assert len(out[21]) == 2 and len(out[22]) == 1
    assert out[21][1].won is True


def test_history_without_participation_has_no_key():
    """빈 튜플을 채워 넣지 않는다. "조회했는데 없었다" 가 그대로 드러나야 한다."""
    assert queries.to_histories([]) == {}


def test_null_bidding_ratio_becomes_zero():
    """NULLIF 로 0 을 걸렀으므로 NULL 이 올 수 있다. 소극 참여로 본다."""
    rows = [{"member_id": 21, "auction_id": 1, "seller_id": 500,
             "participated_at": T0, "won": False, "bidding_ratio": None}]

    assert queries.to_histories(rows)[21][0].bidding_ratio == 0.0


def test_subscriptions_collapse_to_a_set_per_member():
    rows = [
        {"subscriber_id": 21, "broadcaster_id": 500},
        {"subscriber_id": 21, "broadcaster_id": 600},
        {"subscriber_id": 22, "broadcaster_id": 500},
    ]
    out = queries.to_subscriptions(rows)

    assert out[21] == frozenset({500, 600})
    assert out[22] == frozenset({500})


# --- 코퍼스 ----------------------------------------------------------------


def test_corpus_needs_enough_samples():
    """두세 건의 평균은 값이 아니라 잡음이다. 그 잡음으로 판정하면 안 된다."""
    thin = {"n": queries.MIN_CORPUS_AUCTIONS - 1, "mean_bids": 18.0,
            "mean_start_price": 50_000.0}

    assert queries.to_corpus(thin, category_id=7) is None


def test_corpus_is_built_when_samples_suffice():
    fat = {"n": queries.MIN_CORPUS_AUCTIONS, "mean_bids": 18.0,
           "mean_start_price": 50_000.0}
    corpus = queries.to_corpus(fat, category_id=7)

    assert corpus is not None
    assert (corpus.mean_bids, corpus.mean_start_price) == (18.0, 50_000.0)


def test_missing_corpus_is_none_not_zero():
    """**0 으로 채우지 않는다.** 0 은 "평균이 0" 이라는 주장이고, 그러면
    Auction_Bids 가 모든 경매에서 최대값이 된다."""
    assert queries.to_corpus(None, category_id=7) is None
    assert queries.to_corpus({"n": 100, "mean_bids": 0, "mean_start_price": 0}, 7) is None


# --- 시각 ------------------------------------------------------------------


def test_as_of_defaults_to_end_time():
    a = queries.to_auction(auction_row())

    assert queries.resolve_as_of(None, a) == AS_OF


def test_aware_time_is_converted_not_stripped(monkeypatch):
    """**tzinfo 만 떼면 안 된다. 시각을 옮겨야 한다.**

    백엔드는 DB 의 naive TIMESTAMP 를 Instant 로 바꿔 UTC 로 보낸다. KST 14:00 이
    05:00Z 로 오는데, 여기서 tzinfo 만 떼면 05:00 이 되어 9시간이 밀린다. 그러면
    모든 입찰이 경매 시작 이전으로 계산되고 Early_Bidding·Last_Bidding 이 1.0
    (최대 위험)이 된다 — 예외도 로그도 없이.
    """
    monkeypatch.setenv("DIB_DB_TIMEZONE", "Asia/Seoul")
    naive_kst = datetime(2026, 9, 8, 14, 0, 0)          # DB 에서 읽은 값
    wire = datetime(2026, 9, 8, 5, 0, 0, tzinfo=timezone.utc)  # 같은 순간, 백엔드 표현

    assert queries.align_to(wire, naive_kst) == naive_kst


def test_naive_time_is_localised_for_aware_reference(monkeypatch):
    monkeypatch.setenv("DIB_DB_TIMEZONE", "Asia/Seoul")
    aware_utc = datetime(2026, 9, 8, 5, 0, 0, tzinfo=timezone.utc)

    aligned = queries.align_to(datetime(2026, 9, 8, 14, 0, 0), aware_utc)

    assert aligned == aware_utc


def test_align_is_a_no_op_when_kinds_already_match():
    naive = datetime(2026, 9, 8, 14, 0, 0)
    assert queries.align_to(naive, T0) == naive

    aware = datetime(2026, 9, 8, 5, 0, 0, tzinfo=timezone.utc)
    assert queries.align_to(aware, aware) == aware


def test_as_of_is_aligned_to_the_auction(monkeypatch):
    """요청의 as_of 는 tz 를 달고 오고 경매는 naive 다. 섞이면 TypeError 가 난다."""
    monkeypatch.setenv("DIB_DB_TIMEZONE", "Asia/Seoul")
    a = queries.to_auction(auction_row())
    requested = datetime(2026, 9, 8, 5, 2, 0, tzinfo=timezone.utc)  # KST 14:02

    resolved = queries.resolve_as_of(requested, a)

    assert resolved.tzinfo is None
    assert (resolved.hour, resolved.minute) == (14, 2)
    assert resolved > a.started_at


# --- provider 조립 ----------------------------------------------------------


class FakeCursor:
    """SQL 별로 정해진 행을 돌려준다. 어떤 파라미터로 불렸는지 기록한다.

    **두 개 이상의 키에 걸리면 실패시킨다.** 조용히 첫 번째 것을 돌려주면 엉뚱한
    행이 흘러들어가고, 그러면 테스트가 통과했는데 검증한 것은 다른 조회가 된다.
    실제로 CORPUS_SQL 과 AUCTION_SQL 이 같은 조인 문구를 공유해 한 번 당했다.
    """

    def __init__(self, responses: dict[str, list[dict]], calls: list):
        self._responses = responses
        self._calls = calls
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        hits = [key for key in self._responses if key in sql]
        assert len(hits) <= 1, f"SQL 이 여러 응답에 걸립니다: {hits}"
        self._rows = list(self._responses[hits[0]]) if hits else []
        self._calls.append((sql, dict(params or {})))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, responses: dict[str, list[dict]]):
        self.calls: list = []
        self.closed = False
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self._responses, self.calls)

    def close(self):
        self.closed = True


def provider_with(monkeypatch, responses) -> tuple[PostgresProvider, FakeConnection]:
    conn = FakeConnection(responses)
    p = PostgresProvider("postgresql://fake")
    monkeypatch.setattr(p, "_connect", lambda: conn)
    return p, conn


FULL = {
    "WHERE a.auction_id = %(auction_id)s": [auction_row()],
    "FROM bid\nWHERE auction_id": [
        {"bid_id": 1, "auction_id": 10, "member_id": 21, "amount": 11_000, "created_at": T0},
        {"bid_id": 2, "auction_id": 10, "member_id": 22, "amount": 12_000, "created_at": T0},
    ],
    "FROM member\nWHERE": [
        {"member_id": 21, "created_at": T0 - timedelta(days=100)},
        {"member_id": 22, "created_at": T0 - timedelta(days=3)},
    ],
    "WITH mine AS": [
        {"member_id": 21, "auction_id": 9, "seller_id": 500,
         "participated_at": T0 - timedelta(days=1), "won": False, "bidding_ratio": 0.6},
    ],
    "FROM subscription": [{"subscriber_id": 22, "broadcaster_id": 500}],
    "AVG(a.bid_count)": [
        {"n": 50, "mean_bids": 18.0, "mean_start_price": 50_000.0}
    ],
}


def test_load_assembles_detection_input(monkeypatch):
    p, _ = provider_with(monkeypatch, FULL)
    inp = p.load(10, None)

    assert inp is not None
    assert inp.auction.seller_id == 500
    assert [b.bid_id for b in inp.bids] == [1, 2]
    assert set(inp.members) == {21, 22}
    assert inp.histories[21][0].auction_id == 9
    assert inp.is_subscribed(22, 500) and not inp.is_subscribed(21, 500)
    assert inp.corpus is not None and inp.corpus.category_id == 7


def test_unknown_auction_returns_none(monkeypatch):
    p, _ = provider_with(monkeypatch, {})

    assert p.load(999, None) is None


def test_auction_not_started_returns_none(monkeypatch):
    """started_at 이 NULL 이면 모든 시점 비율이 계산되지 않는다."""
    p, _ = provider_with(
        monkeypatch,
        {"WHERE a.auction_id = %(auction_id)s": [auction_row(started_at=None)]},
    )

    assert p.load(10, None) is None


def test_no_bids_skips_the_member_queries(monkeypatch):
    """ANY(빈 배열) 로 전 테이블을 훑는 계획이 나올 수 있다."""
    p, conn = provider_with(
        monkeypatch,
        {
            "WHERE a.auction_id = %(auction_id)s": [auction_row()],
            "FROM bid\nWHERE auction_id": [],
        },
    )
    inp = p.load(10, None)

    assert inp is not None and inp.bids == ()
    assert not any("FROM member" in sql for sql, _ in conn.calls)
    assert not any("WITH mine" in sql for sql, _ in conn.calls)


def test_every_history_query_is_bounded_by_as_of(monkeypatch):
    """**시점 누수 방지.** 하나라도 빠지면 미래 정보가 섞인다.

    평가할 때는 오히려 잘 맞는 것처럼 보여서 눈으로는 안 잡힌다.
    """
    p, conn = provider_with(monkeypatch, FULL)
    p.load(10, None)

    bounded = [
        (sql, params)
        for sql, params in conn.calls
        if any(k in sql for k in ("FROM bid", "WITH mine", "FROM subscription", "AVG("))
    ]
    assert len(bounded) == 4, "입찰·이력·구독·코퍼스 네 조회를 모두 봐야 한다"
    for sql, params in bounded:
        assert params.get("as_of") == AS_OF, f"as_of 가 안 걸린 조회: {sql[:40]}"


def test_corpus_excludes_the_auction_itself(monkeypatch):
    """자기 값이 자기 기준선을 끌어올리면 붐빈 경매일수록 평범해 보인다."""
    p, conn = provider_with(monkeypatch, FULL)
    p.load(10, None)

    sql, params = next((c for c in conn.calls if "AVG(a.bid_count)" in c[0]))
    assert "a.auction_id <> %(auction_id)s" in sql
    assert params["auction_id"] == 10


def test_thin_corpus_still_returns_input_without_ml(monkeypatch):
    """코퍼스가 모자라도 규칙 점수는 나가야 한다."""
    thin = {**FULL, "AVG(a.bid_count)": [{"n": 2, "mean_bids": 5.0, "mean_start_price": 100.0}]}
    p, _ = provider_with(monkeypatch, thin)
    inp = p.load(10, None)

    assert inp is not None and inp.corpus is None


def test_query_failure_becomes_provider_error(monkeypatch):
    """조회 실패와 "경매 없음" 은 다르다. 섞으면 장애가 정상 응답으로 나간다."""
    conn = FakeConnection(FULL)

    def boom():
        raise RuntimeError("connection reset")

    p = PostgresProvider("postgresql://fake")
    monkeypatch.setattr(p, "_connect", lambda: conn)
    monkeypatch.setattr(conn, "cursor", boom)

    with pytest.raises(ProviderError):
        p.load(10, None)


def test_connection_is_closed_even_on_failure(monkeypatch):
    conn = FakeConnection(FULL)
    p = PostgresProvider("postgresql://fake")
    monkeypatch.setattr(p, "_connect", lambda: conn)
    monkeypatch.setattr(conn, "cursor", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    with pytest.raises(ProviderError):
        p.load(10, None)
    assert conn.closed


def test_detection_runs_on_provider_output(monkeypatch):
    """조회 결과가 엔진에 그대로 들어가는지 끝까지 확인한다.

    자료구조만 맞고 값이 엉뚱하면 여기서 예외가 난다.
    """
    from fraud import RuleConfig, detect

    p, _ = provider_with(monkeypatch, FULL)
    inp = p.load(10, None)

    result = detect(inp, RuleConfig.load())

    assert result.auction_id == 10
    assert result.auction_error is None
