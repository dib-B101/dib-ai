"""인기순 랭킹 검증.

이 로직은 세 곳에서 쓰인다 — Cold Start 폴백, 평가 baseline, 장애 폴백.
한 곳이 깨지면 세 곳이 같이 깨지므로 경계 조건을 촘촘히 본다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from reco import Candidate, RecoConfig, percentile_ranks, rank, urgency

NOW = datetime(2026, 9, 9, 12, 0, 0)


@pytest.fixture(scope="module")
def cfg() -> RecoConfig:
    return RecoConfig.load()


def make(auction_id: int, *, ends_in_min: float = 60, views=0, marks=0, bids=0, bidders=0):
    return Candidate(
        auction_id=auction_id,
        product_id=auction_id,
        seller_id=500,
        category_id=1,
        started_at=NOW - timedelta(hours=1),
        auction_time=7200,
        ended_at=NOW + timedelta(minutes=ends_in_min),
        view_count=views,
        bookmark_count=marks,
        bid_count=bids,
        bidder_count=bidders,
    )


# ---------------------------------------------------------------- 백분위

def test_percentile_spreads_over_zero_to_one():
    assert percentile_ranks([10, 20, 30]) == [0.0, 0.5, 1.0]


def test_percentile_gives_ties_the_same_value():
    """같은 조회수인데 순위가 갈리면 안 된다."""
    assert percentile_ranks([5, 5, 9]) == [0.0, 0.0, 1.0]


def test_percentile_is_flat_when_all_values_equal():
    """서비스 초기에는 모든 지표가 0 이다.

    이때 억지로 순서를 만들면 안 된다. 전원 0 이 되어 이 항이 순위에 영향을 주지 않는다.
    """
    assert percentile_ranks([0, 0, 0, 0]) == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("values, expected", [([], []), ([7], [0.0])])
def test_percentile_handles_small_inputs(values, expected):
    assert percentile_ranks(values) == expected


# ---------------------------------------------------------------- 마감 임박도

def test_urgency_decreases_with_time_left():
    tau = 3600
    assert urgency(60, tau) > urgency(3600, tau) > urgency(36000, tau)


def test_urgency_never_exceeds_one():
    """역수로 정의하면 1초 남은 경매가 무한대가 되어 다른 항을 전부 눌러버린다."""
    for remaining in (0.001, 1, 10):
        assert urgency(remaining, 3600) <= 1.0


def test_urgency_of_ended_auction_is_max():
    assert urgency(0, 3600) == 1.0
    assert urgency(-500, 3600) == 1.0


# ---------------------------------------------------------------- 랭킹

def test_ended_auctions_are_excluded(cfg):
    """종료된 경매는 마감 임박도가 최대라, 안 거르면 목록 맨 위에 올라온다."""
    result = rank(
        [make(1, ends_in_min=30), make(2, ends_in_min=-1, views=9999, marks=999)],
        cfg,
        NOW,
    )

    assert [s.auction_id for s in result.items] == [1]
    assert result.excluded == {"ended": 1}


def test_all_ended_returns_empty_not_error(cfg):
    """추천이 비는 것과 터지는 것은 다르다. 빈 목록으로 응답해야 한다."""
    result = rank([make(1, ends_in_min=-5)], cfg, NOW)

    assert result.items == ()
    assert result.excluded == {"ended": 1}


def test_popular_item_outranks_ignored_one_at_same_deadline(cfg):
    result = rank(
        [
            make(1, ends_in_min=60, views=10, marks=0, bids=0, bidders=0),
            make(2, ends_in_min=60, views=5000, marks=200, bids=40, bidders=15),
        ],
        cfg,
        NOW,
    )
    assert [s.auction_id for s in result.items] == [2, 1]


def test_imminent_item_outranks_distant_one_at_same_popularity(cfg):
    result = rank(
        [make(1, ends_in_min=600, views=100), make(2, ends_in_min=2, views=100)],
        cfg,
        NOW,
    )
    assert [s.auction_id for s in result.items] == [2, 1]


def test_ties_break_by_auction_id_for_reproducibility(cfg):
    """같은 입력이면 항상 같은 순서가 나와야 평가와 디버깅이 가능하다."""
    same = [make(7), make(3), make(5)]
    assert [s.auction_id for s in rank(same, cfg, NOW).items] == [3, 5, 7]


def test_limit_caps_result(cfg):
    result = rank([make(i) for i in range(1, 11)], cfg, NOW, limit=3)
    assert len(result.items) == 3


def test_boost_hook_lets_personalization_plug_in(cfg):
    """개인화는 이 자리에 끼워 넣는다. 랭킹 코드를 다시 쓰지 않기 위한 설계다."""
    plain = rank([make(1), make(2)], cfg, NOW)
    boosted = rank([make(1), make(2)], cfg, NOW, boost=lambda c: 1.0 if c.auction_id == 2 else 0.0)

    assert [s.auction_id for s in plain.items] == [1, 2]
    assert [s.auction_id for s in boosted.items] == [2, 1]


def test_new_listing_gets_credit_as_deadline_nears(cfg):
    """지표가 0 인 신규 상품도 마감이 다가오면 점수가 오른다.

    인기도·경쟁도가 0 이어도 마감 임박도는 인기와 무관하게 계산되므로, 신규 상품이
    구조적으로 0 점에 묶이지는 않는다.
    """
    fresh_far = rank([make(1, ends_in_min=600)], cfg, NOW).items[0]
    fresh_near = rank([make(1, ends_in_min=1)], cfg, NOW).items[0]

    assert fresh_near.score > fresh_far.score > 0


def test_max_popularity_outranks_max_urgency_by_design(cfg):
    """가중치 설계상 마감 임박(0.4)만으로는 인기+경쟁 최상위(0.6)를 못 이긴다.

    1분 남았지만 아무도 안 본 경매보다, 6시간 남았지만 경쟁이 붙은 경매를 보여주는
    것이 낫다는 판단이다. 이 균형을 바꾸려면 config/reco.yaml 의 가중치를 조정한다.
    """
    result = rank(
        [
            make(1, ends_in_min=600, views=5000, marks=300, bids=50, bidders=20),
            make(2, ends_in_min=1, views=0, marks=0, bids=0, bidders=0),
        ],
        cfg,
        NOW,
    )
    assert [s.auction_id for s in result.items] == [1, 2]


def test_remaining_seconds_falls_back_to_auction_time():
    """ended_at 이 아직 없으면 started_at + auction_time 으로 유도한다."""
    c = Candidate(
        auction_id=1, product_id=1, seller_id=1, category_id=1,
        started_at=NOW - timedelta(minutes=10), auction_time=3600, ended_at=None,
    )
    assert c.remaining_seconds(NOW) == pytest.approx(3000)


# ---------------------------------------------------------------- 설정

def test_config_rejects_weights_that_do_not_sum_to_one(tmp_path):
    """가중치 오타 하나로 순서가 조용히 망가지면 원인을 찾기 어렵다."""
    bad = tmp_path / "reco.yaml"
    bad.write_text(
        "version: t\n"
        "urgency: {tau_seconds: 3600}\n"
        "popularity: {view_count: 1.0}\n"
        "competition: {bid_count: 1.0}\n"
        "weights: {urgency: 0.9, popularity: 0.9, competition: 0.9}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="가중치 합"):
        RecoConfig.load(bad)


def test_config_rejects_non_positive_tau(tmp_path):
    bad = tmp_path / "reco.yaml"
    bad.write_text(
        "version: t\n"
        "urgency: {tau_seconds: 0}\n"
        "popularity: {view_count: 1.0}\n"
        "competition: {bid_count: 1.0}\n"
        "weights: {urgency: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tau_seconds"):
        RecoConfig.load(bad)


def test_shipped_config_is_valid(cfg):
    assert cfg.version.startswith("reco-")
    assert sum(cfg.weights.values()) == pytest.approx(1.0)
