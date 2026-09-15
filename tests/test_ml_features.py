"""ML 서빙 피처 검증.

**학습은 eBay 원본 값으로 하고 서빙은 여기서 계산한 값을 쓴다.** 식이 어긋나면
예외 없이 그럴듯한 점수가 나와서 알아채기 어렵다. 그래서 방향과 경계를 촘촘히 본다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from fraud import ml_features as MF
from fraud.schema import Auction, Bid, CorpusStats, DetectionInput, HistoryEntry, Member

NOW = datetime(2026, 9, 15, 12, 0, 0)
DURATION = 600  # 10분


def auction(start_price: int = 10_000) -> Auction:
    return Auction(
        auction_id=1, seller_id=500, category_id=1, start_price=start_price,
        started_at=NOW - timedelta(seconds=DURATION), auction_time=DURATION, ended_at=NOW,
    )


def bid_at(member_id: int, second: float, bid_id: int = 1) -> Bid:
    """경매 시작 후 `second` 초 시점의 입찰."""
    return Bid(bid_id, 1, member_id, 1000 * bid_id,
               NOW - timedelta(seconds=DURATION - second))


def history(sellers, won=False, ratio=0.4) -> tuple[HistoryEntry, ...]:
    return tuple(
        HistoryEntry(900 + i, s, NOW - timedelta(days=i + 1), won, ratio)
        for i, s in enumerate(sellers)
    )


# ---------------------------------------------------------------- 시점 피처

def test_last_bidding_is_high_when_stopping_early():
    """값이 클수록 **일찍 멈춘** 것이다. 가격만 올려놓고 빠지는 패턴."""
    early_quit = MF.last_bidding([bid_at(1, 60)], 1, auction())
    stayed = MF.last_bidding([bid_at(1, 60), bid_at(1, 590, 2)], 1, auction())

    assert early_quit > stayed
    assert early_quit == pytest.approx(0.9)   # 600 중 60초에 멈춤 → 90% 를 비활동


def test_early_bidding_is_high_when_joining_early():
    early = MF.early_bidding([bid_at(1, 30)], 1, auction())
    late = MF.early_bidding([bid_at(1, 540)], 1, auction())

    assert early > late
    assert early == pytest.approx(0.95)


def test_time_features_use_original_end_not_extended():
    """연장된 ended_at 을 분모로 쓰면 같은 입찰이 경매마다 다른 비율이 된다."""
    extended = Auction(
        auction_id=1, seller_id=500, category_id=1, start_price=10_000,
        started_at=NOW - timedelta(seconds=DURATION), auction_time=DURATION,
        ended_at=NOW + timedelta(seconds=300),   # 연장됨
    )
    assert MF.early_bidding([bid_at(1, 30)], 1, extended) == pytest.approx(0.95)


@pytest.mark.parametrize("fn, expected", [(MF.last_bidding, 1.0), (MF.early_bidding, 0.0)])
def test_no_bid_means_fully_inactive(fn, expected):
    assert fn([bid_at(2, 100)], 1, auction()) == expected


# ---------------------------------------------------------------- 코퍼스 대비

def test_auction_bids_is_zero_below_average():
    """평균 이하면 0. 신호가 없다는 뜻이지 낮은 값이 아니다."""
    assert MF.auction_bids(10, 18.0) == 0.0
    assert MF.auction_bids(18, 18.0) == 0.0


def test_auction_bids_grows_with_crowding():
    assert MF.auction_bids(36, 18.0) == pytest.approx(0.5)
    assert MF.auction_bids(90, 18.0) == pytest.approx(0.8)


def test_starting_price_is_high_when_cheap():
    """싸게 시작해 사람을 끌어들이는 것이 전형적인 수법이라 방향이 뒤집혀 있다."""
    assert MF.starting_price_average(10_000, 50_000) == pytest.approx(0.8)
    assert MF.starting_price_average(50_000, 50_000) == 0.0
    assert MF.starting_price_average(90_000, 50_000) == 0.0


@pytest.mark.parametrize("fn", [MF.auction_bids, MF.starting_price_average])
def test_zero_average_does_not_divide_by_zero(fn):
    assert fn(10, 0.0) == 0.0


# ---------------------------------------------------------------- 이력 피처

def test_winning_ratio_counts_only_active_participation():
    """한두 번 찔러본 경매까지 세면 누구나 낙찰률이 낮게 나온다."""
    passive = history([1, 2, 3], won=False, ratio=0.05)   # 전부 소극 참여
    assert MF.winning_ratio(passive) == 0.0

    active_lost = history([1, 2, 3], won=False, ratio=0.4)
    assert MF.winning_ratio(active_lost) == 1.0

    active_won = history([1, 2, 3], won=True, ratio=0.4)
    assert MF.winning_ratio(active_won) == 0.0


def test_winning_ratio_without_history_is_zero():
    assert MF.winning_ratio(()) == 0.0


# ---------------------------------------------------------------- 편중도 재정의

def test_bidder_tendency_flags_concentration_without_subscription():
    """구독도 안 했는데 한 판매자만 따라다니면 공범 의심이다."""
    h = history([500, 500, 500, 500])
    assert MF.bidder_tendency(h, 500, 1, {}) == 1.0


def test_bidder_tendency_clears_loyal_subscriber():
    """같은 이력이라도 구독 중이면 정상 행동이다. 이게 재정의의 핵심이다."""
    h = history([500, 500, 500, 500])
    assert MF.bidder_tendency(h, 500, 1, {1: frozenset({500})}) == 0.0


def test_bidder_tendency_excludes_subscribed_sellers_from_denominator():
    """구독한 판매자 참여를 분모에 남기면 비구독 편중이 희석된다."""
    h = history([500, 500, 700, 700, 700, 700])   # 700 은 구독 중
    # 구독 건을 빼면 남는 것은 500 두 건뿐 → 편중 100%
    assert MF.bidder_tendency(h, 500, 1, {1: frozenset({700})}) == 1.0
    # 빼지 않았다면 2/6 = 0.33 이었을 것이다
    assert MF.bidder_tendency(h, 500, 1, {}) == pytest.approx(2 / 6)


def test_bidder_tendency_without_history_is_zero():
    assert MF.bidder_tendency((), 500, 1, {}) == 0.0


# ---------------------------------------------------------------- 전체 계산

def make_input(corpus: CorpusStats | None, subs=None) -> DetectionInput:
    bids = tuple(bid_at(m, s, i + 1) for i, (m, s) in
                 enumerate([(1, 10), (2, 30), (1, 33), (2, 60), (1, 63)]))
    return DetectionInput(
        auction=auction(),
        bids=bids,
        members={1: Member(1, NOW - timedelta(days=3))},
        histories={1: history([500] * 6)},
        as_of=NOW,
        subscriptions=subs or {},
        corpus=corpus,
    )


def test_compute_returns_none_without_corpus():
    """기준값을 모르는 것과 평균과 같은 것은 다르다. 0 으로 채우면 안 된다."""
    assert MF.compute(make_input(None), 1) is None


def test_compute_returns_all_feature_names():
    features = MF.compute(make_input(CorpusStats(1, 18.0, 50_000.0)), 1)

    assert set(features) == set(MF.FEATURE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in features.values())


def test_to_vector_follows_given_order():
    """순서를 호출자가 넘기게 한 것은 모델과 어긋나는 사고를 막기 위해서다."""
    features = {"a": 1.0, "b": 2.0, "c": 3.0}

    assert MF.to_vector(features, ["a", "b", "c"]) == [1.0, 2.0, 3.0]
    assert MF.to_vector(features, ["c", "a", "b"]) == [3.0, 1.0, 2.0]


def test_subscription_changes_the_feature_vector():
    """입찰 행동이 같아도 구독 여부로 편중도가 갈린다."""
    corpus = CorpusStats(1, 18.0, 50_000.0)
    stranger = MF.compute(make_input(corpus), 1)
    subscriber = MF.compute(make_input(corpus, {1: frozenset({500})}), 1)

    assert stranger["Bidder_Tendency"] == 1.0
    assert subscriber["Bidder_Tendency"] == 0.0
    # 나머지 피처는 그대로여야 한다
    for name in set(MF.FEATURE_NAMES) - {"Bidder_Tendency"}:
        assert stranger[name] == subscriber[name]
