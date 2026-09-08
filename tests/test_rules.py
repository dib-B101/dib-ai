"""규칙별 단위 테스트.

각 시나리오는 겨냥한 규칙에서만 점수가 높고, 정상 시나리오에서는 낮아야 한다.
"""

from __future__ import annotations

import pytest

from fraud import RuleConfig, detect

import fixtures as fx


@pytest.fixture(scope="module")
def cfg() -> RuleConfig:
    return RuleConfig.load()


def score_of(result, member_id: int, rule_id: str) -> float | None:
    """특정 입찰자의 특정 규칙 점수. 게이트에 걸렸으면 None."""
    for r in result.results:
        if r.member_id != member_id:
            continue
        for h in r.hits:
            if h.rule_id == rule_id:
                return h.score
        return None
    raise AssertionError(f"member {member_id} 결과 없음")


def rule_score_of(result, member_id: int) -> float:
    for r in result.results:
        if r.member_id == member_id:
            return r.rule_score
    raise AssertionError(f"member {member_id} 결과 없음")


# ---------------------------------------------------------------- R1

def test_r1_pingpong_scores_high(cfg):
    """3초 간격으로 균일하게 재탈환하면 높은 점수."""
    result = detect(fx.pingpong_auction(), cfg)
    assert score_of(result, 21, "R1_RECLAIM_SPEED") > 0.8


def test_r1_normal_scores_low(cfg):
    """수백 초 간격이면 낮은 점수."""
    result = detect(fx.normal_auction(), cfg)
    assert score_of(result, 11, "R1_RECLAIM_SPEED") < 0.2


def test_r1_gated_when_too_few_reclaims(cfg):
    """재탈환 1회뿐이면 판단하지 않는다 (0점이 아니라 미판정)."""
    result = detect(fx.pingpong_auction(), cfg)
    for r in result.results:
        if r.member_id == 23:  # 한 번만 입찰한 회원
            assert "R1_RECLAIM_SPEED" in r.skipped_rules
            break
    else:
        raise AssertionError("member 23 결과 없음")


# ---------------------------------------------------------------- R2

def test_r2_dominant_scores_high(cfg):
    """전체 10건 중 5건을 차지하면 상한에 가까워 높은 점수."""
    result = detect(fx.dominant_bidder_auction(), cfg)
    assert score_of(result, 31, "R2_BID_DOMINANCE") > 0.7


def test_r2_normal_scores_low(cfg):
    """5명이 2건씩 고르게 나누면 균등값이라 0점."""
    result = detect(fx.normal_auction(), cfg)
    assert score_of(result, 11, "R2_BID_DOMINANCE") == pytest.approx(0.0, abs=1e-9)


def test_r2_never_exceeds_one(cfg):
    """정규화 결과가 1을 넘지 않는다."""
    for name, make in fx.ALL_SCENARIOS.items():
        result = detect(make(), cfg)
        for r in result.results:
            for h in r.hits:
                assert 0.0 <= h.score <= 1.0, f"{name}/{r.member_id}/{h.rule_id}={h.score}"


# ---------------------------------------------------------------- R3

def test_r3_new_account_scores_high(cfg):
    """가입 0.5일차가 시작가의 몇 배까지 올리면 점수가 붙는다."""
    result = detect(fx.new_account_auction(), cfg)
    assert score_of(result, 41, "R3_NEW_ACCOUNT_HIGH_BID") > 0.7


def test_r3_old_account_scores_zero(cfg):
    """같은 금액이어도 오래된 계정이면 0점 — 신규성과 곱하기 때문."""
    result = detect(fx.new_account_auction(), cfg)
    assert score_of(result, 42, "R3_NEW_ACCOUNT_HIGH_BID") == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- R4

def test_r4_loyal_bidder_scores_high(cfg):
    """최근 10경매가 전부 같은 판매자면 높은 점수."""
    result = detect(fx.loyal_bidder_auction(), cfg)
    assert score_of(result, 51, "R4_SELLER_CONCENTRATION") > 0.9


def test_r4_gated_when_history_too_short(cfg):
    """과거 참여가 최소 조건 미만이면 판단하지 않는다."""
    inp = fx.loyal_bidder_auction()
    thin = {m: h[:2] for m, h in inp.histories.items()}   # 2건으로 줄인다
    inp = fx.DetectionInput(
        auction=inp.auction, bids=inp.bids, members=inp.members,
        histories=thin, as_of=inp.as_of,
    )
    result = detect(inp, cfg)
    for r in result.results:
        assert "R4_SELLER_CONCENTRATION" in r.skipped_rules


def test_r4_confidence_scales_with_history(cfg):
    """편중도가 같아도 이력이 짧으면 점수가 낮아야 한다."""
    long_hist = detect(fx.loyal_bidder_auction(), cfg)
    long_score = score_of(long_hist, 51, "R4_SELLER_CONCENTRATION")

    inp = fx.loyal_bidder_auction()
    short = dict(inp.histories)
    short[51] = fx.make_history(51, [fx.SELLER] * 4)      # 100% 편중이지만 4건뿐
    inp = fx.DetectionInput(
        auction=inp.auction, bids=inp.bids, members=inp.members,
        histories=short, as_of=inp.as_of,
    )
    short_score = score_of(detect(inp, cfg), 51, "R4_SELLER_CONCENTRATION")
    assert short_score < long_score


# ---------------------------------------------------------------- R6

def test_r6_late_surge_scores_high(cfg):
    """마감 직전 급등을 주도한 입찰자는 높은 점수."""
    result = detect(fx.late_surge_auction(), cfg)
    assert score_of(result, 61, "R6_LATE_SURGE") > 0.5


def test_r6_gated_when_no_late_bids(cfg):
    """마감 직전 입찰이 없으면 판단하지 않는다."""
    result = detect(fx.normal_auction(), cfg)
    for r in result.results:
        assert "R6_LATE_SURGE" in r.skipped_rules


def test_r6_uses_original_end_at(cfg):
    """end_at 이 연장으로 밀려도 결과가 바뀌지 않아야 한다."""
    from datetime import timedelta

    inp = fx.late_surge_auction()
    base = score_of(detect(inp, cfg), 61, "R6_LATE_SURGE")

    extended = fx.Auction(
        auction_id=inp.auction.auction_id,
        seller_id=inp.auction.seller_id,
        category_id=inp.auction.category_id,
        start_price=inp.auction.start_price,
        started_at=inp.auction.started_at,
        auction_time=inp.auction.auction_time,
        ended_at=inp.auction.original_end_at + timedelta(minutes=30),  # 연장됨
    )
    inp2 = fx.DetectionInput(
        auction=extended, bids=inp.bids, members=inp.members,
        histories=inp.histories, as_of=inp.as_of,
    )
    assert score_of(detect(inp2, cfg), 61, "R6_LATE_SURGE") == pytest.approx(base)


# ---------------------------------------------------------------- 종합

def test_normal_scenario_stays_low(cfg):
    """정상 시나리오는 모든 참여자의 총점이 낮아야 한다."""
    result = detect(fx.normal_auction(), cfg)
    for r in result.results:
        assert r.rule_score < 0.3, f"member {r.member_id} = {r.rule_score}"


def test_suspicious_scenarios_beat_normal(cfg):
    """의심 시나리오의 대상자는 정상 시나리오 최고점보다 높아야 한다."""
    normal_max = max(r.rule_score for r in detect(fx.normal_auction(), cfg).results)
    targets = {
        "pingpong": (fx.pingpong_auction, 21),
        "dominant": (fx.dominant_bidder_auction, 31),
        "new_account": (fx.new_account_auction, 41),
        "loyal": (fx.loyal_bidder_auction, 51),
        "late_surge": (fx.late_surge_auction, 61),
    }
    for name, (make, member_id) in targets.items():
        got = rule_score_of(detect(make(), cfg), member_id)
        assert got > normal_max, f"{name}: {got} <= 정상 최고 {normal_max}"


def test_r5_is_disabled(cfg):
    """R5 는 데이터 미수집으로 비활성 상태여야 한다."""
    assert cfg.rules["R5_MULTI_ACCOUNT"].enabled is False
    result = detect(fx.normal_auction(), cfg)
    for r in result.results:
        assert all(h.rule_id != "R5_MULTI_ACCOUNT" for h in r.hits)
