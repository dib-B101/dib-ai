"""엔진 계약 테스트 — 티켓의 완료 조건에 1:1로 대응한다."""

from __future__ import annotations

from datetime import timedelta

import pytest

from fraud import RuleConfig, combine, detect
import fraud.rules as rules_mod

import fixtures as fx


@pytest.fixture(scope="module")
def cfg() -> RuleConfig:
    return RuleConfig.load()


# 완료 조건 1: 탐지 규칙별 점수와 탐지 사유가 기록된다
def test_records_per_rule_score_and_reason(cfg):
    result = detect(fx.dominant_bidder_auction(), cfg)
    target = next(r for r in result.results if r.member_id == 31)

    assert target.hits, "규칙별 점수가 하나도 없다"
    for hit in target.hits:
        assert 0.0 <= hit.score <= 1.0
        assert hit.reason.strip(), f"{hit.rule_id} 의 사유가 비어 있다"
        # 사유에는 실제 수치가 들어가야 한다 (관리자 검토·이의제기 대응용)
        assert any(ch.isdigit() for ch in hit.reason)

    detail = target.to_detail()
    assert set(detail) >= {"rules", "features", "flags", "skipped_rules"}
    assert detail["rules"], "detail.rules 가 비어 있다"


# 완료 조건 2: 임계값을 설정으로 조정할 수 있다
def _r1_only_config(version: str, **params) -> RuleConfig:
    base = {"min_reclaims": 2, "dispersion_threshold": 1.0, "uniformity_influence": 0.0}
    return RuleConfig.from_dict({
        "version": version,
        "gates": {"min_bids": 4, "min_bidders": 3},
        "rules": {
            "R1_RECLAIM_SPEED": {
                "enabled": True, "weight": 1.0, "params": {**base, **params},
            }
        },
        "bands": {"low": [0.0, 0.3], "medium": [0.3, 0.6], "high": [0.6, 1.01]},
    })


def test_threshold_is_configurable(cfg):
    """기준을 좁히면 같은 입력의 R1 점수가 내려가야 한다."""
    strict = _r1_only_config("test-strict", max_fast_seconds=2, min_fast_seconds=2)
    loose = _r1_only_config("test-loose", max_fast_seconds=600, min_fast_seconds=600)

    inp = fx.pingpong_auction()
    s_strict = next(r for r in detect(inp, strict).results if r.member_id == 21).rule_score
    s_loose = next(r for r in detect(inp, loose).results if r.member_id == 21).rule_score
    assert s_strict < s_loose


def test_r1_threshold_scales_with_auction_length(cfg):
    """같은 3초 재탈환이라도 경매가 길수록 더 이상하다.

    라이브 경매는 몇 분간 모두가 화면을 보고 있어 3초 반응이 정상이다.
    며칠짜리 경매에서의 3초 반응은 사람이 아니라 자동화에 가깝다.
    절대 초로 임계값을 박으면 이 차이를 구분할 수 없다.

    (기본 설정에는 max_fast_seconds cap 이 있어 긴 경매는 한계에서 멈춘다.
     여기서는 스케일링 자체를 검증하려고 cap 을 사실상 해제한다.)
    """
    scaled = _r1_only_config("test-scaled", fast_ratio=0.05,
                             min_fast_seconds=1, max_fast_seconds=100_000)

    short = fx.pingpong_auction(duration_hours=1)
    long_ = fx.pingpong_auction(duration_hours=48)      # 같은 입찰, 긴 경매

    s_short = next(r for r in detect(short, scaled).results if r.member_id == 21).rule_score
    s_long = next(r for r in detect(long_, scaled).results if r.member_id == 21).rule_score
    assert s_short < s_long


def test_config_version_is_recorded(cfg):
    result = detect(fx.normal_auction(), cfg)
    assert result.rule_config_version == cfg.version


# 완료 조건 3: 동일 입력에 대해 재현 가능한 결과를 반환한다
def test_deterministic(cfg):
    inp = fx.pingpong_auction()
    a = detect(inp, cfg)
    b = detect(inp, cfg)
    assert [r.to_detail() for r in a.results] == [r.to_detail() for r in b.results]
    assert [r.rule_score for r in a.results] == [r.rule_score for r in b.results]


def test_as_of_is_injected_not_now(cfg):
    """as_of 를 바꾸면 계정 나이가 달라져 R3 점수가 달라진다.

    내부에서 now() 를 부르면 이 테스트가 통과할 수 없다.
    """
    inp = fx.new_account_auction()
    early = next(r for r in detect(inp, cfg).results if r.member_id == 41).rule_score

    later = fx.DetectionInput(
        auction=inp.auction, bids=inp.bids, members=inp.members,
        histories=inp.histories, as_of=inp.as_of + timedelta(days=30),
    )
    assert next(r for r in detect(later, cfg).results if r.member_id == 41).rule_score < early


def test_future_history_is_ignored(cfg):
    """as_of 이후의 이력이 섞여 있어도 무시해야 한다 (미래 정보 누수 차단)."""
    inp = fx.loyal_bidder_auction()
    base = next(r for r in detect(inp, cfg).results if r.member_id == 51).rule_score

    polluted = dict(inp.histories)
    polluted[51] = polluted[51] + tuple(
        fx.HistoryEntry(auction_id=99900 + i, seller_id=fx.OTHER_SELLER,
                        participated_at=inp.as_of + timedelta(days=i + 1))
        for i in range(20)
    )
    inp2 = fx.DetectionInput(
        auction=inp.auction, bids=inp.bids, members=inp.members,
        histories=polluted, as_of=inp.as_of,
    )
    assert next(r for r in detect(inp2, cfg).results if r.member_id == 51).rule_score == base


# 완료 조건 4: 초기 단계에서는 거래를 자동 차단하지 않는다
def test_no_blocking_decision(cfg):
    """결과에는 점수와 사유만 있고 차단·제재 필드가 없다."""
    result = detect(fx.pingpong_auction(), cfg)
    for r in result.results:
        fields = set(r.__slots__)
        assert not (fields & {"blocked", "banned", "action", "suspend"})
        assert 0.0 <= r.rule_score <= 1.0


# 완료 조건 5: 정상·의심 샘플 데이터에 대한 규칙 테스트가 있다  → test_rules.py

# 완료 조건 6: 탐지 실패가 정상 입찰 처리 자체를 중단시키지 않는다
def test_rule_exception_is_isolated(cfg, monkeypatch):
    """규칙 하나가 터져도 나머지는 계속 계산되고 예외가 밖으로 나가지 않는다."""
    def boom(ctx):
        raise RuntimeError("의도적 실패")

    monkeypatch.setitem(rules_mod.RULE_FUNCS, "R2_BID_DOMINANCE", boom)
    result = detect(fx.dominant_bidder_auction(), cfg)

    assert result.auction_error is None
    target = next(r for r in result.results if r.member_id == 31)
    assert any("R2_BID_DOMINANCE" in e for e in target.errors)
    assert any(h.rule_id != "R2_BID_DOMINANCE" for h in target.hits), "다른 규칙이 멈췄다"


def test_engine_never_raises(cfg):
    """입력이 망가져도 예외 대신 auction_error 로 돌려준다."""
    broken = fx.DetectionInput(
        auction=fx.make_auction(),
        bids=fx.normal_auction().bids,
        members={},
        histories=None,          # type: ignore[arg-type]  의도적으로 잘못된 입력
        as_of=fx.T0,
    )
    result = detect(broken, cfg)
    assert result.auction_error is not None
    assert result.results == ()


# 최소 분석 조건
def test_tiny_auction_is_skipped(cfg):
    """입찰자가 적으면 입찰 비중이 기계적으로 커지므로 아예 판정하지 않는다."""
    result = detect(fx.tiny_auction(), cfg)
    assert result.results == ()
    assert result.skipped_bidders
    assert all("최소 분석 조건" in reason for reason in result.skipped_bidders.values())


def test_missing_rule_is_reported_not_crashed():
    """설정에 켜져 있지만 구현이 없는 규칙은 건너뛰고 기록만 남긴다."""
    cfg = RuleConfig.from_dict({
        "version": "test-missing",
        "gates": {"min_bids": 4, "min_bidders": 3},
        "rules": {"R5_MULTI_ACCOUNT": {"enabled": True, "weight": 1.0, "params": {}}},
        "bands": {"low": [0.0, 0.3], "medium": [0.3, 0.6], "high": [0.6, 1.01]},
    })
    result = detect(fx.normal_auction(), cfg)
    assert result.auction_error is None
    for r in result.results:
        assert r.skipped_rules.get("R5_MULTI_ACCOUNT") == "구현되지 않은 규칙"
        assert r.rule_score == 0.0


# 점수 결합
def test_combine_weights():
    assert combine(0.8, 0.4, 0.7, 0.3) == pytest.approx(0.68)
    assert combine(0.8, None, 0.7, 0.3) == pytest.approx(0.8)  # 모델 미서빙 기간


def test_combine_is_invertible():
    """두 점수를 각각 저장하면 가중치를 역산할 수 있다."""
    rule, ml, w = 0.79, 0.62, 0.7
    risk = combine(rule, ml, w, 1 - w)
    assert (risk - ml) / (rule - ml) == pytest.approx(w)


def test_band_mapping(cfg):
    assert cfg.band_of(0.1) == "low"
    assert cfg.band_of(0.45) == "medium"
    assert cfg.band_of(0.9) == "high"
