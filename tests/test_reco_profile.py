"""개인화 추천 검증 (STEP 4·5).

이 기능이 조용히 틀리는 방식은 세 가지다.

**이미 본 상품을 다시 추천한다.** 프로필이 그 상품들의 평균이라 유사도가 당연히
최상위로 나온다. 안 빼면 추천 목록이 이미 입찰한 상품으로 채워진다.

**로그가 없을 때 0 벡터를 만든다.** 모든 상품과 유사도 0 이 되어 "아무것도 안
좋아함" 으로 읽히는데, 그건 "모름" 과 다르다.

**시간 감쇠가 안 걸린다.** 반년 전 관심과 어제 관심이 같은 무게가 된다.

셋 다 예외가 안 나고 그럴듯한 목록이 나온다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from reco import BehaviorEvent, ProductVector, RecoConfig, affinity, build_profile
from reco.profile import DEFAULT_EVENT_WEIGHTS, decay
from serve import app

NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
HOME = "/internal/reco/home"


def event(kind: str, auction_id: int, days_ago: float = 0.0) -> BehaviorEvent:
    return BehaviorEvent(kind, auction_id, NOW - timedelta(days=days_ago))


def vec(auction_id: int, text, image=None) -> ProductVector:
    return ProductVector(auction_id, tuple(text), tuple(image) if image else None)


# 축: [스마트폰, 노트북, 가전]
PHONE = vec(1, (1.0, 0.0, 0.0))
LAPTOP = vec(2, (0.6, 0.8, 0.0))
FRIDGE = vec(3, (0.0, 0.0, 1.0))
STORE = {v.auction_id: v for v in (PHONE, LAPTOP, FRIDGE)}


@pytest.fixture(scope="module")
def cfg() -> RecoConfig:
    return RecoConfig.load()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# --- 시간 감쇠 --------------------------------------------------------------


def test_decay_halves_each_half_life():
    assert decay(0, 14) == pytest.approx(1.0)
    assert decay(14, 14) == pytest.approx(0.5)
    assert decay(28, 14) == pytest.approx(0.25)


def test_decay_never_reaches_zero():
    """선형 감쇠는 어느 순간 0 이 되어 오래된 행동이 통째로 사라진다.

    지수 감쇠는 작아지되 남아서, 로그가 적은 초기에도 쓸 수 있다.
    """
    assert decay(3650, 14) > 0


def test_recent_behaviour_outweighs_old(cfg):
    """반년 전 냉장고보다 어제 본 휴대폰이 커야 한다."""
    events = [event("BID", 3, days_ago=180), event("VIEW", 1, days_ago=1)]
    profile = build_profile(1, events, STORE, NOW, half_life_days=14)

    assert profile is not None
    # 휴대폰 축(0번)이 가전 축(2번)보다 커야 한다
    assert profile.text[0] > profile.text[2]


# --- 행동 가중치 ------------------------------------------------------------


def test_stronger_actions_weigh_more():
    """입찰이 조회보다 강한 신호다."""
    assert DEFAULT_EVENT_WEIGHTS["BID"] > DEFAULT_EVENT_WEIGHTS["WATCH"]
    assert DEFAULT_EVENT_WEIGHTS["WATCH"] > DEFAULT_EVENT_WEIGHTS["VIEW"]


def test_impression_carries_no_signal():
    """노출됐을 뿐이다. 관심의 증거가 아니다."""
    assert DEFAULT_EVENT_WEIGHTS["IMPRESSION"] == 0
    assert build_profile(1, [event("IMPRESSION", 1)], STORE, NOW) is None


def test_repeated_actions_accumulate():
    """보고 찜하고 입찰한 상품은 하나만 한 상품보다 관심이 크다."""
    one = build_profile(1, [event("VIEW", 1), event("BID", 3)], STORE, NOW)
    many = build_profile(
        1, [event("VIEW", 1), event("WATCH", 1), event("BID", 1), event("BID", 3)], STORE, NOW
    )

    assert one is not None and many is not None
    assert many.text[0] > one.text[0], "휴대폰 쪽으로 더 기울어야 한다"


def test_unwatch_cancels_a_watch():
    """찜 해제는 이전 찜을 되돌리는 명시적 행동이다."""
    events = [event("WATCH", 1, 1), event("UNWATCH", 1, 0), event("BID", 3, 1)]
    profile = build_profile(1, events, STORE, NOW)

    assert profile is not None
    assert profile.text[2] > profile.text[0], "가전만 남아야 한다"


# --- 프로필을 못 만드는 경우 ------------------------------------------------


def test_no_events_gives_no_profile():
    """신규 사용자. **0 벡터를 만들지 않는다.**"""
    assert build_profile(1, [], STORE, NOW) is None


def test_missing_embeddings_give_no_profile():
    """행동한 상품에 임베딩이 없다. 배치가 아직 안 돌았다."""
    assert build_profile(1, [event("BID", 99)], STORE, NOW) is None


def test_only_negative_signals_give_no_profile():
    """무엇을 좋아하는지 알 수 없다. 싫어하는 것만으로 추천할 수는 없다."""
    assert build_profile(1, [event("UNWATCH", 1), event("SCROLL_PASS", 2)], STORE, NOW) is None


def test_mismatched_dimensions_are_skipped():
    """차원이 다른 벡터는 섞지 않는다. 모델이 바뀐 흔적이다."""
    mixed = {**STORE, 4: vec(4, (1.0, 0.0))}
    profile = build_profile(1, [event("BID", 1), event("BID", 4)], mixed, NOW)

    assert profile is not None and len(profile.text) == 3


# --- 이미 본 상품 제외 ------------------------------------------------------


def test_already_seen_items_are_excluded(cfg):
    """**입찰한 경매를 다시 추천하면 안 된다.**

    프로필이 그 상품들의 평균이라 유사도가 최상위로 나온다. 안 빼면 추천 목록이
    이미 아는 상품으로 채워진다.
    """
    profile = build_profile(1, [event("BID", 1)], STORE, NOW)
    scores = affinity(profile, [PHONE, LAPTOP, FRIDGE], cfg.similarity_text_weight)

    assert 1 not in scores, "입찰한 경매가 추천에 남았다"
    assert set(scores) == {2, 3}


def test_seen_includes_negatively_weighted_items():
    """스크롤로 지나친 상품도 '이미 본 것' 이다. 다시 올릴 이유가 없다."""
    profile = build_profile(1, [event("BID", 1), event("SCROLL_PASS", 2)], STORE, NOW)

    assert profile is not None
    assert 2 in profile.seen


def test_impression_does_not_mark_an_item_as_seen():
    """**노출은 행동이 아니다.** 목록에 떴다는 이유로 다시 안 보여주면 안 된다.

    `SCROLL_PASS` 와의 차이가 이 테스트의 요점이다. 스크롤로 지나친 것은 사용자가
    그 자리에 있었고 넘겼다는 뜻이지만, 노출은 **화면 아래쪽에 렌더링됐을 뿐**
    사용자가 봤는지조차 모른다. 둘을 같이 취급하면 한 번 실린 적 있는 상품이
    영영 추천에서 사라진다.
    """
    profile = build_profile(1, [event("BID", 1), event("IMPRESSION", 3)], STORE, NOW)

    assert profile is not None
    assert 3 not in profile.seen, "노출만 된 상품이 추천에서 빠졌다"


def test_impression_only_item_stays_recommendable(cfg):
    """노출만 된 상품은 후보로 살아 있어야 한다. seen 제외의 실제 효과다."""
    profile = build_profile(1, [event("BID", 1), event("IMPRESSION", 2)], STORE, NOW)
    scores = affinity(profile, [LAPTOP, FRIDGE], cfg.similarity_text_weight)

    assert 2 in scores, "노출됐다는 이유로 추천 후보에서 빠졌다"


def test_affinity_ranks_similar_items_higher(cfg):
    """휴대폰에 관심을 보였으면 노트북이 냉장고보다 위여야 한다."""
    profile = build_profile(1, [event("BID", 1)], STORE, NOW)
    scores = affinity(profile, [LAPTOP, FRIDGE], cfg.similarity_text_weight)

    assert scores[2] > scores[3]


# --- API ---------------------------------------------------------------------


def test_member_without_logs_falls_back_to_popularity(client):
    """합성 데모에는 행동 로그가 없다. 개인화가 안 되어도 추천은 나가야 한다."""
    body = client.get(HOME, params={"member_id": 12345}).json()

    assert body["strategy"] == "popularity"
    assert body["items"], "폴백이 비면 화면이 빈다"


def test_anonymous_request_still_works(client):
    body = client.get(HOME).json()

    assert body["strategy"] == "popularity"
    assert body["items"]


def test_personalization_weight_is_lower_than_similarity(cfg):
    """유사 상품은 "지금 이걸 보고 있다" 는 확실한 신호지만,
    관심 프로필은 과거 행동에서 추정한 값이라 덜 확실하다."""
    assert cfg.personalization_weight < cfg.similarity_weight
