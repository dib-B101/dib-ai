"""모델 로딩과 서빙 연결 검증.

학습된 모델 파일이 있어야 도는 테스트는 건너뛴다. CI 에서 4MB 짜리 모델을
받아오게 만들 수는 없고, 모델 품질은 학습 스크립트가 따로 본다.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fraud_ml.predict import MODEL_FILE, FraudModel, ModelNotAvailable

pytestmark = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="학습된 모델이 없습니다 (scripts/train_bootstrap_model.py)",
)


@pytest.fixture(scope="module")
def model() -> FraudModel:
    return FraudModel.load()


def test_model_loads_with_version_and_features(model):
    assert model.version
    assert len(model.features) >= 5


def test_serving_features_match_training_set(model):
    """이름이 하나라도 어긋나면 모델이 다른 값을 읽는다."""
    from fraud.ml_features import FEATURE_NAMES

    assert set(model.features) == set(FEATURE_NAMES)


def test_score_is_a_probability(model):
    features = {name: 0.5 for name in model.features}

    score = model.score(features)
    assert 0.0 <= score <= 1.0


def test_score_many_matches_score_one_by_one(model):
    rows = [{name: v for name in model.features} for v in (0.1, 0.5, 0.9)]

    batch = model.score_many(rows)
    assert batch == pytest.approx([model.score(r) for r in rows])


def test_score_many_handles_empty(model):
    assert model.score_many([]) == []


def test_concentration_raises_the_score(model):
    """편중도가 높을수록 점수가 올라야 한다. 방향이 뒤집히면 모델이 반대로 읽는 것이다."""
    low = {name: 0.2 for name in model.features}
    high = {**low, "Bidder_Tendency": 0.95, "Bidding_Ratio": 0.6}

    assert model.score(high) > model.score(low)


def test_missing_model_file_raises_clearly(tmp_path):
    """모델이 없어도 서버는 떠야 한다. 무엇이 없는지 알려주고 끝난다."""
    with pytest.raises(ModelNotAvailable, match="모델 파일이 없습니다"):
        FraudModel.load(model_path=tmp_path / "없음.joblib")


def test_feature_mismatch_is_rejected(tmp_path):
    """학습 피처와 서빙 피처가 다르면 조용히 틀린 점수가 나온다. 로드에서 막는다."""
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({"model_version": "x", "features": ["Bidding_Ratio", "없는피처"]}),
        encoding="utf-8",
    )
    with pytest.raises(ModelNotAvailable, match="학습 피처와 서빙 피처가 다릅니다"):
        FraudModel.load(meta_path=meta)


# ---------------------------------------------------------------- API 연결

def detect(monkeypatch=None, **overrides) -> dict:
    import fraud_api.main as api

    for name, value in overrides.items():
        monkeypatch.setattr(api, name, value)
    with TestClient(api.app) as c:
        return c.post("/internal/fraud/detect", json={"auction_id": 10002}).json()


def test_api_returns_ml_score_when_enabled(monkeypatch):
    """w_ml 을 올리면 ml_score 와 model_version 이 채워진다."""
    body = detect(monkeypatch, W_ML=0.3, W_RULE=0.7)

    assert body["model_version"]
    assert body["weights"] == {"w_rule": 0.7, "w_ml": 0.3}
    assert body["ml_shadow"] is False, "점수에 반영됐으므로 섀도가 아니다"
    assert all(r["ml_score"] is not None for r in body["results"])


# ---------------------------------------------------------------- 섀도 모드
#
# 모델을 켜려면 우리 데이터에서 룰보다 나은지 보여야 하는데, 가중치가 0 이라고
# 모델을 아예 안 부르면 비교할 자료가 영영 안 쌓인다. 섀도는 그 교착을 푼다.


def test_shadow_computes_ml_without_changing_the_verdict(monkeypatch):
    """**판정은 한 자리도 바뀌지 않는다.** 이게 깨지면 섀도가 아니라 조용한 롤아웃이다."""
    body = detect(monkeypatch, W_ML=0.0, W_RULE=1.0, ML_SHADOW=True)

    assert body["ml_shadow"] is True
    assert body["weights"]["w_ml"] == 0.0
    assert all(r["ml_score"] is not None for r in body["results"]), "비교할 값이 없다"
    assert all(r["risk_score"] == r["rule_score"] for r in body["results"])


def test_shadow_records_which_model_produced_the_score(monkeypatch):
    """어느 모델이 낸 점수인지 모르면 나중에 비교를 못 한다. 섀도로 모으는 이유가 그것이다."""
    body = detect(monkeypatch, W_ML=0.0, W_RULE=1.0, ML_SHADOW=True)

    assert body["model_version"]


def test_shadow_scores_equal_the_enabled_scores(monkeypatch):
    """섀도가 다른 값을 낸다면 지금 모으는 자료로 나중을 예측할 수 없다."""
    shadow = detect(monkeypatch, W_ML=0.0, W_RULE=1.0, ML_SHADOW=True)
    live = detect(monkeypatch, W_ML=0.3, W_RULE=0.7, ML_SHADOW=True)

    def by_member(body):
        return {r["member_id"]: r["ml_score"] for r in body["results"]}

    assert by_member(shadow) == pytest.approx(by_member(live))


def test_shadow_off_skips_the_model_entirely(monkeypatch):
    """끄면 추론 자체를 안 한다. 모델이 느려지거나 터질 때 빠져나갈 문이 있어야 한다."""
    body = detect(monkeypatch, W_ML=0.0, W_RULE=1.0, ML_SHADOW=False)

    assert body["ml_shadow"] is False
    assert body["model_version"] is None
    assert all(r["ml_score"] is None for r in body["results"])


def test_pingpong_scores_higher_than_bystander(monkeypatch):
    """두 트랙이 독립적으로 같은 결론에 닿아야 한다."""
    import fraud_api.main as api

    monkeypatch.setattr(api, "W_ML", 0.3)
    monkeypatch.setattr(api, "W_RULE", 0.7)

    with TestClient(api.app) as c:
        body = c.post("/internal/fraud/detect", json={"auction_id": 10002}).json()

    scores = {r["member_id"]: r for r in body["results"]}
    # 21·22 가 핑퐁 계정, 23 은 우연히 낀 일반 입찰자
    assert scores[21]["ml_score"] > scores[23]["ml_score"]
    assert scores[22]["ml_score"] > scores[23]["ml_score"]
