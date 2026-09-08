"""API 계약 검증.

엔드포인트가 백엔드와 약속한 형태를 지키는지 본다. 규칙 로직 자체는
test_rules.py / test_engine.py 가 검증하므로 여기서는 다루지 않는다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_api.demo import NORMAL_AUCTION_ID, PINGPONG_AUCTION_ID
from fraud_api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_config_and_rules(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["rule_config_version"].startswith("rules-")
    assert len(body["enabled_rules"]) > 0
    # 비활성 규칙은 나오지 않아야 한다
    assert "R5_MULTI_ACCOUNT" not in body["enabled_rules"]


def test_detect_returns_fraud_detection_columns(client):
    """응답이 fraud_detection 컬럼과 1:1 로 대응해야 한다."""
    r = client.post("/internal/fraud/detect", json={"auction_id": PINGPONG_AUCTION_ID})
    assert r.status_code == 200
    body = r.json()

    assert body["auction_id"] == PINGPONG_AUCTION_ID
    assert body["rule_config_version"].startswith("rules-")
    assert body["auction_error"] is None
    assert body["results"], "핑퐁 시나리오는 판정 결과가 있어야 한다"

    row = body["results"][0]
    for col in ("member_id", "risk_score", "rule_score", "ml_score", "band", "detail"):
        assert col in row, f"{col} 이 응답에 없다"

    assert row["ml_score"] is None, "모델 트랙 가동 전에는 null 이어야 한다"
    assert row["band"] in ("low", "medium", "high")
    assert 0.0 <= row["rule_score"] <= 1.0


def test_detail_carries_rule_scores_and_version(client):
    """detail 은 fraud_detection.detail (JSONB) 에 그대로 들어간다."""
    r = client.post("/internal/fraud/detect", json={"auction_id": PINGPONG_AUCTION_ID})
    detail = r.json()["results"][0]["detail"]

    assert "rules" in detail, "규칙별 점수 — 총점에서 역산이 불가능하므로 반드시 필요하다"
    assert "features" in detail
    assert "flags" in detail
    assert "skipped_rules" in detail
    assert detail["rule_config_version"].startswith("rules-")


def test_pingpong_scores_higher_than_normal(client):
    """의심 시나리오가 정상보다 높은 점수를 받아야 한다."""
    def top_score(auction_id: int) -> float:
        body = client.post(
            "/internal/fraud/detect", json={"auction_id": auction_id}
        ).json()
        return max((x["rule_score"] for x in body["results"]), default=0.0)

    assert top_score(PINGPONG_AUCTION_ID) > top_score(NORMAL_AUCTION_ID)


def test_same_request_is_reproducible(client):
    """같은 입력이면 언제 호출하든 같은 결과가 나와야 한다."""
    payload = {"auction_id": PINGPONG_AUCTION_ID, "as_of": "2026-09-08T14:03:00"}
    first = client.post("/internal/fraud/detect", json=payload).json()
    second = client.post("/internal/fraud/detect", json=payload).json()
    assert first == second


def test_unknown_auction_returns_404(client):
    r = client.post("/internal/fraud/detect", json={"auction_id": 999_999})
    assert r.status_code == 404


def test_reasons_are_human_readable(client):
    """탐지 사유는 관리자 화면에 그대로 노출된다. 수치가 들어 있어야 한다."""
    r = client.post("/internal/fraud/detect", json={"auction_id": PINGPONG_AUCTION_ID})
    reasons = [x for row in r.json()["results"] for x in row["reasons"]]
    assert reasons, "점수가 있으면 사유도 있어야 한다"
    assert any(any(ch.isdigit() for ch in text) for text in reasons)


def test_openapi_document_is_generated(client):
    """백엔드는 /docs 를 보고 연동한다. 스키마가 깨지면 안 된다."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/internal/fraud/detect" in paths
    assert "/health" in paths
