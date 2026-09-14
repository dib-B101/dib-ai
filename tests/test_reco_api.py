"""추천 API 계약 검증.

백엔드는 `items` 순서대로 노출한다. 순서와 필드 의미가 바뀌면 여기서 깨져야 한다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reco_api.main import app

HOME = "/internal/reco/home"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_exposes_config_and_weights(client):
    body = client.get("/reco/health").json()

    assert body["status"] == "ok"
    assert body["config_version"].startswith("reco-")
    assert sum(body["weights"].values()) == pytest.approx(1.0)


def test_home_returns_ranked_items(client):
    body = client.get(HOME).json()

    assert body["strategy"] == "popularity"
    assert body["items"], "데모 후보가 있으므로 비면 안 된다"
    assert [i["rank"] for i in body["items"]] == list(range(1, len(body["items"]) + 1))

    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True), "점수 내림차순이어야 한다"


def test_home_excludes_ended_auction(client):
    """데모에는 인기 최상위지만 이미 끝난 경매가 하나 들어 있다."""
    body = client.get(HOME).json()

    assert body["excluded"].get("ended") == 1
    assert 20005 not in [i["auction_id"] for i in body["items"]]


def test_detail_explains_the_rank(client):
    """사용자에게 보여주진 않지만, 왜 이 순위인지 설명할 수 없으면 튜닝도 못 한다."""
    detail = client.get(HOME).json()["items"][0]["detail"]

    assert {"urgency", "popularity", "competition", "remaining_seconds"} <= detail.keys()
    assert detail["remaining_seconds"] > 0


def test_limit_is_respected(client):
    assert len(client.get(f"{HOME}?limit=2").json()["items"]) == 2


@pytest.mark.parametrize("limit", [0, 101, -1])
def test_invalid_limit_is_rejected(client, limit):
    assert client.get(f"{HOME}?limit={limit}").status_code == 422


def test_member_id_is_accepted_but_changes_nothing_yet(client):
    """개인화 전이므로 누가 요청하든 같은 순서가 나온다.

    파라미터를 미리 받아 두면 개인화가 붙을 때 백엔드 연동을 다시 하지 않아도 된다.
    """
    without = client.get(HOME).json()
    with_member = client.get(f"{HOME}?member_id=42").json()

    assert [i["auction_id"] for i in without["items"]] == [
        i["auction_id"] for i in with_member["items"]
    ]


def test_all_three_apis_are_served_together():
    """백엔드 입장에서 AI 는 서비스 하나다. 포트를 세 개 열지 않는다."""
    import serve

    schema = serve.app.openapi()["paths"]
    assert "/internal/fraud/detect" in schema
    assert "/internal/moderation/review" in schema
    assert HOME in schema
