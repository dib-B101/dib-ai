"""검수 API 계약 검증.

백엔드가 이 응답을 그대로 저장한다. 필드가 사라지거나 의미가 바뀌면 여기서 깨져야 한다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moderation.keywords import KeywordFilter
from moderation.llm import LLMVerdict
from moderation.pipeline import ModerationPipeline
from moderation.schema import Verdict
from moderation_api.main import app, get_pipeline

REVIEW = "/internal/moderation/review"


class StubLLM:
    """API 계층만 시험한다. 실제 호출은 test_moderation.py 가 다룬다."""

    name = "stub"
    _model = "stub-1"

    def __init__(self, verdict=Verdict.NORMAL, confidence=0.95, category=None):
        self._v = LLMVerdict(verdict, category, confidence, "테스트 사유", "stub-1")

    def judge(self, product, hint=None):
        return self._v


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def use(llm) -> None:
    pipeline = ModerationPipeline(KeywordFilter.load(), llm)
    app.dependency_overrides[get_pipeline] = lambda: pipeline


def test_health_reports_dictionary_and_llm(client):
    body = client.get("/moderation/health").json()

    assert body["status"] == "ok"
    assert body["keyword_config_version"].startswith("keywords-")
    assert body["keyword_patterns"] > 0
    # 키가 없는 테스트 환경이므로 AI 는 꺼져 있다. 그래도 ok 여야 한다.
    assert body["llm"] is None


def test_blocked_product_maps_to_rejected(client):
    use(StubLLM())
    body = client.post(
        REVIEW, json={"product_id": 1, "title": "전자담배 액상 팝니다"}
    ).json()

    assert body["verdict"] == "금지"
    assert body["product_status"] == "REJECTED"
    assert body["stage"] == "rule", "1차에서 끝나야 한다 — AI 를 부르면 안 된다"
    assert body["category"] == "담배"
    assert body["reason"]


def test_normal_product_maps_to_registered(client):
    use(StubLLM(Verdict.NORMAL, 0.95))
    body = client.post(
        REVIEW,
        json={"product_id": 2, "title": "아이폰 15 프로", "description": "정품 미개봉"},
    ).json()

    assert body["verdict"] == "정상"
    assert body["product_status"] == "REGISTERED"
    assert body["stage"] == "ai"


def test_low_confidence_maps_to_pending(client):
    """확신이 낮으면 차단도 승인도 하지 않는다. 관리자 큐로 간다."""
    use(StubLLM(Verdict.BLOCKED, 0.60, "담배"))
    body = client.post(REVIEW, json={"product_id": 3, "title": "아이폰 15 프로"}).json()

    assert body["verdict"] == "검토 필요"
    assert body["product_status"] == "PENDING"
    assert body["detail"]["downgraded"] is True


def test_llm_failure_returns_200_not_500(client):
    """검수 장애가 상품 등록을 막아서는 안 된다."""

    class Broken:
        def judge(self, product, hint=None):
            raise RuntimeError("게이트웨이 장애")

    use(Broken())
    response = client.post(REVIEW, json={"product_id": 4, "title": "아이폰 15 프로"})

    assert response.status_code == 200
    body = response.json()
    assert body["product_status"] == "PENDING"
    assert body["stage"] == "fallback"
    assert "error" in body["detail"]


def test_same_content_gives_same_hash(client):
    """가격만 바뀐 수정에 AI 를 다시 부르지 않기 위한 캐시 키다."""
    use(StubLLM())
    payload = {
        "product_id": 5,
        "title": "아이폰 15 프로",
        "description": "정품",
        "image_urls": ["a.jpg", "b.jpg"],
    }
    first = client.post(REVIEW, json=payload).json()["content_hash"]

    payload["image_urls"] = ["b.jpg", "a.jpg"]  # 순서만 다름
    assert client.post(REVIEW, json=payload).json()["content_hash"] == first

    payload["description"] = "리퍼"
    assert client.post(REVIEW, json=payload).json()["content_hash"] != first


def test_detail_carries_dictionary_version(client):
    """과거 판정을 해석하려면 어떤 사전이었는지 알아야 한다."""
    use(StubLLM())
    body = client.post(REVIEW, json={"product_id": 6, "title": "아이폰 15"}).json()

    assert body["detail"]["keyword_config_version"].startswith("keywords-")


def test_title_is_required(client):
    assert client.post(REVIEW, json={"product_id": 7}).status_code == 422


def test_both_apis_are_served_together():
    """백엔드 입장에서 AI 는 서비스 하나다. 포트를 두 개 열지 않는다.

    라우트 목록이 아니라 실제 응답으로 확인한다. FastAPI 버전에 따라 포함된
    라우터가 중첩 객체로 남아 목록에서는 보이지 않는다.
    """
    import serve

    with TestClient(serve.app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/moderation/health").status_code == 200
        # 404 가 아니면 라우팅은 붙어 있다 (본문 검증은 각 API 테스트가 한다)
        assert c.post("/internal/fraud/detect", json={}).status_code != 404
        assert c.post(REVIEW, json={}).status_code != 404

    schema = serve.app.openapi()["paths"]
    assert "/internal/fraud/detect" in schema
    assert REVIEW in schema, "/docs 에 두 API 가 함께 나와야 한다"
