"""REST API 명세서 92~95번 계약 검증.

백엔드가 이 네 개를 붙일 때 깨지면 안 되는 것만 본다.

    92  POST /internal/v1/ai/bid-anomalies      SERVICE_HMAC · 202
    93  콜백 payload (AI → BE)                  camelCase · features 11개 키
    94  POST /internal/v1/ai/recommendations    SERVICE_HMAC · 202
    95  콜백 payload (AI → BE)                  items[].reason 포함

**콜백은 실제로 HTTP 로 나간다.** `callback.send` 를 가짜로 바꿔치기하면 서명이나
직렬화가 틀려도 통과해 버리므로, httpx 전송 계층만 가로채고 나머지는 그대로 돌린다.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from internal_api import hmac_auth
from internal_api.main import _seen_jobs
from serve import app

SECRET = "test-service-secret"
AI_SECRET = "test-ai-secret"
CALLBACK_URL = "https://backend.test/api/v1/internal/callbacks/ai/bid-anomalies"

BID_ANOMALIES = "/internal/v1/ai/bid-anomalies"
RECOMMENDATIONS = "/internal/v1/ai/recommendations"


@pytest.fixture(autouse=True)
def secrets(monkeypatch):
    monkeypatch.setenv(hmac_auth.SERVICE_SECRET_ENV, SECRET)
    monkeypatch.setenv(hmac_auth.AI_SECRET_ENV, AI_SECRET)
    monkeypatch.delenv("DIB_CALLBACK_ALLOWED_HOSTS", raising=False)
    _seen_jobs.clear()


@pytest.fixture
def sent(monkeypatch):
    """나간 콜백을 모은다. httpx 전송 계층만 가로채 서명·직렬화는 실제로 돌린다."""
    captured: list[dict] = []

    def fake_post(url, *, content, headers, timeout):
        captured.append({"url": url, "body": content, "headers": headers})
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def post_signed(client, path: str, payload: dict, *, secret: str = SECRET):
    body = json.dumps(payload).encode("utf-8")
    return client.post(path, content=body, headers=hmac_auth.sign(secret, body))


def bid_request(**over) -> dict:
    return {
        "jobId": "job-1",
        "auctionId": 10002,  # demo 의 핑퐁 경매
        "memberId": 21,
        "bidId": 1000201,
        "bids": [],
        "callbackUrl": CALLBACK_URL,
        **over,
    }


def reco_request(**over) -> dict:
    return {
        "jobId": "job-r1",
        "memberId": 77,
        "scope": "ALL",
        "behaviorWindow": {"days": 30},
        "candidateAuctionIds": [],
        "callbackUrl": "https://backend.test/api/v1/internal/callbacks/ai/recommendations",
        **over,
    }


# --- 인증 -------------------------------------------------------------------


def test_unsigned_request_is_rejected(client):
    """서명 없이 부르면 401. 내부 API 라도 열어 두지 않는다."""
    assert client.post(BID_ANOMALIES, json=bid_request()).status_code == 401


def test_wrong_secret_is_rejected(client):
    assert post_signed(client, BID_ANOMALIES, bid_request(), secret="틀린키").status_code == 401


def test_tampered_body_is_rejected(client):
    """서명은 본문 원문에 대한 것이다. 한 글자만 바뀌어도 걸려야 한다."""
    body = json.dumps(bid_request()).encode("utf-8")
    headers = hmac_auth.sign(SECRET, body)

    assert client.post(BID_ANOMALIES, content=body + b" ", headers=headers).status_code == 401


def test_stale_timestamp_is_rejected(client, monkeypatch):
    """가로챈 요청을 나중에 그대로 다시 보내는 것을 막는다."""
    import time

    body = json.dumps(bid_request()).encode("utf-8")
    headers = hmac_auth.sign(SECRET, body, timestamp=int(time.time()) - 10_000)

    assert client.post(BID_ANOMALIES, content=body, headers=headers).status_code == 401


def test_missing_secret_returns_503_not_open_access(client, monkeypatch):
    """시크릿을 안 넣으면 **막힌다.** 설정 누락이 무인증 배포가 되면 안 된다."""
    monkeypatch.delenv(hmac_auth.SERVICE_SECRET_ENV, raising=False)

    assert post_signed(client, BID_ANOMALIES, bid_request()).status_code == 503


# --- 92 접수 ----------------------------------------------------------------


def test_accepts_and_returns_job_id(client, sent):
    r = post_signed(client, BID_ANOMALIES, bid_request())

    assert r.status_code == 202
    assert r.json() == {"jobId": "job-1", "status": "ACCEPTED"}


def test_malformed_payload_is_invalid_payload(client):
    r = post_signed(client, BID_ANOMALIES, {"jobId": "job-2"})

    assert r.status_code == 400
    assert "INVALID_PAYLOAD" in r.json()["detail"]


def test_unknown_auction_is_model_unavailable(client):
    """202 를 준 뒤에는 실패를 알릴 방법이 없다. 미리 알 수 있는 실패는 미리 막는다."""
    r = post_signed(client, BID_ANOMALIES, bid_request(auctionId=999_999))

    assert r.status_code == 503
    assert "MODEL_UNAVAILABLE" in r.json()["detail"]


def test_callback_url_must_be_http(client):
    r = post_signed(client, BID_ANOMALIES, bid_request(callbackUrl="ftp://x/y"))

    assert r.status_code == 400


def test_callback_host_allowlist_is_enforced(client, monkeypatch):
    """백엔드가 보낸 값이라도 아무 데로나 보내지 않는다."""
    monkeypatch.setenv("DIB_CALLBACK_ALLOWED_HOSTS", "backend.test")

    assert post_signed(client, BID_ANOMALIES, bid_request()).status_code == 202
    assert post_signed(
        client, BID_ANOMALIES, bid_request(jobId="job-x", callbackUrl="https://evil.test/cb")
    ).status_code == 400


def test_same_job_id_is_not_analyzed_twice(client, sent):
    """백엔드가 재시도해도 두 번 분석하지 않는다."""
    assert post_signed(client, BID_ANOMALIES, bid_request()).status_code == 202
    assert post_signed(client, BID_ANOMALIES, bid_request()).status_code == 202

    assert len(sent) == 1


# --- 93 콜백 ----------------------------------------------------------------


def test_callback_payload_matches_spec(client, sent):
    post_signed(client, BID_ANOMALIES, bid_request())

    assert len(sent) == 1
    body = json.loads(sent[0]["body"])

    assert sent[0]["url"] == CALLBACK_URL
    assert set(body) == {
        "jobId", "auctionId", "memberId", "bidId", "features",
        "predictedLabel", "decisionThreshold", "modelVersion", "featureVersion",
    }
    assert body["jobId"] == "job-1"
    assert body["bidId"] == 1000201, "촉발한 입찰 ID 를 그대로 돌려줘야 한다"
    assert body["predictedLabel"] in (0, 1)
    assert body["featureVersion"]


def test_callback_features_have_all_eleven_keys(client, sent):
    """명세가 정한 11개 키를 전부 채운다. 백엔드가 컬럼을 그대로 매핑한다."""
    post_signed(client, BID_ANOMALIES, bid_request())
    features = json.loads(sent[0]["body"])["features"]

    assert set(features) == {
        "bidderTendency", "biddingRatio", "lastBidding", "auctionBids",
        "startingPriceAverage", "earlyBidding", "winningRatio", "auctionDuration",
        "ruleScore", "mlScore", "riskScore",
    }


def test_auction_duration_is_null_not_zero(client, sent):
    """우리가 쓰지 않는 피처다. **0 은 '가장 짧은 경매' 라는 뜻이 되어 거짓말이다.**"""
    post_signed(client, BID_ANOMALIES, bid_request())

    assert json.loads(sent[0]["body"])["features"]["auctionDuration"] is None


def test_features_are_sent_even_when_model_weight_is_zero(client, sent):
    """`FRAUD_W_ML=0` 이 기본값이다. **그래도 피처는 채워 보낸다.**

    백엔드가 이 값을 fraud_detection 에 저장하고 관리자 화면에서 근거로 본다.
    점수에 안 쓴다는 것과 값을 안 남긴다는 것은 다르다.
    """
    post_signed(client, BID_ANOMALIES, bid_request())
    features = json.loads(sent[0]["body"])["features"]

    computed = [k for k, v in features.items() if v is not None]
    assert "bidderTendency" in computed and "winningRatio" in computed
    assert features["ruleScore"] is not None


def test_model_version_is_absent_when_model_did_not_contribute(client, sent):
    """`mlScore` 가 없는데 버전만 남으면 "이 모델이 낸 점수" 로 오독된다."""
    post_signed(client, BID_ANOMALIES, bid_request())
    body = json.loads(sent[0]["body"])

    if body["features"]["mlScore"] is None:
        assert body["modelVersion"] is None


def four_bids() -> list[dict]:
    """게이트(최소 4건 · 3명)를 통과하는 최소 입찰 묶음.

    21 이 4건 중 1건만 부르므로 조회기 데이터(10건 중 5건)와 확실히 구분된다.
    """
    return [
        {"bidId": 1, "memberId": 21, "amount": 11000, "createdAt": "2026-09-08T14:00:20Z"},
        {"bidId": 2, "memberId": 22, "amount": 12000, "createdAt": "2026-09-08T14:00:40Z"},
        {"bidId": 3, "memberId": 23, "amount": 13000, "createdAt": "2026-09-08T14:01:10Z"},
        {"bidId": 4, "memberId": 22, "amount": 14000, "createdAt": "2026-09-08T14:01:50Z"},
    ]


def test_payload_bids_replace_the_provider_copy(client, sent):
    """백엔드가 보낸 입찰 목록이 정답이다.

    조회기의 핑퐁 경매는 21 이 10건 중 5건을 부른다(비율 0.5). 본문으로 4건 중
    1건만 보내면 **0.25 가 나와야 한다** — 안 바뀌면 `bids[]` 를 받아만 두고
    버리고 있는 것이다.
    """
    post_signed(client, BID_ANOMALIES, bid_request(jobId="job-a"))
    original = json.loads(sent[0]["body"])["features"]["biddingRatio"]

    post_signed(client, BID_ANOMALIES, bid_request(jobId="job-b", bids=four_bids()))
    replaced = json.loads(sent[1]["body"])["features"]["biddingRatio"]

    assert original == pytest.approx(0.5), "조회기 데이터: 21 이 10건 중 5건"
    assert replaced == pytest.approx(0.25), "본문 데이터: 21 이 4건 중 1건"


def test_timezone_aware_payload_does_not_crash(client, sent):
    """본문은 tz 를 달고 오고 조회기 데이터는 naive 다. 섞이면 TypeError 가 난다.

    그 예외는 콜백이 안 나가는 형태로만 드러나 원인을 찾기 어렵다.
    """
    aware = [{**b, "createdAt": b["createdAt"].replace("Z", "+09:00")} for b in four_bids()]
    post_signed(client, BID_ANOMALIES, bid_request(jobId="job-tz", bids=aware))

    assert len(sent) == 1, "예외가 났으면 콜백이 안 나간다"


def test_bids_below_the_gate_send_no_callback(client, sent):
    """입찰이 최소 조건(4건 · 3명)에 못 미치면 판정하지 않는다.

    통계적으로 의미가 없는 경매를 억지로 점수 매기면 오탐만 늘어난다.
    """
    post_signed(client, BID_ANOMALIES, bid_request(jobId="job-thin", bids=four_bids()[:2]))

    assert sent == []


def test_callback_is_signed_with_ai_secret(client, sent):
    """백엔드가 AI_HMAC 으로 검증한다. 우리가 보낸 원문 그대로 맞아야 한다."""
    post_signed(client, BID_ANOMALIES, bid_request())
    call = sent[0]

    hmac_auth.verify(AI_SECRET, call["body"], call["headers"])


def test_callback_signature_covers_the_exact_bytes_sent(client, sent):
    """서명한 문자열과 보낸 문자열이 다르면 백엔드에서 검증이 깨진다."""
    post_signed(client, BID_ANOMALIES, bid_request())
    call = sent[0]

    with pytest.raises(hmac_auth.HmacError):
        hmac_auth.verify(AI_SECRET, call["body"] + b"x", call["headers"])


def test_skipped_bidder_sends_no_callback(client, sent):
    """판정 대상이 아닌 입찰자는 **0 점으로 보내지 않는다.**

    "판단하지 않음" 을 "위험하지 않음" 으로 저장하면 신규 사용자일수록 안전해 보이는
    왜곡이 생긴다. 보낼 값이 없으면 안 보낸다.
    """
    post_signed(client, BID_ANOMALIES, bid_request(memberId=999_999))

    assert sent == []


# --- 94 · 95 추천 -----------------------------------------------------------


def test_recommendation_is_accepted(client, sent):
    r = post_signed(client, RECOMMENDATIONS, reco_request())

    assert r.status_code == 202
    assert r.json() == {"jobId": "job-r1", "status": "ACCEPTED"}


def test_recommendation_callback_matches_spec(client, sent):
    post_signed(client, RECOMMENDATIONS, reco_request())
    body = json.loads(sent[0]["body"])

    assert set(body) == {"jobId", "memberId", "items"}
    assert body["memberId"] == 77
    assert body["items"], "데모 후보가 있으므로 비면 안 된다"
    assert set(body["items"][0]) == {"auctionId", "score", "reason"}
    assert body["items"][0]["reason"], "명세가 reason 을 요구한다"


def test_candidate_ids_restrict_the_result(client, sent):
    """백엔드가 이미 노출 정책으로 걸러 낸 목록이다. 우리가 넓히지 않는다."""
    post_signed(client, RECOMMENDATIONS, reco_request(candidateAuctionIds=[20003]))
    body = json.loads(sent[0]["body"])

    assert [i["auctionId"] for i in body["items"]] == [20003]


def test_recommendation_scope_filters_before_ranking(client, sent):
    """일반 홈 추천에 라이브 편성 경매가 섞이면 안 된다."""
    post_signed(
        client,
        RECOMMENDATIONS,
        reco_request(jobId="job-general", scope="GENERAL"),
    )
    general_ids = [i["auctionId"] for i in json.loads(sent[0]["body"])["items"]]

    post_signed(
        client,
        RECOMMENDATIONS,
        reco_request(jobId="job-live", scope="LIVE"),
    )
    live_ids = [i["auctionId"] for i in json.loads(sent[1]["body"])["items"]]

    assert set(general_ids) <= {20001, 20004}
    assert set(live_ids) <= {20002, 20003}
    assert general_ids and live_ids


def test_invalid_recommendation_scope_is_rejected(client):
    response = post_signed(
        client,
        RECOMMENDATIONS,
        reco_request(jobId="job-bad-scope", scope="UNKNOWN"),
    )

    assert response.status_code == 400
    assert "INVALID_PAYLOAD" in response.json()["detail"]


def test_ended_auction_is_excluded_even_if_requested(client, sent):
    """20005 는 이미 끝났다. 지정해서 요청해도 나가면 안 된다."""
    post_signed(client, RECOMMENDATIONS, reco_request(candidateAuctionIds=[20005]))
    body = json.loads(sent[0]["body"])

    assert body["items"] == []


def test_behavior_window_is_accepted_but_unused(client, sent):
    """조회 구간은 서버 설정이 정하므로 요청값에 따라 순서가 달라지지 않는다."""
    post_signed(client, RECOMMENDATIONS, reco_request(jobId="job-r2", behaviorWindow=None))
    post_signed(client, RECOMMENDATIONS, reco_request(jobId="job-r3", behaviorWindow={"days": 7}))

    def order(body):
        return [i["auctionId"] for i in json.loads(body)["items"]]

    assert order(sent[0]["body"]) == order(sent[1]["body"])


def test_backend_utc_timestamps_do_not_shift_the_features(client, sent):
    """백엔드는 DB 의 naive TIMESTAMP 를 UTC 문자열로 바꿔 보낸다.

    조회기의 경매 시각은 naive 다. 변환 없이 tzinfo 만 떼면 입찰이 경매 시작보다
    몇 시간 앞으로 계산되고, **모든 입찰자의 Early_Bidding·Last_Bidding 이
    1.0(최대 위험)이 된다.** 예외도 로그도 없어 눈으로는 안 잡힌다.

    데모 경매는 14:00:00 에 시작한다(naive). 같은 순간을 UTC 로 표현해 보낸다.
    """
    import os
    from datetime import datetime, timedelta

    os.environ["DIB_DB_TIMEZONE"] = "Asia/Seoul"
    try:
        start_kst = datetime(2026, 9, 8, 14, 0, 0)   # demo 경매의 시작 시각(naive)
        plan = ((20, 21), (40, 22), (70, 23), (110, 21))   # 게이트: 4건 · 3명
        bids = [
            {
                "bidId": i,
                "memberId": member,
                "amount": 10_000 + i * 1_000,
                # KST 14:00:20 을 백엔드는 UTC 05:00:20Z 로 보낸다
                "createdAt": (start_kst + timedelta(seconds=off) - timedelta(hours=9))
                .isoformat() + "Z",
            }
            for i, (off, member) in enumerate(plan, start=1)
        ]
        post_signed(client, BID_ANOMALIES, bid_request(jobId="job-utc", bids=bids))
    finally:
        os.environ.pop("DIB_DB_TIMEZONE", None)

    features = json.loads(sent[0]["body"])["features"]

    assert features["earlyBidding"] < 1.0, "9시간 밀리면 1.0 이 된다"
    assert features["lastBidding"] < 1.0


def test_failed_callback_lets_the_backend_retry(client, monkeypatch):
    """콜백을 못 보냈으면 접수 기록을 지운다.

    **안 그러면 백엔드의 재시도가 무력해진다.** 백엔드는 같은 jobId
    (`bid-anomaly-{auctionId}-{memberId}`) 로 다시 보내는데, 우리가 "이미 접수함"
    으로 202 만 주고 아무것도 안 하면 결과가 영영 가지 않는다.
    """
    attempts: list[bytes] = []
    alive = {"ok": False}

    def flaky(url, *, content, headers, timeout):
        attempts.append(content)
        code = 200 if alive["ok"] else 500
        return httpx.Response(code, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", flaky)
    monkeypatch.setattr("internal_api.callback.MAX_ATTEMPTS", 1)

    post_signed(client, BID_ANOMALIES, bid_request())
    assert len(attempts) == 1, "한 번 시도하고 실패했다"

    alive["ok"] = True
    post_signed(client, BID_ANOMALIES, bid_request())

    assert len(attempts) == 2, "재시도가 실제로 분석을 다시 돌려야 한다"


def test_successful_callback_still_blocks_a_retry(client, sent):
    """성공한 뒤의 재시도는 막는다. 같은 결과를 두 번 보낼 이유가 없다."""
    post_signed(client, BID_ANOMALIES, bid_request())
    post_signed(client, BID_ANOMALIES, bid_request())

    assert len(sent) == 1


def test_gated_bidder_is_not_retried(client, sent):
    """보낼 값이 없는 것은 실패가 아니다. 재시도해도 결과는 같다."""
    post_signed(client, BID_ANOMALIES, bid_request(memberId=999_999))
    post_signed(client, BID_ANOMALIES, bid_request(memberId=999_999))

    assert sent == []


class _EventProvider:
    name = "in-memory(events)"

    def __init__(self, inner, events):
        self._inner, self._events = inner, events

    def load_active(self, now, limit):
        return self._inner.load_active(now, limit)

    def load_vectors(self, auction_ids):
        return self._inner.load_vectors(auction_ids)

    def load_events(self, member_id, since, limit):
        return self._events.get(member_id, ())


@pytest.fixture
def with_events(monkeypatch):
    from datetime import datetime, timezone
    from reco import BehaviorEvent
    from reco_api import main as reco_app

    inner = reco_app._state["provider"]
    events = {7: (BehaviorEvent("BID", 20001, datetime.now(timezone.utc)),)}
    monkeypatch.setitem(reco_app._state, "provider", _EventProvider(inner, events))


def test_async_path_uses_personalization(client, sent, with_events):
    post_signed(client, RECOMMENDATIONS, reco_request(jobId="job-p1", memberId=7))
    ids = [i["auctionId"] for i in json.loads(sent[0]["body"])["items"]]
    assert ids[0] == 20003
    assert 20001 not in ids


def test_member_without_events_still_gets_recommendations(client, sent, with_events):
    post_signed(client, RECOMMENDATIONS, reco_request(jobId="job-p2", memberId=999))
    assert json.loads(sent[0]["body"])["items"]
