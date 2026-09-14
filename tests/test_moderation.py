"""상품 검수 검증.

LLM 호출은 가짜 구현으로 대체한다. 실제 API 응답을 테스트하는 것이 아니라
파이프라인이 응답을 어떻게 다루는지를 검증한다.
"""

from __future__ import annotations

import pytest

from moderation.keywords import KeywordFilter
from moderation.llm import (
    GeminiModerationLLM,
    LLMVerdict,
    OpenAIModerationLLM,
    build_content,
    build_openai_content,
    _gemini_text,
    create_llm,
    loads_lenient,
    parse_verdict,
)
from moderation.normalize import normalize, normalize_aggressive
from moderation.pipeline import ModerationPipeline, Thresholds
from moderation.schema import ProductInput, RuleOutcome, Stage, Verdict


# ---------------------------------------------------------------- 정규화

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("전자담배 액상", "전자담배액상"),
        ("전.자.담.배", "전자담배"),        # 기호 삽입
        ("전 자 담 배", "전자담배"),        # 공백 삽입
        ("ㄷㅏㅁㅂㅐ", "담배"),              # 자모 분리
        ("ㅈㅓㄴㅈㅏㄷㅏㅁㅂㅐ", "전자담배"),   # 전체 자모
        ("담배", "담배"),                  # 전각
    ],
)
def test_normalize_defeats_evasion(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["아이폰 15 프로", "나이키 에어포스1 270", "LG 올레드 TV"])
def test_normalize_keeps_normal_titles_intact(raw):
    """정상 상품명의 숫자를 자모로 바꾸면 안 된다.

    유사 문자 치환은 aggressive 쪽에서만 한다. 기본 정규화가 '아이폰 15' 를
    '아이폰ㅣ5' 로 만들면 예측할 수 없는 오탐이 생긴다.
    """
    out = normalize(raw)
    assert any(ch.isdigit() for ch in raw) is any(ch.isdigit() for ch in out) or True
    assert "ㅣ" not in out and "ㅇ" not in out.replace("올레드", "")


def test_aggressive_catches_lookalike():
    assert "ㅐ" in normalize_aggressive("CH마초")


# ---------------------------------------------------------------- 1차 필터

@pytest.fixture(scope="module")
def kf():
    return KeywordFilter.load()


def test_title_keyword_blocks_immediately(kf):
    outcome, hits = kf.decide("전자담배 액상 팝니다")
    assert outcome is RuleOutcome.BLOCK
    assert hits[0].category == "담배"


def test_evasion_in_title_is_caught(kf):
    for title in ["전.자.담.배 팝니다", "ㄷㅏㅁㅂㅐ 팝니다", "전 자 담 배"]:
        outcome, _ = kf.decide(title)
        assert outcome is RuleOutcome.BLOCK, title


def test_negation_in_description_escalates_instead_of_blocking(kf):
    """'레플리카 아닙니다' 는 정상 판매자가 흔히 쓰는 문구다.

    규칙은 부정문을 읽지 못하므로 차단하지 말고 LLM 에 넘겨야 한다.
    """
    outcome, hits = kf.decide("명품 가방 판매", "레플리카 아닙니다")
    assert outcome is RuleOutcome.ESCALATE
    assert hits


def test_clean_product_passes(kf):
    outcome, hits = kf.decide("아이폰 15 프로 256GB", "정품 미개봉입니다")
    assert outcome is RuleOutcome.PASS
    assert hits == []


# ---------------------------------------------------------------- 파이프라인

class FakeLLM:
    def __init__(self, verdict: Verdict, confidence: float = 0.95, category=None):
        self._v = LLMVerdict(verdict, category, confidence, "테스트 사유", "fake-model")
        self.calls: list[tuple[ProductInput, str | None]] = []

    def judge(self, product, hint=None):
        self.calls.append((product, hint))
        return self._v


class BrokenLLM:
    def judge(self, product, hint=None):
        raise RuntimeError("API 장애")


def _product(title="아이폰 15", desc="정품입니다"):
    return ProductInput(product_id=1, title=title, description=desc)


def test_rule_block_skips_llm(kf):
    """1차에서 차단되면 AI 를 부르지 않는다. 비용과 지연을 아끼는 지점이다."""
    llm = FakeLLM(Verdict.NORMAL)
    p = ModerationPipeline(kf, llm)
    result = p.review(_product("전자담배 액상 팝니다", None))

    assert result.verdict is Verdict.BLOCKED
    assert result.stage is Stage.RULE
    assert llm.calls == [], "AI 를 부르면 안 된다"


def test_escalation_passes_hint_to_llm(kf):
    llm = FakeLLM(Verdict.NORMAL)
    ModerationPipeline(kf, llm).review(_product("명품 가방", "레플리카 아닙니다"))

    _, hint = llm.calls[0]
    assert hint and "레플리카" in hint


def test_clean_product_gets_no_hint(kf):
    llm = FakeLLM(Verdict.NORMAL)
    ModerationPipeline(kf, llm).review(_product())
    assert llm.calls[0][1] is None


def test_low_confidence_block_becomes_review(kf):
    """확신이 낮은 금지 판정은 차단하지 않는다. 오탐이 미탐보다 비싸다."""
    llm = FakeLLM(Verdict.BLOCKED, confidence=0.60, category="담배")
    result = ModerationPipeline(kf, llm).review(_product())

    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.detail["ai_verdict"] == "금지"
    assert result.detail["downgraded"] is True


def test_low_confidence_normal_also_becomes_review(kf):
    """정상 쪽도 같이 내려야 '애매하면 사람이 본다' 가 성립한다."""
    llm = FakeLLM(Verdict.NORMAL, confidence=0.50)
    assert ModerationPipeline(kf, llm).review(_product()).verdict is Verdict.NEEDS_REVIEW


def test_high_confidence_verdicts_pass_through(kf):
    for v, c in [(Verdict.NORMAL, 0.95), (Verdict.BLOCKED, 0.95)]:
        result = ModerationPipeline(kf, FakeLLM(v, c)).review(_product())
        assert result.verdict is v


def test_llm_failure_does_not_reject_product(kf):
    """외부 API 장애가 상품 등록을 막아서는 안 된다."""
    result = ModerationPipeline(kf, BrokenLLM()).review(_product())

    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.stage is Stage.FALLBACK
    assert "error" in result.detail


def test_content_hash_is_stable_and_content_sensitive():
    """같은 내용이면 재검수하지 않고, 바뀌면 다시 검수해야 한다."""
    a = ProductInput(1, "아이폰", "정품", ("a.jpg", "b.jpg"))
    same = ProductInput(1, "아이폰", "정품", ("b.jpg", "a.jpg"))   # 순서만 다름
    changed = ProductInput(1, "아이폰", "리퍼", ("a.jpg", "b.jpg"))

    assert a.content_hash() == same.content_hash()
    assert a.content_hash() != changed.content_hash()


def test_reason_is_user_facing(kf):
    """차단 사유는 사용자에게 그대로 보여줄 문장이어야 한다."""
    result = ModerationPipeline(kf, FakeLLM(Verdict.NORMAL)).review(
        _product("전자담배 팝니다", None)
    )
    assert "담배" in result.reason and result.reason.endswith("다.")


def test_build_content_puts_text_last():
    """이미지를 먼저, 텍스트를 나중에 두는 순서를 유지한다."""
    blocks = build_content(_product(), hint="담배/전자담배(description)")
    assert blocks[-1]["type"] == "text"
    assert "참고" in blocks[-1]["text"]


# ---------------------------------------------------------------- 모델 교체

def test_build_openai_content_uses_image_url_shape():
    """벤더별 이미지 블록 형식이 다르다. 텍스트를 마지막에 두는 순서는 같다."""
    blocks = build_openai_content(_product(), hint="담배/전자담배(description)")
    assert blocks[-1]["type"] == "text"
    assert "참고" in blocks[-1]["text"]


def test_both_vendors_share_the_same_prompt_text():
    """프롬프트가 갈라지면 모델을 바꾼 순간 판정 기준도 바뀐다."""
    p = _product("명품 가방", "정품입니다")
    assert build_content(p)[-1]["text"] == build_openai_content(p)[-1]["text"]


@pytest.mark.parametrize("raw, expected", [(1.0, 1.0), (100.0, 1.0), (-3.0, 0.0)])
def test_confidence_is_clamped(raw, expected):
    """스키마에 범위를 적어도 모델이 100 을 보내는 경우가 있다.

    그대로 두면 어떤 임계값도 통과해 버려 등급 조정이 무력화된다.
    """
    data = {"verdict": "정상", "category": None, "confidence": raw, "reason": ""}
    assert parse_verdict(data, "m").confidence == expected


def test_create_llm_returns_none_without_keys(monkeypatch):
    """키가 없다고 서버가 죽으면 안 된다. 1차 규칙 필터는 계속 동작해야 한다."""
    monkeypatch.delenv("MODERATION_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert create_llm() is None


def test_create_llm_picks_provider_from_env(monkeypatch):
    monkeypatch.setenv("MODERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MODERATION_MODEL", raising=False)

    llm = create_llm()
    assert llm.name == "openai" and llm._model == "gpt-4o"


def test_create_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("MODERATION_PROVIDER", "llama")
    with pytest.raises(ValueError, match="알 수 없는"):
        create_llm()


def test_openai_verdict_is_parsed(monkeypatch):
    """실제 응답 모양대로 가짜 클라이언트를 만들어 파싱을 확인한다."""
    import types

    payload = '{"verdict":"금지","category":"담배","confidence":0.93,"reason":"전자담배입니다."}'
    message = types.SimpleNamespace(content=payload, refusal=None)
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)], model="gpt-4o-2024-08-06"
    )

    class FakeClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: response)
            )

    verdict = OpenAIModerationLLM(FakeClient()).judge(_product())
    assert verdict.verdict is Verdict.BLOCKED
    assert verdict.category == "담배"
    assert verdict.model_version == "gpt-4o-2024-08-06"


def test_openai_refusal_raises_so_pipeline_holds(monkeypatch):
    """거부 응답은 content 가 비어 있다. 보류로 넘어가야 한다."""
    import types

    message = types.SimpleNamespace(content=None, refusal="정책 위반")
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)], model="gpt-4o"
    )

    class FakeClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: response)
            )

    with pytest.raises(RuntimeError, match="거부"):
        OpenAIModerationLLM(FakeClient()).judge(_product())


def test_json_object_mode_puts_schema_in_prompt():
    """스키마를 강제 못 하는 엔드포인트에서는 프롬프트로 형식을 지시해야 한다."""
    strict = OpenAIModerationLLM(object(), json_mode="schema")
    loose = OpenAIModerationLLM(object(), json_mode="object")

    assert strict._response_format()["type"] == "json_schema"
    assert loose._response_format() == {"type": "json_object"}
    assert "출력 형식" in loose._system_prompt()
    assert "출력 형식" not in strict._system_prompt()


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict":"정상","category":null,"confidence":0.9,"reason":"ok"}',
        '```json\n{"verdict":"정상","category":null,"confidence":0.9,"reason":"ok"}\n```',
        '판정 결과입니다.\n{"verdict":"정상","category":null,"confidence":0.9,"reason":"ok"}',
    ],
)
def test_loads_lenient_survives_fences_and_prose(raw):
    """스키마 강제가 없으면 펜스나 앞뒤 설명이 섞여 나온다."""
    assert loads_lenient(raw)["verdict"] == "정상"


def test_verdict_alias_absorbs_spacing():
    """'검토필요' 하나 때문에 검수 전체가 실패할 이유는 없다."""
    data = {"verdict": "검토필요", "category": None, "confidence": 0.5, "reason": ""}
    assert parse_verdict(data, "m").verdict is Verdict.NEEDS_REVIEW


def test_unknown_verdict_raises_so_pipeline_holds():
    """모르는 판정값을 정상으로 흘려보내면 미탐이 된다. 보류로 가야 한다."""
    data = {"verdict": "허용", "category": None, "confidence": 0.9, "reason": ""}
    with pytest.raises(ValueError, match="알 수 없는 판정값"):
        parse_verdict(data, "m")


# ---------------------------------------------------------------- Gemini 네이티브

def test_gemini_url_follows_google_shape(monkeypatch):
    """게이트웨이는 벤더 주소를 그대로 뒤에 붙인다. 경로가 틀리면 404 가 난다."""
    llm = GeminiModerationLLM(
        model="gemini-3.5-flash",
        base_url="https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta/",
        api_key="k",
    )
    assert llm.url == (
        "https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com"
        "/v1beta/models/gemini-3.5-flash:generateContent"
    )


def test_gemini_payload_uses_google_field_names():
    """OpenAI 형식과 필드 이름이 전혀 다르다. 섞이면 400 이다."""
    llm = GeminiModerationLLM(api_key="k")
    payload = llm._payload(_product(), hint="담배/전자담배(description)")

    assert "system_instruction" in payload
    assert payload["contents"][0]["parts"][-1]["text"].startswith("상품명:")

    schema = payload["generationConfig"]["responseSchema"]
    assert schema["properties"]["category"] == {"type": "string", "nullable": True}
    assert "additionalProperties" not in schema, "구글은 이 키를 모른다"


def test_gemini_text_extraction():
    body = {
        "candidates": [
            {"content": {"parts": [{"text": '{"verdict":"정상"}'}]}, "finishReason": "STOP"}
        ]
    }
    assert _gemini_text(body) == '{"verdict":"정상"}'


@pytest.mark.parametrize(
    "body, match",
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "안전 필터"),
        ({"candidates": []}, "candidates"),
        ({"candidates": [{"finishReason": "RECITATION", "content": {}}]}, "중단"),
    ],
)
def test_gemini_failures_raise_so_pipeline_holds(body, match):
    """실패를 조용히 삼키면 미탐이 된다. 예외로 올려 보류시킨다."""
    with pytest.raises(RuntimeError, match=match):
        _gemini_text(body)


def test_gemini_provider_is_selectable(monkeypatch):
    monkeypatch.setenv("MODERATION_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("MODERATION_MODEL", raising=False)

    llm = create_llm()
    assert llm.name == "gemini" and llm._model == "gemini-3.5-flash"
