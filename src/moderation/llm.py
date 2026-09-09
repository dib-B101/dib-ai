"""2차 AI 검수 — 멀티모달 LLM 호출.

**왜 LLM 인가**

분류 모델을 파인튜닝하지 않는 이유는 학습 데이터가 0건이기 때문이다. 금지 품목은
카테고리가 넓고(담배·주류·의약품·레플리카·동물·개인정보), 무엇보다 우회 표현은
문맥 이해가 필요하다.

    "행복한 연기 팝니다"

금칙어가 하나도 없다. 단어 목록으로는 절대 못 잡고, 임베딩 유사도로도 안 잡힌다.
문장의 뜻을 이해해야 한다.

그리고 LLM 만이 **왜 막았는지 한국어로 설명**한다. 그 문장이 그대로 사용자 안내
문구가 되고, 관리자 검토 근거가 되고, 이의제기 대응 자료가 된다. 분류 모델은
확률값만 뱉는다.

**왜 특정 벤더에 묶지 않는가**

파이프라인은 ``ModerationLLM`` 프로토콜만 알고 어느 모델인지는 모른다. 프롬프트와
응답 스키마가 공용이므로 Claude 와 GPT 를 환경변수로 갈아끼울 수 있다. 한쪽 API 가
죽거나 요금 정책이 바뀌어도 파이프라인 코드는 그대로 둔다.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import ProductInput, Verdict

log = logging.getLogger("moderation.llm")

ANTHROPIC_MODEL = "claude-opus-5"
OPENAI_MODEL = "gpt-4o"

# 이미지 토큰은 대략 (가로 × 세로) / 750 이다.
# 1024px 6장이면 약 4,200 토큰이지만 512px 로 줄이면 약 1,050 토큰이다.
# 담배갑·술병·약통 식별은 512px 로 충분하므로 입력이 절반 이하로 줄어든다.
MAX_IMAGE_EDGE = 512
MAX_IMAGES = 6

SYSTEM_PROMPT = """\
당신은 C2C 중고 경매 플랫폼의 상품 검수 담당자입니다.
등록하려는 상품이 **거래 제한 품목**인지 판정합니다.

## 거래 제한 품목

담배·전자담배·니코틴 액상, 주류, 의약품(처방약·다이어트약·스테로이드),
마약류, 총포·도검·전기충격기 등 무기류, 신분증·계정 등 개인정보,
위조품(레플리카·짝퉁), 살아있는 동물, 장기·대리시험 등 법령 위반 거래.

## 판정 기준

- **금지** — 거래 제한 품목이 명확하다
- **검토 필요** — 의심되지만 확신할 수 없다. 정보가 부족한 경우도 포함한다
- **정상** — 제한 품목이 아니다

## 중요한 원칙

우회 표현을 문맥으로 읽으십시오. "행복한 연기", "어른들의 음료" 처럼 금칙어 없이
제한 품목을 가리키는 표현이 있습니다.

동시에 **오탐을 최소화**하십시오. 정상 상품을 막는 비용이 더 큽니다.
- 부정문에 주의하십시오. "레플리카 아닙니다", "사시미칼 아니고 캠핑용" 은 정상입니다
- 제한 품목을 **연상시킬 뿐** 실제로는 다른 물건인 경우가 많습니다.
  재떨이·라이터·와인잔·술병 장식품은 그 자체로 제한 품목이 아닙니다
- 확신이 서지 않으면 금지가 아니라 **검토 필요**로 두십시오

reason 은 사용자에게 그대로 보여줄 문장입니다. 한 문장으로, 무엇이 왜 문제인지
구체적으로 쓰십시오. 정상이면 간단히 근거만 적으십시오.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["정상", "검토 필요", "금지"]},
        "category": {
            "type": ["string", "null"],
            "description": "제한 품목 분류. 정상이면 null",
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "category", "confidence", "reason"],
    "additionalProperties": False,
}

# 스키마를 강제할 수 없는 엔드포인트에서 형식을 지시하는 문구.
SCHEMA_INSTRUCTION = """
## 출력 형식

다른 말 없이 아래 JSON 객체만 출력하십시오.

{"verdict": "정상" | "검토 필요" | "금지",
 "category": 제한 품목 분류 문자열 또는 null,
 "confidence": 0 이상 1 이하의 숫자,
 "reason": "사용자에게 보여줄 한 문장"}
"""


@dataclass(frozen=True, slots=True)
class LLMVerdict:
    verdict: Verdict
    category: str | None
    confidence: float
    reason: str
    model_version: str


class ModerationLLM(Protocol):
    """호출 인터페이스. 테스트에서는 가짜 구현으로 갈아끼운다."""

    def judge(self, product: ProductInput, hint: str | None = None) -> LLMVerdict: ...


# ------------------------------------------------------------------ 입력 구성

IMAGE_FETCH_TIMEOUT = 5.0


def _read_bytes(path: str | Path) -> bytes | None:
    """로컬 경로든 URL 이든 원본 바이트를 가져온다.

    백엔드는 S3 URL 을 보내고 테스트는 로컬 파일을 쓴다. 둘 다 받는다.
    """
    text = str(path)
    if text.startswith(("http://", "https://")):
        import urllib.request

        try:
            with urllib.request.urlopen(text, timeout=IMAGE_FETCH_TIMEOUT) as resp:
                return resp.read()
        except Exception:
            log.warning("이미지 다운로드 실패: %s", text)
            return None

    p = Path(text)
    return p.read_bytes() if p.exists() else None


def encode_image(path: str | Path) -> str | None:
    """이미지를 512px JPEG base64 로 만든다. 실패하면 None 이다.

    이미지 한 장이 깨져도 예외를 던지지 않고 건너뛴다. 나머지 이미지와 텍스트로
    판정한다 — 사진 하나 때문에 상품 등록이 막히는 편이 더 나쁘다.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        log.warning("Pillow 가 없어 이미지를 건너뜁니다")
        return None

    raw = _read_bytes(path)
    if raw is None:
        return None

    try:
        import io

        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
    except Exception:
        log.warning("이미지 인코딩 실패: %s", path)
        return None

    return base64.b64encode(data).decode()


def build_text(product: ProductInput, hint: str | None = None) -> str:
    """상품 정보를 한 덩어리 텍스트로 만든다. 두 벤더가 공유한다."""
    lines = [
        f"상품명: {product.title}",
        f"설명: {product.description or '(없음)'}",
    ]
    if product.category_name:
        lines.append(f"카테고리: {product.category_name}")
    if hint:
        lines.append(
            f"\n참고: 1차 규칙 필터에서 다음 신호가 감지되었습니다 — {hint}\n"
            "다만 부정문이거나 무관한 문맥일 수 있으니 직접 판단하십시오."
        )
    return "\n".join(lines)


def _encoded_images(product: ProductInput) -> list[str]:
    out: list[str] = []
    for url in product.image_urls[:MAX_IMAGES]:
        encoded = encode_image(url)
        if encoded:
            out.append(encoded)
    return out


def build_content(product: ProductInput, hint: str | None = None) -> list[dict]:
    """Anthropic 형식 content 블록. 이미지를 먼저, 텍스트를 나중에 둔다."""
    blocks: list[dict] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        }
        for data in _encoded_images(product)
    ]
    blocks.append({"type": "text", "text": build_text(product, hint)})
    return blocks


def build_openai_content(product: ProductInput, hint: str | None = None) -> list[dict]:
    """OpenAI 형식 content 블록.

    ``detail: "low"`` 는 이미지 한 장을 크기와 무관하게 85 토큰으로 고정한다.
    담배갑·술병 식별에는 충분하고, 6장을 다 넣어도 약 510 토큰이다.
    """
    blocks: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": "low"},
        }
        for data in _encoded_images(product)
    ]
    blocks.append({"type": "text", "text": build_text(product, hint)})
    return blocks


_VERDICT_ALIASES = {
    "정상": Verdict.NORMAL,
    "검토 필요": Verdict.NEEDS_REVIEW,
    "검토필요": Verdict.NEEDS_REVIEW,
    "금지": Verdict.BLOCKED,
}


def loads_lenient(text: str) -> dict:
    """모델 응답에서 JSON 객체를 꺼낸다.

    스키마를 강제하지 못하는 엔드포인트에서는 ```json 펜스나 앞뒤 설명이 섞여 나온다.
    가장 바깥 중괄호 쌍만 잘라서 파싱한다.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse_verdict(data: dict, model_version: str) -> LLMVerdict:
    """벤더 응답 JSON 을 공용 값 객체로 옮긴다.

    두 가지를 방어한다. **confidence 를 0~1 로 자른다** — 스키마에 범위를 적어도
    모델이 100 을 보내는 경우가 있고, 그대로 두면 임계값 판정이 무력화된다.
    그리고 **판정 문자열의 사소한 표기 차이를 흡수한다** — "검토필요" 로 붙여 쓴
    응답 하나 때문에 검수 전체가 실패할 이유는 없다.
    """
    raw = str(data.get("verdict", "")).strip()
    verdict = _VERDICT_ALIASES.get(raw)
    if verdict is None:
        raise ValueError(f"알 수 없는 판정값: {raw!r}")

    confidence = float(data.get("confidence", 0.0))
    return LLMVerdict(
        verdict=verdict,
        category=data.get("category"),
        confidence=min(max(confidence, 0.0), 1.0),
        reason=data.get("reason", ""),
        model_version=model_version,
    )


# ------------------------------------------------------------------ 구현체

class AnthropicModerationLLM:
    """Claude 호출.

    금지 품목 정책이 담긴 시스템 프롬프트는 매 요청 동일하므로 캐싱한다.
    상품 등록마다 같은 텍스트를 재청구할 이유가 없다.
    """

    name = "anthropic"

    def __init__(self, client=None, model: str = ANTHROPIC_MODEL) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def judge(self, product: ProductInput, hint: str | None = None) -> LLMVerdict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                "effort": "low",
            },
            messages=[{"role": "user", "content": build_content(product, hint)}],
        )
        text = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )
        return parse_verdict(loads_lenient(text), getattr(response, "model", self._model))


class OpenAIModerationLLM:
    """OpenAI 형식 Chat Completions 호출.

    OpenAI 본사뿐 아니라 **OpenAI 호환 게이트웨이**도 여기로 붙는다. ``base_url`` 만
    바꾸면 같은 엔드포인트로 GPT·Gemini·Claude 를 모두 부를 수 있다. 사내 게이트웨이는
    보통 이 형태라 벤더별 클라이언트를 따로 만들 필요가 없다.

        OPENAI_BASE_URL=https://.../v1
        MODERATION_MODEL=gemini-3.5-flash

    구조화 출력은 두 방식을 지원한다. 게이트웨이가 OpenAI 를 그대로 중계하지 않는
    경우가 많아서다 — Gemini·Claude 를 OpenAI 형식으로 감싼 엔드포인트는 ``json_schema``
    를 거절하고 ``json_object`` 만 받는 일이 흔하다.

        schema   json_schema + strict. 스키마를 벗어난 응답을 아예 생성하지 못한다
        object   json_object. 스키마를 프롬프트로 지시하고 파싱은 우리가 검증한다

    어느 쪽이 되는지는 ``scripts/check_moderation_llm.py`` 로 한 번 확인하면 된다.
    """

    name = "openai"

    def __init__(
        self,
        client=None,
        model: str = OPENAI_MODEL,
        json_mode: str = "schema",
        base_url: str | None = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self._client = client
        self._model = model
        self._json_mode = json_mode

    def _response_format(self) -> dict:
        if self._json_mode == "object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "moderation_verdict",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        }

    def _system_prompt(self) -> str:
        # json_object 모드에는 스키마 강제가 없으므로 프롬프트로 형식을 지시한다.
        if self._json_mode == "object":
            return SYSTEM_PROMPT + SCHEMA_INSTRUCTION
        return SYSTEM_PROMPT

    def judge(self, product: ProductInput, hint: str | None = None) -> LLMVerdict:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            response_format=self._response_format(),
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": build_openai_content(product, hint)},
            ],
        )
        message = response.choices[0].message
        # 안전 정책상 거부하면 content 가 비어 있다. 파이프라인이 보류로 처리하도록
        # 예외로 올린다 — 빈 문자열을 파싱하려다 엉뚱한 곳에서 터지는 편보다 낫다.
        if getattr(message, "refusal", None):
            raise RuntimeError(f"모델이 응답을 거부했습니다: {message.refusal}")
        return parse_verdict(
            loads_lenient(message.content), getattr(response, "model", self._model)
        )


# ------------------------------------------------------------------ 선택

_PROVIDERS: dict[str, tuple[type, str, str]] = {
    "anthropic": (AnthropicModerationLLM, "ANTHROPIC_API_KEY", ANTHROPIC_MODEL),
    "openai": (OpenAIModerationLLM, "OPENAI_API_KEY", OPENAI_MODEL),
}


def create_llm(
    provider: str | None = None, model: str | None = None
) -> ModerationLLM | None:
    """환경변수로 검수 모델을 고른다.

        MODERATION_PROVIDER=openai      생략하면 키가 있는 쪽을 자동 선택
        MODERATION_MODEL=gemini-3.5-flash   생략하면 벤더 기본값
        MODERATION_JSON_MODE=schema     schema | object (openai 계열만)
        OPENAI_BASE_URL=https://.../v1  사내 게이트웨이를 쓸 때

    OpenAI 호환 게이트웨이는 ``openai`` 제공자에 ``OPENAI_BASE_URL`` 만 지정하면
    된다. 그 뒤로는 모델 이름만 바꿔 GPT·Gemini·Claude 를 오간다.

    키가 하나도 없으면 None 을 돌려주고 파이프라인은 모든 상품을 "검토 필요"로
    보류한다. **키가 없다고 서버가 죽지는 않는다** — 1차 규칙 필터는 그대로
    동작하므로 명백한 위반은 여전히 걸러진다.
    """
    name = (provider or os.getenv("MODERATION_PROVIDER") or "").strip().lower()

    if not name:
        for candidate, (_, env_key, _model) in _PROVIDERS.items():
            if os.getenv(env_key):
                name = candidate
                break

    if not name:
        log.warning("API 키가 없어 2차 AI 검수를 건너뜁니다 (1차 규칙 필터만 동작)")
        return None

    if name not in _PROVIDERS:
        raise ValueError(
            f"알 수 없는 검수 모델 제공자: {name} (가능: {', '.join(_PROVIDERS)})"
        )

    cls, env_key, default_model = _PROVIDERS[name]
    if not os.getenv(env_key):
        raise RuntimeError(f"{name} 를 쓰려면 {env_key} 환경변수가 필요합니다")

    chosen = model or os.getenv("MODERATION_MODEL") or default_model
    kwargs: dict = {"model": chosen}
    if name == "openai":
        kwargs["json_mode"] = os.getenv("MODERATION_JSON_MODE", "schema").strip().lower()

    log.info("2차 AI 검수 준비 — provider=%s, model=%s", name, chosen)
    return cls(**kwargs)
