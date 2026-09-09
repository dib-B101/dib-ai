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

def encode_image(path: str | Path) -> str | None:
    """로컬 이미지를 512px JPEG base64 로 만든다. 실패하면 None 이다.

    이미지 한 장이 깨져도 예외를 던지지 않고 건너뛴다. 나머지 이미지와 텍스트로
    판정한다 — 사진 하나 때문에 상품 등록이 막히는 편이 더 나쁘다.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        log.warning("Pillow 가 없어 이미지를 건너뜁니다")
        return None

    p = Path(path)
    if not p.exists():
        return None

    try:
        import io

        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
    except Exception:
        log.warning("이미지 인코딩 실패: %s", p)
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


def parse_verdict(data: dict, model_version: str) -> LLMVerdict:
    """벤더 응답 JSON 을 공용 값 객체로 옮긴다.

    confidence 를 0~1 로 자른다. 스키마에 범위를 적어도 모델이 100 같은 값을
    보내는 경우가 있고, 그대로 두면 임계값 판정이 무력화된다.
    """
    confidence = float(data.get("confidence", 0.0))
    return LLMVerdict(
        verdict=Verdict(data["verdict"]),
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
        return parse_verdict(json.loads(text), getattr(response, "model", self._model))


class OpenAIModerationLLM:
    """GPT 호출.

    Anthropic 과 달리 캐시 지점을 직접 지정하지 않는다. 1,024 토큰이 넘는 프롬프트의
    공통 접두사를 자동으로 캐싱하므로 시스템 프롬프트를 맨 앞에 두는 것으로 족하다.
    (우리 프롬프트는 그 문턱보다 짧아 실제로는 캐시가 안 걸릴 수 있다.)

    ``strict: true`` 는 스키마를 벗어난 응답을 아예 생성하지 못하게 막는다. 파싱
    실패를 걱정하지 않아도 되는 대신, 모든 필드가 ``required`` 이고
    ``additionalProperties: false`` 여야 한다는 제약이 붙는다.
    """

    name = "openai"

    def __init__(self, client=None, model: str = OPENAI_MODEL) -> None:
        if client is None:
            import openai

            client = openai.OpenAI()
        self._client = client
        self._model = model

    def judge(self, product: ProductInput, hint: str | None = None) -> LLMVerdict:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "moderation_verdict",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_openai_content(product, hint)},
            ],
        )
        message = response.choices[0].message
        # 안전 정책상 거부하면 content 가 비어 있다. 파이프라인이 보류로 처리하도록
        # 예외로 올린다 — 빈 문자열을 파싱하려다 엉뚱한 곳에서 터지는 편보다 낫다.
        if getattr(message, "refusal", None):
            raise RuntimeError(f"모델이 응답을 거부했습니다: {message.refusal}")
        return parse_verdict(
            json.loads(message.content), getattr(response, "model", self._model)
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
        MODERATION_MODEL=gpt-4o         생략하면 벤더 기본값

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
    log.info("2차 AI 검수 준비 — provider=%s, model=%s", name, chosen)
    return cls(model=chosen)
