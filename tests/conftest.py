"""테스트를 개발자의 로컬 설정에서 격리한다.

``.env`` 를 그대로 두면 테스트 결과가 각자의 파일 내용에 따라 달라진다. 그보다
나쁜 것은 **실수로 진짜 API 를 호출하는 것**이다 — 검수 파이프라인을 가짜 LLM 으로
바꾸는 것을 한 번만 잊어도 크레딧이 나가고, CI 에서는 키가 없어 실패한다.

그래서 테스트 중에는 ``.env`` 를 읽지 않고 관련 환경변수도 모두 비운다.
"""

from __future__ import annotations

import os

import pytest

ISOLATED = (
    "MODERATION_PROVIDER",
    "MODERATION_MODEL",
    "MODERATION_JSON_MODE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
)


@pytest.fixture(autouse=True, scope="session")
def isolate_local_config() -> None:
    os.environ["DIB_SKIP_DOTENV"] = "1"
    for name in ISOLATED:
        os.environ.pop(name, None)
