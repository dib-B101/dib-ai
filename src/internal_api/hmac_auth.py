"""백엔드와 AI 사이의 HMAC 서명.

명세서가 `SERVICE_HMAC`(백엔드 → AI)과 `AI_HMAC`(AI → 백엔드)을 요구한다. **다만
서명 방식 자체는 명세에 없다.** 헤더 이름도 알고리즘도 적혀 있지 않아, 아래 규약을
정하고 환경변수로 바꿀 수 있게 열어 두었다. 백엔드와 맞춰야 하는 유일한 항목이다.

    서명 대상   "{timestamp}.{요청 본문 원문}"
    서명        HMAC-SHA256(secret, 서명 대상) 의 16진수
    헤더        X-DIB-Timestamp: 유닉스 초
                X-DIB-Signature: sha256=<16진수>

**본문 원문(raw bytes)에 서명한다.** JSON 을 파싱했다가 다시 직렬화하면 키 순서나
공백이 달라져 서명이 깨진다.

**timestamp 를 서명에 포함한다.** 서명만 검사하면 가로챈 요청을 그대로 다시 보내는
재전송 공격을 막을 수 없다. 허용 오차를 벗어난 요청은 거절한다.

시크릿이 설정되지 않으면 **열어 두지 않고 막는다.** 인증이 없는 편이 편하지만, 설정
누락이 조용히 무인증 배포로 이어지는 쪽이 훨씬 위험하다.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

log = logging.getLogger("internal_api.hmac")

TIMESTAMP_HEADER = os.getenv("DIB_HMAC_TIMESTAMP_HEADER", "X-DIB-Timestamp")
SIGNATURE_HEADER = os.getenv("DIB_HMAC_SIGNATURE_HEADER", "X-DIB-Signature")

# 시계가 조금 어긋나도 통과해야 하지만, 넓히면 재전송 가능 시간이 그대로 늘어난다.
MAX_SKEW_SECONDS = int(os.getenv("DIB_HMAC_MAX_SKEW", "300"))

SERVICE_SECRET_ENV = "DIB_SERVICE_HMAC_SECRET"   # 백엔드가 우리를 부를 때
AI_SECRET_ENV = "DIB_AI_HMAC_SECRET"             # 우리가 백엔드를 부를 때


class HmacError(Exception):
    """서명 검증 실패. 사유를 호출자가 HTTP 상태로 옮긴다."""


def _secret(env_name: str) -> str | None:
    value = (os.getenv(env_name) or "").strip()
    return value or None


def sign(secret: str, body: bytes, timestamp: int | None = None) -> dict[str, str]:
    """보낼 요청에 붙일 헤더."""
    ts = str(timestamp if timestamp is not None else int(time.time()))
    mac = hmac.new(
        secret.encode("utf-8"), f"{ts}.".encode("utf-8") + body, hashlib.sha256
    )
    return {TIMESTAMP_HEADER: ts, SIGNATURE_HEADER: f"sha256={mac.hexdigest()}"}


def verify(secret: str, body: bytes, headers) -> None:
    """검증에 실패하면 `HmacError`. 통과하면 조용히 돌아온다."""
    provided = headers.get(SIGNATURE_HEADER)
    raw_ts = headers.get(TIMESTAMP_HEADER)
    if not provided or not raw_ts:
        raise HmacError(f"{SIGNATURE_HEADER} · {TIMESTAMP_HEADER} 헤더가 필요합니다")

    try:
        ts = int(raw_ts)
    except ValueError as exc:
        raise HmacError(f"{TIMESTAMP_HEADER} 가 유닉스 초가 아닙니다") from exc

    if abs(time.time() - ts) > MAX_SKEW_SECONDS:
        raise HmacError(f"요청 시각이 {MAX_SKEW_SECONDS}초 허용 범위를 벗어났습니다")

    expected = sign(secret, body, ts)[SIGNATURE_HEADER]

    # 문자열을 == 로 비교하면 앞에서부터 몇 글자가 맞았는지가 응답 시간에 드러난다.
    if not hmac.compare_digest(expected, provided):
        raise HmacError("서명이 일치하지 않습니다")


def service_secret() -> str:
    """백엔드 요청을 검증할 시크릿. 없으면 `HmacError`."""
    secret = _secret(SERVICE_SECRET_ENV)
    if secret is None:
        raise HmacError(
            f"{SERVICE_SECRET_ENV} 가 설정되지 않아 요청을 검증할 수 없습니다"
        )
    return secret


def ai_secret() -> str | None:
    """콜백에 서명할 시크릿. 없으면 None — 호출자가 콜백을 포기한다."""
    return _secret(AI_SECRET_ENV)
