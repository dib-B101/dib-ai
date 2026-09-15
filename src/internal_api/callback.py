"""분석 결과를 백엔드로 돌려보낸다 (명세 93 · 95번).

**요청에 실려 온 `callbackUrl` 로 보낸다.** 우리가 주소를 설정으로 들고 있지 않는
이유는, 그래야 스테이징과 운영이 각자 자기 주소를 주고 우리는 손댈 것이 없기 때문이다.

다만 아무 주소로나 보내면 안 된다. 백엔드가 보낸 값이라도 잘못 설정되면 우리가
외부로 요청을 날리는 통로가 된다. 그래서 **허용 호스트 목록으로 막는다.**

실패해도 예외를 밖으로 올리지 않는다. 이미 202 를 돌려준 뒤라 호출자에게 알릴
방법이 없고, **콜백 실패가 서버를 죽이면 다른 분석까지 멈춘다.** 로그를 남기고
재시도한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from urllib.parse import urlparse

import httpx

from . import hmac_auth

log = logging.getLogger("internal_api.callback")

TIMEOUT_SECONDS = float(os.getenv("DIB_CALLBACK_TIMEOUT", "10"))
MAX_ATTEMPTS = int(os.getenv("DIB_CALLBACK_ATTEMPTS", "3"))


def allowed_hosts() -> frozenset[str]:
    """콜백을 보내도 되는 호스트. 비어 있으면 검사하지 않는다.

    `DIB_CALLBACK_ALLOWED_HOSTS=api.dib.io,localhost` 처럼 준다. 비워 두면 통과
    시키는 것은 로컬 개발 때문인데, **운영에서는 반드시 채워야 한다.**
    """
    raw = (os.getenv("DIB_CALLBACK_ALLOWED_HOSTS") or "").strip()
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def check_url(url: str) -> None:
    """보낼 수 없는 주소면 `ValueError`. 202 를 주기 전에 검사한다."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"callbackUrl 이 올바른 http(s) 주소가 아닙니다: {url}")

    hosts = allowed_hosts()
    if hosts and parsed.hostname.lower() not in hosts:
        raise ValueError(
            f"허용되지 않은 콜백 호스트입니다: {parsed.hostname}. "
            "DIB_CALLBACK_ALLOWED_HOSTS 를 확인하십시오"
        )


def send(url: str, payload: dict, *, label: str) -> bool:
    """성공하면 True. 실패는 로그만 남기고 False.

    본문을 **한 번 직렬화해 그대로 서명하고 그대로 보낸다.** 서명한 문자열과 실제
    전송한 문자열이 다르면 백엔드에서 검증이 깨진다.
    """
    secret = hmac_auth.ai_secret()
    if secret is None:
        log.error(
            "%s 콜백을 보내지 못했습니다 — %s 가 없어 서명할 수 없습니다",
            label, hmac_auth.AI_SECRET_ENV,
        )
        return False

    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json", **hmac_auth.sign(secret, body)}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                url, content=body, headers=headers, timeout=TIMEOUT_SECONDS
            )
            if response.status_code < 300:
                log.info("%s 콜백 성공 (%d회차)", label, attempt)
                return True

            # 4xx 는 다시 보내도 같은 답이 온다. 서명이나 본문이 틀린 것이므로
            # 재시도로 시간을 쓰는 대신 바로 포기하고 로그를 남긴다.
            if response.status_code < 500:
                log.error(
                    "%s 콜백 거절 %d — 재시도하지 않습니다. 응답=%s",
                    label, response.status_code, response.text[:300],
                )
                return False

            log.warning("%s 콜백 %d회차 실패 %d", label, attempt, response.status_code)
        except httpx.HTTPError as exc:
            log.warning("%s 콜백 %d회차 오류: %s", label, attempt, exc)

        if attempt < MAX_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))

    log.error("%s 콜백을 %d회 시도했으나 모두 실패했습니다", label, MAX_ATTEMPTS)
    return False
