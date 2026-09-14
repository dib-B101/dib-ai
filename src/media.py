"""이미지 읽기 공용 유틸.

검수와 임베딩이 같은 일을 한다 — 백엔드는 S3 URL 을 주고 테스트는 로컬 파일을 쓴다.
두 곳에 같은 코드를 두면 한쪽만 고쳐지는 일이 생기므로 여기로 모은다.

**읽기 실패를 예외로 올리지 않는다.** 사진 한 장 때문에 상품 등록이나 임베딩 배치가
통째로 멈추면 안 된다. 호출자가 None 을 보고 건너뛴다.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("media")

FETCH_TIMEOUT = 5.0


def read_image_bytes(src: str | Path, timeout: float = FETCH_TIMEOUT) -> bytes | None:
    """로컬 경로든 http(s) URL 이든 원본 바이트를 가져온다. 실패하면 None."""
    text = str(src)

    if text.startswith(("http://", "https://")):
        import urllib.request

        try:
            with urllib.request.urlopen(text, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            log.warning("이미지 다운로드 실패: %s", text)
            return None

    p = Path(text)
    if not p.exists():
        return None
    try:
        return p.read_bytes()
    except OSError:
        log.warning("이미지 읽기 실패: %s", p)
        return None


def load_image(src: str | Path, timeout: float = FETCH_TIMEOUT):
    """PIL RGB 이미지로 연다. 실패하면 None.

    RGB 로 변환하는 이유는 PNG 투명 채널이나 흑백 이미지가 섞여 들어오기 때문이다.
    모델은 채널 수가 고정이라 변환하지 않으면 배치가 통째로 깨진다.
    """
    raw = read_image_bytes(src, timeout)
    if raw is None:
        return None

    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            return im.convert("RGB")
    except Exception:
        log.warning("이미지 디코딩 실패: %s", src)
        return None
