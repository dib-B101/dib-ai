"""저장소 루트의 ``.env`` 를 환경변수로 올린다.

키를 코드에 적지 않기 위해서다. 서버와 스크립트가 시작할 때 한 번 부른다.
파일이 없어도 조용히 넘어간다 — 배포 환경에서는 컨테이너가 환경변수를 직접 넣고
``.env`` 는 없는 것이 정상이다.

**이미 설정된 환경변수를 덮어쓰지 않는다.** 배포 환경의 값이 개발용 파일에 밀리면
안 된다.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("envfile")

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> Path | None:
    """읽은 파일 경로. 없거나 python-dotenv 가 없으면 None."""
    target = path or ROOT / ".env"
    if not target.exists():
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        log.warning("python-dotenv 가 없어 %s 를 읽지 않습니다", target)
        return None

    load_dotenv(target, override=False)
    return target
