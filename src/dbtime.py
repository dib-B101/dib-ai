"""DB 시각과 코드 시각을 맞춘다.

ERD 의 모든 시각 컬럼이 시간대 없는 `TIMESTAMP` 다. 드라이버는 이를 naive
`datetime` 으로 준다. 반면 우리 코드와 백엔드가 보내는 값은 시간대를 달고 있다.

    DB            2026-09-08 14:00:00      naive · 한국 시각
    백엔드 전송    2026-09-08T05:00:20Z     aware · UTC
    우리 코드      datetime.now(timezone.utc)

**셋을 섞으면 두 가지로 터진다.** 하나는 `TypeError: can't subtract offset-naive
and offset-aware datetimes` 로 요란하게, 다른 하나는 **tzinfo 만 떼었을 때 9시간이
조용히 밀려서** 터진다. 후자가 훨씬 위험하다 — 예외도 로그도 없이 모든 입찰자의
`Early_Bidding` · `Last_Bidding` 이 1.0(최대 위험)이 된다.

그래서 **경계에서 한 번만 변환하고, 그 뒤로는 한 종류만 쓴다.** 탐지는 경매 시각을
기준으로 맞추고(`align_to`), 추천은 전부 UTC 로 올린다(`to_utc`).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def db_timezone():
    """DB 의 naive `TIMESTAMP` 가 어느 지역 시각인지.

    `DIB_DB_TIMEZONE` 으로 지정하고, 없으면 이 서버의 지역 시각으로 본다. 백엔드가
    `ZoneId.systemDefault()` 로 UTC 문자열을 만들므로 **AI 서버와 백엔드가 같은
    시간대에서 돌면 기본값으로 맞는다.** 컨테이너를 UTC 로 띄우는 등 둘이 어긋나면
    이 값을 명시해야 한다.
    """
    name = (os.getenv("DIB_DB_TIMEZONE") or "").strip()
    return ZoneInfo(name) if name else None


def to_utc(moment: datetime | None) -> datetime | None:
    """DB 에서 읽은 시각을 tz 를 붙인 UTC 로. 이미 붙어 있으면 변환만 한다.

    **tzinfo 를 붙이는 것이 아니라 시각을 옮긴다.** naive 값은 DB 지역 시각으로
    해석한 뒤 UTC 로 바꾼다.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        zone = db_timezone()
        moment = moment.replace(tzinfo=zone) if zone else moment.astimezone()
    return moment.astimezone(timezone.utc)


def align_to(moment: datetime, reference: datetime) -> datetime:
    """`moment` 를 `reference` 와 같은 tz 종류로 **변환한다.**

    탐지에서 쓴다. 경매 시각(DB 에서 온 naive)을 기준으로 삼아, 요청에 실려 온
    aware 값을 같은 눈금에 올린다.
    """
    zone = db_timezone()

    if reference.tzinfo is not None:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=zone) if zone else moment.astimezone()
        return moment.astimezone(reference.tzinfo)

    if moment.tzinfo is None:
        return moment
    if zone:
        return moment.astimezone(zone).replace(tzinfo=None)
    return moment.astimezone().replace(tzinfo=None)
