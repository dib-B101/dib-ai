"""금칙어 매칭 전에 텍스트를 정규화한다.

단순 문자열 포함 검사는 30초면 뚫린다. 실제 우회는 이런 식이다.

    전.자.담.배      특수문자 삽입
    전 자 담 배      공백 삽입
    ㄷㅏㅁㅂㅐ        자모 분리
    담배            전각 문자
    대마 → CH마      유사 문자

두 단계로 나눈다.

    normalize()             공백·기호 제거 + 자모 조합
    normalize_aggressive()  거기에 유사 문자 치환까지

유사 문자 치환을 기본으로 넣으면 ``아이폰 15`` 가 ``아이폰ㅣ5`` 가 되어 정상
상품명이 망가진다. 그래서 둘 다 만들어 두고 **어느 쪽에서든 걸리면 매칭**으로 본다.

초성 추정(``ㄷㅂ`` → 담배)은 하지 않는다. 담배인지 도배인지 알 수 없어 오탐이 급증한다.
"""

from __future__ import annotations

import re
import unicodedata

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

_CHO_IDX = {c: i for i, c in enumerate(_CHO)}
_JUNG_IDX = {c: i for i, c in enumerate(_JUNG)}
_JONG_IDX = {c: i for i, c in enumerate(_JONG) if c != " "}

_SYL_BASE, _SYL_LAST = 0xAC00, 0xD7A3

# NFKC 는 호환 자모(U+31xx)를 조합형 자모(U+11xx)로 바꾸고 왼쪽부터 탐욕적으로
# 합친다. 그래서 ``ㄷㅏㅁㅂㅐ`` 가 ``다ᄆ배`` 가 되고 ``ㅁ`` 이 종성으로 붙지 못한다.
# 조합형 자모를 호환 자모로 되돌린 뒤 직접 조합한다.
_CONJOIN_TO_COMPAT = {}
for _i, _c in enumerate(_CHO):
    _CONJOIN_TO_COMPAT[chr(0x1100 + _i)] = _c
for _i, _c in enumerate(_JUNG):
    _CONJOIN_TO_COMPAT[chr(0x1161 + _i)] = _c
for _i, _c in enumerate(_JONG[1:]):
    _CONJOIN_TO_COMPAT[chr(0x11A8 + _i)] = _c

_LOOKALIKE = {
    "o": "ㅇ", "0": "ㅇ", "○": "ㅇ", "◯": "ㅇ",
    "l": "ㅣ", "i": "ㅣ", "1": "ㅣ", "|": "ㅣ",
    "h": "ㅐ",
}

_NOISE = re.compile(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+")


def _is_syllable_without_jong(ch: str) -> bool:
    code = ord(ch)
    return _SYL_BASE <= code <= _SYL_LAST and (code - _SYL_BASE) % 28 == 0


def _compose_jamo(text: str) -> str:
    """흩어진 자모를 음절로 되돌린다.

    두 경우를 처리한다.

        ㄷ + ㅏ [+ ㅁ]   초성·중성(·종성) 조합
        다 + ㅁ          이미 만들어진 음절에 종성 붙이기

    종성 판단은 **그다음 문자가 중성인지**로 갈린다. ``다ㅁㅂㅐ`` 에서 ``ㅁ`` 뒤가
    ``ㅂ``(중성 아님)이므로 종성이고, ``다ㅁㅏ`` 였다면 ``ㅁ`` 은 다음 음절의 초성이다.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]

        # 앞 음절에 종성 붙이기
        if (
            out
            and ch in _JONG_IDX
            and _is_syllable_without_jong(out[-1])
            and not (i + 1 < n and text[i + 1] in _JUNG_IDX)
        ):
            out[-1] = chr(ord(out[-1]) + _JONG_IDX[ch])
            i += 1
            continue

        # 초성 + 중성 (+ 종성)
        if ch in _CHO_IDX and i + 1 < n and text[i + 1] in _JUNG_IDX:
            jong_idx, consumed = 0, 2
            if i + 2 < n and text[i + 2] in _JONG_IDX:
                nxt = text[i + 3] if i + 3 < n else ""
                if nxt not in _JUNG_IDX:
                    jong_idx = _JONG_IDX[text[i + 2]]
                    consumed = 3
            out.append(
                chr(_SYL_BASE + (_CHO_IDX[ch] * 21 + _JUNG_IDX[text[i + 1]]) * 28 + jong_idx)
            )
            i += consumed
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _base(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = "".join(_CONJOIN_TO_COMPAT.get(ch, ch) for ch in s)
    return s


def normalize(text: str) -> str:
    """공백·기호를 지우고 흩어진 자모를 합친다. 원문은 건드리지 않는다."""
    if not text:
        return ""
    return _compose_jamo(_NOISE.sub("", _base(text)))


def normalize_aggressive(text: str) -> str:
    """위에 더해 유사 문자까지 치환한다. 오탐 가능성이 있어 보조로만 쓴다."""
    if not text:
        return ""
    s = _base(text)
    s = "".join(_LOOKALIKE.get(ch, ch) for ch in s)
    return _compose_jamo(_NOISE.sub("", s))


def variants(text: str) -> tuple[str, str]:
    """매칭에 쓸 두 가지 표현. 어느 쪽에서든 걸리면 매칭으로 본다."""
    return normalize(text), normalize_aggressive(text)
