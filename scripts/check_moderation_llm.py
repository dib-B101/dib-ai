"""검수 모델 실호출 점검.

게이트웨이·모델을 바꿀 때마다 한 번씩 돌린다. 확인하는 것은 세 가지다.

1. **연결이 되는가** — base_url·키·모델 이름이 맞는지
2. **구조화 출력이 되는가** — json_schema 를 거절하는 엔드포인트가 흔하다.
   실패하면 MODERATION_JSON_MODE=object 를 안내한다
3. **판정이 쓸 만한가** — 우회 표현을 잡고 부정문에 속지 않는지

사용법::

    cp .env.example .env      # 그리고 키·모델을 채운다
    python scripts/check_moderation_llm.py

설정은 저장소 루트의 ``.env`` 에서 읽는다. 이 파일 안의 값을 고치지 말 것 —
아래 CASES 는 점검용 표본이지 설정이 아니다.

이미지는 넣지 않는다. 텍스트 판정이 먼저 되어야 이미지를 볼 의미가 있다.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envfile import load_env  # noqa: E402
from moderation.keywords import KeywordFilter  # noqa: E402
from moderation.llm import create_llm  # noqa: E402
from moderation.pipeline import ModerationPipeline  # noqa: E402
from moderation.schema import ProductInput, Verdict  # noqa: E402

# (상품명, 설명, 기대 판정, 이 사례가 확인하려는 것)
CASES = [
    ("아이폰 15 프로 256GB", "정품 미개봉입니다. 직거래 선호합니다.",
     Verdict.NORMAL, "평범한 상품을 막지 않는가"),
    ("전자담배 액상 팝니다", "니코틴 9mg 입니다.",
     Verdict.BLOCKED, "1차 규칙 필터가 AI 없이 잡는가"),
    ("행복한 연기 팝니다", "한 보루에 4만원, 직거래만 합니다.",
     Verdict.BLOCKED, "금칙어 없는 우회 표현을 문맥으로 읽는가"),
    ("어른들의 음료 한 박스", "12병 미개봉. 도수 있는 제품입니다.",
     Verdict.BLOCKED, "우회 표현 2 — 주류"),
    ("명품 가방 판매", "레플리카 아닙니다. 정품 보증서 함께 드립니다.",
     Verdict.NORMAL, "부정문에 속지 않는가"),
    ("캠핑용 나이프", "사시미칼 아니고 캠핑용입니다. 날 길이 8cm.",
     Verdict.NORMAL, "부정문 2"),
    ("빈티지 재떨이", "유리 재떨이입니다. 흠집 없습니다.",
     Verdict.NORMAL, "연상시킬 뿐인 물건을 막지 않는가"),
]

MARK = {True: "O", False: "X"}


def main() -> int:
    # Windows 콘솔 기본 인코딩(cp949)은 일부 기호를 못 찍고 그대로 죽는다.
    # 점검 스크립트가 인코딩 때문에 실패하면 안 되므로 출력만 UTF-8 로 돌린다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    loaded = load_env()
    print(f"설정 파일   {loaded or '(.env 없음 — 환경변수를 직접 읽습니다)'}")

    provider = (os.getenv("MODERATION_PROVIDER") or "").strip().lower()
    print(f"provider   {provider or '(자동 선택)'}")
    print(f"model      {os.getenv('MODERATION_MODEL') or '(벤더 기본값)'}")

    try:
        llm = create_llm()
    except Exception as exc:
        print(f"\n모델을 만들지 못했습니다 — {type(exc).__name__}: {exc}")
        return 1

    if llm is None:
        print("\nAPI 키가 없습니다.\n")
        print("  1. cp .env.example .env")
        print("  2. .env 를 열어 제공자에 맞는 키와 주소를 채우십시오")
        print("  3. 다시 실행하십시오")
        return 1

    # 제공자마다 읽는 환경변수가 다르다. 만들어진 객체에서 직접 꺼내야
    # "설정한 값이 실제로 쓰였는지" 를 확인할 수 있다.
    if endpoint := getattr(llm, "url", None):
        print(f"endpoint   {endpoint}")
    else:
        base = getattr(getattr(llm, "_client", None), "base_url", None)
        print(f"endpoint   {base or '(벤더 기본 엔드포인트)'}")
    if getattr(llm, "name", None) == "openai":
        print(f"json_mode  {os.getenv('MODERATION_JSON_MODE', 'schema')}")
    print()

    pipeline = ModerationPipeline(KeywordFilter.load(), llm)

    print(f"{'기대':<6} {'실제':<6} {'':2} {'단계':<9} {'초':>5}  상품명")
    print("-" * 78)

    agree = 0
    started = time.perf_counter()

    for i, (title, desc, expected, checking) in enumerate(CASES, 1):
        t0 = time.perf_counter()
        result = pipeline.review(ProductInput(i, title, desc))
        elapsed = time.perf_counter() - t0

        ok = result.verdict is expected
        agree += ok
        print(
            f"{expected.value:<6} {result.verdict.value:<6} {MARK[ok]:2} "
            f"{result.stage.value:<9} {elapsed:5.1f}  {title}"
        )
        print(f"{'':22}└ {checking}")
        if result.reason:
            conf = f"{result.confidence:.2f}" if result.confidence is not None else "—"
            print(f"{'':22}  conf {conf} · {result.reason}")
        if err := result.detail.get("error"):
            print(f"{'':22}  실패: {err}")
        print()

    total = time.perf_counter() - started
    print("-" * 78)
    print(f"{agree}/{len(CASES)} 일치 · 총 {total:.1f}초 (건당 평균 {total / len(CASES):.1f}초)")

    if agree < len(CASES):
        print("\n기대와 다른 판정은 프롬프트를 고칠 근거이지 곧바로 버그는 아니다.")
        print("특히 '검토 필요'로 나온 것은 확신이 낮다는 뜻이라 설계 의도대로다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
