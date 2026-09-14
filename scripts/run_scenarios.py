"""합성 시나리오를 모두 돌려 결과를 표로 출력한다.

    python scripts/run_scenarios.py

DB 없이 규칙 동작을 눈으로 확인하고, 임계값을 바꿔가며 점수 변화를 볼 때 쓴다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import fixtures as fx  # noqa: E402
from fraud import RuleConfig, detect  # noqa: E402


def main() -> None:
    cfg = RuleConfig.load()
    print(f"config version : {cfg.version}")
    print(f"활성 규칙       : {', '.join(s.rule_id for s in cfg.enabled_rules)}")
    print(f"최소 분석 조건  : 입찰 {cfg.min_bids}건 이상, 입찰자 {cfg.min_bidders}명 이상")

    for name, make in fx.ALL_SCENARIOS.items():
        result = detect(make(), cfg)
        print()
        print("=" * 78)
        print(f"[{name}]  auction {result.auction_id}")
        print("=" * 78)

        if result.auction_error:
            print(f"  경매 단위 실패: {result.auction_error}")
            continue
        if not result.results:
            reason = next(iter(result.skipped_bidders.values()), "사유 없음")
            print(f"  판정 안 함 — {reason}")
            continue

        for r in sorted(result.results, key=lambda x: -x.rule_score):
            band = cfg.band_of(r.rule_score)
            mark = "***" if band == "high" else (" * " if band == "medium" else "   ")
            print(f"{mark} member {r.member_id:<4} rule_score {r.rule_score:.3f}  [{band}]")
            for hit in sorted(r.hits, key=lambda h: -h.score * h.weight):
                if hit.score <= 0:
                    continue
                print(f"      {hit.rule_id:<24} {hit.score:.3f} × w{hit.weight}  {hit.reason}")
            if r.errors:
                print(f"      오류: {'; '.join(r.errors)}")


if __name__ == "__main__":
    main()
