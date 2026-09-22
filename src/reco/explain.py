"""추천 사유 한 문장.

명세서 95번 콜백이 `items[].reason` 을 요구한다. 점수를 구성한 항 중 **가장 크게
기여한 하나**를 골라 문장으로 바꾼다.

이상거래 탐지의 사유와는 성격이 다르다. 탐지 사유는 관리자가 판단 근거로 읽지만,
추천 사유는 **왜 이 순서인지 우리가 설명하고 튜닝하기 위한 값**이다. 사용자에게
그대로 노출할 것을 전제로 쓰지 않았다 — "입찰 경쟁이 붙어 있습니다" 같은 문장은
구매를 압박하는 문구로 읽힐 수 있어서, 노출 여부는 기획이 정할 일이다.

기여도는 **가중치 × 값**으로 잰다. 값만 크고 가중치가 낮은 항을 고르면 실제로
순위를 만든 이유와 다른 설명이 나간다.
"""

from __future__ import annotations

from .config import RecoConfig
from .schema import Scored


def _remaining_text(seconds: float) -> str:
    if seconds < 60:
        return "곧 마감됩니다"
    if seconds < 3600:
        return f"마감까지 {int(seconds // 60)}분 남았습니다"
    return f"마감까지 {int(seconds // 3600)}시간 남았습니다"


def reason_for(scored: Scored, cfg: RecoConfig) -> str:
    """이 경매가 위로 올라온 이유 한 문장."""
    w = cfg.weights
    w_sim = cfg.similarity_weight if scored.similarity else 0.0
    base = 1.0 - w_sim

    contributions = {
        "similarity": w_sim * scored.similarity,
        "urgency": base * w.get("urgency", 0.0) * scored.urgency,
        "popularity": base * w.get("popularity", 0.0) * scored.popularity,
        "competition": base * w.get("competition", 0.0) * scored.competition,
    }
    top = max(contributions, key=lambda k: contributions[k])

    # 전부 0 이면 고를 근거가 없다. 지표가 아직 없는 신규 상품이 여기 걸린다.
    if contributions[top] <= 0:
        return "새로 올라온 상품입니다"

    if top == "similarity":
        return "지금 보고 있는 상품과 비슷합니다"
    if top == "urgency":
        return _remaining_text(scored.remaining_seconds)
    if top == "popularity":
        return "조회와 찜이 많은 상품입니다"
    return "입찰 경쟁이 붙어 있습니다"
