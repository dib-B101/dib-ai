"""SHAP 기여도를 관리자가 읽을 문장으로 바꾼다.

규칙 엔진은 규칙 자체가 문장을 만들지만, 모델은 숫자만 낸다. 그런데 정책상
관리자 검토와 이의제기 대응에 근거가 필요하므로 모델 쪽에도 사유가 있어야 한다.

문장은 저장하지 않고 조회 시점에 만든다. 피처값과 기여도만 있으면 언제든
다시 만들 수 있고, 문구를 고칠 때 과거 데이터를 건드릴 필요도 없다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# 피처가 "높을 때" 무엇을 뜻하는지. 기여도가 양수일 때 쓰는 문구다.
_HIGH = {
    "Bidder_Tendency": "특정 판매자 경매에 반복 참여한 정도가 높습니다",
    "Bidding_Ratio": "이 경매의 입찰을 많이 차지했습니다",
    "Last_Bidding": "경매 후반까지 입찰을 이어갔습니다",
    "Auction_Bids": "평균보다 입찰이 많이 몰린 경매입니다",
    "Starting_Price_Average": "유사 경매 평균보다 시작가가 낮았습니다",
    "Early_Bidding": "경매 초반부터 입찰에 참여했습니다",
    "Winning_Ratio": "참여한 경매 대비 낙찰이 적습니다",
    "Auction_Duration": "경매 기간이 깁니다",
    "Successive_Outbidding": "자기 자신을 연속으로 추월했습니다",
}

# 낮을 때. 위험 방향이 반대인 경우를 위해 따로 둔다.
_LOW = {
    "Bidder_Tendency": "판매자 편중이 낮습니다",
    "Bidding_Ratio": "입찰 비중이 낮습니다",
    "Last_Bidding": "일찍 입찰을 멈췄습니다",
    "Auction_Bids": "입찰이 적은 경매입니다",
    "Starting_Price_Average": "시작가가 평균 이상이었습니다",
    "Early_Bidding": "뒤늦게 입찰에 참여했습니다",
    "Winning_Ratio": "참여한 경매를 대체로 낙찰받았습니다",
    "Auction_Duration": "경매 기간이 짧습니다",
    "Successive_Outbidding": "연속 추월이 없었습니다",
}


def top_reasons(
    shap_row: Sequence[float],
    features: Sequence[str],
    values: Mapping[str, float],
    *,
    limit: int = 3,
    min_contribution: float = 0.01,
) -> list[dict]:
    """위험도를 **올린** 요인만 기여도 순으로 돌려준다.

    내린 요인은 관리자 화면에서 쓸모가 없다. "왜 위험한가" 를 묻는 자리이지
    "왜 안전한가" 를 묻는 자리가 아니다.
    """
    order = np.argsort(np.abs(np.asarray(shap_row, dtype=float)))[::-1]
    out: list[dict] = []
    for i in order:
        contribution = float(shap_row[i])
        if contribution <= min_contribution:
            continue
        name = features[i]
        table = _HIGH if contribution > 0 else _LOW
        out.append(
            {
                "feature": name,
                "value": round(float(values.get(name, float("nan"))), 4),
                "contribution": round(contribution, 4),
                "text": table.get(name, f"{name} 기여"),
            }
        )
        if len(out) >= limit:
            break
    return out
