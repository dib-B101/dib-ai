"""상품 유사도 — 텍스트와 이미지를 각각 재서 합친다.

**벡터를 합치는 것이 아니라 유사도 숫자를 합친다.** 텍스트(1024)와 이미지(768)는
차원이 다를 뿐 아니라 각 축이 뜻하는 바가 전혀 달라서, 두 벡터를 더하는 것은
키(cm)와 몸무게(kg)를 더하는 것과 같다.

    ① 각 공간 안에서만 비교      cos(text, text) → 숫자 하나
                                cos(image, image) → 숫자 하나
    ② 백분위로 정규화            후보 집합 안에서 몇 등인지로 바꾼다
    ③ 가중합                    0.6 × 텍스트 + 0.4 × 이미지

②를 빠뜨리면 안 된다. 두 유사도의 분포가 다르기 때문이다. 이미지 유사도가 좁은
구간에 몰려 있으면 어느 상품이든 값이 비슷해 **가중치 0.4 가 이름뿐인 숫자가 된다.**
실측에서도 텍스트가 0.36~0.70 에 몰려, 무관한 상품끼리도 0.4 대가 나왔다.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .popularity import percentile_ranks
from .schema import ProductVector


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """두 벡터의 코사인 유사도 [-1, 1].

    저장 전에 L2 정규화하므로 실제로는 내적만 하면 되지만, 정규화를 빠뜨린 벡터가
    섞여 들어와도 조용히 틀리지 않도록 길이를 나눈다.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def blend(
    text_sims: Mapping[int, float],
    image_sims: Mapping[int, float],
    w_text: float,
) -> dict[int, float]:
    """두 유사도를 백분위로 정규화한 뒤 가중합한다.

    이미지 벡터가 없는 상품이 섞이면 **그 상품은 텍스트만으로 판단한다.** 0 으로
    채우면 "닮지 않음" 으로 오해되어 사진이 없다는 이유만으로 순위가 밀린다.

    백분위는 순서만 보존한다. 그래서 돌려주는 값은 **후보 집합 안에서의 상대 순위이지
    "이만큼 닮았다" 는 절대값이 아니다.** 아무것도 안 닮은 집합을 넣어도 1등은 1.0 을
    받는다. 후보 풀을 미리 좁혀야 하는 이유이며, pgvector 로 넘어가면 ANN 이
    `LIMIT` 으로 가까운 것만 추려 주므로 이 문제가 함께 줄어든다.
    """
    keys = list(text_sims)
    if not keys:
        return {}

    text_pct = dict(zip(keys, percentile_ranks([text_sims[k] for k in keys])))

    with_image = [k for k in keys if k in image_sims]
    image_pct = dict(
        zip(with_image, percentile_ranks([image_sims[k] for k in with_image]))
    )

    out = {}
    for k in keys:
        if k in image_pct:
            out[k] = w_text * text_pct[k] + (1 - w_text) * image_pct[k]
        else:
            out[k] = text_pct[k]
    return out


def similarities(
    seed: ProductVector,
    others: Sequence[ProductVector],
    w_text: float,
) -> dict[int, float]:
    """기준 상품과 후보들의 종합 유사도.

    씨앗 자신은 결과에서 빠진다. 자기 자신은 유사도 1 이라 반드시 1위가 되는데,
    **지금 보고 있는 상품을 "이런 상품은 어때요" 에 다시 띄우는 것은 사고다.**

    씨앗에 이미지가 없으면 이미지 항 자체가 사라져 전원 텍스트로만 비교된다.
    일부만 이미지가 없을 때와 달리 불리해지는 쪽이 없으므로 그대로 둔다.
    """
    pool = [v for v in others if v.auction_id != seed.auction_id]
    if not pool:
        return {}

    text_sims = {v.auction_id: cosine(seed.text, v.text) for v in pool}
    image_sims = (
        {
            v.auction_id: cosine(seed.image, v.image)
            for v in pool
            if v.image is not None
        }
        if seed.image is not None
        else {}
    )
    return blend(text_sims, image_sims, w_text)
