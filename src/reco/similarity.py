"""상품 유사도 — 텍스트와 이미지를 각각 재서 합친다.

**벡터를 합치는 것이 아니라 유사도 숫자를 합친다.** 텍스트(1024)와 이미지(768)는
차원이 다를 뿐 아니라 각 축이 뜻하는 바가 전혀 달라서, 두 벡터를 더하는 것은
키(cm)와 몸무게(kg)를 더하는 것과 같다.

    ① 각 공간 안에서만 비교      cos(text, text) → 숫자 하나
                                cos(image, image) → 숫자 하나
    ② 각각 표준화                후보 집합 안에서 (값 − 평균) / 표준편차
    ③ 가중합                    0.6 × 텍스트 + 0.4 × 이미지
    ④ 백분위                    랭킹 점수와 섞을 수 있도록 [0, 1] 로

②를 빠뜨리면 안 된다. 두 유사도의 **평균과 폭이 다르기 때문이다.** 실측(공개
패션 데이터 288건)에서는 이랬다.

            평균     표준편차   5~95% 폭
    텍스트   0.469    0.064     0.209
    이미지   0.541    0.088     0.294

평균이 다른 것이 특히 문제다. 이미지 쪽이 0.07 높아서, 원값을 그대로 가중합하면
**사진이 없어 텍스트만으로 계산된 상품이 일률적으로 손해를 본다.** 실제로 사진 없는
상품의 평균 피추천 순위가 288개 중 173등까지 밀렸다 — 중앙값이 143등이니 사진이
없다는 이유만으로 뒤로 밀린 것이다. 표준화하면 129등으로 돌아온다.

②를 **백분위가 아니라 표준화로** 하는 이유는 백분위가 순서만 남기고 "얼마나 더
닮았는가" 를 버리기 때문이다. 같은 데이터에서 백분위는 정확도가 일관되게 낮았다.

    Precision@5     대분류    소분류
    원값            0.744     0.228
    백분위          0.724     0.205     ← 매번 가장 낮다
    표준화          0.743     0.228

측정 스크립트는 `scripts/fetch_otpensource.py` + `scripts/eval_embedding.py` 다.
데이터가 패션 한정이라 **"아이폰과 갤럭시를 구분한다" 는 근거는 아니다.** 세 방식의
우열을 가리는 용도로만 쓴다.

④는 `rank()` 가 이 값을 경매 지표 점수와 가중합하기 때문에 필요하다. 표준화된 값은
범위가 정해져 있지 않아 그대로 섞으면 가중치가 무의미해진다. 백분위는 **순서를 그대로
보존하므로** 유사도끼리의 우열은 ②·③ 이 정한 그대로 남는다.
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


def _standardize(values: Sequence[float]) -> list[float]:
    """후보 집합 안에서 (값 − 평균) / 표준편차.

    전부 같은 값이면 표준편차가 0 이다. 이때는 전원 0 을 준다 — 나눗셈을 막기도
    하지만, **모두 똑같이 닮았을 때 억지로 순서를 만들지 않는다**는 뜻이기도 하다.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def blend(
    text_sims: Mapping[int, float],
    image_sims: Mapping[int, float],
    w_text: float,
) -> dict[int, float]:
    """두 유사도를 표준화한 뒤 가중합하고, 마지막에 백분위로 [0, 1] 에 맞춘다.

    이미지 벡터가 없는 상품은 **텍스트만으로 판단한다.** 0 으로 채우면 "닮지 않음"
    으로 오해되어 사진이 없다는 이유만으로 순위가 밀린다.

    표준화가 이 폴백을 성립시킨다. 두 유사도 모두 평균 0 이 되므로, 텍스트 하나만
    쓴 값과 둘을 섞은 값이 같은 눈금 위에 놓인다. 원값으로 섞으면 이미지 평균이
    0.07 높아 사진 없는 상품이 일률적으로 뒤로 밀린다(모듈 문서 참고).

    돌려주는 값은 **후보 집합 안에서의 상대 순위이지 "이만큼 닮았다" 는 절대값이
    아니다.** 아무것도 안 닮은 집합을 넣어도 1등은 1.0 을 받는다. 후보 풀을 미리
    좁혀야 하는 이유이며, pgvector ANN 으로 넘어가면 `LIMIT` 이 가까운 것만 추려
    주므로 이 문제가 함께 줄어든다.
    """
    keys = list(text_sims)
    if not keys:
        return {}

    text_z = dict(zip(keys, _standardize([text_sims[k] for k in keys])))

    with_image = [k for k in keys if k in image_sims]
    image_z = dict(zip(with_image, _standardize([image_sims[k] for k in with_image])))

    fused = [
        w_text * text_z[k] + (1 - w_text) * image_z[k] if k in image_z else text_z[k]
        for k in keys
    ]
    return dict(zip(keys, percentile_ranks(fused)))


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
