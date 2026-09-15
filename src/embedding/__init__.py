"""상품 임베딩 (추천 STEP 2).

텍스트와 이미지를 각각 다른 모델로 벡터화한다. 두 벡터는 차원도 축의 의미도 달라
하나로 합치지 않는다 — 유사도를 각각 구해 가중합하는 것이 추천 설계다.
"""

from .config import IMAGE_DIM, IMAGE_MODEL, TEXT_DIM, TEXT_MODEL, resolve_device
from .encoder import ImageEncoder, TextEncoder, build_text

__all__ = [
    "TextEncoder",
    "ImageEncoder",
    "build_text",
    "TEXT_MODEL",
    "TEXT_DIM",
    "IMAGE_MODEL",
    "IMAGE_DIM",
    "resolve_device",
]
