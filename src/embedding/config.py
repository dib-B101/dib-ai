"""임베딩 모델 설정.

**차원은 DB 컬럼 타입의 일부다.** pgvector 는 `VECTOR(1024)` 처럼 차원을 타입에 박아
두므로, 모델을 바꾸면 컬럼 재생성 + 전 상품 재계산 + 인덱스 재구축이 따라온다.
그래서 여기 값과 DB 스키마는 항상 같아야 하고, 인코더가 시작할 때 실제 출력 차원을
검증한다.
"""

from __future__ import annotations

import os

# 텍스트 — 제목 + 설명
#
# BGE-M3 를 쓰는 이유는 최대 8,192 토큰이라 긴 상품 설명이 잘리지 않기 때문이다.
# 한국어 경량 모델(ko-sroberta 등)은 512 토큰이라 설명이 길면 뒷부분이 통째로 누락되고,
# 그만큼 유사도가 왜곡된다.
TEXT_MODEL = os.getenv("EMBED_TEXT_MODEL", "BAAI/bge-m3")
TEXT_DIM = 1024

# 이미지 — 대표 이미지 1장
#
# SigLIP 은 CLIP 의 후속으로 같은 크기에서 검색·분류 성능이 낫다.
IMAGE_MODEL = os.getenv("EMBED_IMAGE_MODEL", "google/siglip-base-patch16-224")
IMAGE_DIM = 768

# 한 번에 모델에 넣을 개수. GPU 메모리와 맞바꾼다.
TEXT_BATCH = int(os.getenv("EMBED_TEXT_BATCH", "16"))
IMAGE_BATCH = int(os.getenv("EMBED_IMAGE_BATCH", "16"))

# cuda / cpu. 비워 두면 GPU 가 있으면 쓰고 없으면 CPU 로 떨어진다.
DEVICE = os.getenv("EMBED_DEVICE", "")


def resolve_device() -> str:
    if DEVICE:
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"
