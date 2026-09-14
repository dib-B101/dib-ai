"""상품을 벡터로 바꾸는 인코더.

텍스트와 이미지를 **각각 다른 모델로** 임베딩한다. 두 벡터는 차원도 다르고 축이 뜻하는
바도 달라서 하나로 합치지 않는다. 유사도를 각각 구한 뒤 가중합하는 것이 추천 설계다.

세 가지를 여기서 보장한다.

**저장 전에 L2 정규화한다.** 길이를 1 로 맞추면 코사인 유사도가 내적과 같아져
pgvector 쿼리가 빨라진다. 정규화를 빼먹으면 긴 설명이 붙은 상품이 길이만으로 유리해진다.

**출력 차원을 검증한다.** 모델을 바꿨는데 DB 컬럼이 그대로면 INSERT 시점에야 터진다.
로드하자마자 확인해서 일찍 실패시킨다.

**모델은 한 번만 로드한다.** 상품 하나 임베딩할 때마다 1GB 짜리 모델을 올릴 수는 없다.
"""

from __future__ import annotations

import logging
from typing import Sequence

from media import load_image

from .config import (
    IMAGE_BATCH,
    IMAGE_DIM,
    IMAGE_MODEL,
    TEXT_BATCH,
    TEXT_DIM,
    TEXT_MODEL,
    resolve_device,
)

log = logging.getLogger("embedding")


def _normalize(matrix):
    """행마다 길이를 1 로 맞춘다."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 전부 0 인 벡터가 들어오면 0 으로 나누게 된다. 그대로 둔다.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def _as_tensor(features):
    """``get_image_features`` 의 반환 형식을 흡수한다.

    transformers 4.x 는 텐서를 그대로 주고, 5.x 는 출력 객체를 준다. 버전이 올라갔을 때
    조용히 깨지지 않도록 둘 다 받는다. 이미지 한 장은 pooler_output 이 벡터다.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def build_text(title: str, description: str | None = None) -> str:
    """임베딩에 넣을 텍스트를 만든다.

    제목을 앞에 둔다. 중고 거래에서 모델명·상태 같은 결정적 정보가 제목에 몰려 있고,
    설명이 잘릴 경우 뒤쪽부터 사라지기 때문이다.
    """
    parts = [(title or "").strip()]
    if description:
        parts.append(description.strip())
    return "\n".join(p for p in parts if p)


class TextEncoder:
    """제목 + 설명 → 1024차원 벡터."""

    dim = TEXT_DIM

    def __init__(self, model_name: str = TEXT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or resolve_device()
        self._model = None

    def load(self) -> None:
        """모델을 메모리에 올린다. 서버 시작 시 한 번 부른다."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        log.info("텍스트 임베딩 모델 로드 — %s (%s)", self.model_name, self.device)
        model = SentenceTransformer(self.model_name, device=self.device)

        actual = model.get_sentence_embedding_dimension()
        if actual != self.dim:
            raise ValueError(
                f"{self.model_name} 의 출력이 {actual}차원입니다. "
                f"DB 컬럼은 VECTOR({self.dim}) 이므로 그대로 저장할 수 없습니다."
            )
        self._model = model

    def encode(self, texts: Sequence[str], batch_size: int = TEXT_BATCH):
        """(n, 1024) float32 배열. 각 행은 L2 정규화되어 있다."""
        if not texts:
            import numpy as np

            return np.empty((0, self.dim), dtype="float32")

        self.load()
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _normalize(vectors)

    def encode_product(self, title: str, description: str | None = None):
        """상품 1건. (1024,) 벡터를 돌려준다."""
        return self.encode([build_text(title, description)])[0]


class ImageEncoder:
    """대표 이미지 → 768차원 벡터.

    여러 장을 평균 내지 않는다. 배경·바닥면·포장지가 섞여 상품 자체의 특징이 희석된다.
    MVP 는 대표 이미지 1장으로 시작하고, 데이터로 확인한 뒤 다중 이미지를 판단한다.
    """

    dim = IMAGE_DIM

    def __init__(self, model_name: str = IMAGE_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or resolve_device()
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        log.info("이미지 임베딩 모델 로드 — %s (%s)", self.model_name, self.device)
        model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        # AutoProcessor 를 쓰면 텍스트 토크나이저까지 따라온다. SigLIP 토크나이저는
        # SentencePiece 를 요구하는데, 우리는 이미지만 인코딩하므로 쓸 일이 없다.
        # 이미지 전처리기만 불러 의존성을 하나 줄인다.
        processor = AutoImageProcessor.from_pretrained(self.model_name)

        actual = model.config.vision_config.hidden_size
        if actual != self.dim:
            raise ValueError(
                f"{self.model_name} 의 출력이 {actual}차원입니다. "
                f"DB 컬럼은 VECTOR({self.dim}) 이므로 그대로 저장할 수 없습니다."
            )

        self._model = model
        self._processor = processor
        self._torch = torch

    def encode_images(self, images: Sequence, batch_size: int = IMAGE_BATCH):
        """PIL 이미지 목록 → (n, 768). 각 행은 L2 정규화되어 있다."""
        import numpy as np

        if not images:
            return np.empty((0, self.dim), dtype="float32")

        self.load()
        torch = self._torch
        out = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = list(images[i : i + batch_size])
                inputs = self._processor(images=chunk, return_tensors="pt").to(self.device)
                out.append(_as_tensor(self._model.get_image_features(**inputs)).cpu().numpy())

        return _normalize(np.vstack(out))

    def encode_paths(self, sources: Sequence[str], batch_size: int = IMAGE_BATCH):
        """경로·URL 목록 → (성공한 인덱스, 벡터 배열).

        **읽지 못한 이미지는 건너뛴다.** 사진 한 장이 깨졌다고 배치 전체를 멈추면
        수천 건짜리 작업이 통째로 실패한다. 어느 것이 빠졌는지는 인덱스로 알려준다.
        """
        import numpy as np

        loaded, kept = [], []
        for idx, src in enumerate(sources):
            image = load_image(src)
            if image is None:
                continue
            loaded.append(image)
            kept.append(idx)

        if not loaded:
            return [], np.empty((0, self.dim), dtype="float32")
        return kept, self.encode_images(loaded, batch_size)

    def encode_product(self, thumbnail_url: str):
        """상품 1건. 읽지 못하면 None."""
        kept, vectors = self.encode_paths([thumbnail_url])
        return vectors[0] if kept else None
