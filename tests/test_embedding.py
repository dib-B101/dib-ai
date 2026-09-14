"""임베딩 유틸 검증.

모델 로드가 필요한 부분은 테스트하지 않는다. 1GB 짜리 모델을 CI 마다 내려받을 수 없고,
모델 자체의 품질은 `scripts/check_embedding.py` 로 사람이 확인한다.

여기서는 **모델과 무관한 로직**만 본다 — 텍스트 조합, 정규화, 이미지 읽기.
조용히 틀려도 눈에 띄지 않는 것들이라 오히려 이쪽이 중요하다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from embedding import build_text
from embedding.encoder import _normalize
from media import load_image, read_image_bytes


# ---------------------------------------------------------------- 텍스트 조합

def test_title_comes_first():
    """설명이 잘릴 때 뒤쪽부터 사라지므로 결정적 정보인 제목을 앞에 둔다."""
    assert build_text("빈티지 카메라", "셔터 정상").startswith("빈티지 카메라")


@pytest.mark.parametrize("desc", [None, "", "   "])
def test_missing_description_leaves_no_trailing_blank(desc):
    """설명이 없을 때 줄바꿈만 남으면 모델에 빈 줄이 들어간다."""
    assert build_text("아이폰 15", desc) == "아이폰 15"


def test_whitespace_is_trimmed():
    assert build_text("  아이폰  ", "  정품  ") == "아이폰\n정품"


# ---------------------------------------------------------------- 정규화

def test_normalize_makes_unit_length():
    """길이를 1 로 맞춰야 코사인 유사도를 내적으로 계산할 수 있다."""
    out = _normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_keeps_direction():
    out = _normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(out[0], [0.6, 0.8])


def test_normalize_survives_zero_vector():
    """빈 텍스트가 0 벡터를 만들 수 있다. 0 으로 나누면 NaN 이 되어 조용히 퍼진다."""
    out = _normalize(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert not np.isnan(out).any()
    assert np.allclose(out[1], [1.0, 0.0])


def test_normalize_returns_float32():
    """pgvector 는 float4 다. float64 로 넘기면 변환 비용만 늘어난다."""
    assert _normalize(np.array([[1.0, 2.0]])).dtype == np.float32


# ---------------------------------------------------------------- 이미지 읽기

@pytest.fixture
def jpg(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), "red").save(path)
    return path


def test_reads_local_file(jpg):
    assert read_image_bytes(jpg)[:2] == b"\xff\xd8"   # JPEG 매직 넘버


def test_missing_file_returns_none_instead_of_raising(tmp_path):
    """사진 한 장 때문에 임베딩 배치나 상품 등록이 멈추면 안 된다."""
    assert read_image_bytes(tmp_path / "없음.jpg") is None


def test_load_image_converts_to_rgb(tmp_path):
    """투명 채널이나 흑백이 섞이면 모델 배치가 통째로 깨진다."""
    path = tmp_path / "gray.png"
    Image.new("L", (16, 16), 128).save(path)

    assert load_image(path).mode == "RGB"


def test_load_image_returns_none_for_non_image(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("이건 이미지가 아닙니다", encoding="utf-8")

    assert load_image(path) is None


def test_unreachable_url_returns_none():
    """S3 가 잠깐 죽어도 예외가 아니라 None 이어야 한다."""
    assert read_image_bytes("http://127.0.0.1:1/없는서버.jpg", timeout=0.2) is None
