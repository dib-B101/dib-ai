"""유사 상품 추천 검증.

이 기능이 조용히 틀리는 방식은 두 가지다.

**벡터가 없는 상품을 0 으로 채우는 것.** 0 은 "안 닮음" 으로 읽혀, 아직 임베딩이
안 돌았다는 이유만으로 순위가 밀린다. 모르는 것과 안 닮은 것은 다르다.

**기준 상품 자신이 결과에 남는 것.** 유사도 1 이라 항상 1위가 되는데, 지금 보고 있는
상품을 "이런 상품은 어때요" 에 다시 띄우는 것은 사고다.

둘 다 예외가 안 나고 그럴듯한 목록이 나와서 눈으로는 안 잡힌다. 그래서 테스트로 잡는다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from reco import Candidate, ProductVector, RecoConfig, blend, cosine, rank, similarities
from reco_api.main import app

NOW = datetime(2026, 9, 9, 12, 0, 0)
SIMILAR = "/internal/reco/similar"


@pytest.fixture(scope="module")
def cfg() -> RecoConfig:
    return RecoConfig.load()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make(auction_id: int, *, ends_in_min: float = 60, views=0, marks=0, bids=0, bidders=0):
    return Candidate(
        auction_id=auction_id,
        product_id=auction_id,
        seller_id=500,
        category_id=1,
        started_at=NOW - timedelta(hours=1),
        auction_time=7200,
        ended_at=NOW + timedelta(minutes=ends_in_min),
        view_count=views,
        bookmark_count=marks,
        bid_count=bids,
        bidder_count=bidders,
    )


# --- cosine ---------------------------------------------------------------


def test_identical_vectors_are_one():
    assert cosine((0.6, 0.8), (0.6, 0.8)) == pytest.approx(1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_length_is_divided_out():
    """저장 전에 정규화하지만, 안 된 벡터가 섞여도 조용히 틀리면 안 된다."""
    assert cosine((3.0, 0.0), (0.5, 0.0)) == pytest.approx(1.0)


def test_degenerate_inputs_return_zero():
    assert cosine((), (1.0,)) == 0.0
    assert cosine((1.0, 0.0), (1.0,)) == 0.0, "차원이 다르면 비교 자체가 무의미하다"
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0, "0 벡터는 방향이 없다"


# --- blend ----------------------------------------------------------------


def test_blend_output_is_scaled_for_mixing():
    """돌려주는 값은 [0, 1] 로 펴진다.

    `rank()` 가 이 값을 경매 지표 점수와 가중합하므로 눈금이 맞아야 한다. 텍스트
    유사도가 0.68/0.66/0.64 처럼 좁은 구간에 몰려도 순위가 벌어져야, 0.5 라는
    가중치가 이름뿐인 숫자가 되지 않는다.
    """
    out = blend({1: 0.68, 2: 0.66, 3: 0.64}, {}, w_text=1.0)

    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(0.0)


def test_missing_image_falls_back_to_text_only():
    """사진이 없다는 이유만으로 순위가 밀리면 안 된다.

    2번은 텍스트가 1등인데 이미지 벡터가 없다. 0 으로 채웠다면 0.6×1.0 + 0.4×0 = 0.6
    이 되어 이미지가 있는 1번(0.6×0.5 + 0.4×1.0 = 0.7)에 진다.
    """
    out = blend({1: 0.5, 2: 0.9, 3: 0.1}, {1: 0.9, 3: 0.2}, w_text=0.6)

    assert out[2] == pytest.approx(1.0), "텍스트에서 1등이면 최종도 1등이어야 한다"
    assert out[2] > out[1], "사진 없다고 1등이 밀리면 안 된다"


def test_missing_image_is_not_penalized_by_scale_difference():
    """두 유사도의 **평균이 달라서** 생기는 편향을 표준화가 없앤다.

    이미지 유사도(0.86~0.90)가 텍스트(0.46~0.50)보다 한참 높은 상황이다. 원값으로
    가중합하면 2번은 0.48 인데 3번은 0.6×0.46 + 0.4×0.86 = 0.62 가 되어, **텍스트로는
    2등인 2번이 꼴찌로 밀린다.** 사진이 없다는 것 말고는 이유가 없다.

    실측에서도 같은 일이 있었다 — 이미지 평균이 0.07 높아 사진 없는 상품의 평균
    피추천 순위가 288개 중 173등까지 밀렸다(중앙값 143등).
    """
    out = blend({1: 0.50, 2: 0.48, 3: 0.46}, {1: 0.90, 3: 0.86}, w_text=0.6)

    assert out[1] > out[2] > out[3], "텍스트 순서(1 > 2 > 3)가 그대로 유지돼야 한다"


def test_gap_size_within_a_modality_survives_the_blend():
    """표준화는 순서뿐 아니라 **간격**도 합산까지 가져간다.

    텍스트에서 2번(0.93)과 3번(0.82)은 근소한 차이고, 1번(0.46)만 한참 뒤다.
    이미지에서는 3번(0.51)이 2번(0.35)보다 확실히 앞선다. 종합하면 **3번이
    정답이다** — 텍스트는 2번과 거의 같은데 이미지는 분명히 낫다.

    각 항목을 백분위로 바꾸면 2번과 3번의 텍스트 차이 0.11 이 1번과의 차이 0.36 과
    똑같은 한 계단으로 눌려, 2번이 1위가 된다. 표준화는 이 간격을 유지한다.
    """
    out = blend(
        {1: 0.46, 2: 0.93, 3: 0.82},
        {1: 0.59, 2: 0.35, 3: 0.51},
        w_text=0.6,
    )

    assert out[3] > out[2], "텍스트 차이가 근소하면 이미지 우위가 뒤집을 수 있어야 한다"
    assert out[1] == pytest.approx(0.0), "양쪽에서 처진 1번은 꼴찌여야 한다"


def test_blend_of_empty_candidates_is_empty():
    assert blend({}, {}, w_text=0.6) == {}


# --- similarities ---------------------------------------------------------


def _vec(auction_id: int, text, image=None) -> ProductVector:
    return ProductVector(auction_id=auction_id, text=tuple(text), image=image)


def test_seed_is_excluded_from_its_own_result():
    seed = _vec(1, (1.0, 0.0))
    out = similarities(seed, [seed, _vec(2, (0.9, 0.1)), _vec(3, (0.0, 1.0))], w_text=0.6)

    assert 1 not in out, "보고 있는 상품을 다시 추천하면 안 된다"
    assert set(out) == {2, 3}


def test_nearest_product_ranks_first():
    seed = _vec(1, (1.0, 0.0))
    out = similarities(seed, [_vec(2, (0.99, 0.14)), _vec(3, (0.0, 1.0))], w_text=1.0)

    assert out[2] > out[3]


def test_seed_without_image_compares_by_text_only():
    """기준 상품에 사진이 없으면 이미지 항 자체가 사라진다.

    이때는 전원이 텍스트로만 비교되므로 불리해지는 쪽이 없다.
    """
    seed = _vec(1, (1.0, 0.0), image=None)
    out = similarities(
        seed,
        [_vec(2, (0.9, 0.1), image=(1.0, 0.0)), _vec(3, (0.0, 1.0), image=(1.0, 0.0))],
        w_text=0.6,
    )

    assert out[2] == pytest.approx(1.0) and out[3] == pytest.approx(0.0)


def test_empty_pool_returns_empty():
    seed = _vec(1, (1.0, 0.0))
    assert similarities(seed, [seed], w_text=0.6) == {}


# --- rank() 와의 결합 -------------------------------------------------------


def test_big_similarity_gap_beats_popularity(cfg):
    """유사도 차이가 크면 유사도가 이긴다. 유사 상품 추천이니 당연한 방향이다.

    w_sim = 0.5 에서 유사도 1.0 대 0.0 의 기여 차이는 0.5 다. 인기·경쟁을 양쪽 끝까지
    벌려도 기여 차이는 0.5 × (0.3 + 0.3) = 0.3 에 그친다. **가중치를 이렇게 정한 이상
    이 결과가 설계대로다.**
    """
    candidates = [make(2), make(3, views=9000, marks=400, bids=70, bidders=25)]
    result = rank(candidates, cfg, NOW, similarity={2: 1.0, 3: 0.0}, strategy="similar")

    assert result.strategy == "similar"
    assert result.items[0].auction_id == 2


def test_small_similarity_gap_loses_to_popularity(cfg):
    """반대로 유사도가 엇비슷하면 경매 지표가 순서를 정한다.

    **이게 섞는 이유다.** 유사도만 보면 닮기만 하고 아무도 안 보는 경매가 위로 올라온다.
    여기서는 유사도 기여 차이가 0.5 × 0.2 = 0.1 이라 인기·경쟁의 0.3 에 밀린다.
    """
    candidates = [make(2), make(3, views=9000, marks=400, bids=70, bidders=25)]
    result = rank(candidates, cfg, NOW, similarity={2: 0.6, 3: 0.4}, strategy="similar")

    assert result.items[0].auction_id == 3


def test_similarity_breaks_ties_between_equal_auctions(cfg):
    """경매 지표가 같으면 유사도가 순서를 정한다."""
    candidates = [make(2, views=100, marks=5), make(3, views=100, marks=5)]
    result = rank(candidates, cfg, NOW, similarity={2: 0.0, 3: 1.0}, strategy="similar")

    assert result.items[0].auction_id == 3
    assert result.items[0].similarity == pytest.approx(1.0)


def test_popularity_ranking_ignores_similarity_weight(cfg):
    """유사도를 안 넘기면 STEP 1 결과가 그대로여야 한다.

    STEP 3 을 붙이면서 인기순이 바뀌면 Cold Start 폴백과 평가 baseline 이 같이 흔들린다.
    """
    candidates = [make(2, views=10), make(3, views=9000, marks=400)]

    assert rank(candidates, cfg, NOW).items == rank(
        candidates, cfg, NOW, similarity=None
    ).items


def test_similarity_appears_in_breakdown(cfg):
    result = rank([make(2)], cfg, NOW, similarity={2: 0.75}, strategy="similar")

    assert result.items[0].breakdown()["similarity"] == pytest.approx(0.75)


# --- API ------------------------------------------------------------------


def test_similar_returns_same_category_first(client):
    """데모 벡터는 같은 카테고리끼리 가깝게 만들어 두었다.

    20001(아이폰) 기준이면 20003(갤럭시)이 1위여야 한다. 아니면 벡터가 아니라
    배선이 틀린 것이다.
    """
    body = client.get(SIMILAR, params={"auction_id": 20001}).json()

    assert body["strategy"] == "similar"
    assert body["items"][0]["auction_id"] == 20003


def test_similar_excludes_the_seed_auction(client):
    body = client.get(SIMILAR, params={"auction_id": 20001}).json()

    assert 20001 not in [i["auction_id"] for i in body["items"]]


def test_similar_excludes_ended_auctions(client):
    body = client.get(SIMILAR, params={"auction_id": 20001}).json()

    assert 20005 not in [i["auction_id"] for i in body["items"]]
    assert body["excluded"]["ended"] >= 1


def test_similar_falls_back_to_popularity_without_seed_vector(client):
    """임베딩이 없는 상품을 봐도 추천은 나가야 한다.

    빈 목록을 돌려주면 화면이 비고, 백엔드는 장애인지 데이터가 없는 건지 모른다.
    `strategy` 로 구분할 수 있게 해 둔다.
    """
    body = client.get(SIMILAR, params={"auction_id": 99999}).json()

    assert body["strategy"] == "popularity"
    assert body["items"], "폴백은 비면 안 된다"


def test_similar_items_are_score_sorted_and_ranked(client):
    body = client.get(SIMILAR, params={"auction_id": 20001, "limit": 3}).json()
    items = body["items"]

    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)
    assert len(items) <= 3


def test_similar_requires_auction_id(client):
    assert client.get(SIMILAR).status_code == 422
