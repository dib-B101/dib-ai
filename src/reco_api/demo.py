"""합성 추천 후보.

DB 없이 랭킹이 어떻게 동작하는지 눈으로 확인하기 위한 데이터다. 세 가지 상황을
일부러 섞어 두었다.

    인기 많지만 시간이 많이 남음   → 인기도는 높고 마감 임박도는 낮다
    관심 없지만 곧 끝남           → 그 반대
    이미 끝남                    → 후보에서 빠져야 한다
    라이브 방송 중                → scope=LIVE / GENERAL 로 갈린다

마지막 것이 중요하다. 종료된 경매는 남은 시간이 0 이하라 마감 임박도가 최대가 되므로,
걸러내지 않으면 **끝난 경매가 목록 맨 위에 올라온다.**

아래쪽에는 유사 상품 추천용 합성 벡터도 함께 둔다.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from reco import Candidate, ProductVector


def demo_candidates(now: datetime | None = None) -> tuple[Candidate, ...]:
    now = now or datetime.now(timezone.utc)

    def at(**kw) -> datetime:
        return now + timedelta(**kw)

    return (
        # 인기 최상위. 아직 6시간 남아 급하지 않다
        Candidate(
            auction_id=20001, product_id=1, seller_id=501, category_id=1,
            started_at=at(hours=-2), auction_time=28800, ended_at=at(hours=6),
            view_count=4200, bookmark_count=180, bid_count=31, bidder_count=12,
        ),
        # 관심은 적지만 2분 뒤 마감. 같은 방송의 두 번째 경매다
        Candidate(
            auction_id=20002, product_id=2, seller_id=502, category_id=2,
            started_at=at(hours=-1), auction_time=3720, ended_at=at(minutes=2),
            live_broadcast_id=3001,
            view_count=90, bookmark_count=3, bid_count=2, bidder_count=2,
        ),
        # 중간. 40분 남았고 경쟁이 붙어 있다. **라이브 방송 중인 경매다** —
        # scope=LIVE / GENERAL 분리가 동작하는지 확인하는 용도다
        Candidate(
            auction_id=20003, product_id=3, seller_id=501, category_id=1,
            started_at=at(hours=-1), auction_time=6000, ended_at=at(minutes=40),
            live_broadcast_id=3001,
            view_count=1500, bookmark_count=64, bid_count=18, bidder_count=9,
        ),
        # 갓 올라와 아무 지표도 없다. 정보 없음이 곧 나쁨은 아니다
        Candidate(
            auction_id=20004, product_id=4, seller_id=503, category_id=3,
            started_at=at(minutes=-5), auction_time=7200, ended_at=at(minutes=115),
            view_count=0, bookmark_count=0, bid_count=0, bidder_count=0,
        ),
        # 이미 30분 전에 끝났다. 인기가 가장 높지만 제외되어야 한다
        Candidate(
            auction_id=20005, product_id=5, seller_id=504, category_id=2,
            started_at=at(hours=-3), auction_time=9000, ended_at=at(minutes=-30),
            view_count=9800, bookmark_count=400, bid_count=77, bidder_count=25,
        ),
    )


# ---------------------------------------------------------------------------
# 유사 상품 추천용 합성 벡터
#
# **실제 임베딩이 아니다.** 진짜 벡터는 텍스트 1024차원(BGE-M3), 이미지 768차원
# (SigLIP) 이고 `product_embedding` 테이블에서 온다. 여기서는 사람이 읽고 검증할 수
# 있도록 축마다 뜻을 붙인 저차원 벡터를 손으로 만들었다.
#
# 목적은 "좋은 추천이 나온다" 를 보이는 것이 아니라 **배선이 맞는지** 보이는 것이다.
# 같은 카테고리끼리 가까운 값을 주었으므로, 유사도가 제대로 흐르면 같은 카테고리가
# 위로 올라온다. 안 올라오면 벡터가 아니라 코드가 틀린 것이다.
# ---------------------------------------------------------------------------

_TEXT_AXES = ("스마트폰", "노트북", "의류", "신발", "가구", "기타")
_IMAGE_AXES = ("기기화면", "신발형태", "목재질감", "배경")


def _unit(values: tuple[float, ...]) -> tuple[float, ...]:
    """L2 정규화. 실제 파이프라인도 저장 전에 정규화한다."""
    norm = math.sqrt(sum(v * v for v in values))
    return tuple(v / norm for v in values) if norm else values


def demo_vectors() -> tuple[ProductVector, ...]:
    return (
        # 아이폰. 20003 과 가깝고 나머지와는 멀다
        ProductVector(
            auction_id=20001,
            text=_unit((0.95, 0.30, 0.0, 0.0, 0.0, 0.05)),
            image=_unit((0.95, 0.0, 0.0, 0.30)),
        ),
        # 나이키 운동화. 20005 와 가깝다
        ProductVector(
            auction_id=20002,
            text=_unit((0.0, 0.0, 0.35, 0.93, 0.0, 0.05)),
            image=_unit((0.0, 0.96, 0.0, 0.25)),
        ),
        # 갤럭시. 같은 카테고리인 20001 과 가장 가깝다
        ProductVector(
            auction_id=20003,
            text=_unit((0.90, 0.35, 0.0, 0.0, 0.0, 0.10)),
            image=_unit((0.92, 0.0, 0.0, 0.35)),
        ),
        # 원목 책상. **이미지 벡터가 없다** — 사진을 못 읽었거나 배치가 아직 안 돈
        # 상품이다. 0 으로 채우지 않고 없는 채로 둬서 텍스트만으로 판단되게 한다.
        ProductVector(
            auction_id=20004,
            text=_unit((0.0, 0.10, 0.0, 0.0, 0.98, 0.10)),
            image=None,
        ),
        # 아디다스 운동화. 이미 끝난 경매라 후보에서 빠지지만, 벡터는 남겨 두어
        # "종료 제외가 유사도 계산보다 먼저" 인지 확인할 수 있게 한다
        ProductVector(
            auction_id=20005,
            text=_unit((0.0, 0.0, 0.40, 0.90, 0.0, 0.05)),
            image=_unit((0.0, 0.93, 0.10, 0.30)),
        ),
    )
