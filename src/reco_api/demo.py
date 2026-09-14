"""합성 추천 후보.

DB 없이 랭킹이 어떻게 동작하는지 눈으로 확인하기 위한 데이터다. 세 가지 상황을
일부러 섞어 두었다.

    인기 많지만 시간이 많이 남음   → 인기도는 높고 마감 임박도는 낮다
    관심 없지만 곧 끝남           → 그 반대
    이미 끝남                    → 후보에서 빠져야 한다

마지막 것이 중요하다. 종료된 경매는 남은 시간이 0 이하라 마감 임박도가 최대가 되므로,
걸러내지 않으면 **끝난 경매가 목록 맨 위에 올라온다.**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from reco import Candidate


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
        # 관심은 적지만 2분 뒤 마감
        Candidate(
            auction_id=20002, product_id=2, seller_id=502, category_id=2,
            started_at=at(hours=-1), auction_time=3720, ended_at=at(minutes=2),
            view_count=90, bookmark_count=3, bid_count=2, bidder_count=2,
        ),
        # 중간. 40분 남았고 경쟁이 붙어 있다
        Candidate(
            auction_id=20003, product_id=3, seller_id=501, category_id=1,
            started_at=at(hours=-1), auction_time=6000, ended_at=at(minutes=40),
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
