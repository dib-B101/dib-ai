"""ML 모델에 넣을 피처를 우리 데이터에서 계산한다.

원 논문(UCI Shill Bidding Dataset)의 정의를 그대로 옮긴다. **학습은 eBay 원본 값으로
하고 서빙은 여기서 계산한 값을 쓰므로, 식이 어긋나면 모델이 엉뚱하게 읽는다.**
그래서 정의를 주석으로 남기고 데이터로 검증한 근거까지 적어 둔다.

    Bidder_Tendency          특정 판매자 경매 참여 수 / 전체 경매 참여 수
    Bidding_Ratio            이 입찰자의 입찰 수 / 경매 전체 입찰 수
    Last_Bidding             (경매 종료 − 마지막 입찰) / 전체 시간
    Early_Bidding            1 − (첫 입찰 − 경매 시작) / 전체 시간
    Auction_Bids             1 − 평균 입찰 수 / 이 경매 입찰 수
    Starting_Price_Average   1 − 이 시작가 / 평균 시작가
    Winning_Ratio            1 − 낙찰 수 / 적극 참여 경매 수

**전부 "클수록 위험" 방향이다.** `Last_Bidding` 이 특히 헷갈린다 — 값이 크면 경매 후반에
입찰을 **멈춘** 것이다. 가격만 올려놓고 빠지는 패턴이라 위험 신호가 된다.

`Auction_Duration` 은 계산하지 않는다. eBay 는 1~10 '일' 인데 우리는 라이브(분)와 일반
경매(판매자 자유 설정)가 섞여, 이 값이 "긴 경매인가" 가 아니라 "경매 유형" 을 가리키게
된다. 학습 때 없던 의미다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .features import bidding_ratio, ordered_bids
from .schema import Auction, Bid, CorpusStats, DetectionInput, HistoryEntry

# 서빙 피처 계산식의 버전. 명세 93번 콜백의 `featureVersion` 으로 나간다.
# **식을 바꾸면 반드시 올린다.** 같은 이름의 피처가 다른 뜻을 갖게 되면, 과거에
# 저장된 fraud_detection 행을 지금 기준으로 해석해 틀린 결론을 내게 된다.
FEATURE_VERSION = "dib-fv1"

# 원 논문이 "적극 참여" 를 가르는 기준. 한두 번 찔러본 경매를 낙찰률 분모에서 뺀다.
ACTIVE_PARTICIPATION_RATIO = 0.1

# 우리가 계산하는 피처 **집합**. 순서는 여기서 정하지 않는다 —
# 모델마다 학습 시 컬럼 순서가 다를 수 있고, 이름이 같아도 순서가 어긋나면
# 값이 뒤섞여 조용히 틀린 점수가 나온다. 순서는 model_meta.json 이 정한다.
FEATURE_NAMES = (
    "Bidding_Ratio",
    "Last_Bidding",
    "Auction_Bids",
    "Starting_Price_Average",
    "Early_Bidding",
    "Winning_Ratio",
    "Bidder_Tendency",
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _elapsed_ratio(moment, auction: Auction) -> float:
    """경매 시작부터 이 시각까지가 전체 시간의 몇 할인가.

    분모는 `ended_at` 이 아니라 **연장 전 예정 종료시각**이다. 연장이 반영되면 같은
    입찰이 경매마다 다른 비율로 계산되어 학습 분포와 어긋난다.
    """
    total = (auction.original_end_at - auction.started_at).total_seconds()
    if total <= 0:
        return 0.0
    return _clamp01((moment - auction.started_at).total_seconds() / total)


def last_bidding(bids: Sequence[Bid], member_id: int, auction: Auction) -> float:
    """경매 종료보다 얼마나 일찍 입찰을 멈췄는가. 클수록 일찍 멈춘 것."""
    mine = [b for b in bids if b.member_id == member_id]
    if not mine:
        return 1.0  # 입찰이 없으면 처음부터 끝까지 비활동
    return _clamp01(1.0 - _elapsed_ratio(max(b.created_at for b in mine), auction))


def early_bidding(bids: Sequence[Bid], member_id: int, auction: Auction) -> float:
    """경매 시작 후 얼마나 빨리 들어왔는가. 클수록 일찍 들어온 것."""
    mine = [b for b in bids if b.member_id == member_id]
    if not mine:
        return 0.0
    return _clamp01(1.0 - _elapsed_ratio(min(b.created_at for b in mine), auction))


def auction_bids(bid_count: int, mean_bids: float) -> float:
    """이 경매가 평균보다 얼마나 붐볐는가. 평균 이하면 0."""
    if mean_bids <= 0 or bid_count <= mean_bids:
        return 0.0
    return _clamp01(1.0 - mean_bids / bid_count)


def starting_price_average(start_price: float, mean_start_price: float) -> float:
    """시작가가 평균보다 얼마나 낮은가. 평균 이상이면 0.

    싸게 시작해 사람을 끌어들이는 것이 전형적인 수법이라 **낮을수록 위험**이다.
    그래서 값의 방향이 뒤집혀 있다.
    """
    if mean_start_price <= 0 or start_price >= mean_start_price:
        return 0.0
    return _clamp01(1.0 - start_price / mean_start_price)


def winning_ratio(history: Sequence[HistoryEntry]) -> float:
    """적극 참여한 경매 중 낙찰받지 못한 비율. 클수록 안 사면서 올리기만 한 것.

    분모를 전체 참여가 아니라 **적극 참여**로 잡는 이유는, 한두 번 찔러보고 만 경매까지
    세면 누구나 낙찰률이 낮게 나오기 때문이다.
    """
    active = [h for h in history if h.bidding_ratio > ACTIVE_PARTICIPATION_RATIO]
    if not active:
        return 0.0  # 적극 참여한 적이 없으면 판단 근거가 없다
    return _clamp01(1.0 - sum(1 for h in active if h.won) / len(active))


def bidder_tendency(
    history: Sequence[HistoryEntry],
    seller_id: int,
    member_id: int,
    subscriptions: Mapping[int, frozenset[int]],
) -> float:
    """이 판매자에게 얼마나 쏠려 있는가. **구독 관계는 빼고 센다.**

    원 정의를 그대로 쓰면 구독 모델에서 단골 구매자가 고위험으로 찍힌다. 구독한
    판매자의 경매에만 참여하는 것은 정상 행동이기 때문이다.

    구독 건을 분자·분모에서 빼면 "구독도 안 했는데 이 판매자만 따라다니는가" 가 되어
    **"높으면 의심" 이라는 원래 의미가 유지된다.** eBay 데이터에는 구독 개념이 없어
    뺄 것이 없으므로 학습 값과도 어긋나지 않는다.
    """
    subscribed = subscriptions.get(member_id, frozenset())
    if seller_id in subscribed:
        return 0.0  # 구독 중인 판매자면 편중 자체를 따지지 않는다

    unsubscribed = [h for h in history if h.seller_id not in subscribed]
    if not unsubscribed:
        return 0.0
    return _clamp01(
        sum(1 for h in unsubscribed if h.seller_id == seller_id) / len(unsubscribed)
    )


def compute(inp: DetectionInput, member_id: int) -> dict[str, float] | None:
    """입찰자 1명의 모델 입력값. 코퍼스 기준값이 없으면 None.

    **None 을 0 으로 대체하지 않는다.** 두 피처가 "평균 대비" 라서 0 은 "평균과 같다"
    는 뜻이 된다. 기준값을 모르는 것과 평균과 같은 것은 다르다.
    """
    corpus: CorpusStats | None = inp.corpus
    if corpus is None:
        return None

    auction, bids = inp.auction, inp.bids
    history = inp.histories.get(member_id, ())

    return {
        "Bidding_Ratio": bidding_ratio(bids, member_id),
        "Last_Bidding": last_bidding(bids, member_id, auction),
        "Auction_Bids": auction_bids(len(bids), corpus.mean_bids),
        "Starting_Price_Average": starting_price_average(
            auction.start_price, corpus.mean_start_price
        ),
        "Early_Bidding": early_bidding(bids, member_id, auction),
        "Winning_Ratio": winning_ratio(history),
        "Bidder_Tendency": bidder_tendency(
            history, auction.seller_id, member_id, inp.subscriptions
        ),
    }


def to_vector(features: Mapping[str, float], order: Sequence[str]) -> list[float]:
    """모델이 학습 때 쓴 순서로 늘어놓는다.

    순서를 호출자가 넘기게 한 것은 **하드코딩된 순서가 모델과 어긋나는 사고**를 막기
    위해서다. 이름이 같아도 자리가 다르면 값이 뒤섞이는데, 예외 없이 그럴듯한 점수가
    나와서 눈치채기 어렵다.
    """
    return [float(features[name]) for name in order]
