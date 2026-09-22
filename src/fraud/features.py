"""피처 계산기.

규칙 엔진과 (나중에) ML 모델이 함께 쓴다. 순수 함수만 두고 부수효과를 만들지 않는다.

입찰 순서에 관하여
    bid.amount 에 UNIQUE(auction_id, amount) 가 걸려 있고 영국식 경매는 가격이
    단조 증가하므로, amount 오름차순이 곧 시간 순서다. 별도 bid_seq 컬럼이 필요 없다.
    DUTCH / SEALED 경매를 도입하면 이 전제가 깨지므로 그때 bid_seq 를 추가해야 한다.
"""

from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Iterable, Sequence

from .schema import Auction, Bid, HistoryEntry


def ordered_bids(bids: Iterable[Bid]) -> list[Bid]:
    """입찰을 시간 순서(= 금액 오름차순)로 정렬한다."""
    return sorted(bids, key=lambda b: (b.amount, b.bid_id))


def bidder_ids(bids: Sequence[Bid]) -> list[int]:
    return sorted({b.member_id for b in bids})


def bidding_ratio(bids: Sequence[Bid], member_id: int) -> float:
    """이 경매의 전체 입찰 중 해당 입찰자가 차지한 비율."""
    if not bids:
        return 0.0
    mine = sum(1 for b in bids if b.member_id == member_id)
    return mine / len(bids)


def dominance_ceiling(total_bids: int) -> float:
    """우리 정책에서 한 사람이 가질 수 있는 입찰 비중의 상한.

    최고 입찰자는 추가 입찰이 불가하므로 연속 두 번 입찰할 수 없다.
    따라서 최대는 번갈아 입찰하는 경우이며 (n+1)/2n 이고, n 이 커질수록 0.5 로 수렴한다.
    """
    if total_bids <= 0:
        return 1.0
    return (total_bids + 1) / (2 * total_bids)


def reclaim_latencies(ordered: Sequence[Bid], member_id: int) -> list[float]:
    """최고가를 빼앗긴 뒤 되찾기까지 걸린 시간(초) 목록.

    우리 정책상 같은 사람이 연속으로 입찰할 수 없으므로, 이 입찰자의 두 입찰 사이에는
    반드시 다른 사람의 입찰이 있다. 그 첫 번째 입찰이 "빼앗긴 시점" 이다.
    """
    idxs = [i for i, b in enumerate(ordered) if b.member_id == member_id]
    out: list[float] = []
    for prev_i, next_i in zip(idxs, idxs[1:]):
        if next_i <= prev_i + 1:
            continue  # 정책상 없어야 하지만 방어적으로 건너뛴다
        lost_at = ordered[prev_i + 1].created_at      # 나를 추월한 첫 입찰
        regained_at = ordered[next_i].created_at
        latency = (regained_at - lost_at).total_seconds()
        if latency >= 0:
            out.append(latency)
    return out


def robust_dispersion(values: Sequence[float]) -> float:
    """중앙값 대비 MAD(중앙값 절대편차). 작을수록 간격이 균일하다.

    표준편차 기반 변동계수를 쓰지 않는 이유: 재탈환을 3회 빠르게 하고 마지막에
    한참 쉰 경우, 그 이상치 하나가 표준편차를 부풀려 "균일하지 않다" 로 뒤집는다.
    실제로는 3회의 즉각 재탈환이 신호이므로 이상치에 둔감한 MAD 를 쓴다.
    """
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    if median <= 0:
        return 0.0
    mad = statistics.median([abs(v - median) for v in values])
    return mad / median


def longest_pingpong_run(ordered: Sequence[Bid]) -> dict[frozenset[int], int]:
    """동일한 두 사람이 제3자 없이 배타적으로 주고받은 최장 구간 길이.

    우리 정책상 모든 입찰 시퀀스는 구조적으로 교대 형태다. 따라서 "번갈아 입찰했는가" 를
    재면 모든 경매가 걸린다. 재야 할 것은 "동일한 두 사람만" 얼마나 길게 주고받았는가다.

    이 값만으로는 진짜 구매자 두 명의 경쟁과 구분되지 않으므로, 판정에 쓸 때는
    판매자 편중도·낙찰 회피와 함께 봐야 한다.
    """
    seq = [b.member_id for b in ordered]
    best: dict[frozenset[int], int] = {}
    i = 0
    while i < len(seq) - 1:
        a, b = seq[i], seq[i + 1]
        if a == b:
            i += 1
            continue
        j = i + 2
        while j < len(seq) and seq[j] == (a if (j - i) % 2 == 0 else b):
            j += 1
        run = j - i
        if run >= 4:  # 최소 2왕복 이상만 의미가 있다
            key = frozenset((a, b))
            best[key] = max(best.get(key, 0), run)
        i = max(j - 1, i + 1)
    return best


def pingpong_ratio(ordered: Sequence[Bid], member_id: int) -> tuple[float, int | None]:
    """이 입찰자가 낀 최장 핑퐁 구간의 비율과 상대방 member_id."""
    if not ordered:
        return 0.0, None
    runs = longest_pingpong_run(ordered)
    best_run, partner = 0, None
    for pair, run in sorted(runs.items(), key=lambda kv: (-kv[1], sorted(kv[0]))):
        if member_id in pair and run > best_run:
            best_run = run
            partner = next(iter(pair - {member_id}), None)
    return best_run / len(ordered), partner


def seller_concentration(history: Sequence[HistoryEntry], seller_id: int) -> float:
    """과거 참여 경매 중 해당 판매자 상품이 차지한 비율."""
    if not history:
        return 0.0
    same = sum(1 for h in history if h.seller_id == seller_id)
    return same / len(history)


def copair_repeat_count(
    history: Sequence[HistoryEntry], current_auction_id: int, seller_id: int
) -> int:
    """같은 판매자의 다른 경매에 반복 등장한 횟수."""
    return sum(
        1 for h in history if h.seller_id == seller_id and h.auction_id != current_auction_id
    )


def late_surge(
    ordered: Sequence[Bid], auction: Auction, window_seconds: float, member_id: int
) -> dict[str, float]:
    """최초 예정 종료시각 기준 마지막 구간의 가격 상승과 이 입찰자의 기여분.

    end_at 이 아니라 original_end_at 을 쓴다. 30초 연장 정책 때문에 end_at 은
    계속 움직여서 같은 입력에 같은 결과가 나오지 않는다.
    """
    empty = {"surge_ratio": 0.0, "contribution": 0.0, "window_bids": 0.0}
    if not ordered:
        return empty

    window_start = auction.original_end_at - timedelta(seconds=window_seconds)
    before = [b for b in ordered if b.created_at < window_start]
    in_window = [b for b in ordered if b.created_at >= window_start]
    if not in_window:
        return empty

    price_before = before[-1].amount if before else auction.start_price
    price_final = ordered[-1].amount
    total_gain = price_final - price_before
    if price_before <= 0 or total_gain <= 0:
        return {**empty, "window_bids": float(len(in_window))}

    my_gain = 0
    prev = price_before
    for b in in_window:
        if b.member_id == member_id:
            my_gain += b.amount - prev
        prev = b.amount

    return {
        "surge_ratio": total_gain / price_before,
        "contribution": max(0.0, my_gain / total_gain),
        "window_bids": float(len(in_window)),
    }


def account_age_days(member_created_at, as_of) -> float:
    return max(0.0, (as_of - member_created_at).total_seconds() / 86400.0)


def max_bid_multiple(bids: Sequence[Bid], member_id: int, start_price: int) -> float:
    """이 입찰자의 최고 입찰가가 시작가의 몇 배인가."""
    mine = [b.amount for b in bids if b.member_id == member_id]
    if not mine or start_price <= 0:
        return 0.0
    return max(mine) / start_price


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
