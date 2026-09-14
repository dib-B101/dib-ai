"""추천 API 의 요청·응답 스키마.

이 모듈이 백엔드와의 계약이다. FastAPI 가 여기서 OpenAPI 문서를 자동 생성하므로,
필드 설명은 백엔드 개발자가 읽을 것을 전제로 쓴다.

추천은 **경매 ID 순서만** 돌려준다. 상품명·가격·이미지는 백엔드가 이미 갖고 있으므로
다시 실어 보내면 응답만 무거워지고 두 곳에서 같은 데이터를 관리하게 된다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecoItem(BaseModel):
    rank: int = Field(..., description="1부터 시작하는 순위", examples=[1])
    auction_id: int = Field(..., description="이 순서대로 노출하면 된다", examples=[10432])
    score: float = Field(..., description="랭킹 점수. 정렬에만 쓰고 사용자에게 보이지 않는다")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "점수를 구성한 항목들(마감 임박도·인기도·경쟁도·남은 시간). "
            "**사용자에게 노출하지 않는다.** 왜 이 순위인지 우리가 설명하고 튜닝하기 위한 값이다"
        ),
    )


class RecoResponse(BaseModel):
    as_of: datetime = Field(..., description="이 순서를 만든 기준 시각")
    strategy: str = Field(
        ...,
        description=(
            "어떤 방식으로 뽑았는지. `popularity` = 인기순(개인화 없음). "
            "개인화가 가동되면 값이 바뀐다"
        ),
        examples=["popularity"],
    )
    config_version: str = Field(
        ..., description="랭킹 가중치 설정 버전. 과거 결과를 해석할 때 필요하다"
    )
    items: list[RecoItem] = Field(default_factory=list)
    excluded: dict[str, int] = Field(
        default_factory=dict,
        description="후보에서 제외된 사유와 건수. 예: 이미 종료된 경매",
        examples=[{"ended": 3}],
    )


class RecoHealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    config_version: str
    weights: dict[str, float] = Field(..., examples=[{"urgency": 0.4}])
    provider: str = Field(..., description="현재 연결된 데이터 조회기")
