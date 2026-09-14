"""HTTP 요청·응답 스키마.

이 모듈이 백엔드와의 계약이다. FastAPI 가 여기서 OpenAPI 문서를 자동 생성하므로,
필드 설명은 백엔드 개발자가 읽을 것을 전제로 쓴다.

응답 필드는 fraud_detection 테이블 컬럼과 1:1 로 대응시켰다. 백엔드는 변환 없이
그대로 저장할 수 있다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    auction_id: int = Field(..., description="분석할 경매 ID", examples=[10432])
    as_of: datetime | None = Field(
        None,
        description=(
            "기준 시각. 생략하면 경매의 실제 종료 시각을 쓴다. "
            "이 값 이후의 데이터는 계산에서 제외되므로, 같은 값으로 재호출하면 "
            "언제 호출하든 같은 결과가 나온다."
        ),
        examples=["2026-09-08T14:03:00Z"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"auction_id": 10432, "as_of": "2026-09-08T14:03:00Z"}]
        }
    }


class BidderRisk(BaseModel):
    """입찰자 1명의 판정 결과. fraud_detection 1행에 대응한다."""

    member_id: int = Field(..., description="→ fraud_detection.member_id")
    risk_score: float = Field(
        ..., description="→ fraud_detection.risk_score. 규칙·모델 점수의 가중합"
    )
    rule_score: float = Field(..., description="→ fraud_detection.rule_score")
    ml_score: float | None = Field(
        None,
        description=(
            "→ fraud_detection.ml_score. 모델 트랙 미가동 상태이므로 현재는 항상 null 이다. "
            "가동되면 이 필드만 채워지고 나머지 계약은 바뀌지 않는다."
        ),
    )
    band: str = Field(
        ...,
        description="관리자 검토 구간. low / medium / high. **자동 제재에 사용하지 않는다**",
        examples=["high"],
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="점수가 높은 순으로 정렬한 탐지 사유. 관리자 화면에 그대로 노출할 수 있다",
    )
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "→ fraud_detection.detail (JSONB). 규칙별 점수·피처값·플래그·건너뛴 규칙. "
            "백엔드는 이 내용을 해석할 필요 없이 그대로 저장한다"
        ),
    )


class DetectResponse(BaseModel):
    auction_id: int
    as_of: datetime = Field(..., description="실제로 적용된 기준 시각")
    rule_config_version: str = Field(
        ..., description="규칙 임계값 설정 버전. 과거 판정을 해석할 때 필요하다"
    )
    model_version: str | None = Field(
        None, description="→ fraud_detection.model_version. 모델 트랙 가동 전에는 null"
    )
    weights: dict[str, float] = Field(
        ..., description="risk_score 를 만들 때 쓴 가중치", examples=[{"w_rule": 1.0, "w_ml": 0.0}]
    )
    results: list[BidderRisk] = Field(
        default_factory=list, description="판정된 입찰자. 저장 대상이다"
    )
    skipped_bidders: dict[int, str] = Field(
        default_factory=dict,
        description=(
            "판정하지 않은 입찰자와 그 사유. **저장하지 않는다.** "
            "이력이 부족한 신규 사용자 등이 여기 들어간다 — "
            "'판단하지 않음'과 '위험하지 않음'은 다르므로 0점으로 저장하면 안 된다"
        ),
    )
    auction_error: str | None = Field(
        None,
        description=(
            "경매 단위 실패 사유. 값이 있으면 results 는 비어 있다. "
            "탐지 실패가 경매 종료나 낙찰 처리를 막아서는 안 되므로 "
            "이 경우에도 200 으로 응답한다"
        ),
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    rule_config_version: str
    enabled_rules: list[str]
    provider: str = Field(..., description="현재 연결된 데이터 조회기")
