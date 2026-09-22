"""검수 API 의 요청·응답 스키마.

이 모듈이 백엔드와의 계약이다. FastAPI 가 여기서 OpenAPI 문서를 자동 생성하므로,
필드 설명은 백엔드 개발자가 읽을 것을 전제로 쓴다.

``verdict`` 를 그대로 저장하기보다 ``product_status`` 를 함께 내려준다. 백엔드가
"검토 필요가 어느 상태였더라" 를 매번 찾아보지 않게 하려는 것이다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    product_id: int = Field(..., description="검수할 상품 ID", examples=[8812])
    title: str = Field(..., description="→ product.title", examples=["아이폰 15 프로 256GB"])
    description: str | None = Field(
        None, description="→ product.description", examples=["정품 미개봉입니다."]
    )
    image_urls: list[str] = Field(
        default_factory=list,
        description=(
            "→ product_image.image_url. http(s) URL 또는 로컬 경로를 받는다. "
            f"앞에서부터 최대 6장만 본다. 내려받지 못한 이미지는 조용히 건너뛴다 — "
            "사진 하나 때문에 상품 등록이 막히면 안 된다"
        ),
        examples=[["https://cdn.example.com/p/8812/1.jpg"]],
    )
    category_name: str | None = Field(
        None, description="→ category.name. 판정 정확도를 높이는 참고 정보", examples=["디지털기기"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": 8812,
                    "title": "아이폰 15 프로 256GB",
                    "description": "정품 미개봉입니다. 직거래 선호합니다.",
                    "image_urls": ["https://cdn.example.com/p/8812/1.jpg"],
                    "category_name": "디지털기기",
                }
            ]
        }
    }


class ReviewResponse(BaseModel):
    product_id: int

    verdict: str = Field(
        ...,
        description="판정. 정상 / 검토 필요 / 금지",
        examples=["정상"],
    )
    product_status: str = Field(
        ...,
        description=(
            "→ product.status 에 그대로 넣을 값. "
            "정상 → REGISTERED, 검토 필요 → PENDING, 금지 → REJECTED"
        ),
        examples=["REGISTERED"],
    )
    stage: str = Field(
        ...,
        description=(
            "어디서 결정되었는지. "
            "rule = 1차 규칙 필터(AI 호출 없음), ai = 2차 AI 검수, "
            "fallback = AI 호출 실패로 보류"
        ),
        examples=["ai"],
    )
    category: str | None = Field(
        None, description="거래 제한 품목 분류. 정상이면 null", examples=["담배"]
    )
    confidence: float | None = Field(
        None,
        description="AI 판정의 확신도. 1차 규칙 필터에서 끝나면 1.0, 실패하면 null",
        examples=[0.93],
    )
    reason: str = Field(
        "",
        description=(
            "**사용자에게 그대로 보여줄 문장이다.** 관리자 검토 근거와 "
            "이의제기 대응 자료로도 쓴다"
        ),
        examples=["상품명에 거래 제한 품목(담배)이 포함되어 등록할 수 없습니다."],
    )
    content_hash: str = Field(
        ...,
        description=(
            "상품명·설명·이미지의 해시. 저장해 두면 재검수 여부를 판단할 수 있다. "
            "이 값이 그대로면 가격만 바뀐 수정이므로 다시 부를 필요가 없다"
        ),
    )
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "1차 필터 결론·걸린 금칙어·모델 버전·등급 조정 여부. "
            "백엔드는 해석할 필요 없이 그대로 저장한다"
        ),
    )


class ModerationHealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    keyword_config_version: str
    keyword_patterns: int = Field(..., description="변형 전개 후 등록된 패턴 수")
    llm: str | None = Field(
        None,
        description=(
            "연결된 2차 AI 검수 모델. **null 이면 AI 검수가 꺼진 상태**로, "
            "1차 규칙 필터에서 차단되지 않은 상품이 전부 검토 필요로 보류된다"
        ),
        examples=["openai/gemini-3.5-flash"],
    )
