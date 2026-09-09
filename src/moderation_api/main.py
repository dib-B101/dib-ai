"""상품 검수 API.

백엔드가 상품 등록·수정 시 호출한다. 판정과 사유만 돌려주며 상태 변경은 하지 않는다.

    POST /internal/moderation/review   상품 1건 검수
    GET  /moderation/health            헬스체크
    GET  /docs                         자동 생성 API 문서 (백엔드는 여기를 본다)

**거래 제한 품목인지만 판정한다.** 상품 정보 불일치(사진과 설명이 다르다, 가격이
이상하다)는 검수 범위가 아니다.

설계 원칙 세 가지가 응답에 드러난다.

1. 검수 실패가 상품 등록을 막지 않는다.
   AI 호출이 실패해도 200 으로 응답하고 stage=fallback, verdict=검토 필요로 보낸다.
   등록을 거부하는 대신 관리자 큐로 넘긴다.

2. 확신이 없으면 차단하지 않는다.
   오탐이 미탐보다 비싸다. 정상 상품을 막으면 판매자가 이탈하지만, 금지 품목이
   한 번 통과해도 신고·사후 탐지로 잡는다.

3. 같은 내용을 다시 검수하지 않는다.
   content_hash 를 저장해 두면 가격만 바뀐 수정에 AI 를 부르지 않을 수 있다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from envfile import load_env
from moderation.keywords import KeywordFilter
from moderation.llm import create_llm
from moderation.pipeline import ModerationPipeline
from moderation.schema import ProductInput, Verdict

from .models import ModerationHealthResponse, ReviewRequest, ReviewResponse

log = logging.getLogger("moderation_api")

# 판정 → product.status. 백엔드가 매번 찾아보지 않도록 응답에 함께 담는다.
PRODUCT_STATUS = {
    Verdict.NORMAL: "REGISTERED",
    Verdict.NEEDS_REVIEW: "PENDING",
    Verdict.BLOCKED: "REJECTED",
}

router = APIRouter()
_state: dict[str, object] = {}


def startup() -> None:
    """검수기를 준비한다. 서버 수명 동안 한 번만 부른다.

    금칙어 오토마타 구축과 API 클라이언트 생성은 요청마다 할 일이 아니다.
    """
    load_env()
    keywords = KeywordFilter.load()

    try:
        llm = create_llm()
    except Exception as exc:
        # 설정이 틀렸다고 서버가 못 뜨면 안 된다. 1차 규칙 필터는 여전히 쓸모 있다.
        log.exception("2차 AI 검수 구성 실패 — 1차 규칙 필터만 동작합니다")
        llm = None
        _state["llm_error"] = f"{type(exc).__name__}: {exc}"

    _state["keywords"] = keywords
    _state["llm"] = llm
    _state["pipeline"] = ModerationPipeline(keywords, llm)
    log.info(
        "검수기 준비 완료 — 사전=%s, 패턴 %d개, AI=%s",
        keywords.version,
        keywords.size,
        _llm_label(llm) or "없음",
    )


def shutdown() -> None:
    _state.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


def _llm_label(llm) -> str | None:
    if llm is None:
        return None
    return f"{getattr(llm, 'name', '?')}/{getattr(llm, '_model', '?')}"


def get_pipeline() -> ModerationPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "검수기가 준비되지 않았습니다"
        )
    return pipeline  # type: ignore[return-value]


@router.get("/moderation/health", response_model=ModerationHealthResponse, tags=["ops"])
def health() -> ModerationHealthResponse:
    """인프라가 감시할 헬스체크.

    AI 검수가 꺼져 있어도 ok 다 — 1차 규칙 필터만으로도 서비스는 동작한다.
    다만 `llm` 이 null 이면 검토 필요가 대량으로 쌓이므로 확인해야 한다.
    """
    keywords: KeywordFilter = _state.get("keywords")  # type: ignore[assignment]
    if keywords is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "금칙어 사전이 로드되지 않았습니다"
        )

    return ModerationHealthResponse(
        status="ok",
        keyword_config_version=keywords.version,
        keyword_patterns=keywords.size,
        llm=_llm_label(_state.get("llm")),
    )


@router.post(
    "/internal/moderation/review",
    response_model=ReviewResponse,
    tags=["moderation"],
    summary="상품 1건의 거래 제한 품목 여부 판정",
)
def review_product(
    req: ReviewRequest,
    pipeline: ModerationPipeline = Depends(get_pipeline),
) -> ReviewResponse:
    """상품 등록·수정 시 호출한다.

    같은 내용으로 재호출하면 같은 `content_hash` 가 나온다. 이전에 저장한 값과
    같으면 판정도 그대로이므로 호출을 생략해도 된다 — 가격만 바꾼 수정에 AI 를
    부를 이유가 없다.

    **판정 실패로 500 을 내지 않는다.** AI 호출이 실패하면 stage=fallback,
    verdict=검토 필요로 200 응답한다. 검수 장애가 상품 등록을 막아서는 안 된다.
    """
    result = pipeline.review(
        ProductInput(
            product_id=req.product_id,
            title=req.title,
            description=req.description,
            image_urls=tuple(req.image_urls),
            category_name=req.category_name,
        )
    )

    keywords: KeywordFilter = _state["keywords"]  # type: ignore[assignment]
    return ReviewResponse(
        product_id=result.product_id,
        verdict=result.verdict.value,
        product_status=PRODUCT_STATUS[result.verdict],
        stage=result.stage.value,
        category=result.category,
        confidence=result.confidence,
        reason=result.reason,
        content_hash=result.content_hash,
        detail={**result.detail, "keyword_config_version": keywords.version},
    )


app = FastAPI(
    title="DIB 상품 검수 API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "상품이 **거래 제한 품목**인지 판정한다. 상품 정보 불일치는 검수하지 않는다.\n\n"
        "1차 규칙 필터가 명백한 위반을 AI 호출 없이 걸러내고, 애매한 것만 "
        "멀티모달 LLM 으로 넘긴다.\n\n"
        "**확신이 없으면 차단하지 않는다** — 오탐이 미탐보다 비싸므로 "
        "애매한 건 `검토 필요`로 두고 관리자가 본다."
    ),
)
app.include_router(router)
