"""이상거래 탐지와 상품 검수를 한 서버에 띄운다.

    python -m uvicorn serve:app --port 8000

백엔드 입장에서 AI 는 서비스 하나다. 포트를 두 개 열면 서비스 등록도 헬스체크도
두 벌이 되므로, 배포는 이 앱 하나로 한다. `/docs` 에 두 API 가 함께 나온다.

개별 앱(`fraud_api.main:app`, `moderation_api.main:app`, `reco_api.main:app`)도 그대로 살아 있다.
한쪽만 띄워 시험할 때 쓴다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from envfile import load_env
from fraud_api.main import lifespan as fraud_lifespan
from fraud_api.main import router as fraud_router
from moderation_api.main import lifespan as moderation_lifespan
from moderation_api.main import router as moderation_router
from reco_api.main import lifespan as reco_lifespan
from reco_api.main import router as reco_router

log = logging.getLogger("serve")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """두 앱의 준비·정리를 순서대로 감싼다.

    한쪽 준비가 실패하면 서버가 뜨지 않는다. 두 기능 모두 자체적으로 실패를
    흡수하도록 만들어 두었으므로(설정 오류는 경고 후 축소 동작), 여기까지
    예외가 올라온다면 정말로 뜨면 안 되는 상태다.
    """
    load_env()
    async with fraud_lifespan(app), moderation_lifespan(app), reco_lifespan(app):
        yield


app = FastAPI(
    title="DIB AI API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "DIB 의 AI 컴포넌트. 두 가지를 제공한다.\n\n"
        "- **이상거래 탐지** — 경매 종료 후 입찰자별 허위입찰 위험도\n"
        "- **상품 검수** — 상품이 거래 제한 품목인지 판정\n\n"
        "둘 다 **판정만 하고 제재하지 않는다.** 자동 차단 여부는 백엔드 정책이다."
    ),
)

app.include_router(fraud_router)
app.include_router(moderation_router)
app.include_router(reco_router)
