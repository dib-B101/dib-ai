# ── DIB AI (FastAPI) ──
# 이상거래 탐지 · 상품 검수 · 추천을 한 앱(serve:app)으로 띄운다.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# lightgbm·pyahocorasick 등이 휠을 못 찾으면 소스 빌드로 떨어진다
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# envfile.py 가 ROOT 를 parents[1] 로 잡는다. src 가 /app/src 여야 config 와 .env 를 /app 에서 찾는다
COPY config ./config
COPY artifacts ./artifacts
COPY src ./src

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
