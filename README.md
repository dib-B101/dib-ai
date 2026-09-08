# dib-ai

DIB 시스템의 AI 컴포넌트. 이상거래 탐지, 상품 검수, 개인화 추천을 담당한다.

## Git 정책

이 저장소는 [`dib-orchestration`의 중앙 Git Flow 정책](https://github.com/dib-B101/dib-orchestration/blob/main/docs/GIT_FLOW.md)을 따른다.

## 구성

| 영역 | 상태 | 위치 |
| --- | --- | --- |
| 이상거래 탐지 — 규칙 기반 베이스라인 | 구현 완료 | `src/fraud/` |
| 이상거래 탐지 — HTTP API | 구현 완료 | `src/fraud_api/` |
| 이상거래 탐지 — ML 트랙 | EDA 완료, 서빙 미구현 | `notebooks/` |
| 상품 검수 | 미착수 | TBD |
| 개인화 추천 | 미착수 (라이브 전환에 따라 재설계 필요) | TBD |

## 폴더 구조

```
dib-ai/
├── config/
│   └── rules.yaml            임계값·가중치. 코드 수정 없이 조정한다
├── src/
│   ├── fraud/                규칙 엔진 (라이브러리)
│   │   ├── __init__.py       공개 API (RuleConfig, detect, combine)
│   │   ├── schema.py         입출력 자료구조. DB 를 모른다
│   │   ├── config.py         YAML 로더
│   │   ├── features.py       피처 계산기. 나중에 ML 트랙과 공용
│   │   ├── rules.py          규칙 5종
│   │   └── engine.py         오케스트레이션 · 실패 격리 · 점수 결합
│   └── fraud_api/            HTTP 계층
│       ├── main.py           FastAPI 앱 · 엔드포인트
│       ├── models.py         요청·응답 스키마. 백엔드와의 계약
│       ├── provider.py       데이터 조회기. DB 가 생기면 여기만 갈아끼운다
│       └── demo.py           합성 경매. bid 테이블 없이 API 를 호출해 볼 수 있다
├── tests/
│   ├── fixtures.py           합성 시나리오 7종
│   ├── test_rules.py         규칙별 검증
│   ├── test_engine.py        티켓 완료 조건 검증
│   └── test_api.py           API 계약 검증
├── scripts/
│   └── run_scenarios.py      시나리오 실행 데모
├── notebooks/
│   └── shill_bidding_eda.ipynb   eBay 데이터셋 EDA
├── data/
│   └── Shill Bidding Dataset.csv
├── pytest.ini
└── requirements.txt
```

학습된 모델 바이너리(`artifacts/`)는 저장소에 포함하지 않는다. 노트북을 실행하면 재생성된다.

## 실행

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

python -m pytest                    # 테스트 40개
python scripts/run_scenarios.py     # 시나리오별 점수 확인

# API 서버
PYTHONPATH=src uvicorn fraud_api.main:app --reload --port 8000
#   http://localhost:8000/docs   ← 백엔드는 여기를 보고 연동한다
```

---

# API

백엔드가 경매 종료 후 호출한다. **연동 시 `/docs` 를 보면 된다** — FastAPI 가 코드에서
OpenAPI 문서를 자동 생성하므로 별도 문서를 관리하지 않는다.

| 엔드포인트 | 용도 |
| --- | --- |
| `POST /internal/fraud/detect` | 경매 1건의 입찰자별 위험도 산출 |
| `GET /health` | 헬스체크. 인프라가 감시한다 |
| `GET /docs` | 자동 생성 API 문서 |

```bash
curl -X POST localhost:8000/internal/fraud/detect \
  -H "Content-Type: application/json" \
  -d '{"auction_id": 10002}'
```

**응답은 `fraud_detection` 컬럼과 1:1 로 대응한다.** 백엔드는 변환 없이 저장한다.

| 응답 필드 | 컬럼 |
| --- | --- |
| `results[].member_id` | `member_id` |
| `results[].risk_score` | `risk_score` |
| `results[].rule_score` | `rule_score` |
| `results[].ml_score` | `ml_score` (모델 트랙 가동 전 `null`) |
| `results[].detail` | `detail` (JSONB) |

`skipped_bidders` 는 **저장하지 않는다.** 이력이 부족해 판정하지 않은 입찰자이며,
0 점으로 저장하면 신규 사용자일수록 안전해 보이는 왜곡이 생긴다.

`auction_error` 가 있으면 탐지가 실패한 것이지만 **응답 코드는 200 이다.**
탐지 실패가 경매 종료나 낙찰 처리를 막아서는 안 되기 때문이다.

## 데이터 조회기

규칙 엔진은 DB 를 모른다. `provider.py` 가 그 경계다.

```
지금   InMemoryProvider   합성 경매 2건. bid 테이블 없이 호출해 볼 수 있다
나중   PostgresProvider   실제 조회. 엔진과 엔드포인트는 건드리지 않는다
```

피처 계산식은 AI 쪽에서 가장 자주 바뀌는 부분이다. 백엔드가 데이터를 모아 보내는
구조였다면 식을 하나 고칠 때마다 백엔드 배포가 필요하다. 그래서 **AI 가 읽기 전용
계정으로 직접 조회**하는 쪽을 택했다.

데모 경매 — `10001` 정상, `10002` 핑퐁 의심

---

# 이상거래 탐지 (규칙 기반 베이스라인)

라벨이 없는 서비스 초기에도 의심 입찰을 식별하기 위한 규칙 엔진.
DB 없이 동작하는 순수 함수로 구현되어, 입찰 로그가 쌓이기 전에도 개발·검증할 수 있다.

## 사용법

```python
from fraud import RuleConfig, detect, combine

cfg = RuleConfig.load()               # config/rules.yaml
result = detect(detection_input, cfg) # 예외를 던지지 않는다

for r in result.results:
    risk = combine(r.rule_score, ml_score, w_rule=0.7, w_ml=0.3)
    save(
        auction_id=result.auction_id,
        member_id=r.member_id,
        rule_score=r.rule_score,       # fraud_detection.rule_score
        ml_score=ml_score,             # fraud_detection.ml_score
        risk_score=risk,               # fraud_detection.risk_score
        detail=r.to_detail(),          # fraud_detection.detail (JSONB) — 컬럼 추가 필요
    )
```

`detect()` 는 `auction_id` 와 `as_of` 만 알면 되고 DB 접근은 호출자가 담당한다.
FastAPI 로 감싸든 배치로 돌리든 엔진은 그대로 쓴다.

## 규칙

| ID | 내용 | 가중치 |
|---|---|---|
| `R1_RECLAIM_SPEED` | 최고가를 빼앗긴 뒤 되찾는 속도와 균일성 | 0.25 |
| `R2_BID_DOMINANCE` | 한 경매에서 입찰을 얼마나 독식했는가 | 0.25 |
| `R3_NEW_ACCOUNT_HIGH_BID` | 신규 계정의 공격적 고액 입찰 | 0.15 |
| `R4_SELLER_CONCENTRATION` | 특정 판매자 경매에 반복 참여 | 0.20 |
| `R5_MULTI_ACCOUNT` | 동일 환경 다계정 — **비활성** | 0.00 |
| `R6_LATE_SURGE` | 최초 마감 직전의 급격한 가격 상승 | 0.15 |

### R1 은 원 티켓에서 재정의했다

티켓의 "짧은 시간의 반복 입찰" 은 우리 서비스에서 **구조적으로 발생할 수 없다.**
최고 입찰자는 추가 입찰이 막혀 있어 `A → A → A` 가 나오지 않는다. 한 사람이 두 번
입찰하려면 반드시 그 사이에 다른 사람이 끼어야 한다.

```
A → B → A → B → A      가능
A → A → A              정책상 불가
```

그래서 실제로 관측 가능한 것은 **"빼앗기자마자 얼마나 빨리 되찾는가"** 이며,
이 값이 짧고 균일하면 자동화 또는 계정 간 핑퐁을 의심한다.

균일성은 가산항이 아니라 **속도의 배수**로 적용한다. 느리게 재탈환하면 아무리
규칙적이어도 위험하지 않다 — 그냥 성실한 구매자다.

산포도는 표준편차가 아니라 **MAD(중앙값 절대편차)** 로 잰다. 3회 즉시 재탈환한 뒤
한참 쉰 경우, 그 이상치 하나가 표준편차를 부풀려 판정을 뒤집기 때문이다.

### R5 는 구현하지 않았다

device fingerprint / IP 를 수집하지 않아 계산이 불가능하다. 컬럼만 추가하면 되는
문제도 아니다.

- **개인정보 수집 항목**이라 처리방침에 명시해야 한다. AI 파트 단독 결정 사안이 아니다
- IP 는 공유 와이파이·통신사 NAT 때문에 무관한 사용자가 묶여 오탐이 많다

`config/rules.yaml` 에서 `enabled: false` 로 두었고, 수집이 결정되면
`RULE_FUNCS` 에 함수를 추가하고 플래그만 켜면 된다.

## 설계 원칙

**재현성** — 기준 시각 `as_of` 는 주입받고 내부에서 `now()` 를 부르지 않는다.
규칙 실행 순서도 `rule_id` 정렬로 고정한다. 같은 입력이면 언제 돌려도 같은 결과가 나온다.

**실패 격리** — 규칙 하나가 예외를 던져도 나머지는 계속 계산되고, 오류는
`BidderResult.errors` 에 기록된다. `detect()` 는 어떤 경우에도 예외를 밖으로 던지지 않는다.
탐지 실패가 경매 종료나 낙찰 처리를 막아서는 안 되기 때문이다.

**자동 차단 없음** — 점수와 사유만 돌려준다. 제재 판단은 관리자 몫이다.

**"판단 안 함" ≠ "위험하지 않음"** — 데이터가 부족한 규칙은 0점이 아니라 `skipped_rules`
로 빠지고 가중 평균의 분모에서도 제외된다. 0점으로 세면 이력이 적을수록 점수가
낮아지는 왜곡이 생긴다.

## 알려진 특성

**가중 평균이라 단일 규칙만 강하면 총점이 희석된다.** 예를 들어 입찰의 50% 를
독식(R2 = 0.857)해도 다른 규칙이 0 이면 총점은 0.25 수준이다. 여러 신호가 겹쳐야
높은 점수가 나오는 설계이며, 오탐을 줄이는 대신 단일 패턴 공격에는 둔감하다.

밴드 임계값(`config/rules.yaml` 의 `bands`)이나 결합 방식(가중 평균 → 최댓값 등)은
관리자 검토 큐의 실제 물량을 보고 조정한다.

## ERD 의존성

`dib.sql` 기준으로 아직 정리되지 않은 항목이 있다.

| 항목 | 상태 |
| --- | --- |
| `fraud_detection.rule_score` · `ml_score` | 반영됨 |
| `fraud_detection.detail` (JSONB) | **미반영** — 신규 피처와 규칙별 점수를 저장할 곳이 없다 |
| `auction.original_end_at` | 삭제됨. `started_at + auction_time` 으로 유도한다. `auction_time` 은 생성 시 확정되고 연장해도 바뀌지 않음을 확인했다 |
| `auction.bid_unit` | 삭제됨 — 최소 입찰 단위 기반 피처는 계산 불가 |
| `auction.live_broadcast_id` | `NOT NULL` — 라이브 밖 경매를 만들 수 없다. nullable 전환 필요 |

## 앞으로

| 시점 | 할 일 |
|---|---|
| `bid` 테이블 생성 후 | 피처 계산기를 실제 쿼리에 연결. 규칙 로직은 그대로 |
| `AUCTION_CLOSED` 이벤트 후 | 탐지 트리거를 이벤트 구독으로 전환 |
| `BID_FAILED` 이벤트 후 | R1 강화 — 실패한 입찰 시도까지 포함 |
| 관리자 라벨 축적 후 | `rule_score` 와 `ml_score` 를 각각 채점해 가중치 재설정 |
