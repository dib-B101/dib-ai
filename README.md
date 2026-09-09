# dib-ai

DIB 시스템의 AI 컴포넌트. 이상거래 탐지, 상품 검수, 개인화 추천을 담당한다.

## Git 정책

이 저장소는 [`dib-orchestration`의 중앙 Git Flow 정책](https://github.com/dib-B101/dib-orchestration/blob/main/docs/GIT_FLOW.md)을 따른다.

## 구성

| 영역 | 상태 | 위치 |
| --- | --- | --- |
| 이상거래 탐지 — 규칙 기반 베이스라인 | 구현 완료 | `src/fraud/` |
| 이상거래 탐지 — HTTP API | 구현 완료 | `src/fraud_api/` |
| 이상거래 탐지 — ML 트랙 | 부트스트랩 학습 완료, 서빙 미연결 | `src/fraud_ml/` |
| 상품 검수 — 1차 규칙 필터 | 구현 완료 | `src/moderation/` |
| 상품 검수 — 2차 AI 검수 | 구현 완료, 실호출 미검증 | `src/moderation/llm.py` |
| 개인화 추천 | 미착수 (라이브 전환에 따라 재설계 필요) | TBD |

## 폴더 구조

```
dib-ai/
├── config/
│   ├── rules.yaml            임계값·가중치. 코드 수정 없이 조정한다
│   └── banned_keywords.yaml  금칙어 사전. 운영 중 추가는 여기만 고친다
├── src/
│   ├── fraud/                규칙 엔진 (라이브러리)
│   │   ├── __init__.py       공개 API (RuleConfig, detect, combine)
│   │   ├── schema.py         입출력 자료구조. DB 를 모른다
│   │   ├── config.py         YAML 로더
│   │   ├── features.py       피처 계산기. 나중에 ML 트랙과 공용
│   │   ├── rules.py          규칙 5종
│   │   └── engine.py         오케스트레이션 · 실패 격리 · 점수 결합
│   ├── fraud_api/            HTTP 계층
│   │   ├── main.py           FastAPI 앱 · 엔드포인트
│   │   ├── models.py         요청·응답 스키마. 백엔드와의 계약
│   │   ├── provider.py       데이터 조회기. DB 가 생기면 여기만 갈아끼운다
│   │   └── demo.py           합성 경매. bid 테이블 없이 API 를 호출해 볼 수 있다
│   ├── fraud_ml/             ML 트랙 (부트스트랩)
│   │   ├── bootstrap.py      학습 · 확률 보정 · 절제 실험
│   │   └── explain.py        SHAP 기여도 → 한국어 탐지 사유
│   └── moderation/           상품 검수
│       ├── schema.py         입출력 자료구조. DB 도 HTTP 도 모른다
│       ├── normalize.py      한글 우회 표현 정규화
│       ├── keywords.py       Aho-Corasick 금칙어 필터
│       ├── llm.py            2차 AI 검수. Claude / GPT 교체 가능
│       └── pipeline.py       1차 → 2차 오케스트레이션
├── tests/
│   ├── fixtures.py           합성 시나리오 7종
│   ├── test_rules.py         규칙별 검증
│   ├── test_engine.py        티켓 완료 조건 검증
│   ├── test_api.py           API 계약 검증
│   └── test_moderation.py    검수 파이프라인 검증
├── scripts/
│   ├── run_scenarios.py      규칙 시나리오 실행 데모
│   └── train_bootstrap_model.py   ML 부트스트랩 학습
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

python -m pytest                        # 테스트 78개
python scripts/run_scenarios.py         # 규칙 시나리오별 점수 확인
python scripts/train_bootstrap_model.py # ML 부트스트랩 학습

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

데모 경매 — `10001` 정상, `10002` 핑퐁 의심, `10003` 구독자(R4 skip)

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
| `R4_SELLER_CONCENTRATION` | **구독하지 않은** 판매자에게 반복 참여 | 0.20 |
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

### R4 는 구독 관계를 본다

라이브 방송 구독 모델에서는 구독자가 좋아하는 방송자의 경매에만 참여하는 것이
**정상 행동**이다. 편중도만 보면 충성 고객이 그대로 고위험으로 찍힌다.

```
구독함     → 판단하지 않음 (skip)
구독 안 함 → 편중도를 그대로 평가
```

0 점이 아니라 skip 이다. "구독했으니 안전하다" 가 아니라 "이 신호로는 판단할 수
없다" 이기 때문이며, 가중 평균의 분모에서도 빠져 다른 규칙이 온전한 무게를 갖는다.

ML 트랙에서 `Bidder_Tendency` 를 제외한 것과 같은 이유다. 두 트랙이 같은 신호를
반대로 해석하면 구독자가 한쪽에서만 고위험으로 나온다.

**한계** — 공모자가 위장하려고 구독할 수 있다. 다만 구독은 공개 관계라 관리자가
확인할 수 있고, 핑퐁·재탈환 같은 다른 규칙은 그대로 작동한다. 판정 결과의
`flags.subscribed_to_seller` 에 구독 여부를 남겨 관리자가 판단할 수 있게 한다.

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

---

# ML 트랙 (부트스트랩)

`python scripts/train_bootstrap_model.py`

**이 모델은 배포용이 아니다.** eBay 데이터셋은 도메인이 달라 성능 숫자를 그대로
인용할 수 없다. 목적은 파이프라인 검증과 서빙 입력 스키마 확정이다.

## 절제 실험

| 구성 | 피처 | PR-AUC | Precision | Recall |
| --- | --- | --- | --- | --- |
| A. 전체 | 9개 | 0.9947 | 0.995 | 0.990 |
| B. `Successive_Outbidding` 제외 | 8개 | 0.6791 | 0.519 | 0.911 |
| **C. B + `Bidder_Tendency` 제외** | **7개** | **0.6983** | **0.615** | 0.792 |
| D. 상위 3개만 | 3개 | 0.5800 | 0.523 | 0.776 |

무작위 기준선 PR-AUC = 0.0965

**A 의 0.99 는 인용하지 말 것.** `Successive_Outbidding` 이 라벨과 거의 일대일로
붙어 있어서 나오는 숫자다. `SO > 0` 이라는 if 문 한 줄만으로 shill 675건 중 673건이
잡힌다. 그리고 그 피처는 우리 정책상 항상 0 이라 쓸 수 없다.

## 왜 C 를 채택했나

`Bidder_Tendency`(판매자 편중도)를 빼면 **성능과 정밀도가 함께 올랐다.**

```
PR-AUC     0.6791 → 0.6983   (+2.8%)
Precision  0.519  → 0.615    (+18.5%)
```

원래는 구독 충돌을 피하려고 성능을 포기하는 트레이드오프로 봤는데, 실제로는 순이득이었다.

> 라이브 구독 모델에서는 구독자가 좋아하는 방송자의 경매에만 참여하는 것이 **정상**이다.
> eBay 모델은 "판매자 편중이 높으면 위험" 으로 배우므로, 그대로 두면 **충성 고객이
> 고위험으로 찍힌다.**

관리자 검토 큐는 재현율보다 정밀도가 중요하다. 오탐 한 건마다 사람이 시간을 쓴다.

같은 이유로 규칙 엔진의 `R4_SELLER_CONCENTRATION` 도 재검토 대상이다.

## 확률 보정

```
Brier  0.0554 → 0.0447   (보정 후 19% 개선)
```

정책의 `0.3 / 0.6` 등급 경계는 **보정된 확률에서만 의미가 있다.** 보정 전 점수로
밴드를 나누면 "0.6" 이 위험도 60% 를 뜻하지 않는다.

`sklearn 1.6` 에서 `cv="prefit"` 이 제거되어 `FrozenEstimator` 로 감싼다.

## 탐지 사유

`explain.py` 가 SHAP 기여도를 한국어 문장으로 바꾼다. 규칙 엔진의 `reasons()` 와
형태를 맞춰 관리자 화면에서 두 트랙을 나란히 보여줄 수 있다.

```
· 이 경매의 입찰을 많이 차지했습니다        (기여 +6.546, 값 0.4444)
· 참여한 경매 대비 낙찰이 적습니다          (기여 +2.120, 값 1.0)
```

문장은 **저장하지 않고 조회 시 생성**한다. 피처값과 기여도만 있으면 다시 만들 수 있고,
문구를 고칠 때 과거 데이터를 건드릴 필요가 없다.

## 운영 관점

최적점이 Precision 0.615 / Recall 0.792 다. **고위험 판정 1.6건 중 1건만 실제 의심**이라
자동 제재는 불가능하고 관리자 검토 큐 정렬용이다.

## 서빙 계약

`artifacts/model_meta.json` 의 `serving_features_contract` 가 백엔드가 준비해야 할
값이다. 학습 피처와 일치하는지 스크립트가 매번 확인한다.

---

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

---

# 상품 검수

**거래 제한 품목인지만 판정한다.** 상품 정보 불일치(사진과 설명이 다르다, 가격이
이상하다)는 검수 범위가 아니다.

## 2단 구조

```
상품 등록
    ↓
1차 규칙 필터  (0.001초, 무료)
    ├─ BLOCK     제목에 금칙어 그대로     → AI 호출 없이 즉시 차단
    ├─ ESCALATE  설명에만 / 우회 의심     → 신호를 붙여 AI 로
    └─ PASS      아무것도 안 걸림         → 일반 AI 검수
    ↓
2차 AI 검수  (멀티모달 LLM, 상품명 + 설명 + 이미지)
    ↓
confidence 로 등급 조정
    ↓
정상(REGISTERED) / 검토 필요(PENDING) / 금지(REJECTED)
```

1차의 목적은 **AI 호출을 줄이는 것**이다. 명백한 위반은 여기서 끝내고 애매한 것만
LLM 으로 넘긴다. 금칙어가 수천 개로 늘어도 텍스트를 한 번만 훑도록 Aho-Corasick 을
쓴다 — 정규식 반복문은 금칙어 수에 비례해 느려진다.

## 왜 통과/차단 2단이 아닌가

처음에는 2단으로 만들었는데 이런 게 걸렸다.

```
"명품 가방 판매"  /  "레플리카 아닙니다"
"캠핑용 나이프"   /  "사시미칼 아니고 캠핑용입니다"
```

정상 판매자가 흔히 쓰는 문구다. **규칙은 부정문을 읽지 못하므로 판단하지 말고 LLM 에
넘긴다.** 그래서 `BLOCK` / `ESCALATE` / `PASS` 3단이다.

## 우회 표현 정규화

단순 문자열 포함 검사는 30초면 뚫린다.

| 우회 방식 | 예시 | 대응 |
| --- | --- | --- |
| 특수문자 삽입 | `전.자.담.배` | 기호 제거 |
| 공백 삽입 | `전 자 담 배` | 공백 제거 |
| 자모 분리 | `ㄷㅏㅁㅂㅐ` | 음절 재조합 |
| 전각 문자 | `담배` | NFKC |
| 유사 문자 | `CH마초` | 유사 문자 치환 |

자모 재조합은 직접 구현했다. NFKC 에 맡기면 `ㄷㅏㅁㅂㅐ` 가 `다ᄆ배` 가 된다 —
왼쪽부터 탐욕적으로 합치느라 `ㅁ` 을 종성으로 붙이지 못한다.

유사 문자 치환은 **기본 정규화에 넣지 않는다.** `1 → ㅣ`, `0 → ㅇ` 를 항상 적용하면
`아이폰 15` 가 `아이폰ㅣ5` 가 되어 정상 상품명이 망가진다. 그래서 두 가지 표현을
만들어 두고 어느 쪽에서든 걸리면 매칭으로 보되, 유사 문자 쪽에서만 걸린 것은
`evasion` 으로 표시해 차단하지 않고 AI 로 넘긴다.

초성 추정(`ㄷㅂ` → 담배)은 하지 않는다. 담배인지 도배인지 알 수 없어 오탐이 급증한다.

## 왜 LLM 인가

분류 모델을 파인튜닝하지 않는 이유는 **학습 데이터가 0건**이기 때문이다. 그리고 이런
문장은 애초에 단어 목록이나 임베딩 유사도로 잡히지 않는다.

```
"행복한 연기 팝니다"
```

금칙어가 하나도 없다. 문장의 뜻을 이해해야 한다.

LLM 만이 **왜 막았는지 한국어로 설명**한다는 점도 크다. 그 문장이 그대로 사용자 안내
문구가 되고, 관리자 검토 근거가 되고, 이의제기 대응 자료가 된다. 분류 모델은 확률값만
뱉는다.

## 모델 교체

파이프라인은 `ModerationLLM` 프로토콜만 알고 어느 모델인지는 모른다. 프롬프트와 응답
스키마가 공용이라 환경변수로 갈아끼운다.

```bash
MODERATION_PROVIDER=openai      # anthropic | openai. 생략하면 키 있는 쪽 자동 선택
MODERATION_MODEL=gpt-4o         # 생략하면 벤더 기본값
OPENAI_API_KEY=sk-...           # 또는 ANTHROPIC_API_KEY
```

**키가 없어도 서버는 뜬다.** 1차 규칙 필터는 그대로 동작하므로 명백한 위반은 계속
걸러지고, 나머지는 전부 "검토 필요"로 보류된다.

벤더별 차이는 어댑터 안에만 있다.

| | Anthropic | OpenAI |
| --- | --- | --- |
| 구조화 출력 | `output_config.format` | `response_format.json_schema` (`strict: true`) |
| 이미지 블록 | `source.base64` | `image_url` (data URI) |
| 프롬프트 캐싱 | `cache_control` 로 지점 지정 | 1,024 토큰 초과 시 자동 |
| 이미지 토큰 | (가로 × 세로) / 750 | `detail: low` 로 85 토큰 고정 |

이미지는 512px 로 줄여 보낸다. 담배갑·술병·약통 식별에는 충분하고 입력이 절반 이하로
줄어든다.

## 설계 원칙

**오탐이 미탐보다 비싸다.** 정상 상품을 막으면 판매자가 이탈하지만, 금지 품목이 한 번
통과해도 신고·사후 탐지로 잡는다. 그래서 `confidence` 가 낮으면 **금지와 정상 양쪽 다**
검토 필요로 내린다. 금지만 내리면 미탐이 그대로 통과한다.

| 판정 | 임계값 | 미만이면 |
| --- | --- | --- |
| 금지 | `confidence ≥ 0.85` | 검토 필요 |
| 정상 | `confidence ≥ 0.70` | 검토 필요 |

**AI 장애가 상품 등록을 막지 않는다.** 외부 API 는 언제든 죽는다. 실패하면 등록을
거부하는 대신 검토 필요로 보류하고 관리자·재시도로 넘긴다. 이미지 한 장이 깨져도
마찬가지로 건너뛰고 나머지로 판정한다.

**같은 내용을 다시 검수하지 않는다.** 상품명·설명·이미지의 해시(`content_hash`)를
저장해 두고, 가격만 바꾼 수정은 LLM 을 부르지 않는다.

## 금칙어 사전

`config/banned_keywords.yaml` — 9개 카테고리, 변형 전개 후 60개 패턴.

```
담배 · 주류 · 의약품 · 마약류 · 무기류 · 개인정보 · 위조품 · 동물 · 기타
```

운영 중 추가는 이 파일만 고친다. 코드 수정도 재배포도 필요 없다.

## 알려진 특성

- **실제 API 호출은 아직 검증하지 않았다.** 응답 파싱·거부 처리·재시도 경로는 가짜
  클라이언트로만 확인했다. 키를 넣고 한 번 돌려봐야 한다.
- 사전은 초기 60개 패턴이라 커버리지가 넓지 않다. 1차에서 놓친 것은 2차가 받는
  구조이므로 치명적이지는 않지만, 그만큼 LLM 호출이 늘어난다.
- 임계값 0.85 / 0.70 은 근거 없는 초기값이다. 관리자 판정 로그가 쌓이면 조정한다.

## 앞으로

| 시점 | 할 일 |
| --- | --- |
| API 키 투입 후 | 실호출 검증. 프롬프트·임계값 1차 조정 |
| `POST /internal/moderation/review` | HTTP 계층 추가. 백엔드 연동 |
| 관리자 판정 로그 축적 후 | 임계값 재설정. 오탐 사례를 프롬프트에 반영 |
