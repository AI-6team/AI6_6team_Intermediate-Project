# EXP19 Phase D: Generalization-First 0.99 재달성 계획

## 목표
- 질문별 하드코딩 없이 범용 파이프라인으로 성능 재달성.
- 평가 기준:
  - dev `kw_v5 ≥ 0.99`
  - holdout `kw_v5 ≥ 0.95`
  - sealed holdout `kw_v5 ≥ 0.95` (최소 0.93)

## 현재 기준선 (2026-02-23)
- testset best composite: `1.0000` (30/30)
- holdout(raw GT): `0.8821`
- holdout(v2 GT): `0.9671`

해석:
- 점수 개선의 큰 부분이 GT 표현 통일 효과.
- 운영 일반화 확정을 위해 GT post-hoc 수정 없는 sealed 평가가 필요.

## 진행 상태
- [x] D1 벤치마크 잠금 완료 (2026-02-23)
- [x] D2 범용 프롬프트 단일화 (2026-02-23) — dev=0.9374, holdout=0.8611, gate ❌
- [x] D3 retrieval 범용 개선 (2026-02-23) — dev=0.9091, holdout=0.7979, gate ❌
- [x] D4 프롬프트 V3 + SC 5-shot (2026-02-23) — dev=0.9330, **holdout=0.9545**, holdout gate ✅ **Holdout Best**
- [x] D5 gentle 쿼리 확장 (2026-02-23) — dev=0.9002, holdout=0.9211, gate ❌, D4 대비 -3.3pp
- [x] D6 Prompt V4 + top_k=20 (2026-02-23) — overall=0.9509, dev=0.9534, holdout=0.9434
- [x] D7 Structure-Aware Retrieval (2026-02-24) — **overall=0.9784**, **dev=0.9914**, holdout=0.9394, ⭐ **Overall+Dev Best, dev gate ✅ 최초 통과**
- [x] D8 Sealed Verification (2026-02-24) — overall=0.9627, dev=0.9854, holdout=0.9434, **sealed=0.9140**, ❌ **전체 gate 미통과** (SC 변동 + sealed compliance hard 약점)

## 실패/리스크 분석 (스크립트 기반)

### 1) 질문 특화 전략 의존 리스크
- `scripts/run_exp19_to_099.py`는 Q7/Q9류 targeted prompt로 점수 상승.
- 동일 방식은 질문 ID 의존도가 높아 새로운 문서/질문으로 확장 시 재현성 저하 가능.

### 2) SC 확장(shot/temperature) 불안정성
- EXP16/17/19 로그에서 SC 고샷이 일부 문항 regression을 유발.
- 따라서 SC는 고정 샷(3-shot) + 제한된 temperature로 관리해야 함.

### 3) retrieval 단순 확장 효과 한계
- pool_size 확대는 노이즈 유입으로 역효과가 발생한 케이스가 존재.
- 단순 top-k 확장보다 query decomposition/structure-aware retrieval이 유효.

### 4) GT 규칙 불일치가 평가 왜곡
- holdout raw GT는 과도하게 길고 접속사/서술어 포함.
- 동일 파이프라인 평가라도 GT 스타일 차이로 점수 편향 발생.

## 실행 원칙 (Overfitting 방지)
1. 질문/문항 ID 기반 분기 금지.
2. prompt는 문서 타입(텍스트 중심/테이블 중심) 수준까지만 분기.
3. GT는 평가 전 확정 후 잠금(중간 수정 금지).
4. 모델/샘플링 설정 고정(재현성 우선).
5. 최종 선택은 단일 지표가 아니라 `kw_v5 + 카테고리 균형` 동시 기준.

## 실행 단계

## D1. 벤치마크 잠금
1. `train/dev/holdout/sealed_holdout` 분리.
2. 모든 GT에 공통 작성 규칙 적용:
   - 1~2문장
   - 핵심 키워드 유지(사업명/금액/기간/기관명)
   - 페이지/장절/불필요 접속어 제거
3. `sealed_holdout`은 개발 중 평가 금지.

산출물:
- `data/experiments/golden_testset_holdout_v3_locked.csv`
- `data/experiments/golden_testset_sealed_v1.csv`

실행 결과 (완료):
- 스크립트: `scripts/run_exp19_phase_d_lock_benchmark.py`
- 추가 산출물:
  - `data/experiments/golden_testset_dev_v1_locked.csv`
  - `data/experiments/exp19_phase_d_split_manifest.json`
- 분할 결과:
  - dev: 30문항
  - holdout_locked: 10문항 (`hold_F~J` 각 2문항)
  - sealed_holdout: 10문항 (`hold_F~J` 각 2문항)
- manifest 해시:
  - holdout_locked: `0f85fddc7f23ad47c0d17ad2a84c8998ca23d822d09ea4928387320552515d68`
  - sealed_holdout: `ab2c889bced0885ce3f8b03b0a261c1dd9e4f76623f51dd15d399a55cc411ba2`

## D2. 범용 프롬프트 단일화
1. targeted prompt 제거.
2. 범용 지시문 1개로 통일:
   - 사실 근거 우선
   - 항목형 질문은 항목명 누락 없이 요약
   - 근거 없는 확장 설명 금지
3. 문서 타입별 힌트만 허용(테이블/숫자 추출 우선 등).

산출물:
- `scripts/prompts/exp19_phase_d_prompt_v1.txt`

실행 결과 (완료):
- overall kw_v5=0.9183, dev=0.9374 (22/30 perfect), holdout=0.8611 (3/10 perfect)
- gate 미통과 (dev<0.99, holdout<0.95)
- 강점: compliance, evaluation, procurement, schedule, security 모두 1.0 (dev)
- 약점: budget(0.810), general(0.778), maintenance(0.500)
- 산출물: `data/experiments/exp19_phase_d_metrics.csv`, `data/experiments/exp19_phase_d_report.json`

## D3. retrieval 범용 개선
1. query decomposition 추가:
   - 질문을 `핵심 엔티티 + 속성(기간/금액/기준)`로 분해
2. table-heavy 문서는 숫자/기간/단위 토큰 보존 가중.
3. rerank 입력에서 노이즈 heading/footer 제거.

산출물:
- `scripts/run_exp19_phase_d_eval.py`
- `data/experiments/exp19_phase_d_metrics.csv`

실행 결과 (완료):
- 프롬프트 v2 + query_expansion=true (최대 4 변형)
- overall kw_v5=0.8813, dev=0.9091 (24/30 perfect), holdout=0.7979 (4/10 perfect)
- gate 미통과 (dev<0.99, holdout<0.95)
- D2 대비: overall -3.7pp, perfect +3개 (28 vs 25)
- 멀티쿼리 효과: 복잡한 질문(보안, 하자담보) +0.4~0.8pp, 단순 사실 질문(예산, 사업명) -0.6~1.0pp 회귀
- retrieval 시간 대폭 개선: mean 1.1s (D2: 6.2s), max 3.0s (D2: 127.3s)
- 산출물: `data/experiments/exp19_phase_d_metrics_d3.csv`, `data/experiments/exp19_phase_d_report_d3.json`

D2 vs D3 비교 결론:
- D2가 overall 성능 우수 (+3.7pp), D3는 perfect 수 우수 (+3개)
- D3는 단순 사실 질문에서 치명적 회귀(0.0점) 발생 → 쿼리 확장이 노이즈 유입
- 다음 단계: 프롬프트 강화 + SC 확장 + gentle 쿼리 확장으로 양쪽 장점 결합

## D4. 프롬프트 V3 + SC 5-shot (generation 강화)

D2/D3 오답 분석 기반 generation 품질 개선:

1. **프롬프트 V3** (7규칙):
   - 규칙2: "원문 표기를 그대로 인용" (v1의 "우선 포함"보다 강력)
   - 규칙3(신규): 단일 사실 질문 → 해당 정보만 간결 답변
   - 규칙4: 항목 나열 시 상위 제목(1., 가., 나.) 포함 필수
   - 규칙5(신규): 장/절 내 모든 하위 항목 제목 빠짐없이 나열
   - 규칙6: 불필요한 접속사(그리고, 또한, 아울러) 명시적 제거
2. **SC 5-shot** (temp=[0.0, 0.1, 0.2, 0.3, 0.5]):
   - temp=0.0(결정적) 추가 → 단순 사실 추출 안정화
   - 3-shot 대비 oracle 선택지 +2 → perfect 확률 증가
   - max temp=0.5 유지 (EXP16의 SC 5-shot 역효과는 고온도 때문)
3. 쿼리 확장: **Off** (D2 baseline retrieval)

기대: doc_E evaluation(0.636→1.0), doc_D evaluation(0.833→1.0) 등 generation 실패 해결.

산출물:
- `scripts/prompts/exp19_phase_d_prompt_v3.txt`
- `data/experiments/exp19_phase_d_metrics_d4.csv`
- `data/experiments/exp19_phase_d_report_d4.json`

실행 결과 (완료 — ⭐ Phase D Best):
- overall kw_v5=0.9384, dev=0.9330 (26/30 perfect), **holdout=0.9545 (7/10 perfect)**
- **holdout gate 최초 통과** (≥0.95): D2~D3에서 불가능했던 일반화 성능 달성
- dev gate 미통과 (0.9330 < 0.99, gap 6.7pp)
- 강점: budget(1.0), compliance(1.0), general(1.0), procurement(1.0), schedule(1.0), technical(1.0)
- 약점: security(0.211), maintenance(0.400), evaluation(0.689) — 메타질문+deep enum 한계
- D2 대비: overall +2.0pp, holdout +9.3pp, perfect 25→33개 (+8)
- 잔여 실패 4개: 메타질문 3개(Q22, Q23, Q30) + deep enum 1개(Q24)

## D5. Gentle 쿼리 확장 (retrieval 보강)

D4 + 원본 쿼리 지배적 gentle 확장으로 retrieval 개선:

1. 쿼리 확장: **On** (variant_weights=[1.0, 0.3, 0.2, 0.2])
   - 원본 쿼리 weight=1.0, 보조 변형 weight=0.2~0.3
   - 원본 top-ranked = 0.0164 vs 보조 top-ranked = 0.0049 (3.3x 차이)
   - D3의 [1.0, 0.9, 0.8, 0.7] 대비 보조 영향력 1/3로 축소
2. 효과:
   - 단순 사실 질문: 원본 랭킹이 지배 → D2급 성능 유지
   - 복잡 도메인 질문: 보조 변형이 누락 chunk 보충 → D3급 개선
3. 리스크: 보조 영향이 너무 작아 D4와 차이 미미할 수 있음

기대: doc_D security(0.211→1.0) 등 retrieval 실패도 해결하면서 D3 회귀 방지.

산출물:
- `data/experiments/exp19_phase_d_metrics_d5.csv`
- `data/experiments/exp19_phase_d_report_d5.json`

실행 결과 (완료 — ❌ 기각):
- overall kw_v5=0.9055, dev=0.9002 (25/30 perfect), holdout=0.9211 (6/10 perfect)
- 두 gate 모두 미통과 (dev<0.99, holdout<0.95)
- D4 대비: overall -3.3pp, perfect 33→31개 (-2)
- 치명적 회귀: Q17 doc_C 최초 구축 시기 (1.0→0.2, 모든 SC shot 동일 실패)
- D4 실패 문항 개선: 0건 (security, maintenance, evaluation 전부 동일)
- 멀티쿼리 최종 판정: 원본 가중치 62.5%에서도 성능 하락 → **한국어 RFP에서 멀티쿼리 근본적 비효과**

## D6. Prompt V4 (구조 인용 강화) + top_k 확장 (generation + retrieval 동시 개선)

D4 잔여 실패 4문항(dev gap 6.7pp) 정밀 분석 기반:

### D4 실패 분류

| 문항 | Score | 실패 유형 | 원인 |
|------|-------|----------|------|
| Q22 (doc_D 평가방법 장 위치) | 0.833 | Generation | "다. 평가 기준" 하위 절 누락 |
| Q23 (doc_D 하자담보 규정 위치) | 0.400 | Retrieval | chunk에 "9장 기타 사항" 헤더 미포함 |
| Q24 (doc_D 보안 세부 항목) | 0.211 | Retrieval | 잘못된 섹션(비공개자료 vs 보안관리) 검색 |
| Q30 (doc_E 평가방식 장 위치) | 0.545 | Generation | 하위 절 번호·제목 미나열 |

### 변경사항

1. **Prompt V4** (V3 대비 변경):
   - 규칙4 강화: "일부만 언급하고 나머지를 생략하지 마세요" 명시적 anti-skip
   - 규칙5 대폭 강화:
     - "위치·구성·내용"으로 트리거 확대 (V3: "내용"만)
     - "정확한 장번호(예: 제7장, Ⅶ장)를 반드시 포함" (V3: 없음)
     - "모든 하위 절/항의 번호와 제목" (V3: "하위 항목 제목")
     - 예시 형식 추가: `"7장 제안안내사항의 나. 평가 방법, 다. 평가 기준"`
   - 규칙6: 1~3문장 → 1~5문장으로 완화 (열거형 답변 허용)

2. **top_k: 15 → 20**:
   - 5개 chunk 추가로 Q24의 누락 섹션 포함 확률 증가
   - Q23의 장 헤더 포함 chunk 확보 가능성
   - 리스크: 노이즈 증가 → SC 5-shot으로 완화

3. 유지: SC 5-shot, pool_size=50, 멀티쿼리 Off

### 기대 효과

| 문항 | D4 | D6 예상 | 개선 요인 |
|------|-----|---------|----------|
| Q22 | 0.833 | 1.000 | Prompt V4 규칙5 강화 (하위 절 전수 나열) |
| Q23 | 0.400 | 0.6~0.8 | top_k=20으로 장 헤더 chunk 추가 확보 |
| Q24 | 0.211 | 0.5~0.8 | top_k=20으로 올바른 보안관리 섹션 확보 |
| Q30 | 0.545 | 0.8~1.0 | Prompt V4 규칙5 강화 + 예시 형식 |

최선 시나리오: dev 0.9330 + 2.0/30 ≈ 0.9997 → dev gate 통과 가능
현실 시나리오: dev 0.9330 + 1.0/30 ≈ 0.9663 → dev gate 미달이나 유의미 개선
최악 시나리오: top_k 증가로 기존 perfect 문항 회귀 → D4와 동등 이하

산출물:
- `scripts/prompts/exp19_phase_d_prompt_v4.txt`
- `data/experiments/exp19_phase_d_metrics_d6.csv`
- `data/experiments/exp19_phase_d_report_d6.json`

### D6 실행 결과 ⭐ Overall Best

**실행**: 2026-02-23 22:33~23:09 (약 37분), 40문항 × SC 5-shot = 200 API calls

| 지표 | D4 | D6 | Delta |
|------|----|----|-------|
| Overall | 0.9384 | **0.9509** | **+1.25pp** |
| Dev | 0.9330 | **0.9534** | **+2.04pp** |
| Dev Perfect | 26/30 | **27/30** | +1 |
| Holdout | **0.9545** | 0.9434 | -1.11pp |
| Holdout Perfect | 7/10 | 7/10 | 0 |
| Dev Gate (≥0.99) | ❌ | ❌ | gap: 4.66pp |
| Holdout Gate (≥0.95) | ✅ | ❌ | gap: 0.66pp |

**타겟 문항 결과**:

| 문항 | D4 | D6 | Delta | 판정 |
|------|----|----|-------|------|
| Q22 evaluation doc_D | 0.833 | **1.000** | +16.7pp | **FIXED** — V4 구조 인용 효과 |
| Q23 maintenance doc_D | 0.400 | 0.400 | 0 | 미변화 — chunk에 장 헤더 부재 (retrieval 한계) |
| Q24 security doc_D | 0.211 | 0.474 | +26.3pp | 개선 — top_k=20 추가 chunk 효과 |
| Q30 evaluation doc_E | 0.545 | 0.727 | +18.2pp | 개선 — V4 규칙5 부분 효과 |

**Holdout 변동**: Q35 (hold_H 공동수급) 1.000→0.778 퇴보, Q36 (hold_H 기능) 0.889→1.000 개선

**판정**: D6는 overall/dev 최고이나 holdout 소폭 퇴보. D4는 holdout 최고. 두 후보 중 최종 선택 필요.

## D7. Structure-Aware Retrieval ⭐ Dev Gate 최초 통과

**전략**: VDB 재구축 없이 런타임에 문서 구조 정보를 context에 주입.
1. **TOC 자동 감지**: VDB 내 목차 chunk를 패턴 매칭으로 식별 (10/10 문서 성공)
2. **Chapter Prefix**: 검색된 chunk에 소속 장 정보 접두사 삽입
3. **Prompt V5**: 규칙8 신설 (목차 기반 위치 답변 강화)
4. top_k=20, SC 5-shot, query_expansion=Off 유지

산출물:
- `scripts/prompts/exp19_phase_d_prompt_v5.txt`
- `data/experiments/exp19_phase_d_metrics_d7.csv`
- `data/experiments/exp19_phase_d_report_d7.json`

실행 결과 (완료 — ⭐ Overall+Dev Best):
- overall kw_v5=**0.9784**, dev=**0.9914** (28/30 perfect), holdout=0.9394 (7/10 perfect)
- **dev gate 최초 통과** (0.9914 ≥ 0.99)
- holdout gate 미통과 (0.9394 < 0.95, gap 1.06pp)
- Q23(하자담보) 0.400→**1.000** FIXED, Q24(보안) 0.474→**1.000** FIXED, Q30(평가) 0.727→0.909 개선
- D6 대비: overall +2.75pp, dev +3.80pp, holdout -0.40pp

## D8. Sealed Holdout Verification ❌ No-Go

**후보**: D7 (dev gate ✅, overall best)
**평가**: 50문항 (dev 30 + holdout_locked 10 + sealed_holdout 10)

산출물:
- `data/experiments/exp19_phase_d_metrics_d8.csv`
- `data/experiments/exp19_phase_d_report_d8.json`

실행 결과 (완료 — ❌ No-Go):
- overall kw_v5=0.9627 (41/50 perfect)
- dev=0.9854 (28/30), holdout=0.9434 (7/10), **sealed=0.9140 (6/10)**
- 전체 gate 미통과: dev<0.99 (SC 변동), holdout<0.95, sealed<0.95
- Sealed 실패 패턴: compliance hard 3건 + technical medium 1건
- SC 변동성: 동일 config D7 run(0.9914) vs D8 run(0.9854) ±0.6pp

## Go/No-Go 기준
- Go:
  - dev `≥ 0.99`
  - holdout `≥ 0.95`
  - sealed holdout `≥ 0.95`
  - 카테고리 편차 `≤ 0.10`
- No-Go:
  - 특정 문항만 개선되고 전체 편차가 커지는 경우
  - holdout 개선 없이 testset만 상승하는 경우

## 즉시 실행 우선순위
1. ~~D1 벤치마크 잠금~~ ✅
2. ~~D2 범용 프롬프트로 baseline 재측정~~ ✅
3. ~~D3 retrieval 개선~~ ✅
4. ~~D4 프롬프트 V3 + SC 5-shot~~ ✅ ⭐ **Best (holdout 0.9545)**
5. ~~D5 gentle 쿼리 확장~~ ✅ ❌ 기각 (-3.3pp)
6. ~~D6 Prompt V4 + top_k=20~~ ✅ overall 0.9509, dev +2pp, holdout -1pp
7. ~~D7 Structure-Aware Retrieval~~ ✅ ⭐ **Overall+Dev Best (0.9784)**, **dev gate 최초 통과 (0.9914)**
8. ~~D8 Sealed Verification~~ ✅ ❌ **No-Go** (sealed=0.9140, 전체 gate 미통과)
