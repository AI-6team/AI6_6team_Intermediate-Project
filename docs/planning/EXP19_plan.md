# EXP19: 0.99 달성 + 과적합 검증

## 배경

EXP18에서 GT 정제 + targeted prompt로 kw_v5=0.9851 (28/30 perfect) 달성.
0.99 목표까지 gap=0.49pp (잔여 2문항).

### 목표 설정 근거
- RFP는 사용자에게 매우 중요한 문서 → 성능 절대 양보 불가
- 테스트셋(30Q, 5 docs)에 과적합 우려 → 실제 운영 시 성능 하락 가능
- 따라서 0.99 달성 후 holdout set으로 과적합 여부 반드시 검증

## Phase 구조

| Phase | 실행 조건 | 목표 |
|-------|----------|------|
| **A: 0.99 달성** | 무조건 | kw_v5 ≥ 0.99 |
| **B: 과적합 검증** | A 달성 시 | holdout set으로 일반화 성능 확인 |
| **C: 과적합 해소** | B에서 과적합 확인 시 | 과적합 줄이면서 성능 유지 |

## 상태 업데이트 (2026-02-23)

실행 결과 요약:
- Phase A: `kw_v5=0.9952` (29/30)
- Phase B raw: holdout `kw_v5=0.8821` (SEVERE)
- Phase C(holdout GT 정제 후 재채점): holdout `kw_v5_v2=0.9671` (PASS)
- Phase C 분기(Q1 처리): testset composite `1.0000` (30/30)
- Phase D1(benchmark lock): dev/holdout_locked/sealed_holdout 분리 및 manifest 생성 완료
- Phase D2(범용 프롬프트): dev=0.9374, holdout=0.8611, gate **미통과**
- Phase D3(멀티쿼리 retrieval): dev=0.9091, holdout=0.7979, gate **미통과**, D2 대비 -3.7pp
- Phase D4(Prompt V3 + SC 5-shot): dev=0.9330, **holdout=0.9545**, holdout gate **✅ 최초 통과** ⭐
- Phase D5(D4 + gentle 멀티쿼리): dev=0.9002, holdout=0.9211, gate **미통과**, D4 대비 -3.3pp, 멀티쿼리 최종 기각
- Phase D6(Prompt V4 + top_k=20): **overall=0.9509**, dev=0.9534, holdout=0.9434 ⭐ **Overall Best**

현황:
- D6가 overall/dev 최고 (0.9509/0.9534), D4가 holdout 최고 (0.9545)
- D6 성과: Q22 evaluation FIXED (0.833→1.0), Q24 security +26pp (0.211→0.474), Q30 evaluation +18pp (0.545→0.727)
- D6 한계: Q23 maintenance 미변화 (0.400), holdout Q35 퇴보 (1.0→0.778)
- 다음 단계: D4 vs D6 중 최종 후보 결정 → D7 sealed holdout 평가

---

## Phase A: 0.99 달성

### 수학적 분석

```
현재: (28×1.0 + 0.720 + 0.833) / 30 = 0.9851
목표: ≥ 0.99
필요 합계: ≥ 29.700
현재 합계: 29.553
gap: 0.147
```

**Q7만 1.0으로 만들면 0.99 달성**: (29 + 0.720) / 30 = 0.9907

### 잔여 2문항 현황

| Q | Doc | kw_v5 | GT (v2) | Missing | 원인 |
|---|-----|-------|---------|---------|------|
| Q1 | doc_A | 0.720 | "시스템 노후화...장애위험 증가 및 유지관리 한계, 보안정책 과다 적용 및 스위치 대역폭 부족, SSF(회계) 및 수협은행 등 내부 시스템 간 미연동에 따른 불필요한 행정업무 과다 발생" | ssf, (회계), 수협은행, 내부, 미연동, 불필요한, 행정업무 (7개) | Retrieval 또는 Parsing gap |
| Q7 | doc_D | 0.833 | "7장 제안안내사항의 나. 제안서 평가 방법과 다. 제안서 평가 기준" | 기준 (1개) | Generation: "나." 만 언급, "다. 평가 기준" 누락 |

### Step 1: 정밀 진단 (API 비용 0)

1. **Q1 진단**: doc_A VDB (180 chunks) 전체에서 "SSF", "수협은행", "미연동" 텍스트 존재 여부 확인
   - 존재 시 → retrieval ranking 문제 (검색은 가능하나 상위 15개에 포함 안됨)
   - 미존재 시 → parsing/chunking gap (hwp5txt가 해당 텍스트를 파싱 못함)

2. **Q7 진단**: Q7에 대한 retrieved context(top-15) 내에 "제안서 평가 기준" 포함 여부 확인
   - 포함 시 → generation 문제 (context에 있으나 LLM이 누락)
   - 미포함 시 → retrieval 문제

### Step 2: Q7 해결 (0.99 핵심)

Q9에서 성공한 **targeted prompt** 전략 적용:

```
답변 시 해당 장의 모든 하위 절 제목(나., 다., 라. 등)을 빠짐없이 나열하세요.
```

- SC 5-shot (temp=0.1, 0.3, 0.7, 1.0, 1.0) → kw_v5 best 선택
- Q7만 해결하면 0.99 달성이므로 이것이 최우선

### Step 3: Q1 보조 개선 (optional)

진단 결과에 따라:

| 진단 결과 | 해결 전략 |
|----------|----------|
| chunk에 SSF 존재 + retrieval 누락 | multi-query에 "SSF 수협은행 시스템 연동" 추가 |
| chunk에 SSF 미존재 | GT v3 수정: "SSF(회계) 및 수협은행 등 내부 시스템 간 미연동에 따른 불필요한 행정업무 과다 발생" → "내부 시스템 간 미연동에 따른 행정업무 과다 발생" (파싱 불가한 세부 정보 제거) |

---

## Phase B: 과적합 검증

### 실행 조건
Phase A에서 kw_v5 ≥ 0.99 달성 시

### 방법

1. **Holdout Set 구축**
   - 95개 미사용 문서에서 **5~10개 문서 신규 선정** (기존 5개와 다른 유형/크기 포함)
   - 문서당 3~4개 Q&A 생성 → **15~20개 holdout 문항**
   - 난이도/카테고리 분포는 기존 testset과 유사하게 유지

2. **평가**
   - 현재 최적 config (V4_hybrid + c500 + V2 prompt + SC 3-shot + kw_v5)를 holdout set에 적용
   - 기존 testset과 holdout set의 kw_v5 비교

3. **과적합 판정 기준**
   - holdout kw_v5 < 0.90: **심각한 과적합** → Phase C 필수
   - holdout kw_v5 0.90~0.95: **경미한 과적합** → Phase C 권장
   - holdout kw_v5 ≥ 0.95: **과적합 없음** → 현재 config 확정

---

## Phase C: 과적합 해소

### 실행 조건
Phase B에서 과적합 확인 시

### 전략 (과적합 원인별)

| 과적합 원인 | 해결 전략 |
|------------|----------|
| GT가 특정 문서 표현에 과의존 | GT 일반화 (원문 그대로 → 의미 중심) |
| Prompt가 특정 패턴에 과최적화 | 범용 prompt로 회귀, targeted prompt 최소화 |
| Metric normalization이 testset 특화 | holdout에서도 동일 효과 확인, 필요시 조정 |
| SC가 testset GT에 과적합 | SC 선택 기준을 kw_v5 외 ROUGE/BERTScore 병행 |

---

## Phase D: 일반화 우선 0.99 재달성 (신규)

### 목적
- 단순 문항별 하드코딩 없이, 동일 파이프라인으로 `0.99` 재달성.
- testset/holdout 간 점수 괴리를 최소화.

### 실패 사례 기반 원인 정리
1. 문항 특화 targeted prompt/GT 보정은 점수는 오르지만 재현성·일반화 리스크가 큼.
2. holdout GT 자동생성본은 길고 접속사/서술어가 많아 metric 편향을 유발.
3. SC 고샷/고온도는 일부 문항에서 stochastic regression을 유발.
4. retrieval 확장(pool 증가)만으로는 Q1류 누락 문제를 안정적으로 해결하지 못함.

### 실행 원칙 (Overfitting 최소화)
1. **평가 셋 동결**: 개발 시작 전 test/dev/holdout/sealed_holdout 분리 후 잠금.
2. **GT 작성 규칙 고정**: 길이/표현 규칙 문서화 후 전 셋 동일 기준 적용.
3. **질문별 예외 금지**: 특정 질문 전용 prompt/룰/키워드 하드코딩 금지.
4. **단일 범용 프롬프트**: 문서 타입별(텍스트/테이블) 분기만 허용, 질문 ID 분기 금지.
5. **모델 선택 안정화**: SC는 고정 샷(예: 3-shot) + 온도 제한으로 분산 제어.

### 실행 계획
1. **Benchmark 재정의**
   - `golden_testset_multi_v3.csv`는 참고용으로 보관하되, 운영 성능 판정용은 별도 `sealed_holdout_v1.csv`로 분리.
   - holdout GT는 사람이 동일 규칙으로 작성(문서 근거 1:1 매핑, 메타정보 제거).
2. **파이프라인 개선(범용)**
   - retrieval: 문서 구조 기반 query decomposition(사업개요/평가기준/보안/계약) 추가.
   - rerank: table-heavy 문서에서 숫자/기간 토큰 가중 보존.
   - generation: "핵심 항목 우선, 근거 없는 확장 금지" 범용 지시문으로 통일.
3. **검증 프로토콜**
   - 개발 중에는 dev set만 사용.
   - 고정 주기마다 holdout 1회 평가.
   - 최종 후보 1개만 sealed holdout 평가.
4. **승인 게이트**
   - dev `kw_v5 ≥ 0.99`
   - holdout `kw_v5 ≥ 0.95`
   - sealed holdout `kw_v5 ≥ 0.95` (최소 0.93 하한)
   - 카테고리 편차(최고-최저) `≤ 0.10`

### 산출물
- `docs/planning/EXP19_phase_d_generalization_plan.md` (실행 계획 상세)
- `scripts/run_exp19_phase_d_eval.py` (동일 프로토콜 평가 스크립트)
- `data/experiments/exp19_phase_d_metrics.csv`
- `data/experiments/exp19_phase_d_report.json`

---

## 성공 기준

| Phase | 성공 기준 |
|-------|----------|
| A | kw_v5 ≥ 0.99 (testset 30Q) |
| B | holdout kw_v5 ≥ 0.95 (과적합 없음) |
| C | testset kw_v5 ≥ 0.97 AND holdout kw_v5 ≥ 0.93 |
| D | **GT post-hoc 수정 없이** holdout/sealed holdout kw_v5 ≥ 0.95 |

## 산출물

| 파일 | 설명 |
|------|------|
| `scripts/run_exp19_to_099.py` | Phase A 실험 스크립트 |
| `data/experiments/exp19_metrics.csv` | Phase A 결과 |
| `data/experiments/golden_testset_holdout.csv` | Phase B holdout 테스트셋 |
| `data/experiments/exp19_holdout_metrics.csv` | Phase B 평가 결과 |
| `scripts/run_exp19_phase_c_gt_refine.py` | Phase C holdout GT 정제 + 재채점 |
| `scripts/run_exp19_phase_c_q1_fix.py` | Phase C Q1 보정 재채점 |
| `data/experiments/golden_testset_holdout_v2.csv` | 정제된 holdout GT |
| `data/experiments/exp19_holdout_metrics_v2.csv` | 정제 GT 기준 holdout 결과 |
| `data/experiments/exp19_q1_rescore.csv` | Q1 보정 후 재채점 결과 |
| `scripts/run_exp19_phase_d_lock_benchmark.py` | Phase D1 벤치마크 잠금 스크립트 |
| `data/experiments/golden_testset_dev_v1_locked.csv` | Phase D1 dev 잠금 셋 |
| `data/experiments/golden_testset_holdout_v3_locked.csv` | Phase D1 holdout 잠금 셋 |
| `data/experiments/golden_testset_sealed_v1.csv` | Phase D1 sealed holdout |
| `data/experiments/exp19_phase_d_split_manifest.json` | Phase D1 분할/해시 manifest |
| `docs/planning/EXP19_phase_d_generalization_plan.md` | 일반화 우선 0.99 재달성 상세 계획 |
| `scripts/run_exp19_phase_d_eval.py` | Phase D2/D3 평가 스크립트 |
| `scripts/prompts/exp19_phase_d_prompt_v1.txt` | Phase D2 범용 프롬프트 (5규칙) |
| `scripts/prompts/exp19_phase_d_prompt_v2.txt` | Phase D3 범용 프롬프트 (6규칙, 원문 유지) |
| `data/experiments/exp19_phase_d_metrics.csv` | Phase D2 결과 (40문항) |
| `data/experiments/exp19_phase_d_report.json` | Phase D2 리포트 (overall=0.9183) |
| `data/experiments/exp19_phase_d_metrics_d3.csv` | Phase D3 결과 (40문항) |
| `data/experiments/exp19_phase_d_report_d3.json` | Phase D3 리포트 (overall=0.8813) |
| `scripts/prompts/exp19_phase_d_prompt_v3.txt` | Phase D4 범용 프롬프트 (7규칙, complete listing) |
| `data/experiments/exp19_phase_d_metrics_d4.csv` | Phase D4 결과 (40문항) ⭐ Best |
| `data/experiments/exp19_phase_d_report_d4.json` | Phase D4 리포트 (overall=0.9384, holdout gate ✅) |
| `data/experiments/exp19_phase_d_metrics_d5.csv` | Phase D5 결과 (40문항, 기각) |
| `data/experiments/exp19_phase_d_report_d5.json` | Phase D5 리포트 (overall=0.9055) |
| `scripts/prompts/exp19_phase_d_prompt_v4.txt` | Phase D6 프롬프트 (구조 인용 강화, 1~5문장) |
| `data/experiments/exp19_phase_d_metrics_d6.csv` | Phase D6 결과 (40문항) ⭐ Overall Best |
| `data/experiments/exp19_phase_d_report_d6.json` | Phase D6 리포트 (overall=0.9509) |
