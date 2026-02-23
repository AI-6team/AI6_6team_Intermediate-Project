# BidFlow RAG 성능 최적화 실험 계획서 (Experiment Plan)

본 문서는 BidFlow 시스템의 RAG 파이프라인을 **정량 측정 → 원인 분리 → 최적 조합 채택**까지 닫는 실험 가이드입니다.
“기능 구현”을 넘어 “품질 고도화(Production-grade) 의사결정”을 목표로 합니다.

---

## 0. 실험 전제 확인 (반드시 기입)

> **미기입 시 실험 시작 금지**: 아래 전제값은 이후 모든 실험의 해석/의사결정을 고정합니다.

### 0.1 Error Analysis (실패 원인 1순위 체크)

* [ ] **A. 검색 실패**(Retrieval Miss / Context Recall 부족)
* [ ] **B. 추출 실패**(정보는 있으나 Slot Omission / Faithfulness 문제)
* 현재 관측 결과(근거/예시 2~3개): ____________________________________________

### 0.2 Document Structure (문서 특성 기록)

* RFP 평균 길이: _______ pages
* 표(Table) 비중: [ ] 상  [ ] 중  [ ] 하
* 표가 깨질 때 치명 슬롯 예시(있다면): _______________________________________

### 0.3 Latency Budget (제약조건)

* 허용 p95 응답시간: _______ 초
* (선택) 1요청당 비용 상한(토큰/원): _______________________________________

---

## 1. 실험 목표 및 성공 기준 (Selection Rule)

실험의 목적은 “점수 올리기”가 아니라 **최종 채택 조합을 결정하는 것**입니다.
따라서 다음 의사결정 규칙을 먼저 고정합니다.

### 1.1 Selection Rule (채택 알고리즘)

* **Gate (통과 기준)**: **Slot Omission Rate < 10%**

  * 필수 항목 누락이 1개라도 발생하면 해당 설정은 즉시 탈락(배포 후보 제외)
* **Primary Objective (핵심 목표)**: **Decision Accuracy 최대화**

  * 권장: Accuracy를 **FN/FP로 분해**하여 치명 오류(업무 리스크)가 큰 쪽을 우선 최소화
* **Tie-breaker (동점 처리)**: Accuracy가 **±1% 이내**면

  1. **Latency(p95)** 우위
  2. **Cost(요청당 토큰/비용)** 우위
* **Tail Risk (대형 사고 방지)**: **Worst 10% Faithfulness** 별도 관리

  * 평균이 좋아도 “큰 사고”가 나는 조합은 배포 후보에서 제외

---

## 2. 실험 환경 (Setup)

### 2.1 Golden Dataset (탐색용 vs 최종검증용)

* **위치**: `data/eval/gold/`
* **탐색용(Exploration)**: 30~50

  * 목적: 빠른 루프로 후보 조합을 좁힘
* **최종검증용(Final Validation)**: **50~100 권장**

  * 목적: 최종 채택 조합을 통계적으로 더 안정적으로 검증

**분포(필수 명시)**

* 공고 유형: IT / 건설 / 용역 등
* 문서 복잡도: 표 비중 상/중/하
* 판정 난이도: 명확 / 모호

**정답 구성(필수)**

* 정답 슬롯 값뿐 아니라, 문서 내 **근거 Quote/Span(원문 위치/발췌)**를 함께 저장

  * Faithfulness/Extraction 검증을 “사람이 재확인 가능한 형태”로 고정

---

### 2.2 모델/인프라 기본값

* Baseline LLM: `gpt-5-mini`
* Baseline Chunking: RecursiveCharacter(아래 Baseline 명시)
* Baseline Retrieval: Hybrid (BM25 + Vector) 기본 결합
* Baseline Prompt: Zero-shot

---

### 2.3 Metrics (Core vs Diagnostic)

**Core Metrics (의사결정용)**

* Slot Omission Rate (필수 / 권장 슬롯 분리 권장)
* Decision Accuracy (**FN/FP 분리 권장**)
* Faithfulness (평균 + **Worst 10%**)
* Latency: p50 / p95
* Cost: 요청당 토큰 사용량(입력/출력) + API 비용 추정

**Diagnostic Metrics (원인 분리용)**

* Context Recall / Context Precision
* Answer Relevance
* **MRR**, 필요 시 NDCG

---

## 3. 실험 규약 (Protocol & Reproducibility)

### 3.1 Phase-based Approach (Frozen 전략)

실험 간 간섭을 최소화하기 위해, 단계별로 “최선 조합을 고정”합니다.

| Phase       | 변경 변수(IV)                | 고정 변수(CV)                       | 핵심 지표                                     |
| ----------- | ------------------------ | ------------------------------- | ----------------------------------------- |
| **Phase A** | Exp-1: Chunking(+Table)  | Retrieval(Base), Prompt(Base)   | Context Recall (+Token/Cost)              |
| **Phase B** | Exp-2: Retrieval         | **Best Chunk**, Prompt(Base)    | Context Precision, **MRR**, Latency       |
| **Phase C** | Exp-3: Prompt/Extraction | **Best Chunk + Best Retrieval** | Slot Omission, Faithfulness, Decision Acc |

---

### 3.2 반복/통계 규약 (현실적 대안 포함)

* 각 조건은 **N회 반복(권장 3회)** 후 평균±표준편차 기록
* (선택) 가능하면 부트스트랩 CI
* **대안 규약(권장)**: 3회 반복 중 **2회 이상 개선**이면 “개선”으로 판정

  * (예: Core metric 기준으로 Gate 통과 + Accuracy 개선이 2회 이상이면 후보 유지)

---

### 3.3 Early Stopping Rule (탐색용에서만 적용)

비용/시간 절약을 위해, **탐색용(Exploration)** 평가 중 아래 조건을 만족하면 조기 탈락시킬 수 있습니다.

**Early Stopping**

* 탐색용 첫 **10개 샘플(랜덤, seed 고정)**에서

  * **Slot Omission Rate > 20%**이면 해당 조건 **즉시 탈락**(전체 평가 생략)

> 주의: 최종검증용(Final Validation)에서는 Early stopping 적용하지 않습니다.

---

### 3.4 재현성(Reproducibility)

* `seed=42` 고정(샘플링/평가 파이프라인 전반)
* 결과 리포트에 **Git commit hash**, 실행 일시, 환경 버전 기록
* 동일 exp_name 재실행 시 동일 seed/동일 입력이면 동일 결과가 나오도록 캐시 정책 권장

---

## 4. 실험 상세 설계 (Exp-0 ~ Exp-3)

## Exp-0: Baseline 측정

현재 성능을 기준점으로 고정합니다.

* **조건**

  * Chunk: 500, Overlap: 50
  * Hybrid 결합: 기본(예: alpha=0.5)
  * Top-K: 10(권장)
  * Model: `gpt-5-mini`
  * Prompt: Zero-shot
* **산출물**

  * `data/experiments/baseline_report.json`

---

## Exp-1: 문맥 최적화 (Chunking + Table Preservation)

### 4.1 Semantic Chunking 포함 여부 (실험 범위 확정)

* Semantic chunking 구현 가능 여부를 먼저 판정하고, 본 사이클 포함/제외를 명시합니다.

  * [ ] 포함(구현 가능/비용 합리적)  → Exp-1 변수에 포함
  * [ ] 제외(구현 불가/비용 과다)  → 이번 사이클 제외(다음 사이클 후보)

### 가설

RFP는 “조항/계층 구조”가 핵심이며, 표가 많은 경우 **표가 깨지는 순간 성능이 급락**한다.
따라서 길이 기반보다 **섹션/조항 기반 분할 + 표 보존**이 유리하다.

### Independent Variables (IV)

1. Chunking 방식

* RecursiveCharacter: chunk_size = 500 / 1000 / 2000
* Header 기반 분할(권장): MarkdownHeaderSplitter (`#`, `##`)
* (선택) Semantic Chunking: 포함 체크 시에만 수행

2. **Table 보존 전략(필수 변수)**

* A. 표를 Markdown/CSV 형태로 **한 덩어리 보존**
* B. 표를 텍스트로 풀되 **행 단위 유지**
* C. 기존 방식(표 깨짐 허용) — 비교군

### Control Variables (CV)

* Retrieval: Baseline 고정
* Prompt: Baseline 고정
* Top-K: Baseline 고정

### Evaluation

* Context Recall (+ Context Precision 보조)
* Token Usage / Cost
* Latency 변화(가능하면 함께 기록)

### 산출물

* `exp1_chunk_<variant>_report.json`

---

## Exp-2: 검색 전략 최적화 (Retrieval Tuning)

### 가설

특수 용어/약어/고유명사가 많은 입찰 문서는 임베딩만으로 매칭력이 떨어질 수 있어, **BM25 비중 및 결합 전략**이 성능을 좌우한다.
Top-K가 커질수록 노이즈가 늘어 **reranker가 ROI가 좋은 레버**가 된다.

### Independent Variables (IV)

1. Hybrid 레버

* Alpha: 0.3 / 0.5 / 0.7 / **0.9**
* 결합 방식: Linear blend(alpha) vs **RRF**

2. Top-K

* 5 / 10 / 20

3. Reranker 도입

* on/off (예: bge-reranker 계열)

### Control Variables (CV)

* Chunking: **Best Chunk 고정**
* Prompt: Baseline 고정

### Evaluation

* Context Precision / Context Recall
* **MRR(필수)**, 필요 시 NDCG
* Latency(p50/p95), Token/Cost
* Reranker 추가 지연(ms) 및 성능 상승폭으로 ROI 평가

### 산출물

* `exp2_retrieval_<variant>_report.json`

---

## Exp-3: 프롬프트/추출 최적화 (Prompt & Extraction Control)

### 가설

Slot Omission 감소의 핵심은 CoT가 아니라
**구조화 출력 + 누락 규약 + 검증 단계**로 실패 형태를 통제하는 것이다.

### Independent Variables (IV)

1. **2-step Workflow (권장)**

* Step 1: 추출(JSON)
* Step 2: 검증(근거 인용/스팬 매칭 확인) 후 수정/확정

2. Structured Output

* **JSON mode / 스키마 강제 on/off** (필수 포함)

3. Few-shot

* zero-shot vs 1~3개 예시

4. CoT

* 옵션 실험군으로만 둠(환각 리스크)

### 2-step 비용/지연 추정(사전 명시)

* 예상 비용 증가율: **+80~100%** (LLM 호출 2회 기반)
* 예상 지연 증가: **LLM 1회 추가 시간**만큼

> 실제 측정치로 리포트에 반드시 기록하여 ROI 판단에 사용합니다.

### Control Variables (CV)

* Chunking: Best Chunk 고정
* Retrieval: Best Retrieval 고정

### Evaluation

* Slot Omission Rate(필수) + 필수/권장 슬롯 분리
* Faithfulness(평균 + Worst 10%)
* Decision Accuracy(FN/FP 분리 권장)
* Latency/Cost(2-step의 증가분 포함)

### Error Taxonomy (로그 분리 규약)

* **Retrieval Miss**: 정답 evidence span이 top-k context에 없음
* **Extraction Miss**: evidence span은 존재하나 JSON에 누락/오추출
* **Hallucination**: evidence 없이 생성(또는 evidence와 불일치)

### 산출물

* `exp3_prompt_<variant>_report.json`

---

## 5. 파이프라인 고도화 & 확장 실험

### Exp-4: 파이프라인 고도화 (V3 통합 - Pipeline Enhancement)

Exp-1~3의 결과를 종합하여, **생성(Generation) 품질 병목**을 해결하기 위한 통합 파이프라인 고도화 실험입니다.

- **배경**: V2 실험 결과 Context Recall(0.73) 대비 Keyword Accuracy(0.49)가 크게 낮아, 검색은 충분하나 생성 단계에서 정보 손실이 발생
- **가설**: 문서 메타데이터 주입 + Verbatim 프롬프트 + LLM Reranker + Query Decomposition + Relevance Grading을 누적 적용하면 생성 품질이 유의미하게 개선된다
- **방법**: Ablation Study (누적 적용)
  - A_baseline: V2 Best Config (zero_shot_ko)
  - B_metadata: + Document Metadata (Fact Sheet)
  - C_verbatim: + Verbatim Extraction Prompt
  - D_reranker: + LLM Reranker (30→15)
  - E_decompose: + Query Decomposition
  - F_full: + Relevance Grading (Full Pipeline)
- **데이터**: 기존 Golden Testset (30문항)
- **지표**: Context Recall, Faithfulness, ResponseRelevancy, Keyword Accuracy, Latency, Cost
- **산출물**: `exp04_report.json`

---

### Exp-4-v3: 매칭 품질 직접 개선 (Matching Quality Improvement)

EXP04-v2에서 아키텍처 복잡화가 top-k 오염만 유발한다는 것이 밝혀짐.
**철학 전환**: "후보를 다양하게" → "후보 품질을 직접 올리고, 원인 분기를 진단으로 확정"

- **Phase 0 (진단)**: Oracle Recall(인덱스 존재 확인) + Recall@15/30/50/100 다점 분석 + (표/본문)×(숫자/리스트/서술) 2D 분해
- **Phase 1 (Reranker, 1순위)**: Cross-encoder reranker (bge-reranker-v2-m3), 후보 풀 30/50/100 → top-15, rerank-only vs rerank+diversity
- **Phase 2 (Embedding, 2순위, 조건부)**: multilingual-e5-large vs baseline, RRF 중심 비교 + alpha 소규모 grid
- **Phase 3 (BM25 토크나이저, 3순위, 조건부)**: Kiwi/Okt 형태소 분석, BM25 단독→Hybrid→rerank 후 3단 기록
- **병렬 개선**: Answer Normalizer (숫자/퍼센트/연차 표현 표준화)로 KW_Acc 평가 정확도 보정
- **산출물**: `exp04v3_diagnostic.json`, `exp04v3_report.json`, `exp04v3_results.csv`

---

### Exp-5: 타겟 품질 개선 (Targeted Quality Improvement)

EXP04-v3의 진단 결과를 기반으로, **실패 유형별 맞춤 최적화**를 수행하는 실험입니다.

- **배경**: EXP04-v3에서 KW_Acc=0.741, CR=0.900, Faithfulness=0.922 달성. 그러나 Oracle Recall(0.917)과 Recall@15(0.703) 사이 0.214 갭이 존재하며, 5개 질문이 지속적으로 실패
- **목표**: KW_Acc >= 0.80, CR >= 0.93, Faithfulness >= 0.95
- **방법**: 4-Phase 구조
  - **Phase 0 (포렌식 진단)**: 30개 질문별 6항목 파이프라인 추적 → 실패 분류 (INDEX_MISSING / RETRIEVAL_MISS / RERANK_MISS / GENERATION_MISS)
  - **Phase 1 (프롬프트 최적화)**: P0_verbatim(재현) / P1_extract(표/숫자 추출 규칙) / P2_evidence(근거-우선 + 최종답 포맷)
  - **Phase 2 (리랭커 동적 컷)**: R0_fixed15(기존 고정) / R1_elbow(점수 기반 동적 컷오프, min=5/max=15)
  - **Phase 3 (통합)**: Phase 1 최선 프롬프트 + Phase 2 최선 전략 결합
- **고정 설정**: OpenAI embedding, alpha=0.7, pool=50, bge-reranker-v2-m3
- **데이터**: 기존 Golden Testset (30문항)
- **산출물**: `exp05_forensic.json`, `exp05_report.json`
- **노트북**: `notebooks/exp05_targeted_quality.ipynb`

---

Exp-1~3은 "단일 문서" 기준 최적화, Exp-4는 "파이프라인 고도화", Exp-5는 "타겟 품질 개선", Exp-6~7은 "일반화/확장 성능" 검증입니다.

### Exp-6: 일반화 검증 (Generalization)
- **목적**: 고려대 RFP 외 다른 도메인(건설, 정부R&D 등)에서도 Best Config가 통하는지 확인
- **데이터**: 2~3개의 추가 PDF 확보 (가능하면 포맷/표 구조가 다른 문서)
- **방법**: Exp-1~5에서 선정된 Best Config로 검증용 문항(각 10~15개) 테스트
- **산출물**: `exp06_generalization_report.json`

### Exp-7: 포맷 확장 (HWP Support)
- **목적**: HWP(한글) 문서 파싱 및 RAG 성능 검증
- **데이터**: HWP 형식의 공공 RFP 1~2개
- **방법**: PDFParser 대신 `HWPParser` 사용, 동일 파이프라인 태움
- **산출물**: `exp07_hwp_report.json`

---

## 6. 실험 실행 가이드 (How-to: Notebook-driven Strategy)

**보고서 작성 및 시각화**의 용이성을 위해, CLI 스크립트 대신 **Jupyter Notebook**을 메인 실험 도구로 사용합니다.

1.  **Refactoring**: 각 모듈(`Loader`, `Retriever`)이 노트북에서 파라미터를 동적으로 변경하며 실행될 수 있도록 클래스화/함수화되어야 합니다.
2.  **Organization**: 각 실험 단계별로 노트북을 분리하여 관리합니다.
    *   `notebooks/exp01_chunking_optimization.ipynb`
    *   `notebooks/exp02_retrieval_strategy.ipynb`
    *   `notebooks/exp03_prompt_engineering.ipynb`
    *   `notebooks/exp04_pipeline_enhancement.ipynb`
    *   `notebooks/exp04v2_retrieval_architecture.ipynb`
    *   `notebooks/exp04v3_matching_quality.ipynb`
    *   `notebooks/exp05_targeted_quality.ipynb`
    *   `notebooks/exp06_generalization_verification.ipynb`
    *   `notebooks/exp07_hwp_format_verification.ipynb`
3.  **Visualization**: 실험 결과 JSON을 즉시 로드하여 `pandas` 테이블 및 `matplotlib` 차트로 시각화합니다. 이 결과물은 그대로 최종 보고서 및 발표 자료에 사용됩니다.

---

## 7. 리포팅 표준 (산출물 포맷 통일)

### 7.1 결과 저장

* 로컬 JSON: `data/experiments/<exp_name>_report.json`
* (옵션) MLflow: run_name = exp_name

### 7.2 최소 리포트 필드(권장)

* meta: exp_name, timestamp, git_commit, seed, env
* config: chunking(table 포함), retrieval(alpha/rrf/topk/rerank), prompt(json/fewshot/2step)
* metrics_core: slot_omission, decision_acc(+FN/FP), faithfulness_mean/worst10, latency_p50/p95, token_in/out, cost_est
* metrics_diagnostic: context_recall/precision, answer_relevance, mrr, ndcg(optional)
* error_analysis: retrieval_miss/extraction_miss/hallucination + 대표 실패 사례

### 7.3 fail_cases_topN 권장치

* **N=5~10 권장**
* 가능하면 유형별(Retrieval/Extraction/Hallucination)로 Top-N 추출

---

## 8. 일정 (의존성/버퍼/병렬 범위 포함)

### 8.1 업데이트 일정

* **1~2주차**: Golden 구축(탐색용 30~50 → 최종검증용 50+) + evidence span 규약 확정
* **3주차**: Phase A(Chunking+Table) → Phase B(Retrieval)
* **4주차**: Phase C(Prompt) + 최종 통합 테스트 + 배포 후보 선정

### 8.2 의존성 및 대응 방안

* Phase B는 Phase A의 Best Chunk가 필요(순차 의존)
* **버퍼**: Golden 구축 2주 내 **2~3일 버퍼** 확보 권장
* **병렬 가능 범위(Phase A 진행 중에도 가능)**

  * Exp-2/Exp-3 실행 코드(argparse), 리포트 JSON 스키마, MLflow 로깅/대시보드 준비
  * Prompt 템플릿(JSON schema, N/A/null 규약) 설계

---

## 9. 최종 체크리스트

* [ ] 섹션 0 전제값 기입 완료
* [ ] Selection Rule 명시 및 자동 채택/탈락 가능
* [ ] Golden에 Quote/Span 저장
* [ ] Phase A/B/C Frozen 전략 준수
* [ ] 비용/지연(p95)/토큰 기록
* [ ] Error Taxonomy로 원인 분리 로그 확보
* [ ] Early stopping은 탐색용에서만 사용

---