# HANDOFF v2: BidFlow RAG 최적화 — 다음 단계 실험 계획서

**Date**: 2026-02-21
**Author**: 3-feedback 종합 분석 기반
**Status**: Phase A 착수 대기
**Previous**: `system-prompt-extraction/HANDOFF.md` (EXP01~05 기록)

---

## 0. 이 문서를 읽는 에이전트/사람에게

이 문서는 **BidFlow RAG 파이프라인 최적화 실험의 다음 단계 계획서**입니다.
EXP01~09까지 완료된 상태에서, 3개의 독립적 피드백을 종합하여 최적 행동 계획을 도출했습니다.
**토큰/세션이 끊겨도 이 문서만 읽으면 바로 이어서 진행할 수 있도록** 작성되었습니다.

### 읽기 순서
1. **섹션 1** (현재 상태 요약) → 빠르게 맥락 파악
2. **섹션 2** (문제 진단) → 왜 이 계획이 필요한지
3. **섹션 3** (통합 행동 계획) → **바로 실행할 내용**
4. **섹션 4~7** (각 Phase 상세) → 해당 Phase 진행 시 참고
5. **섹션 8** (파일 참조) → 코드/데이터 위치

---

## 1. 현재 상태 요약 (EXP01~09)

### 1.1 실험 이력과 최고 성능

| 실험 | 핵심 변경 | KW_Acc | CR | Faithfulness | 비고 |
|------|-----------|--------|-----|-------------|------|
| EXP01 | chunk=500, layout | - | 0.733 | - | V1 baseline |
| EXP01-v2 | T+M (Table+Metadata) | - | **0.790** | - | +3.1% 돌파 |
| EXP02 | alpha=0.5, top_k=15 | - | 0.767 | - | Hybrid search |
| EXP03 | zero_shot_ko prompt | 0.492 | 0.733 | **0.963** | Korean prompt 승리 |
| EXP04 | +Metadata +Verbatim | 0.664 | 0.700 | 0.829 | Ablation study |
| **EXP04-v3** | **Reranker@50** | **0.741** | **0.900** | **0.922** | **단일문서 BEST** |
| EXP05 | Extract prompt + Elbow cut | 0.706 | 0.800 | 0.836 | 미세 조정 |
| EXP06 | Normalization + 3-run avg | 0.724 | - | 0.894 | 안정화 |
| EXP07 | Table-aware 평가 | 0.748 | 0.867 | 0.871 | table gap 확인 |
| EXP08 | Corpus EDA (100문서) | - | - | - | 분포 분석 |
| **EXP09** | **일반화 검증** | **0.267** | **0.332** | **0.690** | **일반화 실패** |

### 1.2 현재 최적 설정 (단일 문서 기준)

```yaml
# EXP04-v3 기준 Best Config (단일 PDF: 고려대학교)
embedding: text-embedding-3-small (OpenAI)
gen_model: gpt-5-mini
reranker: BAAI/bge-reranker-v2-m3
chunk_size: 500
chunk_overlap: 50
parsing: T+M (text + markdown tables)
alpha: 0.7  # 70% dense, 30% BM25
reranker_pool: 50
top_k: 15
prompt: zero_shot_ko + verbatim directive + fact_sheet
normalize_v2: true  # EXP06 확정
```

### 1.3 코퍼스 현황 (EXP08 기준)

- **총 파일**: 100건 (96 HWP + 4 PDF)
- **파싱 성공**: 97건 (hwp5html: 75, hwp5txt: 20, fallback: 1, pdfplumber: 4)
- **파싱 실패**: 3건 (hwp5html timeout)
- **테이블 없는 문서**: 22건 (22%)
- **평균 테이블 수**: 96.8개/문서 (중앙값 103)
- **테이블 복잡도**: simple 25%, medium 54%, complex 21%

### 1.4 Golden Testset 현황

- **현재**: `data/experiments/golden_testset.csv` — **30문항, 단일 PDF**
- **출처**: `고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf`
- **카테고리**: basic, schedule, qualification, evaluation, contract, general, budget, procurement, technical, compliance, security, data, management, maintenance (11개)
- **난이도**: easy 13, medium 12, hard 5

---

## 2. 문제 진단: 왜 다음 실험이 필요한가

### 2.1 핵심 문제 — EXP09 일반화 실패

| 메트릭 | 단일문서 (EXP04-v3) | 100문서 코퍼스 (EXP09) | 격차 |
|--------|---------------------|----------------------|------|
| KW_Acc | 0.741 | 0.267 | **-64%** |
| CR | 0.900 | 0.332 | **-63%** |
| Faithfulness | 0.922 | 0.690 | **-25%** |

### 2.2 EXP09 결과의 신뢰도 문제 (3개 피드백 공통 지적)

EXP09 Phase 2 수치를 액면 그대로 믿으면 안 되는 이유:

1. **평가 파이프라인이 운영 체인과 다름**
   - 노트북 내 명시: "이 셀은 실험용 자동 생성기입니다(운영 체인과 1:1 동일하지 않음)"
   - `exp09_generalization_verification.ipynb` line 701

2. **ops 지표가 dry-run proxy**
   - latency_sec, timeout_rate 등이 실측이 아닌 추정값
   - `exp09_phase2_metrics.csv`의 latency가 28.0, 30.5, 33.5 등 공식화된 값

3. **문서당 1문항, 1-run 실행**
   - 통계적 분산 추정 불가
   - config당 3-run 필요 (EXP06에서 이미 검증한 원칙)

4. **quality metrics가 skeleton에서 채워진 것일 가능성**
   - kw_v2, faithfulness가 0.0 또는 1.0만 반복 (이진값)
   - 연속 스코어가 아닌 규칙 기반 매핑 의심

### 2.3 근본 원인 분석 (3개 피드백 종합)

| 원인 | 설명 | 피드백 출처 |
|------|------|------------|
| **단일문서 testset 편향** | 30문항이 모두 고려대 PDF 1건에서 생성 → 코퍼스 97~99 백분위 극단값 | 전원 합의 |
| **테이블 성능 격차** | text KW 0.780 vs table KW 0.566 = 21.4%p 차이 (EXP06 기준) | 피드백 1, 2 |
| **HWP 파싱 불안정** | hwp5html timeout 3건, 파싱 방법 혼재 (html 75 + txt 20) | 피드백 2 |
| **평가 방법론 결함** | EXP09가 proxy/dry-run 기반으로 실행됨 | 피드백 1, 2 |
| **이미지 정보 미활용** | 70/100 문서에 정보성 이미지(>=50KB) 존재하나 완전 무시 | 피드백 2 |

---

## 3. 통합 행동 계획 (Phase A → D)

### 3.0 설계 원칙

세 피드백의 우선순위가 순환 의존 관계이므로 **가장 작은 단위부터** 해결:

```
┌──────────────────────────────────────────────────────────┐
│  Phase A: 다문서 Golden Testset 구축 (1~2일)              │ ← 모든 평가의 기반  ✅
│  ↓                                                        │
│  Phase B: EXP09 Phase 2 실측 재실행 (2~3일)               │ ← 실제 일반화 성능 확인  ✅
│  ↓                                                        │
│  Phase CE: 파서 전환 + Table Gap 해소 (3~5일)             │ ← 기존 C+E 통합, 최대 ROI
│  ↓                                                        │
│  Phase D: 정리 + Baseline 고정 (1일)                      │ ← 최종 설정 확정
└──────────────────────────────────────────────────────────┘
```

**Phase C+E 통합 근거 (Phase B 결과 기반)**:
- hwp5txt 파서가 테이블 문서에서 56~74% 텍스트 손실 → 파서 전환이 필수 선행 조건
- retrieval 파라미터(alpha, pool, top_k) 미세조정 효과 최대 0.57pp → 파서 자체 개선이 ROI 최대
- Phase C의 col_path/정규화/rewrite는 **파서가 테이블을 추출한 후에야** 의미 있음
- 따라서 Phase E(파서 전환)를 선행하고 Phase C(테이블 최적화)를 통합 실행

### 3.1 Phase 의존성

| Phase | 선행 조건 | 산출물 | 후속 Phase에 미치는 영향 |
|-------|----------|--------|------------------------|
| **A** ✅ | 없음 | `golden_testset_multi.csv` | B, CE, D 모두 이 testset 사용 |
| **B** ✅ | A 완료 | `exp10b_report.json` | 일반화 baseline 수치 확정 (hwp5txt) |
| **CE** | B 완료 | `exp10ce_report.json` | 파서 전환 + table 성능 개선 수치 |
| **D** | CE 완료 | 최종 config + HANDOFF_v3.md | 프로덕션 배포 기준 |

### 3.2 Phase별 노트북 명명 규칙

```
notebooks/exp10a_multi_doc_testset.ipynb        # Phase A  ✅
notebooks/exp10b_generalization_rerun.ipynb      # Phase B  ✅
scripts/run_exp10b.py                            # Phase B  ✅
notebooks/exp10ce_parser_upgrade_table_gap.ipynb  # Phase CE
scripts/run_exp10ce.py                           # Phase CE
```

---

## 4. Phase A: 다문서 Golden Testset 구축

### 4.1 목표

- 코퍼스 대표성을 갖는 **5건 문서 × 6문항 = 30문항** 다문서 testset 생성
- 기존 30문항 (단일문서)과 합쳐 총 60문항 평가 체계 확립

### 4.2 문서 선정 기준 (층화 샘플링)

`exp08_eda_results.csv`에서 아래 5가지 유형을 각 1건씩 선정:

| 유형 | 선정 기준 | 후보 문서 예시 |
|------|----------|---------------|
| **A. text_only** | n_tables == 0, text_len > 30000 | `(사）한국대학스포츠협의회_KUSF...` (0 tables, hwp5txt) |
| **B. table_simple** | 30 < n_tables < 80, complexity=mostly_simple | EDA에서 중앙값 부근의 simple 비중 높은 문서 |
| **C. table_complex** | n_tables > 120, complex 비중 > 30% | `(사)벤처기업협회_...` (226 tables) 또는 유사 |
| **D. mixed** | 50 < n_tables < 120, n_images > 10 | 텍스트+테이블+이미지 균형 문서 |
| **E. hwp_representative** | hwp5html 정상 변환, file_size_kb 중앙값(~944KB) 부근 | 코퍼스 중간 규모 대표 HWP |

### 4.3 문항 생성 규칙

각 문서당 6문항:
- **easy 2문항**: 사업명, 사업기간, 예산 등 (명시적 단답)
- **medium 2문항**: 입찰조건, 기술요구사항 등 (문서 내 여러 곳 참조 필요)
- **hard 2문항**: 평가기준 세부 배점, 보안 요구사항 등 (테이블 내 정보 또는 여러 조건 결합)

### 4.4 필수 포함 필드

```csv
question,ground_truth,category,difficulty,source_page,source_doc,evidence_span
```

- `source_doc`: 출처 문서 파일명 (다문서 식별용)
- `evidence_span`: 정답 근거 원문 (Faithfulness 검증용, experiment_plan.md 권장사항)

### 4.5 실행 방법

```python
# exp10a_multi_doc_testset.ipynb 셀 구조
# Cell 0: 환경 설정 + EDA 결과 로드
# Cell 1: 층화 샘플링으로 5건 문서 자동 선정
# Cell 2: 각 문서 hwp5html 변환 + 텍스트 추출
# Cell 3: LLM 기반 초안 질문 생성 (GPT-5-mini)
#          - 각 문서의 텍스트를 입력하여 카테고리별 질문-답변 쌍 생성
#          - 프롬프트: "이 RFP 문서에서 {category}에 해당하는 질문과 정답을 생성하세요"
# Cell 4: 수동 검증 + 수정 (사람이 확인해야 하는 단계)
# Cell 5: golden_testset_multi.csv 저장
# Cell 6: 기존 golden_testset.csv와 합쳐 golden_testset_combined.csv 생성
```

### 4.6 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/golden_testset_multi.csv` | 다문서 30문항 |
| `data/experiments/golden_testset_combined.csv` | 기존 30 + 신규 30 = 60문항 |
| `notebooks/exp10a_multi_doc_testset.ipynb` | 생성 과정 기록 |

### 4.7 주의사항

- **LLM 생성 후 반드시 사람이 검증**: 자동 생성된 Q&A는 hallucination 가능
- **evidence_span 필수 기입**: 문서 내 정답 근거 위치를 명시해야 이후 실험에서 검증 가능
- **기존 testset과 난이도 분포 맞추기**: easy/medium/hard 비율 유사하게 유지

### 4.8 완료 조건

- [ ] 5건 문서 선정 완료
- [ ] 문서당 6문항 × 5건 = 30문항 생성
- [ ] 사람이 최소 1회 검증 (Q&A 정확성)
- [ ] evidence_span 전체 기입
- [ ] golden_testset_combined.csv 생성 및 60문항 확인

---

## 5. Phase B: EXP09 Phase 2 실측 재실행

### 5.1 목표

- 실제 운영 RAG 체인으로 A/B/C 라우팅 설정 비교
- dry-run proxy 제거, 실측 metrics 확보
- 다문서 testset(Phase A)으로 일반화 성능 정량화

### 5.2 EXP09와의 차이점

| 항목 | EXP09 (기존) | Phase B (신규) |
|------|-------------|---------------|
| 평가 파이프라인 | 실험용 자동 생성기 | **운영 RAG 체인** (rag_chain.py) |
| ops 지표 | dry-run proxy | **실측** (실제 latency, timeout) |
| 문서당 문항 수 | 1문항 | **6문항** (Phase A testset) |
| 반복 횟수 | 1-run | **3-run** 평균 |
| testset | 자동 생성 | **수동 검증된** golden_testset |
| 평가 문서 | 100건 전체 | **5건 대표 문서** (Phase A 선정) |

### 5.3 비교 설정 (A/B/C)

```python
configs = {
    "A_single_pipeline": {
        # 현재 최적 설정 그대로 (EXP04-v3 best)
        "routing": None,
        "alpha": 0.7,
        "reranker_pool": 50,
        "top_k": 15,
        "prompt": "zero_shot_ko_verbatim",
    },
    "B_rule_single_route": {
        # 문서 유형별 단일 라우팅
        "routing": "rule_based_single",
        # text_only → alpha=0.5 (BM25 비중 높임)
        # table_heavy → alpha=0.7 + table-aware prompt
        # mixed → alpha=0.7 기본
    },
    "C_rule_multi_route": {
        # 불확실 구간 다중 라우팅
        "routing": "rule_based_multi",
        # confidence < 0.45 → 2개 route 병행 실행
        # confidence >= 0.70 → 단일 route
    },
}
```

### 5.4 실행 방법

```python
# exp10b_generalization_rerun.ipynb 셀 구조
# Cell 0: 환경 설정 + golden_testset_combined.csv 로드
# Cell 1: 5건 대표 문서 인덱싱 (ChromaDB, 문서별 별도 collection)
#          - 각 문서: hwp5html → chunk(500) → embed → store
# Cell 2: Config A 실행 (3-run × 5문서 × 6문항 = 90 evaluations)
# Cell 3: Config B 실행 (동일 규모)
# Cell 4: Config C 실행 (동일 규모)
# Cell 5: RAGAS 평가 (kw_v2, faithfulness, context_recall)
# Cell 6: 결과 집계 (overall_mean, macro_group_mean, worst_group)
# Cell 7: 단일문서 testset(기존 30문항)도 병행 실행 → 비교 테이블
# Cell 8: 해석 및 결론
```

### 5.5 핵심 KPI 및 판정 기준

```yaml
# EXP09의 decision_rule을 현실적으로 조정
quality_floor:
  kw_v2: 0.50        # 0.75에서 하향 (다문서 첫 측정이므로)
  faithfulness: 0.80  # 0.90에서 하향
  context_recall: 0.60  # 0.88에서 하향

worst_group_floor:
  kw_v2: 0.35
  faithfulness: 0.70
  context_recall: 0.45

ops_ceiling:
  timeout_rate: 0.10    # 10% 이하
  p95_latency_sec: 120  # 2분 이하
```

### 5.6 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10b_report.json` | 종합 결과 |
| `data/experiments/exp10b_metrics.csv` | 상세 per-doc per-question metrics |
| `data/experiments/exp10b_comparison.csv` | 단일문서 vs 다문서 성능 비교 |

### 5.7 완료 조건

- [ ] 5건 문서 인덱싱 완료 (각 문서별 chunk 수 기록)
- [ ] A/B/C config × 3-run × 30문항 = 270 evaluations 완료
- [ ] 기존 단일문서 30문항도 재실행 (비교용)
- [ ] quality_floor 기준 pass/fail 판정
- [ ] worst_group 분석 완료

---

## 6. Phase CE: 파서 전환 + Table Gap 해소 (기존 Phase C + E 통합)

### 6.0 통합 배경

**Phase B 결과에 의한 계획 변경**:

Phase B에서 hwp5txt 파서로 5건 문서를 인덱싱한 결과, 테이블 문서에서 56~74%의 텍스트가 손실됨이 확인됨.
기존 Phase C의 3-axis(col_path, 정규화, query rewrite)는 **파서가 테이블을 추출한 후에야** 의미 있으므로,
파서 전환(기존 Phase E)을 선행하고 테이블 최적화(기존 Phase C)를 통합 실행하는 것으로 계획 변경.

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| Phase C | hwp5txt 위에서 retrieval 최적화 | ~~삭제~~ → Phase CE로 통합 |
| Phase E | D 완료 후 조건부 착수 | ~~삭제~~ → Phase CE로 통합 |
| Phase CE | (신규) | 파서 전환 + col_path + 정규화 |

### 6.1 목표

1. **파서 전환**: hwp5txt → hwp5html + BeautifulSoup (테이블 구조 보존)
2. **table_kw_v2**: Phase B baseline(0.71) → **0.80 이상** 달성
3. **text-table gap**: 21pp → **10pp 이하** 축소
4. **overall_kw_v2**: Phase B baseline(0.75) → **0.80 이상** 달성

### 6.2 파서 전환 설계

#### 현재 문제 (hwp5txt)

| 문서 | hwp5txt 길이 | hwp5html 길이 | 손실률 | 원인 |
|------|-------------|-------------|--------|------|
| doc_A (text_only) | 35,940 | 35,940 | 0% | 테이블 없음 |
| doc_B (table_simple) | 17,330 | 43,573 | **60%** | 75개 테이블 손실 |
| doc_C (table_complex) | 23,884 | 92,239 | **74%** | 246개 테이블 손실 |
| doc_D (mixed) | 31,879 | 72,399 | **56%** | 116개 테이블 + 23개 이미지 손실 |
| doc_E (hwp_representative) | 21,219 | 76,272 | **72%** | 139개 테이블 손실 |

#### 신규 파서: `hwp_html_parser.py`

```python
# src/bidflow/parsing/hwp_html_parser.py
class HWPHtmlParser:
    """
    hwp5html CLI → BeautifulSoup → 텍스트/테이블 분리 추출
    Fallback: hwp5txt (text_only 문서 또는 hwp5html 실패 시)
    """
    def parse(self, file_path: str) -> Tuple[str, List[TableBlock]]:
        # 1. hwp5html --html 실행 → HTML 문자열
        # 2. BeautifulSoup로 파싱
        # 3. 텍스트 추출: <p>, <span> 등 (테이블 외)
        # 4. 테이블 추출: <table> → TableBlock 리스트
        # 5. 테이블별 col_path 구축 (EXP07 방식)
        return text_content, table_blocks
```

#### 출력 스키마

```python
@dataclass
class TableBlock:
    table_id: str           # "t_001"
    page_hint: int          # 대략적 위치 (HTML 섹션 기반)
    headers: List[str]      # col_path 결합 헤더 ["사업개요/총사업비", ...]
    rows: List[Dict]        # [{"사업개요/총사업비": "112.7억원", ...}]
    caption: str            # 테이블 캡션 (있으면)
    raw_html: str           # 원본 HTML (디버깅용)
```

### 6.3 테이블 최적화 설계 (기존 Phase C 3-axis)

#### Axis 1: col_path 보존 (EXP07 방식 적용)

```python
# 다중 헤더 → 경로화
# <th rowspan=2>사업개요</th><th>총사업비</th> → col_path: "사업개요/총사업비"

def build_col_paths(header_rows):
    """다중 헤더 → col_path 배열 표준화 (EXP07 _build_col_paths 이식)"""
    # 병합 셀 상속, NULL 셀 처리
```

#### Axis 2: 숫자/기간 정규화

```python
# 금액: "11,270,000,000원" → "112.7억원"
# 비율: "30%" == "30퍼센트"
# 기간: "24개월" == "2년"
# 날짜: "2024.12.31" == "2024년 12월 31일"
```

#### Axis 3: 테이블 청킹 전략

```python
# 방식 1: table-level 문서 (schema + 전체 데이터)
# 방식 2: row-level 문서 ("col_path: value" 포맷)
# 방식 3: 하이브리드 (schema 1건 + row N건)
# → EXP07에서 효과 검증된 방식 3(하이브리드) 기본 채택
```

### 6.4 실험 매트릭스

| Config | 파서 | 테이블 청킹 | col_path | 정규화 | 설명 |
|--------|------|-----------|---------|--------|------|
| V0_hwp5txt | hwp5txt | N/A | OFF | OFF | Phase B baseline 재현 |
| V1_html_basic | **hwp5html** | text_only | OFF | OFF | 파서 전환 효과 분리 |
| V2_html_table | **hwp5html** | **하이브리드** | **ON** | OFF | + 테이블 구조 보존 |
| V3_html_full | **hwp5html** | **하이브리드** | **ON** | **ON** | 통합 효과 |

### 6.5 평가 방법

- **testset**: `golden_testset_multi.csv` (30문항, 5문서)
- **비교 기준**: Phase B의 Config A baseline (kw_v2=0.7503)
- **메트릭**:
  - overall_kw_v2, table_kw_v2 (doc_B/C/D/E), text_kw_v2 (doc_A)
  - text-table gap
  - parse_time, indexing_time, retrieval_time
  - 문서별 청크 수 변화
- **반복**: 3-run 평균
- **성공 기준**:
  - table_kw_v2 >= 0.80
  - text-table gap <= 10pp
  - overall_kw_v2 >= 0.80
  - parse timeout rate <= 5%
  - p95 latency <= 120s

### 6.6 구현 파일

```
src/bidflow/parsing/hwp_html_parser.py     # 신규: hwp5html + BeautifulSoup 파서
src/bidflow/parsing/table_chunker.py       # 신규: 테이블 청킹 (schema + row-level)
scripts/run_exp10ce.py                     # 실험 스크립트
notebooks/exp10ce_parser_upgrade_table_gap.ipynb  # 실험 노트북
data/experiments/exp10ce_report.json       # 종합 결과
data/experiments/exp10ce_metrics.csv       # 상세 메트릭
data/exp10ce/vectordb_*/                   # Per-config ChromaDB
```

### 6.7 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10ce_report.json` | V0~V3 비교 결과, 채택 config 근거 |
| `data/experiments/exp10ce_metrics.csv` | 문항별 상세 (kw_v2, timing) |
| `src/bidflow/parsing/hwp_html_parser.py` | 프로덕션 파서 (채택 시) |
| `src/bidflow/parsing/table_chunker.py` | 테이블 청킹 모듈 (채택 시) |

### 6.8 완료 조건

- [ ] hwp_html_parser.py 구현 + 5건 문서 파싱 성공
- [ ] V0~V3 4개 config × 3-run 완료
- [ ] table_kw_v2 >= 0.80 달성 config 존재 여부 확인
- [ ] text-table gap 측정 (목표: 10pp 이하)
- [ ] 최선 config 선정 및 근거 기록
- [ ] 프로덕션 파서 교체 결정 (채택/보류)

---

## 7. Phase D: 정리 + Baseline 고정

### 7.1 목표

- Phase A, B, CE 결과 종합하여 최종 baseline 확정
- 프로덕션 config 업데이트 (파서 전환 반영)
- HWP timeout 스모크 테스트 (3건)

### 7.2 할 일 목록

1. **normalize_v2 전체 반영**: EXP06에서 확인된 정규화 효과를 모든 실험 baseline에 고정
2. **최종 config 확정**: Phase B(baseline)와 Phase CE(파서+테이블) 최선 조합을 `configs/prod.yaml`에 반영
3. **HWP timeout 스모크 테스트**:
   - timeout 3건 (`exp08_eda_results.csv`의 error='hwp5html_timeout') 재시도
   - timeout 180s → 300s 상향 테스트
   - fallback 자동 전환 (hwp5html → hwp5txt) 로직 확인
4. **HANDOFF_v3.md 작성**: Phase A~CE 결과를 포함한 최종 핸드오프 문서

### 7.3 prod.yaml 업데이트 대상

```yaml
# 현재 (configs/prod.yaml)
parsing:
  chunk_size: 500
  chunk_overlap: 100   # ← 실험 best는 50 (불일치!)
retrieval:
  top_k: 10            # ← 실험 best는 15 (불일치!)
  rerank: true
model:
  embedding: text-embedding-3-large  # ← 실험은 3-small 사용 (불일치!)

# 업데이트 필요 항목:
# - chunk_overlap: 100 → 50
# - top_k: 10 → 15
# - embedding: 3-large → 3-small (또는 실험과 일치시키기)
# - alpha: 추가 (현재 없음, 0.7 추가)
# - reranker_pool: 추가 (현재 없음, 50 추가)
```

**주의**: prod.yaml과 실험 설정 간 불일치 3건 발견됨. Phase D에서 반드시 동기화할 것.

### 7.4 산출물

| 파일 | 설명 |
|------|------|
| `configs/prod.yaml` | 업데이트된 프로덕션 설정 |
| `docs/planning/HANDOFF_v3_final.md` | 최종 핸드오프 문서 |
| `data/experiments/exp10d_hwp_smoke.json` | HWP timeout 스모크 테스트 결과 |

---

## 8. 파일 참조 (Key Files Reference)

### 8.1 실험 데이터

| 파일 | 설명 |
|------|------|
| `data/experiments/golden_testset.csv` | 기존 30문항 (단일 PDF) |
| `data/experiments/exp08_eda_results.csv` | 100문서 EDA 결과 (문서 선정 기준) |
| `data/experiments/exp10b_report.json` | Phase B 결과 (3 config × 3 run, kw_v2) ✅ |
| `data/experiments/exp10b_metrics.csv` | Phase B 270건 상세 ✅ |
| `data/experiments/exp09_report.json` | EXP09 결과 (신뢰도 주의) |
| `data/experiments/exp09_phase2_metrics.csv` | EXP09 상세 (dry-run 포함) |
| `data/experiments/exp04v3_report.json` | 단일문서 최고 성능 |
| `data/experiments/exp07_report.json` | 테이블 성능 격차 데이터 |

### 8.2 소스 코드 (파이프라인)

| 파일 | 역할 |
|------|------|
| `src/bidflow/retrieval/rag_chain.py` | RAG 체인 (프롬프트 + 생성) |
| `src/bidflow/retrieval/hybrid_search.py` | Hybrid 검색 (BM25 + Vector, RRF) |
| `src/bidflow/retrieval/rerank.py` | Cross-encoder reranker |
| `src/bidflow/extraction/pipeline.py` | 추출 파이프라인 |
| `src/bidflow/extraction/chains.py` | LLM 체인 정의 |
| `src/bidflow/ingest/pdf_parser.py` | PDF 파싱 |
| `src/bidflow/ingest/loader.py` | 문서 로더 (청킹 포함) |
| `src/bidflow/ingest/storage.py` | ChromaDB 벡터 저장소 |
| `src/bidflow/parsing/preprocessor.py` | 텍스트 전처리/정규화 |
| `src/bidflow/parsing/hwp_parser.py` | HWP 파서 (hwp5txt, extract_tables 미구현) |
| `src/bidflow/parsing/hwp_html_parser.py` | **Phase CE 신규**: hwp5html + BeautifulSoup |
| `src/bidflow/parsing/table_chunker.py` | **Phase CE 신규**: 테이블 청킹 (col_path) |
| `scripts/run_exp10b.py` | Phase B 실험 스크립트 ✅ |

### 8.3 설정

| 파일 | 설명 |
|------|------|
| `configs/prod.yaml` | 프로덕션 설정 (**실험과 불일치 3건 있음, Phase D에서 수정**) |
| `configs/dev.yaml` | 개발 설정 |

### 8.4 원본 데이터

| 경로 | 설명 |
|------|------|
| `data/raw/고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf` | 단일문서 실험 대상 PDF |
| `data/raw/files/` | HWP 코퍼스 (96건) |
| `data/raw/files/고려대학교_차세대 포털·학사 정보시스템 구축사업.pdf` | PDF 원본 (files 내 복사본) |

### 8.5 기존 노트북

| 파일 | 설명 |
|------|------|
| `notebooks/exp01_chunking_optimization.ipynb` ~ `exp09_generalization_verification.ipynb` | EXP01~09 |
| `notebooks/exp10_hwp_format_verification.ipynb` | 원래 EXP10 (미실행, Phase A~D로 대체) |

---

## 9. 알려진 함정 (Known Gotchas)

1. **Kernel restart**: 모든 변수 소실됨. 각 Phase별 중간 결과를 JSON으로 반드시 저장할 것
2. **RAGAS 평가 속도**: config당 400~600초. 3-run이면 20~30분/config. Phase B 전체는 ~2시간
3. **ChromaDB lock files**: 크래시 후 `data/chroma_exp/` 내 lock 파일이 남을 수 있음. `robust_rmtree()` 사용
4. **Windows 인코딩**: 터미널에서 한글 깨짐 가능. Jupyter/JSON은 UTF-8로 정상
5. **HWP 변환 메모리**: 큰 HWP (>10MB)는 hwp5html이 메모리를 많이 사용. timeout 가능
6. **prod.yaml 불일치**: 현재 prod.yaml과 실험 최적 설정이 3곳 다름 (chunk_overlap, top_k, embedding). Phase D 전까지 수정하지 말 것 (실험 일관성 유지)
7. **EXP09 결과 인용 금지**: EXP09의 kw_v2/CR/faithfulness 수치를 "일반화 성능"으로 보고서에 인용하면 안 됨. Phase B 실측 결과를 사용할 것

---

## 10. 의사결정 트리 (Decision Tree) — 업데이트

```
Phase A 완료 후: ✅
├── testset 품질 OK (사람 검증 통과) → Phase B 진행 ✅

Phase B 완료 후: ✅
├── quality_floor 통과 (kw_v2=0.75 >= 0.50) ✅
├── worst_group 통과 (0.65 >= 0.35) ✅
├── text-table gap = 21pp (> 10pp) → 파서 전환 필요 확인
└── → Phase CE 진행 (파서 전환 + table gap 해소)

Phase CE 완료 후:
├── table_kw_v2 >= 0.80 + text-table gap <= 10pp
│   ├── 파서 전환 효과 확인 → hwp_html_parser 프로덕션 채택
│   └── Phase D 진행 (baseline 고정)
│
├── table_kw_v2 >= 0.80 but gap > 10pp
│   → 파서 채택 + retrieval 추가 튜닝 (query rewrite 등)
│
└── table_kw_v2 < 0.80
    ├── V1(html_basic)만으로도 큰 개선 → 파서 전환만 채택, 청킹 전략 재검토
    └── V1도 개선 미미 → hwp5html 출력 품질 분석, 대안 파서 검토

Phase D 완료 후:
├── 목표 달성 + 운영 안정성 양호
│   → HANDOFF_v3.md 작성 → 프로덕션 배포
│
└── 운영 안정성 미달 (timeout 등)
    → fallback 전략 강화 (hwp5html timeout → hwp5txt fallback)
```

---

## 11. 빠른 시작 가이드 (Quick Start)

### 새 에이전트가 이어받을 때:

```
1. 이 문서 전체를 읽는다
2. 현재 Phase 확인:
   - golden_testset_multi.csv 존재? → Phase A 완료 ✅
   - exp10b_report.json 존재? → Phase B 완료 ✅
   - exp10ce_report.json 존재? → Phase CE 완료
   - exp10d_hwp_smoke.json 존재? → Phase D 완료
3. 해당 Phase의 완료 조건 체크리스트를 확인한다
4. HISTORY_v2_execution.md에서 진행 상태를 확인한다
5. 미완료 항목부터 이어서 진행한다
```

### 환경 설정:

```bash
# 필수
cd E:/Codeit/AI6_6team_Intermediate-Project/bidflow
# .env에 OPENAI_API_KEY 있는지 확인
# Python 환경: langchain, chromadb, ragas, sentence-transformers, pdfplumber, pyhwp

# 노트북 실행
jupyter notebook notebooks/exp10a_multi_doc_testset.ipynb
```

---

## 12. 부록: 3개 피드백 원문 요약

### 피드백 A (Claude 분석)
- **1순위**: 다문서 Golden Testset 구축
- **2순위**: 인제스천 파이프라인 안정화
- **3순위**: HWP 청킹 검증

### 피드백 B (외부 리뷰어 1)
- **1순위**: EXP09.5 평가 프로토콜 고정 (실제 운영 RAG 체인 사용)
- **2순위**: Table gap 해소 (EXP10 목적 전환)
- **3순위**: HWP 포맷은 스모크 테스트로 축소

### 피드백 C (외부 리뷰어 2)
- **1순위**: EXP09 Phase 2 실측 완료
- **2순위**: EXP07 테이블 성능 확인 (table_kw_v2 0.566 → 0.700)
- **3순위**: 다문서 평가셋 구축

### 통합 결론
→ Phase A(testset) ✅ → Phase B(실측) ✅ → Phase CE(파서+table) → Phase D(고정)
→ "측정 도구 → 실측 → 최대 병목 해결" 순서로 순환 의존 해소
→ **Phase B 결과**: C+E 통합 필요성 확인 (hwp5txt 56~74% 손실)

---

## 13. ~~Phase E (Optional)~~ → Phase CE로 통합됨

> **2026-02-21 업데이트**: Phase B 실험 결과, hwp5txt 파서의 테이블 손실률(56~74%)이
> Phase C(테이블 최적화)의 효과를 근본적으로 제한함을 확인.
> 따라서 Phase E(파서 전환)와 Phase C(테이블 최적화)를 **Phase CE**로 통합.
> 상세 계획은 **섹션 6 (Phase CE)** 참조.
>
> 기존 Phase E의 핵심 개념(col_path, ROI gate, V0/V1 비교)은 모두 Phase CE에 반영됨.

---

*이 문서는 2026-02-21에 작성되었습니다. Phase 진행 시 결과를 이 문서에 추가하거나 HANDOFF_v3.md를 새로 작성하세요.*
