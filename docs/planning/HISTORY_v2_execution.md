# HISTORY: BidFlow EXP10 실행 기록

**참조 계획서**: `HANDOFF_v2_next_experiments.md`
**시작일**: 2026-02-21
**최종 업데이트**: 2026-02-23 (EXP13 완료)

---

## 진행 상태 요약

| Phase | 상태 | 시작일 | 완료일 | 비고 |
|-------|------|--------|--------|------|
| **A: 다문서 Golden Testset** | ✅ 완료 | 2026-02-21 | 2026-02-21 | 30문항 생성, 사람 검증 필요 |
| **B: EXP09 실측 재실행** | ✅ 완료 | 2026-02-21 | 2026-02-21 | 3 config × 3 run × 30Q = 270 evals, 0 errors |
| **CE: 파서 전환 + Table Gap 해소** | ✅ 완료 | 2026-02-21 | 2026-02-21 | 4 config × 3 run × 30Q = 360 evals, V1 best for tables |
| **D: 정리 + Baseline 고정** | ✅ 완료 | 2026-02-21 | 2026-02-21 | V4_hybrid +2.2pp, Gap 0.21→-0.06 |
| **D+: 테이블 필터 실험** | ❌ 실패 | 2026-02-21 | 2026-02-21 | V5 필터링 실패: doc_A 무변화, doc_B -30.6pp |
| **E: Retrieval 재최적화** | ✅ 완료 | 2026-02-22 | 2026-02-22 | c500_pv2 최고 0.8136, +3.9pp vs Phase D |
| **EXP11 (F~J): 종합 최적화** | ✅ 완료 | 2026-02-22 | 2026-02-22 | baseline 미달, kw_v3 정규화만 유효 (+8.3pp) |
| **EXP12: Retrieval 최적화** | ✅ 완료 | 2026-02-22 | 2026-02-22 | multi_query 0.900 (+0.4pp), emb_kure 0.896 (OpenAI 동등, 무료) |
| **EXP13: Contextual Retrieval** | ❌ 부정결과 | 2026-02-23 | 2026-02-23 | 모든 config baseline 미달, ctx_full 0.889 (-0.75pp) |

---

## Phase A: 다문서 Golden Testset 구축 ✅

### A-1. 문서 선정 (층화 샘플링) ✅

EXP08 EDA 결과(`exp08_eda_results.csv`)에서 5가지 유형별 대표 문서 선정:

| 유형 | 선정 기준 | 선정 문서 | n_tables | n_images | text_len | size_kb | extract_method | 상태 |
|------|----------|----------|----------|----------|----------|---------|----------------|------|
| A. text_only | n_tables=0, 최대 text_len | 수협중앙회_수산물사이버직매장 ISMP | 0 | 4 | 35,940 | 557 | hwp5txt | ✅ |
| B. table_simple | 30<tables<80 | 한국교육과정평가원_NCIC 운영 개선 | 75 | 2 | 43,573 | 371 | hwp5html | ✅ |
| C. table_complex | tables>120, 최다급 | 국립중앙의료원_차세대 응급의료 | 246 | 11 | 92,239 | 1,484 | hwp5html | ✅ |
| D. mixed | tables+images 모두 높음 | 한국철도공사_예약발매시스템 ISMP | 116 | 23 | 72,399 | 1,600 | hwp5html | ✅ |
| E. hwp_representative | size_kb≈중앙값(944) | 스포츠윤리센터_LMS 기능개선 | 139 | 5 | 76,272 | 941 | hwp5html | ✅ |

**선정 다양성 확인**:
- 도메인: 수산/이커머스, 교육, 의료, 교통, 스포츠/교육
- 테이블: 0, 75, 116, 139, 246 (전 범위 커버)
- 파싱: hwp5txt 1건 + hwp5html 4건
- 크기: 371~1,600 KB

### A-2. 문서 변환 (HWP → Text) ✅

| 문서 | 변환 방법 | 텍스트 길이 | 테이블 추출 | 상태 |
|------|----------|-----------|-----------|------|
| 수협중앙회 (A) | hwp5txt | 35,940 chars | N/A (text only) | ✅ |
| 한국교육과정평가원 (B) | hwp5html → BeautifulSoup | 43,573 chars | 75 tables | ✅ |
| 국립중앙의료원 (C) | hwp5html → BeautifulSoup | 92,239 chars | 246 tables | ✅ |
| 한국철도공사 (D) | hwp5html → BeautifulSoup | 72,399 chars | 116 tables | ✅ |
| 스포츠윤리센터 (E) | hwp5html → BeautifulSoup | 76,272 chars | 139 tables | ✅ |

변환 텍스트 저장 위치: `data/tmp_testset_gen/doc_{A~E}_text.txt` (각 30,000 chars 캡)

### A-3. 질문 생성 ✅

| 문서 | easy | medium | hard | 합계 | 상태 |
|------|------|--------|------|------|------|
| A. 수협중앙회 | 3 | 2 | 1 | 6 | ✅ |
| B. 한국교육과정평가원 | 2 | 2 | 2 | 6 | ✅ |
| C. 국립중앙의료원 | 2 | 2 | 2 | 6 | ✅ |
| D. 한국철도공사 | 2 | 2 | 2 | 6 | ✅ |
| E. 스포츠윤리센터 | 2 | 2 | 2 | 6 | ✅ |
| **합계** | **11** | **10** | **9** | **30** | ✅ |

**카테고리 분포**: budget(7), technical(6), schedule(6), general(3), procurement(2), compliance(2), evaluation(2), maintenance(1), security(1)

### A-4. Testset 저장 ✅

| 파일 | 문항 수 | 상태 |
|------|---------|------|
| `data/experiments/golden_testset_multi.csv` | 30문항 (5문서) | ✅ |
| `data/experiments/golden_testset_combined.csv` | 60문항 (기존30 + 신규30) | ✅ |

### A 완료 체크리스트
- [x] 5건 문서 선정 완료
- [x] 문서당 6문항 × 5건 = 30문항 생성
- [x] evidence_span 전체 기입
- [x] golden_testset_combined.csv 생성 (60문항)
- [ ] **사람 검증 필요**: Q&A 정확성 확인 (LLM이 아닌 문서 원문에서 직접 추출했으나 재확인 권장)

### A 참고사항

- 난이도 분포: easy 14 / medium 10 / hard 6 (기존 testset: easy 13 / medium 12 / hard 5와 유사)
- easy 문항이 약간 많은 이유: 새 문서의 사업개요 정보가 명확하여 easy 추출이 용이
- hard 문항은 주로 compliance, security, evaluation 카테고리에서 출제
- **주의**: 일부 hard 문항은 목차 기반 답변 (페이지 위치 참조)이므로 Phase B 실행 시 실제 검색 성능과 차이 가능

---

## Phase B: EXP09 Phase 2 실측 재실행 ✅

### B-1. 인덱싱 (HWP → ChromaDB) ✅

5건 문서를 per-document ChromaDB로 인덱싱 (hwp5txt 파서 사용):

| 문서 | 청크 수 | 텍스트 길이(hwp5txt) | Phase A 길이(hwp5html) | 손실률 |
|------|---------|---------------------|----------------------|--------|
| doc_A (text_only) | 87 | 35,940 | 35,940 | 0% |
| doc_B (table_simple) | 48 | 17,330 | 43,573 | **60%** |
| doc_C (table_complex) | 57 | 23,884 | 92,239 | **74%** |
| doc_D (mixed) | 75 | 31,879 | 72,399 | **56%** |
| doc_E (hwp_representative) | 58 | 21,219 | 76,272 | **72%** |

**핵심 발견**: hwp5txt 파서는 테이블 콘텐츠를 대부분 손실 (56~74%). text_only 문서만 손실 없음.
이는 Phase C (Table Gap 해소) 필요성을 강하게 뒷받침.

인덱싱 설정: chunk_size=500, chunk_overlap=50, embedding=text-embedding-3-small
저장 위치: `data/exp10b/vectordb_doc_{A~E}/`

### B-2. 실험 실행 ✅

| 항목 | 값 |
|------|-----|
| 실험 스크립트 | `scripts/run_exp10b.py` |
| 테스트셋 | `golden_testset_multi.csv` (30문항, 5문서) |
| 설정 수 | 3 (A_single_pipeline, B_rule_single_route, C_conservative_wide) |
| 반복 수 | 3 runs per config |
| 총 평가 수 | 270 (3 × 3 × 30) |
| 총 소요 시간 | 2,812초 (46.9분) |
| 에러 수 | 0 |
| LLM | gpt-5-mini (temperature=1) |
| Reranker | BAAI/bge-reranker-v2-m3 |

### B-3. 결과 요약 ✅

#### Config 비교 (kw_v2, 3-run 평균)

| Config | kw_v2 | std | p95 Latency | Timeout | 판정 |
|--------|-------|-----|-------------|---------|------|
| A_single_pipeline | 0.7503 | 0.3025 | 26.8s | 0% | **PASS** |
| B_rule_single_route | 0.7559 | 0.3015 | 21.8s | 0% | **PASS** |
| C_conservative_wide | 0.7560 | 0.2993 | 29.5s | 0% | **PASS** |

**결론**: 3개 Config 모두 Quality Floor 통과. 차이 극소 (최대 0.57pp).

#### 문서별 kw_v2

| Config | doc_A | doc_B | doc_C | doc_D | doc_E | macro |
|--------|-------|-------|-------|-------|-------|-------|
| A | 0.9236 | 0.6537 | 0.7481 | 0.6799 | 0.7463 | 0.7503 |
| B | 0.9236 | **0.7031** | 0.7481 | 0.6584 | 0.7463 | 0.7559 |
| C | 0.9236 | **0.7031** | 0.7481 | 0.6587 | 0.7463 | 0.7560 |

#### Text vs Table Gap

| Config | text_only | table_avg | gap |
|--------|-----------|-----------|-----|
| A | 0.9236 | 0.7070 | **0.2166** |
| B | 0.9236 | 0.7140 | **0.2096** |
| C | 0.9236 | 0.7140 | **0.2096** |

#### 난이도별 kw_v2

| Config | easy | medium | hard |
|--------|------|--------|------|
| A | 0.80 | 0.81 | 0.54 |
| B | 0.80 | 0.81 | 0.56 |
| C | 0.80 | 0.81 | 0.57 |

#### Quality Floor 결과

| 기준 | A | B | C | Floor |
|------|---|---|---|-------|
| kw_v2 overall | 0.7503 | 0.7559 | 0.7560 | ≥0.50 ✅ |
| worst group kw_v2 | 0.6537 | 0.6584 | 0.6587 | ≥0.35 ✅ |
| timeout rate | 0% | 0% | 0% | ≤10% ✅ |
| p95 latency | 26.8s | 21.8s | 29.5s | ≤120s ✅ |
| **종합** | **PASS** | **PASS** | **PASS** | |

### B-4. 핵심 인사이트

1. **Config 차이 극소**: A/B/C 간 kw_v2 차이 최대 0.57pp → retrieval 파라미터(alpha, pool, top_k) 미세조정의 한계
2. **Text-Table Gap 확인**: text_only(0.92) vs table(0.71), 약 21pp 차이 → Phase C의 필요성 확인
3. **Worst group**: doc_B(table_simple, 0.65) 와 doc_D(mixed, 0.66)가 취약 → 테이블 정보 손실이 원인
4. **Hard 문항 부진**: kw_v2=0.54~0.57 → 복잡한 질문(compliance, security, evaluation)에서 성능 저하
5. **hwp5txt 파서 한계 재확인**: 테이블 문서에서 56~74% 텍스트 손실 → Phase E(직접 파싱) ROI gate 조건 충족 가능성
6. **운영 지표 안정**: timeout 0%, p95 latency ≤30s → 프로덕션 배포 가능 수준

### B-5. RAGAS 평가 (보류)

RAGAS 메트릭(faithfulness, context_recall)은 별도 LLM API 호출이 필요하여 비용 고려 시 별도 실행 예정.
Run 0 데이터로 3 config × 30문항 = 90 evaluations 필요.

### B 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10b_metrics.csv` | 270건 상세 결과 (question, answer, kw_v2, timing) |
| `data/experiments/exp10b_report.json` | 실험 리포트 (config 요약, quality floor 결과) |
| `data/exp10b/vectordb_doc_{A~E}/` | Per-document ChromaDB |
| `data/exp10b/indexing_results.json` | 인덱싱 결과 (청크 수) |
| `scripts/run_exp10b.py` | 실험 스크립트 |
| `notebooks/exp10b_generalization_rerun.ipynb` | 실험 노트북 |

---

## Phase CE: 파서 전환 + Table Gap 해소 ✅

**상태**: ✅ 완료 (2026-02-21)

### CE-1. 구현 ✅

| 파일 | 설명 | 상태 |
|------|------|------|
| `src/bidflow/parsing/hwp_html_parser.py` | hwp5html → BeautifulSoup 기반 파서, col_path 구현 | ✅ |
| `src/bidflow/parsing/table_chunker.py` | V1/V2/V3 모드 테이블-aware 청킹 | ✅ |
| `scripts/run_exp10ce.py` | 인덱싱 + 평가 자동화 (360 evals) | ✅ |

### CE-2. 인덱싱 통계

| Config | doc_A(text) | doc_B(table) | doc_C(complex) | doc_D(mixed) | doc_E(hwp) | Total |
|--------|------------|-------------|---------------|-------------|-----------|-------|
| V0_hwp5txt | 43 | 22 | 29 | 36 | 27 | 157 |
| V1_html_basic | 175 | 115 | 237 | 184 | 193 | 904 |
| V2_html_table | 180(84T+96t) | 134(41T+93t) | 322(54T+268t) | 206(74T+132t) | 212(51T+161t) | 1,054 |
| V3_html_full | 180 | 134 | 322* | 206 | 212 | 1,054 |

*V3 doc_C: 인덱싱 시 hwp5html 프로세스 장애로 임베딩 실패 (V2와 동일 구조)

### CE-3. 실험 결과 (360 evaluations, 63.2분)

**Config × Document kw_v2:**

| Config | doc_A(text) | doc_B(table) | doc_C(complex) | doc_D(mixed) | doc_E(hwp) | Overall | Gap |
|--------|------------|-------------|---------------|-------------|-----------|---------|-----|
| V0_hwp5txt | **0.9144** | 0.6908 | 0.7481 | **0.6946** | **0.7343** | **0.7564** | 0.197 |
| V1_html_basic | 0.6389 | **0.9671** | **0.8315** | 0.5953 | 0.6923 | 0.7450 | -0.133 |
| V2_html_table | 0.6343 | 0.9641 | 0.7481 | 0.5045 | 0.6550 | 0.7012 | -0.084 |
| V3_html_full | 0.6435 | 0.9638 | NaN* | 0.5444 | 0.6510 | 0.7007 | -0.076 |

Quality Floor: 4개 config 모두 **PASS** (kw_v2 ≥ 0.50, worst ≥ 0.35)
Phase CE Target: 4개 config 모두 **FAIL** (overall ≥ 0.80 미달성)

### CE-4. 핵심 분석

**1. 테이블 성능 대폭 개선 (V1)**
- doc_B (table_simple): 0.6908 → **0.9671** (+40%)
- doc_C (table_complex): 0.7481 → **0.8315** (+11%)
- hwp5html이 hwp5txt 대비 테이블 내용을 훨씬 잘 추출

**2. 텍스트 성능 하락 (V1)**
- doc_A (text_only): 0.9144 → **0.6389** (-30%)
- hwp5html의 HTML 마크업 잔류가 텍스트 품질을 저하시킴
- doc_D (mixed): 0.6946 → 0.5953 (-14%)

**3. col_path 분리가 오히려 역효과 (V2/V3 < V1)**
- V2 (col_path): 0.7012 vs V1 (flat): 0.7450
- 테이블을 개별 청크로 분리하면 정보가 파편화됨
- "col_path: value" 형식이 LLM 응답 품질을 저하

**4. Text-Table Gap 역전**
- V0: text 0.91 > table 0.72 (gap=0.20, 텍스트 우위)
- V1: text 0.64 < table 0.77 (gap=-0.13, 테이블 우위)
- Gap은 해소되었으나 방향이 반대로 역전

**5. Difficulty 분석**
- V0: easy=0.80, medium=0.81, **hard=0.58**
- V1: easy=0.72, medium=0.80, **hard=0.73** (+15pp 개선)
- 어려운 질문(주로 테이블 기반)에서 V1이 크게 개선

### CE-5. 결론 및 Phase D 방향

**현재 최선 전략: 하이브리드 파서**
- V0 (hwp5txt): text_only 문서에서 최적 (kw_v2=0.91)
- V1 (html_basic): table 문서에서 최적 (kw_v2=0.77~0.97)
- V2/V3 (col_path): 현재 구현에서는 성능 이점 없음 → 보류

**Phase D 권장 방향 (3가지 옵션)**:
1. **하이브리드 접근**: doc_type별 파서 자동 선택 (text→hwp5txt, table→hwp5html)
2. **V1 텍스트 개선**: hwp5html 텍스트 추출 품질 향상 (HTML 태그 정리, 텍스트 전처리 개선)
3. **Baseline 고정**: V0을 baseline으로 확정, V1 결과를 참고용으로 기록

### CE-6. 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10ce_metrics.csv` | 360건 원시 결과 |
| `data/experiments/exp10ce_report.json` | 실험 리포트 |
| `data/exp10ce/index_stats.json` | 인덱싱 통계 |
| `data/exp10ce/vectordb_*` | 20개 ChromaDB 컬렉션 |
| `src/bidflow/parsing/hwp_html_parser.py` | hwp5html 파서 |
| `src/bidflow/parsing/table_chunker.py` | 테이블-aware 청커 |
| `scripts/run_exp10ce.py` | 실험 스크립트 |

---

## Phase D: Hybrid Parser + Baseline 고정 ✅

**상태**: ✅ 완료 (2026-02-21)

### D-1. 하이브리드 파서 (V4) 설계

**전략**: hwp5txt 텍스트 + hwp5html 테이블-only 결합
- 텍스트 청크: hwp5txt (깔끔, 33K chars for doc_A)
- 테이블 청크: hwp5html (테이블 구조 보존, flat 직렬화)
- `table_chunker.py`에 `chunk_v4_hybrid()` 메서드 추가

**V1 텍스트 하락 진단**:
- hwp5html이 1.91x 더 많은 텍스트 추출 (33K → 63K chars, doc_A)
- 43 → 175 chunks로 검색 풀 4x 확대 → 신호 희석
- 키워드는 양쪽 모두 존재 → 텍스트 품질 문제가 아닌 검색 희석 문제

### D-2. 실험 결과 (180 evaluations, 27.8분, 0 errors)

| Config | doc_A(text) | doc_B(table) | doc_C(complex) | doc_D(mixed) | doc_E(hwp) | Overall | Gap |
|--------|------------|-------------|---------------|-------------|-----------|---------|-----|
| V0_hwp5txt | **0.924** | 0.685 | 0.748 | 0.630 | **0.778** | 0.753 | 0.213 |
| V4_hybrid | 0.729 | **0.972** | **0.776** | **0.669** | 0.730 | **0.775** | -0.058 |
| Delta | -0.194 | **+0.287** | +0.028 | +0.039 | -0.048 | **+0.022** | -0.271 |

### D-3. 핵심 분석

**1. V4_hybrid Overall +2.2pp 개선 (0.753 → 0.775)**
- doc_B: +28.7pp (가장 큰 개선, 테이블 정보 완전 복원)
- doc_D: +3.9pp, doc_C: +2.8pp

**2. doc_A 하락 여전 (-19.4pp)**
- hwp5txt 43 chunks + hwp5html 93 table chunks = 136 total
- 테이블 청크 추가로 검색 풀 확대 → text_only 질문에 대한 신호 희석
- 해결 방안: doc_type 판별 후 테이블 청크 추가 여부 결정

**3. Text-Table Gap 거의 해소**
- V0: 0.213 (텍스트 우위) → V4: -0.058 (거의 균형)

### D-4. Baseline 결정

**V4_hybrid를 새 Baseline으로 채택** (overall 0.775)

| 항목 | V0 (old baseline) | V4 (new baseline) | 변화 |
|------|------|------|------|
| Overall kw_v2 | 0.753 | **0.775** | +2.2pp |
| Text-Table Gap | 0.213 | **0.058** | -15.5pp |
| Worst doc | doc_D (0.630) | doc_D (0.669) | +3.9pp |
| Table avg | 0.710 | **0.787** | +7.7pp |

**추후 개선 방향**:
- doc_A 하락 복구: table-count 기반 동적 선택 (테이블 0개 시 hwp5txt only)
- pool_size 최적화: V4의 큰 검색 풀(~140 chunks)에 맞게 pool_size 상향
- 프롬프트 개선: 테이블 컨텍스트 활용 지시문 추가

### D-5. 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10d_metrics.csv` | 180건 원시 결과 |
| `data/experiments/exp10d_report.json` | 실험 리포트 |
| `data/exp10d/index_stats.json` | 인덱싱 통계 |
| `scripts/run_exp10d.py` | Phase D 실험 스크립트 |
| `src/bidflow/parsing/table_chunker.py` | chunk_v4_hybrid() 추가 |

---

## 실행 로그 (시간순)

### 2026-02-21

| 시간 | 작업 | 결과 | 비고 |
|------|------|------|------|
| PM | Phase A-1: 문서 선정 | ✅ 5건 선정 완료 | exp08_eda_results.csv 기반 층화 샘플링 |
| PM | Phase A-2: HWP 변환 | ✅ 5건 모두 성공 | hwp5txt 1건, hwp5html 4건 |
| PM | Phase A-3: Q&A 생성 | ✅ 30문항 생성 | 문서 원문 기반 직접 추출 |
| PM | Phase A-4: CSV 저장 | ✅ multi(30) + combined(60) | golden_testset_combined.csv |
| PM | HISTORY 업데이트 | ✅ | Phase A 완료 기록 |
| PM | Phase B-1: 인덱싱 | ✅ 325 chunks | hwp5txt, per-doc ChromaDB |
| PM | Phase B-2: 스크립트 생성 | ✅ run_exp10b.py | 270 eval 자동화 |
| PM | Phase B-3: 실험 실행 | ✅ 270/270, 0 errors | 46.9분 소요 |
| PM | Phase B-4: 결과 분석 | ✅ 3 config PASS | kw_v2: 0.75~0.76 |
| PM | HISTORY 업데이트 | ✅ | Phase B 완료 기록 |
| PM | Phase CE-1: 파서 구현 | ✅ | hwp_html_parser.py + table_chunker.py |
| PM | Phase CE-2: 파싱 검증 | ✅ | 5건 모두 hwp5html 파싱 성공 |
| PM | Phase CE-3: 인덱싱 | ✅ | V0~V3 × 5docs = 20 collections, 15.5분 |
| PM | Phase CE-4: 평가 실행 | ✅ | 360 evals (4×3×30), 63.2분, 18 errors (V3 doc_C 누락) |
| PM | Phase CE-5: 결과 분석 | ✅ | V1 테이블↑40%, 텍스트↓30%, Gap 역전 |
| PM | HISTORY 업데이트 | ✅ | Phase CE 완료 기록 |
| PM | Phase D-1: V1 텍스트 하락 진단 | ✅ | 검색 희석 확인 (43→175 chunks) |
| PM | Phase D-2: V4_hybrid 구현 | ✅ | hwp5txt text + hwp5html table |
| PM | Phase D-3: 실험 실행 | ✅ | 180 evals, 0 errors, 27.8분 |
| PM | Phase D-4: Baseline 고정 | ✅ | V4_hybrid=0.775 (+2.2pp) |
| PM | HISTORY 업데이트 | ✅ | Phase D 완료 기록 |
| PM | Phase D+: 테이블 필터 구현 | ✅ | is_data_table(), chunk_v5_hybrid_smart() |
| PM | Phase D+: 필터 효과 검증 | ✅ | doc_A: 93→67 tables, doc_B: 84→61 tables |
| PM | Phase D+: V4 vs V5 실험 | ✅ | 180 evals, 31.8분, 0 errors |
| PM | Phase D+: 결과 분석 | ❌ | V5 실패: doc_A 무변화, doc_B -30.6pp, overall -5.3pp |
| PM | HISTORY 업데이트 | ✅ | Phase D+ 실패 기록, V4 최종 확정 |

### 2026-02-22

| 시간 | 작업 | 결과 | 비고 |
|------|------|------|------|
| AM | Phase E-1: chunk_v4_hybrid 버그 발견 | ✅ | chunk_size 무시됨 (HWPParser 하드코딩 1000) |
| AM | Phase E-1: 버그 수정 | ✅ | self.splitter로 직접 분할하도록 변경 |
| AM | Phase E-2: 인덱싱 (3 sizes × 5 docs) | ✅ | 15 VDB 생성 완료 |
| AM | Phase E-4: 평가 실행 (190/450 완료) | ❌ | OpenAI API quota 초과 (insufficient_quota) |
| AM | Phase E-5: 스크립트 개선 | ✅ | 증분저장, 재개, quota 조기중단, N_RUNS=1 |
| AM | HISTORY 업데이트 | ✅ | Phase E 진행상황 기록 |
| PM | Phase E: API quota 복구 | ✅ | OpenAI 크레딧 충전 |
| PM | Phase E: 실험 실행 (150 evals) | ✅ | 28.2분, 0 errors |
| PM | Phase E: 결과 분석 | ✅ | c500_pv2 최고 (0.814, +3.9pp) |
| PM | Phase E: Baseline 결정 | ✅ | c500_pv2 채택 |
| PM | HISTORY 업데이트 | ✅ | Phase E 완료 기록 |

| 실험 | Config | Overall kw_v2 | Text-Table Gap | Evals | Errors |
|------|--------|-------------|----------------|-------|--------|
| Phase B (EXP10b) | V0_hwp5txt | 0.756 | 0.197 | 270 | 0 |
| Phase CE (EXP10ce) | V1_html_basic | 0.745 | -0.133 | 360 | 18 |
| **Phase D (EXP10d)** | **V4_hybrid** | **0.775** | **-0.058** | 180 | 0 |

**V4_hybrid 채택**: hwp5txt text + hwp5html table 결합, overall +2.2pp, gap -15.5pp

---

---

## Phase D+: 테이블 품질 필터 실험 ❌

**상태**: ❌ 실패 — 필터 접근 비효과적 (2026-02-21)

### D+-1. 구현

- `hwp_html_parser.py`에 `is_data_table()` 정적 메서드 추가
  - 판별 기준: data rows≥2, cols≥2, fill_ratio≥0.3, avg_cell_len≥2.0
- `table_chunker.py`에 `chunk_v5_hybrid_smart()` 추가
  - V4 + 레이아웃 테이블 필터링

### D+-2. 필터 효과

| Doc | V4 tables | V5 tables | Removed |
|-----|-----------|-----------|---------|
| doc_A | 93 | 67 | 26 |
| doc_B | 84 | 61 | 23 |
| doc_C | 265 | 0* | - |
| doc_D | 121 | 83 | 38 |
| doc_E | 146 | 92 | 54 |

*doc_C: hwp5html 타임아웃 (필터 무관)

### D+-3. 실험 결과 (180 evals, 31.8분, 0 errors)

| Config | doc_A | doc_B | doc_C | doc_D | doc_E | Overall |
|--------|-------|-------|-------|-------|-------|---------|
| V4_hybrid | 0.722 | **0.972** | 0.748 | 0.640 | 0.734 | **0.763** |
| V5_smart | 0.722 | 0.666 | 0.748 | 0.636 | **0.778** | 0.710 |
| Delta | **0.000** | **-0.306** | 0.000 | -0.004 | +0.044 | **-0.053** |

### D+-4. 실패 원인 분석

1. **doc_A 미복구** (0.722→0.722): 93→67개로 필터링했지만 67개 남은 테이블이 여전히 43개 텍스트 청크를 희석. 구조 필터로는 doc_A의 진짜 데이터 테이블(요구사항표, 조직표 등)을 레이아웃과 구분 불가

2. **doc_B 대폭 하락** (0.972→0.666, -30.6pp): 필터가 doc_B의 실제 데이터 테이블 23개를 오판 제거. doc_B는 table_simple 문서로 테이블이 핵심 정보원

3. **근본 원인 재확인**: doc_A 문제는 "레이아웃 테이블 품질"이 아니라 **hwp5txt 텍스트와 hwp5html 테이블 간 콘텐츠 중복** + 검색 풀 팽창. hwp5txt가 이미 표의 텍스트를 일부 추출하고, hwp5html이 같은 내용을 테이블로 다시 추가

### D+-5. 결론

- **V5_hybrid_smart 폐기**: 전체 성능 V4 대비 -5.3pp 하락
- **V4_hybrid를 최종 Baseline으로 확정**: overall 0.763~0.775 (런 분산)
- doc_A 하락(-0.20pp vs V0)은 doc_B 상승(+0.38pp)과 전체 개선(+0.01~0.02pp)으로 상쇄
- 향후 doc_A 개선은 파서 레벨이 아닌 **retrieval 튜닝**(pool_size 확대, prompt 개선)으로 접근 필요

### D+-6. 산출물

| 파일 | 설명 |
|------|------|
| `data/experiments/exp10d_plus_metrics.csv` | 180건 원시 결과 |
| `data/experiments/exp10d_plus_report.json` | 실험 리포트 |
| `src/bidflow/parsing/hwp_html_parser.py` | is_data_table() 추가 |
| `src/bidflow/parsing/table_chunker.py` | chunk_v5_hybrid_smart() 추가 |
| `scripts/run_exp10d_plus.py` | Phase D+ 실험 스크립트 |

---

## 최종 요약 (Phase A→B→CE→D→D+)

| 실험 | Config | Overall kw_v2 | Text-Table Gap | Evals | Errors |
|------|--------|-------------|----------------|-------|--------|
| Phase B (EXP10b) | V0_hwp5txt | 0.756 | 0.197 | 270 | 0 |
| Phase CE (EXP10ce) | V1_html_basic | 0.745 | -0.133 | 360 | 18 |
| **Phase D (EXP10d)** | **V4_hybrid** | **0.775** | **-0.058** | 180 | 0 |
| Phase D+ (EXP10d+) | V5_smart ❌ | 0.710 | 0.015 | 180 | 0 |

**최종 Baseline: V4_hybrid** (hwp5txt text + hwp5html table 결합)
- Overall: 0.763~0.775 (run variance)
- Text-Table Gap: ~0.05 (거의 해소)
- 주요 trade-off: doc_A -0.20pp (text_only) ↔ doc_B +0.38pp (table)

**다음 개선 방향** (파서 레벨 최적화 한계 도달):
1. Retrieval 파라미터 튜닝 (pool_size 확대, alpha 조정)
2. Prompt Engineering (테이블 컨텍스트 활용 지시문)
3. Chunk 전략 조정 (chunk_size 실험, 중복 제거)

*Phase A→D+ 완료. 파서 최적화 단계 종료, Retrieval/Prompt 최적화 단계로 전환 권장.*

---

## Phase E: Retrieval Pipeline 재최적화 ✅

**상태**: ✅ 완료 (2026-02-22)
**시작일**: 2026-02-22

### E-0. 배경

EXP01-09의 retrieval 파라미터(chunk_size=500, pool_size=50, alpha=0.7 등)가 단일 문서에서 최적화되어 다문서 테스트셋에서 과적합 가능성이 있음. V4_hybrid 파서를 고정하고, 3가지 차원을 재최적화:
1. **chunk_size**: 300 / 500 / 800
2. **pool_size**: 50 / 100
3. **prompt**: V1 (기본) / V2 (테이블 인식)

### E-1. 치명적 버그 발견 및 수정 ✅

**문제**: `chunk_v4_hybrid()`가 `HWPParser().parse()`를 호출하여 하드코딩된 `chunk_size=1000`으로 텍스트를 분할. `TableAwareChunker`에 전달한 chunk_size(300/500/800)가 텍스트 청크에 전혀 적용되지 않았음.

**증거**: 3개 chunk_size 모두 동일한 866 chunks 생성 (text:43개 동일)

**수정**: `chunk_v4_hybrid()`에서 `HWPParser` 의존성 제거. `hwp5txt`로 raw 텍스트 추출 후 `self.splitter` (외부 chunk_size 반영)로 분할하도록 변경.

**수정 파일**: `src/bidflow/parsing/table_chunker.py` (chunk_v4_hybrid 메서드)

**영향 범위**: 이전 Phase D/D+ 실험(chunk_size=500)은 `HWPParser`의 기본값 1000을 사용했음. Phase D는 V0과 V4를 같은 조건(둘 다 1000)으로 비교했으므로 상대 비교 결론(V4>V0)은 유효.

### E-2. 인덱싱 완료 ✅

수정된 chunk_v4_hybrid()로 3개 chunk_size × 5 docs = 15 VDB 생성:

| chunk_size | doc_A text | doc_B text | doc_C text | doc_D text | doc_E text | Total chunks |
|---|---|---|---|---|---|---|
| 300 | 159 | 82 | 106 | 133 | 94 | 1,283 |
| 500 | 87 | 48 | 57 | 75 | 58 | 1,034 |
| 800 | 53 | 26 | 36 | 47 | 33 | 903 |

(테이블 청크: doc_A=93, doc_B=84, doc_C=265, doc_D=121, doc_E=146 — chunk_size 무관)

### E-3. 평가 설정

| Config | chunk_size | pool_size | prompt | alpha | top_k |
|--------|-----------|-----------|--------|-------|-------|
| c300 | 300 | 50 | V1 | 0.7 | 15 |
| c500 | 500 | 50 | V1 | 0.7 | 15 |
| c800 | 800 | 50 | V1 | 0.7 | 15 |
| c500_p100 | 500 | 100 | V1 | 0.7 | 15 |
| c500_pv2 | 500 | 50 | V2(테이블인식) | 0.7 | 15 |

프롬프트 V2 (테이블 인식):
```
아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.
문맥에는 일반 텍스트와 테이블 데이터가 포함될 수 있습니다.
테이블에서 추출된 정보(금액, 기간, 수량, 비율 등)가 있다면 우선적으로 활용하세요.
답변 시 원문의 사업명, 기관명, 금액, 날짜, 숫자 등을 정확히 그대로(Verbatim) 인용하세요.
문맥에 답이 없으면 '해당 정보를 찾을 수 없습니다'라고 답하세요.
```

### E-4. 실험 결과 (150 evals, 28.2분, 0 errors) ✅

#### Overall kw_v2 순위

| Config | kw_v2 | std | vs c500 baseline | 판정 |
|--------|-------|-----|-----------------|------|
| **c500_pv2** | **0.8136** | 0.260 | **+2.0pp** | **최고** |
| c500 | 0.7937 | 0.281 | baseline | 준수 |
| c500_p100 | 0.7858 | 0.294 | -0.8pp | 하락 |
| c800 | 0.7528 | 0.307 | -4.1pp | 하락 |
| c300 | 0.7464 | 0.313 | -4.7pp | 하락 |

#### Config × Document kw_v2

| Config | doc_A(text) | doc_B(table) | doc_C(complex) | doc_D(mixed) | doc_E(hwp) | Overall |
|--------|------------|-------------|---------------|-------------|-----------|---------|
| **c500_pv2** | **0.875** | **0.972** | 0.748 | **0.683** | **0.790** | **0.814** |
| c500 | 0.875 | 0.972 | 0.748 | 0.643 | 0.730 | 0.794 |
| c500_p100 | 0.875 | 0.972 | 0.748 | 0.604 | 0.730 | 0.786 |
| c800 | 0.625 | 0.972 | 0.748 | 0.677 | 0.742 | 0.753 |
| c300 | 0.688 | 0.972 | 0.748 | 0.594 | 0.730 | 0.746 |

#### Text vs Table Gap

| Config | text_only | table_avg | gap |
|--------|-----------|-----------|-----|
| c500_pv2 | 0.875 | 0.798 | +0.077 |
| c500 | 0.875 | 0.773 | +0.102 |
| c500_p100 | 0.875 | 0.764 | +0.112 |
| c800 | 0.625 | 0.785 | -0.160 |
| c300 | 0.688 | 0.761 | -0.074 |

#### Difficulty × Config kw_v2

| Config | easy | medium | hard |
|--------|------|--------|------|
| c500_pv2 | 0.907 | **0.801** | **0.616** |
| c500 | 0.907 | 0.777 | 0.556 |
| c800 | 0.788 | **0.814** | 0.568 |
| c500_p100 | 0.907 | 0.727 | 0.600 |
| c300 | 0.812 | 0.769 | 0.556 |

### E-5. 핵심 분석

**1. 프롬프트 V2가 가장 큰 개선 (+2.0pp)**
- 테이블 인식 지시문이 doc_D(mixed, +4.0pp), doc_E(hwp, +6.0pp)에서 효과적
- 테이블+텍스트 혼합 문서에서 LLM이 테이블 데이터를 우선 활용하도록 유도
- hard 문항: 0.556→0.616 (+6.0pp), medium: 0.777→0.801 (+2.4pp)

**2. chunk_size=500이 최적**
- c300: doc_A -18.7pp (작은 청크 → 맥락 부족)
- c800: doc_A -25.0pp (큰 청크 → 검색 정밀도 저하)
- chunk_size=500이 텍스트/테이블 모두에서 균형 잡힌 성능

**3. pool_size=100은 효과 없음 (-0.8pp)**
- doc_D에서 하락 (0.643→0.604), 더 큰 풀이 노이즈 증가
- pool_size=50이 이미 충분

**4. doc_A 복구 성공 (Phase D 대비)**
- Phase D V4_hybrid: doc_A=0.729 (chunk_size=1000 버그)
- Phase E c500/c500_pv2: doc_A=**0.875** (+14.6pp)
- 버그 수정(chunk_size=500 적용)이 doc_A 성능 복구의 핵심

**5. doc_B/C 안정**
- doc_B: 전 config에서 0.972 (테이블 성능 완전 포화)
- doc_C: 전 config에서 0.748 (복잡한 테이블은 chunk_size/prompt 무관)

### E-6. Baseline 결정

**c500_pv2를 새 Baseline으로 채택** (overall 0.8136)

| 항목 | Phase D V4(old) | Phase E c500_pv2(new) | 변화 |
|------|------|------|------|
| Overall kw_v2 | 0.775 | **0.814** | **+3.9pp** |
| doc_A (text_only) | 0.729 | **0.875** | **+14.6pp** |
| doc_B (table) | 0.972 | 0.972 | 0.0pp |
| doc_D (mixed) | 0.669 | **0.683** | +1.4pp |
| Text-Table Gap | -0.058 | +0.077 | 텍스트 우위로 전환 |
| hard 문항 | - | **0.616** | - |

**최종 최적 설정**:
- Parser: V4_hybrid (hwp5txt text + hwp5html table)
- chunk_size: 500, chunk_overlap: 50
- pool_size: 50, alpha: 0.7, top_k: 15
- Prompt: V2 (테이블 인식)
- Reranker: BAAI/bge-reranker-v2-m3
- Embedding: text-embedding-3-small
- LLM: gpt-5-mini (temperature=1)

### E-7. 산출물

| 파일 | 설명 |
|------|------|
| `scripts/run_exp10e.py` | Phase E 실험 스크립트 |
| `src/bidflow/parsing/table_chunker.py` | chunk_v4_hybrid() 버그 수정 |
| `data/exp10e/vectordb_*` | 15개 ChromaDB (3 sizes × 5 docs) |
| `data/exp10e/index_stats.json` | 인덱싱 통계 |
| `data/experiments/exp10e_metrics.csv` | 150건 평가 결과 |
| `data/experiments/exp10e_report.json` | 실험 리포트 |

---

## 최종 요약 (Phase A→B→CE→D→D+→E→EXP11→EXP12)

| 실험 | Config | Overall kw_v2 | Overall kw_v3 | Evals | Errors | 상태 |
|------|--------|-------------|-------------|-------|--------|------|
| Phase B (EXP10b) | V0_hwp5txt | 0.756 | — | 270 | 0 | ✅ |
| Phase CE (EXP10ce) | V1_html_basic | 0.745 | — | 360 | 18 | ✅ |
| Phase D (EXP10d) | V4_hybrid | 0.775 | — | 180 | 0 | ✅ |
| Phase D+ (EXP10d+) | V5_smart ❌ | 0.710 | — | 180 | 0 | ✅ |
| **Phase E (EXP10e)** | **c500_pv2** | **0.814** | **0.896** | 150 | 0 | **✅** |
| EXP11: prompt_v3 | verbatim 강화 | 0.782 | 0.865 | 30 | 0 | ❌ 하락 |
| EXP11: route_type | 타입별 라우팅 | 0.806 | 0.885 | 30 | 0 | ❌ 하락 |
| EXP11: section_ctx | query expansion | 0.775 | 0.859 | 30 | 0 | ❌ 하락 |
| EXP11: coverage_retry | coverage 검증 | 0.801 | 0.872 | 30 | 0 | ❌ 하락 |
| **EXP12: multi_query** | **LLM 쿼리 3변형** | **0.809** | **0.900** | 30 | 0 | **✅ +0.4pp** |
| EXP12: pool_80 | 리랭커 후보 80개 | 0.805 | 0.898 | 30 | 0 | ✅ +0.2pp |
| EXP12: alpha_05 | BM25 50% | 0.755 | 0.839 | 30 | 0 | ❌ 하락 |
| EXP12: pool_80_k20 | pool=80+top_k=20 | 0.793 | 0.873 | 30 | 0 | ❌ 하락 |

**최종 Best Config: multi_query** (c500_pv2 + LLM 쿼리 3변형)
- Overall kw_v2: **0.809** / kw_v3: **0.900** (Phase E 대비 +0.4pp)
- doc_D 보안 문항: 0.211 → **0.632** (+42.1pp, 치명적 실패 복구)
- Perfect(1.0) 비율: kw_v3 19/30 (63%)
- 비용: 쿼리당 +3 LLM 호출 (reformulation), retrieval 시간 5× (10s vs 2s)

**최적화 여정**: V0(0.756) → V4(0.775) → c500_pv2(0.814/**0.896**) → multi_query(0.809/**0.900**)

**핵심 결론**:
- 프롬프트 엔지니어링(EXP11): 모든 변형 baseline 미달
- Retrieval 최적화(EXP12): multi_query(+0.4pp)와 pool_80(+0.2pp)이 baseline 초과
- 가장 큰 개선: doc_D 보안 문항 +42.1pp (multi_query, pool_80 모두)
- text-embedding-3-large 접근 불가로 Stage 2 미실행
- LLM(gpt-5-mini) 변경 불가 제약 하에서 retrieval 다양성이 가장 효과적

---

## EXP11: 종합 최적화 (Phase F~J)

**상태**: ✅ 완료
**시작일**: 2026-02-22
**완료일**: 2026-02-22

### 배경 및 목표

Phase E에서 c500_pv2가 kw_v2=0.814를 달성했지만 15/30 문항이 여전히 미달. 실패 분석 결과:

| 실패 원인 | 해당 문항 수 | 예상 복구 방법 |
|-----------|------------|---------------|
| 평가 지표 정규화 부족 (구두점/조사/괄호) | 12개 | Phase I: kw_v3 |
| LLM 패러프레이징 (원문 미인용) | 6개 | Phase F: verbatim 강화 |
| 검색 실패 (잘못된 섹션 검색) | 1개 | Phase G: 섹션 라우팅 |
| 리스트형 질문 항목 누락 | 3개 | Phase F: 구조화 출력 |
| 숫자/특수문자 정규화 | 3개 | Phase I: 특수문자 룰 |

**외부 피드백 반영 (5개 Phase)**:
- Phase F: 질문 타입 라우팅 + 구조화 출력 강제
- Phase G: 섹션/헤더 기반 retrieval bias
- Phase H: 생성 후 coverage 검증 + 1회 재생성
- Phase I: kw_v2 정규화 룰 확장
- Phase J: 3-run 재측정 + 분산 확인

### EXP11 실험 설계

**Config 목록 (6개)**:

| Config | 설명 | 새 API 호출 | 핵심 변경 |
|--------|------|------------|----------|
| `ref_v2` | Phase E c500_pv2 결과 재활용 (kw_v2 기준) | ❌ | 없음 (참조용) |
| `ref_v3` | 같은 답변을 kw_v3로 재채점 | ❌ | Phase I 정규화 |
| `prompt_v3` | verbatim 강화 + 리스트 인식 프롬프트 | ✅ 30 calls | Phase F |
| `route_type` | 질문 타입별 프롬프트 라우팅 | ✅ 30 calls | Phase F |
| `section_ctx` | 섹션 헤더 컨텍스트 추가 | ✅ 30 calls | Phase G |
| `coverage_retry` | 생성 후 coverage 검증 + 재생성 | ✅ 30+α calls | Phase H |

**Phase J**: 최고 config을 3-run 재측정 (별도 실행)

**총 API 호출**: ~120-150 calls (약 30분 예상)

### Phase I: 평가 지표 고도화 (kw_v3)

kw_v2 → kw_v3 변경점:
1. **구두점 제거**: 단어 끝 `,` `.` `;` 제거 (`체계화,` → `체계화`)
2. **한국어 조사 제거**: `은/는/이/가/을/를/의/에/에서/으로/와/과/이며` 등
3. **괄호 분리**: `응급의료기본계획('23.3)과` → `응급의료기본계획 ('23.3) 과`
4. **특수문자 정규화**: `Ⅶ`→`7`, `￦`→`₩` 등

시뮬레이션 결과 (exp10e 데이터 재채점):
- kw_v2: 0.814 → kw_v3: **0.876** (+6.2pp, 무료 개선)

### Phase F: 질문 타입 라우팅 + 구조화 출력

질문 분류기:
- **단답형**: 사업명, 기간, 금액 → 기본 프롬프트
- **리스트형**: 항목, 내용, 문제점 → "모든 항목을 빠짐없이 원문 그대로 나열"
- **위치형**: 몇 장, 어디, 규정 → "정확한 장번호, 제목, 페이지 번호 포함"

프롬프트 V3 핵심 추가:
```
답변 규칙:
1. 원문의 표현을 정확히 그대로 인용 (절대 패러프레이징 금지)
2. 여러 항목을 나열할 때는 원문의 모든 항목을 빠짐없이 포함
3. 장/절/페이지를 묻는 질문은 정확한 번호와 제목 포함
```

### Phase G: 섹션 컨텍스트 보강

검색 시 질문에서 섹션 키워드를 추출하여 컨텍스트에 섹션 헤더 정보 추가.
doc_D의 보안 질문(kw_v2=0.053) 같은 구조 질문 타겟.

### Phase H: Coverage 검증 + 재생성

1. 첫 답변 생성 후, 검색된 컨텍스트의 핵심 키워드 대비 답변 coverage 계산
2. coverage < 0.7이면 누락 키워드를 포함한 재프롬프트 실행
3. hard/list 질문에서 효과적

---

### EXP11 실행 결과

**실행 정보**:
- 실행 시간: 2026-02-22 15:15 ~ 15:58 (총 2558초 / 약 43분)
- 총 평가: 120 (5 configs × 30Q), 에러 0건
- API 호출: ~120 (ref_v2는 Phase E 결과 재활용)

#### Phase I 결과 (kw_v3 재채점)

Phase E c500_pv2 답변을 kw_v3로 재채점:
- kw_v2: 0.8136 → **kw_v3: 0.8961** (+8.3pp)
- Perfect(1.0): 15/30 → 19/30
- 시뮬레이션 예측(+6.2pp)을 **초과** 달성

#### Config별 Overall 성능

| Config | kw_v2 | kw_v3 | delta_v3 | Q당 시간 | 핵심 Phase |
|--------|-------|-------|----------|---------|-----------|
| **ref_v2** (baseline) | **0.814** | **0.896** | — | 13.4s | — |
| route_type | 0.806 | 0.885 | -1.1pp | 13.0s | Phase F |
| prompt_v3 | 0.782 | 0.865 | -3.2pp | 11.5s | Phase F |
| coverage_retry | 0.801 | 0.872 | -2.4pp | 47.1s | Phase H |
| section_ctx | 0.775 | 0.859 | -3.7pp | 13.6s | Phase G |

**핵심 결과: 모든 프롬프트 엔지니어링 변형이 baseline보다 하락**

#### Question Type별 분석 (kw_v3)

| Config | direct (19Q) | list (9Q) | location (2Q) |
|--------|-------------|-----------|---------------|
| ref_v2 | **0.942** | 0.840 | 0.718 |
| route_type | 0.925 | 0.830 | **0.763** |
| section_ctx | 0.879 | 0.830 | **0.801** |
| prompt_v3 | 0.905 | 0.830 | 0.641 |
| coverage_retry | 0.929 | 0.824 | 0.558 |

- **location 타입**: section_ctx(0.801), route_type(0.763)이 ref_v2(0.718)보다 개선
- **direct 타입**: 모든 변형이 ref_v2(0.942)보다 하락 — "원문 인용" 강제가 간결한 답변을 방해
- **list 타입**: 거의 차이 없음 (~0.83)

#### Document별 분석 (kw_v3)

| Config | doc_A | doc_B | doc_C | doc_D | doc_E |
|--------|-------|-------|-------|-------|-------|
| ref_v2 | 0.916 | **1.000** | 0.905 | **0.731** | **0.928** |
| route_type | 0.901 | 1.000 | 0.905 | 0.738 | 0.882 |
| coverage_retry | 0.901 | 0.992 | **0.938** | 0.662 | 0.869 |
| section_ctx | 0.901 | 0.917 | 0.905 | 0.676 | 0.895 |
| prompt_v3 | 0.901 | 1.000 | 0.905 | 0.648 | 0.869 |

- **doc_D** (보안 문서): 모든 config에서 가장 어려움 (0.65~0.74)
- **doc_B**: baseline 만점, 대부분의 변형도 높은 성능
- **doc_C**: coverage_retry(0.938)가 유일하게 ref_v2(0.905) 초과

#### Difficulty별 분석 (kw_v3)

| Config | easy | medium | hard |
|--------|------|--------|------|
| ref_v2 | **0.986** | **0.877** | 0.719 |
| route_type | 0.971 | 0.856 | **0.734** |
| section_ctx | 0.936 | 0.818 | **0.747** |
| coverage_retry | 0.971 | 0.858 | 0.665 |
| prompt_v3 | 0.971 | 0.818 | 0.693 |

- **hard 문항**: section_ctx(0.747)와 route_type(0.734)가 ref_v2(0.719)를 약간 초과
- 하지만 easy/medium에서의 손실이 hard 개선을 상쇄

#### Coverage Retry 상세

- 30개 중 **24개 질문에서 retry 발생** (80%)
- retry된 질문 평균 kw_v3: 0.887 vs retry 없는 질문: 0.815
- retry 시간: 53.9s vs 비retry: 20.2s
- coverage 기준이 너무 엄격하여 대부분 retry → 비용만 증가, 개선 미미

#### Phase별 개선 효과 판정

| Phase | 목표 | 결과 | 판정 |
|-------|------|------|------|
| Phase I (kw_v3) | 정규화 개선 | +8.3pp (0.814→0.896) | ✅ **유효** |
| Phase F (prompt_v3) | verbatim 강화 | -3.2pp | ❌ 역효과 |
| Phase F (route_type) | 타입별 라우팅 | -1.1pp (location만 개선) | ⚠️ 부분적 |
| Phase G (section_ctx) | query expansion | -3.7pp (hard만 개선) | ❌ 역효과 |
| Phase H (coverage_retry) | 누락 보완 | -2.4pp, 비용 3.5× | ❌ 비효율적 |
| Phase J (3-run 검증) | 분산 확인 | 생략 (baseline이 최고) | ⏭️ 불필요 |

### EXP11 핵심 교훈

1. **Prompt V2가 이미 최적**: 과도한 제약 추가는 LLM의 자연스러운 답변 능력을 제한
2. **Query expansion은 양날의 검**: 정밀도를 희석시켜 overall 하락 (hard/location만 소폭 개선)
3. **Coverage retry 기준이 과민**: 24/30 retry는 기준값(0.7) 조정이 필요했음
4. **kw_v3 정규화가 최고의 ROI**: 코드 변경 없이 평가 함수만 개선하여 +8.3pp
5. **location 타입에 section_ctx가 유효**: 향후 질문 타입별 선택적 적용 가능성

---

## EXP12: Retrieval 최적화 ✅

**상태**: ✅ 완료 (Stage 1 + Stage 2 KURE-v1)
**시작일**: 2026-02-22
**완료일**: 2026-02-22

### 배경 및 목표

EXP11에서 프롬프트 엔지니어링 4가지 접근 모두 baseline 미달 확인.
→ 병목은 retrieval 품질. LLM 변경 불가(gpt-5-mini 최적) 제약 하에서 retrieval 파라미터 튜닝.

**핵심 실패 패턴** (EXP11 baseline 분석):
- doc_D 67% 실패율 (특히 보안 문항 kw_v3=0.211)
- hard 질문 67% 실패 (섹션 간 합성 필요)
- 고정 top_k=15, pool_size=50이 doc_D 196개 청크에서 부족

### EXP12 실험 설계

**Stage 1: 파라미터 튜닝** (기존 VDB 재사용, 재인덱싱 불필요):

| Config | alpha | pool | top_k | 특수기능 | 근거 |
|--------|-------|------|-------|---------|------|
| ref_v2 | 0.7 | 50 | 15 | — | 베이스라인 |
| alpha_05 | **0.5** | 50 | 15 | — | BM25 키워드 매칭 강화 |
| pool_80 | 0.7 | **80** | 15 | — | 리랭커에게 더 많은 후보 |
| pool_80_k20 | 0.7 | **80** | **20** | — | 후보 + 컨텍스트 확대 |
| multi_query | 0.7 | 50 | 15 | **LLM 쿼리 3변형** | 검색 다양성 확보 |

**Stage 2: 오픈소스 임베딩 (KURE-v1)** (재인덱싱 필요):
- emb_kure: nlpai-lab/KURE-v1 한국어 특화 (1024 dims, GPU 로컬)
- combined_best: KURE-v1 + Stage 1 최적 파라미터 (multi_query params)

> ※ 원래 text-embedding-3-large 계획이었으나 403 접근불가 → KURE-v1 오픈소스로 대체

### Multi-Query 구현 상세

1. gpt-5-mini로 원본 질문의 한국어 변형 3개 생성 (동의어, 다른 조사/어미)
2. 원본 + 변형 4개 쿼리 각각으로 BM25+Vector 검색 실행
3. 결과 합산 후 page_content 기준 중복 제거 (평균 64-74개 고유 후보)
4. 리랭커로 최종 top_k=15 선택

### 실행 결과

**Stage 1**: 4 configs × 30Q = 120 API 호출, ~30분, 에러 0건
**Stage 2**: KURE-v1 오픈소스 임베딩 2 configs × 30Q = 60 API 호출, 에러 0건
**합계**: 7 configs × 30Q = 210 평가, 에러 0건

#### Config별 Overall 성능

| Config | Stage | kw_v2 | kw_v3 | delta_v3 | Q당 시간 |
|--------|-------|-------|-------|----------|---------|
| **multi_query** | 1 | 0.809 | **0.900** | **+0.4pp** | 19.1s |
| pool_80 | 1 | 0.805 | 0.898 | +0.2pp | 14.9s |
| ref_v2 (baseline) | 1 | 0.814 | 0.896 | — | 13.4s |
| **emb_kure** | **2** | 0.809 | **0.896** | **-0.0pp** | **11.4s** |
| combined_best | 2 | 0.799 | 0.891 | -0.5pp | 10.8s |
| pool_80_k20 | 1 | 0.793 | 0.873 | -2.4pp | 12.5s |
| alpha_05 | 1 | 0.755 | 0.839 | -5.7pp | 13.0s |

**핵심 발견**:
- multi_query와 pool_80이 baseline 초과 달성
- **emb_kure (KURE-v1)**: OpenAI embedding과 사실상 동등 (0.896 vs 0.896), 하지만 **API 무료 + 더 빠름**
- combined_best (KURE + multi_query params): 단독 KURE보다 오히려 -0.5pp

#### Config × Document kw_v3

| Config | doc_A | doc_B | doc_C | doc_D | doc_E | overall |
|--------|-------|-------|-------|-------|-------|---------|
| **multi_query** | 0.916 | 1.000 | 0.905 | **0.801** | 0.877 | **0.900** |
| pool_80 | 0.916 | 1.000 | 0.905 | 0.781 | 0.890 | 0.898 |
| ref_v2 | 0.916 | 1.000 | 0.905 | 0.731 | **0.928** | 0.896 |
| **emb_kure** | 0.916 | 1.000 | 0.905 | **0.781** | 0.877 | **0.896** |
| combined_best | 0.916 | 1.000 | 0.905 | 0.745 | 0.890 | 0.891 |
| pool_80_k20 | 0.916 | 1.000 | 0.880 | 0.690 | 0.877 | 0.873 |
| alpha_05 | 0.722 | 1.000 | 0.905 | 0.690 | 0.877 | 0.839 |

- **doc_D**: multi_query +7.0pp, emb_kure +5.0pp, pool_80 +5.0pp
- **doc_E**: ref_v2가 최고 (0.928), emb_kure는 -5.1pp

#### doc_D 문항별 상세 (핵심 실패 영역)

| 문항 | ref_v2 | multi_query | pool_80 | emb_kure | combined_best |
|------|--------|-------------|---------|----------|---------------|
| 보안 준수사항 세부 항목 | **0.211** | **0.632** | **0.632** | **0.632** | 0.421 |
| 제안서 평가방법 위치 | 0.625 | 0.625 | 0.500 | 0.500 | 0.500 |
| 하자담보 책임기간 위치 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| 추진배경 | 0.885 | 0.885 | 0.885 | 0.885 | 0.885 |
| 사업비 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 사업기간 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**보안 문항**: ref_v2 0.211 → multi_query/pool_80/emb_kure 0.632 (+42.1pp)
- KURE-v1도 보안 문항에서 OpenAI 대비 동일한 대폭 개선 달성

#### Config × Difficulty kw_v3

| Config | easy | medium | hard |
|--------|------|--------|------|
| multi_query | **0.986** | **0.877** | 0.738 |
| pool_80 | **0.986** | 0.865 | **0.751** |
| ref_v2 | **0.986** | **0.877** | 0.719 |
| emb_kure | **0.986** | 0.865 | 0.738 |
| combined_best | **0.986** | 0.865 | 0.716 |
| pool_80_k20 | **0.986** | 0.837 | 0.668 |
| alpha_05 | 0.902 | 0.852 | 0.668 |

- **hard**: pool_80(0.751) > multi_query/emb_kure(0.738) > ref_v2(0.719)

#### Config × Question Type kw_v3

| Config | direct | list | location |
|--------|--------|------|----------|
| multi_query | **0.942** | **0.886** | 0.564 |
| pool_80 | 0.935 | **0.886** | 0.603 |
| ref_v2 | **0.942** | 0.840 | **0.718** |
| emb_kure | 0.935 | **0.886** | 0.564 |
| combined_best | 0.935 | 0.863 | 0.603 |
| pool_80_k20 | 0.929 | 0.823 | 0.564 |
| alpha_05 | 0.867 | 0.840 | 0.564 |

- **list 타입**: multi_query/pool_80/emb_kure 모두 +4.6pp (0.840→0.886)
- **location 타입**: ref_v2(0.718)가 여전히 최고 — retrieval 변형이 구조 정보에 약함

### Stage 2 임베딩 분석: KURE-v1 vs OpenAI

| 항목 | text-embedding-3-small (ref_v2) | KURE-v1 (emb_kure) | 차이 |
|------|-------------------------------|---------------------|------|
| kw_v3 | 0.8961 | 0.8957 | **-0.04pp** (무시 가능) |
| Q당 시간 | 13.4s | **11.4s** | **-15%** (API 호출 없음) |
| API 비용 | OpenAI embedding 호출 | **무료** (로컬 GPU) |
| doc_D 성능 | 0.731 | **0.781** | **+5.0pp** |
| location 성능 | **0.718** | 0.564 | **-15.4pp** |
| 차원 수 | 1536 | 1024 | 33% 경량 |

**결론**: KURE-v1은 OpenAI embedding과 전체 성능 동등, doc_D에서 우수, 무료+빠름.
단, location 타입 질문에서 약세 (구조적 문맥 임베딩 차이).

### EXP12 핵심 교훈

1. **Multi-query가 가장 효과적**: 한국어 동의어/표현 다양성이 검색 커버리지 확대에 핵심
2. **pool_80이 2위**: 리랭커 후보 확대(50→80)가 안정적 개선 제공
3. **KURE-v1은 OpenAI 대체 가능**: 전체 성능 동등(-0.04pp), doc_D에서 우수, API 비용 절감
4. **BM25 비중 증가(alpha_05)는 역효과**: 벡터 검색의 의미적 매칭 능력이 BM25보다 우수
5. **top_k 증가(pool_80_k20)는 노이즈 유입**: 20개 컨텍스트는 LLM에 과부하
6. **doc_D 보안 문항(Q23)이 핵심 지표**: multi_query, pool_80, emb_kure 모두 +42.1pp 개선
7. **location 타입은 retrieval로 해결 불가**: 구조 메타데이터가 필요 (장번호, 페이지 정보)
8. **multi_query 비용**: 쿼리당 +3 LLM 호출, retrieval 시간 5× — 정확도-비용 trade-off
9. **combined_best 실패**: 최적 파라미터 조합이 반드시 누적 개선을 보장하지 않음

### EXP12 산출물

| 파일 | 설명 |
|------|------|
| `scripts/run_exp12.py` | EXP12 실험 스크립트 (Stage 1 + Stage 2 KURE-v1) |
| `data/experiments/exp12_metrics.csv` | 210건 평가 결과 (7 configs × 30Q) |
| `data/experiments/exp12_report.json` | 실험 리포트 |
| `data/exp12/vectordb_kure/vectordb_c500_doc_*` | KURE-v1 임베딩 VDB (5문서) |

---

## EXP13: Contextual Retrieval + 한국어 BM25 최적화 ❌

### 실험 목적
Anthropic의 Contextual Retrieval (2024) 기법을 적용하여 각 청크에 LLM 생성 문서 맥락 프리픽스를 추가, 검색 품질 개선을 시도.

### 성공 기준
- Overall kw_v3 > 0.92 (EXP12 best 0.900 대비 +2pp)
- doc_D kw_v3 > 0.85 (현재 ~0.74)
- Q25 보안 준수사항 kw_v3 > 0.7 (현재 0.211)

### 실험 설계 (5 configs)

| Config | 설명 | ctx_vdb | bm25_ko | multi_query |
|--------|------|---------|---------|-------------|
| ref_v2 | EXP12 baseline 재활용 | ❌ | ❌ | ❌ |
| ctx_basic | Contextual prefix만 | ✅ | ❌ | ❌ |
| ctx_bm25_ko | ctx + Kiwi 한국어 BM25 | ✅ | ✅ | ❌ |
| ctx_multi_query | ctx + multi_query | ✅ | ❌ | ✅ |
| ctx_full | ctx + bm25_ko + multi_query | ✅ | ✅ | ✅ |

### Context Prefix 생성
- LLM: gpt-5-mini로 각 청크에 1~2문장 맥락 프리픽스 생성
- 입력: 문서 목차(TOC) + 청크 내용 → 출력: "[이 청크는 ○○ 섹션의 ○○에 대한 내용입니다]"
- 총 1,034 chunks × 5 documents, JSON 캐싱
- 한국어 BM25: kiwipiepy 형태소 분석기 기반 토크나이저

### 결과 (5 configs × 30Q = 150 evals)

| Config | kw_v3 | delta | kw_v2 | perfect_v3 | 판정 |
|--------|-------|-------|-------|------------|------|
| **ref_v2 (baseline)** | **0.8961** | — | 0.8136 | 19/30 | 기준 |
| ctx_basic | 0.8076 | -8.85pp | 0.7437 | 16/30 | ❌ |
| ctx_bm25_ko | 0.8656 | -3.06pp | 0.7756 | 18/30 | ❌ |
| ctx_multi_query | 0.8282 | -6.79pp | 0.7574 | 18/30 | ❌ |
| ctx_full | 0.8886 | -0.75pp | 0.8044 | 20/30 | ❌ |

### Doc별 kw_v3 (ctx_full vs ref_v2)

| Doc | ref_v2 | ctx_full | delta |
|-----|--------|----------|-------|
| doc_A | 0.9164 | 0.8990 | -1.74pp |
| doc_B | 1.0000 | 1.0000 | 0 |
| doc_C | 0.9049 | 0.9049 | 0 |
| doc_D | 0.7311 | 0.7034 | -2.78pp |
| doc_E | 0.9282 | 0.9359 | +0.77pp |

### Q25 보안 준수사항 (doc_D)
모든 config에서 kw_v3 = 0.211 → **개선 없음**

### 분석 및 교훈

1. **Contextual prefix가 검색을 오히려 악화시킴**
   - ctx_basic이 baseline 대비 -8.85pp로 가장 큰 하락
   - prefix 텍스트가 임베딩 벡터를 왜곡하여 기존 잘 되던 쿼리 성능 하락
   - 특히 doc_A에서 -28pp 급락 (simple direct questions가 ctx prefix에 의해 오염)

2. **한국어 BM25 (Kiwi)가 부분 복구 효과**
   - ctx_bm25_ko가 ctx_basic 대비 +5.8pp 회복
   - 형태소 분석 기반 BM25가 prefix 왜곡을 상쇄하는 효과

3. **전체 조합(ctx_full)이 가장 나은 ctx config이나 여전히 baseline 미달**
   - multi_query + bm25_ko가 prefix 왜곡을 최대한 보상
   - 그러나 net effect는 -0.75pp로 개선 아닌 악화

4. **doc_D 난이도 문제는 retrieval이 아닌 generation/정답 구조 문제**
   - Q25 보안 준수사항: 검색에서 관련 청크를 찾아도 LLM이 키워드를 정확히 재현 못함
   - ctx prefix로 해결 불가 — 프롬프트 또는 정답 레이블 재검토 필요

5. **Contextual Retrieval은 한국어 RFP에서 효과 없음**
   - 원인 추정: (1) 이미 chunk_size=500이 충분한 맥락 제공, (2) 한국어 문서 구조가 영어와 달라 prefix 효과 감소, (3) text-embedding-3-small이 이미 양호한 의미 파악
   - 비용 대비 효과 전혀 없음 (1,034 LLM calls for prefix generation + 재임베딩)

### 결론
**EXP13은 부정적 결과**. Contextual Retrieval은 BidFlow의 한국어 RFP 분석에서 효과가 없음을 실증.
- 현재 최적 config는 EXP12의 ref_v2 (kw_v3=0.896) 또는 multi_query (kw_v3=0.900)로 유지
- Q25 보안 문제는 retrieval 개선이 아닌 다른 접근 필요 (프롬프트 engineering 또는 정답 구조 변경)

### 산출물

| 파일 | 설명 |
|------|------|
| `scripts/run_exp13.py` | EXP13 실험 스크립트 (Contextual Retrieval + Korean BM25) |
| `data/experiments/exp13_metrics.csv` | 150건 평가 결과 (5 configs × 30Q) |
| `data/experiments/exp13_report.json` | 실험 리포트 (수정 완료) |
| `data/exp13/contextual_chunks_doc_*.json` | LLM 생성 맥락 프리픽스 캐시 |
| `data/exp13/vectordb_ctx_doc_*/` | Contextual 임베딩 VDB (5문서) |
| `docs/planning/EXP13_plan.md` | 실험 계획서 |
| `docs/planning/EXP13_handoff_prompt.md` | 핸드오프 문서 |
