# BidFlow 팀 프로젝트 병합 핸드오프 프롬프트

아래 프롬프트를 새 채팅에 붙여넣어 사용하세요.

---

## 프롬프트

나는 한국어 입찰제안요청서(RFP) 분석을 위한 RAG 시스템을 만드는 팀 프로젝트를 진행 중이야. 3명의 팀원이 공통 기획서를 기반으로 각자 개발했고, 이제 **각자의 장점을 하나의 최종 프로젝트로 병합**해야 해.

### 프로젝트 위치
```
E:/Codeit/AI6_6team_Intermediate-Project/
├── bidflow/                      # 내 프로젝트 (임창현)
├── RFP_RAG_project(김보윤)/       # 팀원 김보윤
├── 김슬기/                       # 팀원 김슬기
└── AI6기_6팀_중급프로젝트_기획서.md  # 공통 기획서
```

### 내 프로젝트(bidflow) 분석 가이드

내 프로젝트는 **13회 실험(EXP01~13)**을 거쳐 최적화된 RAG 파이프라인이야. 핵심 파일들:

**아키텍처 이해:**
- `src/bidflow/retrieval/hybrid_search.py` — Weighted RRF 하이브리드 검색 (핵심 로직)
- `src/bidflow/retrieval/rerank.py` — BAAI/bge-reranker-v2-m3 리랭커
- `src/bidflow/retrieval/rag_chain.py` — LangChain LCEL 체인
- `src/bidflow/parsing/hwp_parser.py` — hwp5txt 기반 HWP 파서
- `src/bidflow/parsing/hwp_html_parser.py` — hwp5html 기반 테이블 추출
- `src/bidflow/indexing/store.py` — ChromaDB 벡터스토어

**실험 결과 & 최적 설정:**
- `docs/planning/HISTORY_v2_execution.md` — 전체 실험 기록 (반드시 읽기)
- `data/experiments/exp12_metrics.csv` — EXP12 결과 (최고 성능)
- `data/experiments/exp13_metrics.csv` — EXP13 결과 (Contextual Retrieval, 실패)
- `scripts/run_exp12.py` — 실험 스크립트 (invoke_rag, build_retriever 로직 참고)

**최적 config (EXP12 결론):**
| 항목 | 값 |
|------|-----|
| Parser | V4_hybrid (hwp5txt text + hwp5html table) |
| chunk_size | 500 tokens, overlap 50 |
| Embedding | text-embedding-3-small |
| Hybrid | BM25(0.3) + Vector(0.7), pool_size=50 |
| Reranker | BAAI/bge-reranker-v2-m3, top_k=15 |
| LLM | gpt-5-mini |
| Prompt | V2 (table-aware, col_path 직렬화) |
| **Best kw_v3** | **0.900** (multi_query 적용 시) |

**UI/API:**
- `src/bidflow/apps/api/` — FastAPI
- `src/bidflow/apps/ui/` — Streamlit (5 pages)

### 3명의 핵심 차별점 (병합 시 참고)

| 구분 | 내 프로젝트(bidflow) | 김보윤 | 김슬기 |
|------|---------------------|--------|--------|
| **강점** | 13회 실험으로 검증된 최적 파라미터, Weighted RRF, Reranker, col_path 테이블 직렬화, kw_v3 메트릭 | Query Classification(요약/추출/의사결정), Rule-based Decision Engine, SourcedField 근거 추적, DOCX/HWPX 지원 | Front-loading(앞 15p 강제 주입), 정규식 힌트 주입, Grid Search 자동화(rag_optimizer.py) |
| **Vector DB** | ChromaDB | FAISS | ChromaDB |
| **Hybrid** | Weighted RRF (직접 구현) | 가중합 min-max norm | LangChain EnsembleRetriever |
| **Reranker** | ✅ BAAI/bge-reranker-v2-m3 | ❌ 없음 | ✅ BAAI/bge-reranker-v2-m3 |
| **LLM** | gpt-5-mini | gpt-5-mini | gpt-4o |
| **Chunk** | 500 tokens | 800 tokens | 500자 (최적화 후) |
| **특화 기능** | multi_query, 13회 실험 | DecisionEngine(입찰 참여 권고), 경험 DB | Front-loading, 정규식 패턴 감지 |
| **평가** | RAGAS + kw_v3 (30Q×5docs) | 자체 4-metric | PASS/FAIL + Hit Rate |

### 병합 전략 제안

**Base**: 내 프로젝트(bidflow)를 기반으로 사용 (실험적으로 가장 최적화됨)

**김보윤에서 가져올 것:**
1. `QueryAnalyzer` — 쿼리 유형 자동 분류 (summary/extraction/decision)
2. `DecisionEngine` + `rules.yaml` — 입찰 참여 여부 자동 판단
3. `SourcedField` 패턴 — 추출 값마다 근거 chunk_id 추적
4. DOCX/HWPX 파서 — 파일 형식 지원 확장

**김슬기에서 가져올 것:**
1. Front-loading 전략 — 사업개요 등 문서 앞부분 정보 보장
2. 정규식 힌트 주입 — 금액/날짜 패턴 감지 후 LLM 프롬프트에 힌트 제공
3. `rag_optimizer.py` 자동화 패턴 — 설정 탐색 자동화 아이디어

### 요청사항

1. 먼저 위 3개 프로젝트의 코드를 읽고 각 팀원의 접근 방식을 파악해줘
2. 내 프로젝트(bidflow)를 base로, 다른 팀원의 장점 기능을 병합하는 구체적 계획을 세워줘
3. 코드 충돌이 최소화되는 순서로 단계별 병합 계획을 제시해줘
4. 병합 후에도 기존 kw_v3=0.900 성능이 유지되는지 검증 방법도 포함해줘
