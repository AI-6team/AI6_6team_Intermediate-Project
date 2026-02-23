# EXP13 핸드오프 프롬프트

아래 프롬프트를 다음 채팅의 첫 메시지로 사용하세요.

---

## 프롬프트 시작

EXP13: Contextual Retrieval + 한국어 BM25 최적화 실험을 진행해줘.

### 배경
BidFlow RAG 시스템 (한국어 RFP 문서 분석)에서 EXP10~12를 통해 파싱/청킹/retrieval 파라미터/프롬프트/임베딩을 최적화했고, 현재 best는 kw_v3=0.900 (multi_query). 하지만 doc_D 보안 문항(kw_v3=0.211)을 포함한 ~10개 문항이 여전히 실패 중이야. 근본 원인은 **청크에 문서 내 위치/섹션 맥락이 없어서 검색이 안 되는 것**이야.

### 해야 할 일
`docs/planning/EXP13_plan.md`에 상세 계획이 있어. 이걸 참조해서 `scripts/run_exp13.py`를 작성하고 실행해줘.

### 핵심 요구사항

1. **Contextual Prefix 생성** (가장 중요)
   - 기존 VDB(`data/exp10e/vectordb_c500_doc_*`)에서 각 문서의 청크를 로드
   - 각 청크에 대해 gpt-5-mini로 "[문서: {제목} | {장/절} | {소주제}]" 형태의 짧은 맥락 프리픽스 생성
   - 문서 전체 텍스트(또는 목차)를 참조하여 정확한 위치 파악
   - 결과를 `data/exp13/contextual_chunks_{doc_key}.json`에 캐싱
   - prefix + 원본 청크를 합쳐 새 VDB에 인덱싱: `data/exp13/vectordb_ctx_{doc_key}/`

2. **한국어 형태소 분석 BM25** (ctx_bm25_ko config)
   - `kiwipiepy`로 한국어 형태소 분석 토크나이저 구현
   - `BM25Retriever.from_documents(docs, preprocess_func=korean_tokenizer)`
   - 먼저 `pip install kiwipiepy` 필요

3. **5개 설정 평가**
   - `ref_v2`: EXP12 baseline 재활용 (API 호출 불필요)
   - `ctx_basic`: Contextual prefix만 적용
   - `ctx_bm25_ko`: ctx_basic + Kiwi 한국어 BM25
   - `ctx_multi_query`: ctx_basic + multi_query (EXP12에서 가져옴)
   - `ctx_full`: ctx_basic + bm25_ko + multi_query

4. **스크립트 구조는 `scripts/run_exp12.py`를 템플릿으로 사용**
   - `ExperimentRetriever` 클래스, `build_retriever()`, `invoke_rag()` 재활용
   - `keyword_accuracy_v2/v3` 평가 함수 동일
   - 증분 CSV 저장 (문항별)
   - Resume 지원 (이미 완료된 config 스킵)

### 파일 위치
- 프로젝트 루트: `E:/Codeit/AI6_6team_Intermediate-Project/bidflow/`
- 계획서: `docs/planning/EXP13_plan.md`
- 기존 VDB: `data/exp10e/vectordb_c500_doc_{A,B,C,D,E}`
- 테스트셋: `data/experiments/golden_testset_multi.csv` (30문항, 5개 문서)
- EXP12 스크립트 (템플릿): `scripts/run_exp12.py`
- EXP12 결과: `data/experiments/exp12_metrics.csv`
- 실행 이력: `docs/planning/HISTORY_v2_execution.md`

### 문서별 정보 (DOC_CONFIGS)
```python
DOC_CONFIGS = {
    "doc_A": {"name": "수협중앙회", "source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp", "doc_type": "text_only"},
    "doc_B": {"name": "한국교육과정평가원", "source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp", "doc_type": "table_simple"},
    "doc_C": {"name": "국립중앙의료원", "source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp", "doc_type": "table_complex"},
    "doc_D": {"name": "한국철도공사", "source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp", "doc_type": "mixed"},
    "doc_E": {"name": "스포츠윤리센터", "source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp", "doc_type": "hwp_representative"},
}
```

### 현재 최적 설정 (변경 금지)
- Embedding: text-embedding-3-small (OpenAI)
- LLM: gpt-5-mini
- Reranker: BAAI/bge-reranker-v2-m3
- alpha: 0.7, pool_size: 50, top_k: 15
- Prompt: V2 (table-aware)
- Parser: V4_hybrid (hwp5txt text + hwp5html table)
- chunk_size: 500, chunk_overlap: 50

### 성공 기준
- Overall kw_v3 > 0.92 (현재 0.900)
- doc_D kw_v3 > 0.85 (현재 ~0.74)
- Q25(보안) kw_v3 > 0.7 (현재 0.211)

### 실행 순서
1. `pip install kiwipiepy` 확인
2. `scripts/run_exp13.py` 작성
3. `cd bidflow && python -u scripts/run_exp13.py` 실행
4. 결과 분석 후 `docs/planning/HISTORY_v2_execution.md` 업데이트

---
