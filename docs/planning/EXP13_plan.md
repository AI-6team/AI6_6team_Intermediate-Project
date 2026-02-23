# EXP13: Contextual Retrieval + 한국어 BM25 최적화

## 1. 배경

### 1.1 현재 성능 (EXP12 기준)
- **Best**: multi_query kw_v3=**0.900** (baseline 0.896 대비 +0.4pp)
- 30문항 중 **여전히 ~10문항이 kw_v3 < 0.8** (미달)
- 프롬프트 엔지니어링(EXP11)과 retrieval 파라미터 튜닝(EXP12) 모두 한계 도달

### 1.2 핵심 실패 패턴 (ref_v2 baseline 기준)

| 문항 | doc | kw_v3 | 실패 원인 |
|------|-----|-------|----------|
| Q25 (보안 세부항목) | doc_D | 0.211 | **치명적 검색 실패** — 196개 청크에서 "보안" 섹션 관련 청크 식별 불가 |
| Q23 (평가방법 장번호) | doc_D | 0.625 | 목차/장번호 메타데이터 미보존 |
| Q24 (하자담보 위치) | doc_D | 0.667 | 마찬가지 — 섹션 위치 정보 부재 |
| Q31 (평가방식 장번호) | doc_E | 0.714 | 동일한 목차 참조 실패 |
| Q22 (추진배경) | doc_D | 0.885 | 일부 세부항목 누락 |

**공통 근본 원인**: 청크가 원본 문서에서 어느 섹션/장에 속하는지 맥락 정보가 없음.
→ "보안 준수사항"이라는 쿼리가 들어와도, 관련 청크의 page_content에는 "보안"이라는 단어 자체가 없을 수 있음 (해당 섹션 제목은 상위 문맥에만 존재).

### 1.3 미시도 방법

| 방법 | 예상 효과 | 구현 난이도 | 비용 |
|------|----------|------------|------|
| **Contextual Retrieval** | ★★★★★ | 중 | 1회 LLM/청크 (인덱싱 시) |
| **한국어 형태소 분석 BM25** | ★★ | 낮 | 무료 |
| Parent-Child Retrieval | ★★★ | 중 | 무료 |
| HyDE | ★★ | 낮 | 1회 LLM/질문 |

## 2. EXP13 실험 설계

### 2.1 핵심 아이디어: Contextual Retrieval

Anthropic이 제안한 방법 (2024):
> 각 청크를 인덱싱하기 전에, LLM에게 "이 청크가 전체 문서에서 어떤 위치/맥락에 있는지" 설명하는 짧은 프리픽스를 생성하게 하고, 이 프리픽스를 청크 앞에 붙여서 임베딩하고 BM25에도 저장.

**예시**:
```
# Before (현재)
"가. 공통사항\n1) 「보안업무규정」 및 「정보보안기본지침」을 준수한다.\n2) 용역 수행시..."

# After (Contextual)
"[문서: 한국철도공사 예약발매시스템 개량 ISMP 용역 | 8장 보안 준수사항 | 가. 공통사항]
가. 공통사항\n1) 「보안업무규정」 및 「정보보안기본지침」을 준수한다.\n2) 용역 수행시..."
```

이렇게 하면:
- "보안 준수사항" 쿼리 → BM25에서 "보안 준수사항" 키워드 매칭
- 벡터 검색에서도 "8장 보안" 의미가 임베딩에 반영
- doc_D Q25(kw_v3=0.211) 같은 치명적 실패 해결 가능

### 2.2 실험 설정 (5개 config, 2 Stage)

#### Stage 1: Contextual Prefix 생성 + 재인덱싱 (API 호출 필요)

| Config | 변경 사항 | 근거 |
|--------|----------|------|
| `ref_v2` | EXP12 baseline 재활용 | 참조 기준, API 호출 불필요 |
| `ctx_basic` | LLM으로 각 청크에 짧은 맥락 프리픽스 추가 | Contextual Retrieval 핵심 |
| `ctx_bm25_ko` | ctx_basic + Kiwi 한국어 형태소 분석 BM25 | 한국어 토크나이징 개선 |

#### Stage 2: 최적 조합 (Stage 1 결과 기반)

| Config | 변경 사항 | 근거 |
|--------|----------|------|
| `ctx_multi_query` | ctx_basic + multi_query (EXP12 최고) | 두 가지 개선 조합 |
| `ctx_full` | ctx_basic + bm25_ko + multi_query | 전체 스택 결합 |

### 2.3 Contextual Prefix 생성 방법

```python
CONTEXT_PROMPT = """다음은 '{doc_title}' 문서의 청크입니다.
이 청크가 문서 내에서 어떤 위치와 맥락에 있는지 간결하게 설명하세요.
반드시 2문장 이내로, 문서명/장/절/주제를 포함하세요.

<document_chunk>
{chunk_text}
</document_chunk>

맥락 설명:"""
```

**생성 규칙**:
- 전체 문서 텍스트를 참조하여 청크의 위치를 파악 (문서 요약 or 목차 활용)
- 각 청크당 1회 LLM 호출 (gpt-5-mini, temperature=0)
- 결과 형식: `[문서: {제목} | {장/절} | {소주제}]\n{원본 청크}`
- 캐싱: 한 번 생성하면 저장하여 재사용

**비용 추정**:
- 5개 테스트 문서 × 평균 ~60 청크 = ~300 청크
- 300 LLM 호출 × ~500 input tokens = ~150K tokens ≈ $0.02 (매우 저렴)

### 2.4 한국어 BM25 최적화 (Kiwi)

현재 BM25Retriever는 기본 토크나이저(공백 분리)를 사용하여 한국어 조사/어미가 분리되지 않음.

```python
# Before
"보안업무규정을" → ["보안업무규정을"]  # 조사 포함, "보안업무규정" 매칭 실패

# After (Kiwi 형태소 분석)
"보안업무규정을" → ["보안", "업무", "규정", "을"]  # 형태소 단위, 매칭 성공
```

**구현**:
```python
from kiwipiepy import Kiwi
kiwi = Kiwi()

def korean_tokenizer(text):
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith(('N', 'V', 'M'))]
    # 명사, 동사, 수식어만 추출

# LangChain BM25Retriever에 preprocess_func 전달
bm25 = BM25Retriever.from_documents(docs, preprocess_func=korean_tokenizer)
```

### 2.5 실행 계획

```
Phase 1: Contextual Prefix 생성 (~10분)
  - 5개 문서 × ~60 청크 = ~300 LLM 호출
  - 결과 저장: data/exp13/contextual_chunks_{doc_key}.json

Phase 2: 재인덱싱 (~5분)
  - Contextual 청크로 새 VDB 생성
  - data/exp13/vectordb_ctx_{doc_key}/

Phase 3: 평가 실행 (~40분)
  - 4개 설정 × 30문항 = 120 API 호출
  - Stage 2는 Stage 1 결과에서 런타임 결정

합계: ~170 LLM 호출(평가) + ~300 호출(prefix) + multi_query ~90 = ~560 LLM 호출
예상 시간: 50-60분
```

## 3. 구현: `scripts/run_exp13.py`

### 3.1 EXP12 템플릿 대비 주요 변경점

1. **Contextual Prefix 생성기 추가**
   - `generate_context_prefix(doc_title, full_doc_text, chunk_text)` → prefix 문자열
   - 문서 전체 텍스트(또는 목차 요약)를 참조하여 청크 위치 파악
   - 결과를 JSON으로 캐싱 (재실행 시 재활용)

2. **Contextual 청크 인덱싱**
   - 기존 VDB에서 원본 청크 로드 → prefix 생성 → prefix + 원본을 합쳐 새 VDB에 인덱싱
   - `page_content = f"{prefix}\n{original_chunk}"`
   - metadata에 `has_context=True`, `context_prefix=prefix` 추가

3. **한국어 BM25 통합**
   - `kiwipiepy` 패키지 사용
   - `BM25Retriever.from_documents(docs, preprocess_func=korean_tokenizer)`
   - Kiwi 싱글턴 인스턴스 관리

4. **Stage 2 자동 결정**
   - Stage 1 결과에서 ctx_basic vs ctx_bm25_ko 중 더 좋은 것 선택
   - 거기에 multi_query 결합

### 3.2 파일 목록

| 파일 | 작업 |
|------|------|
| `scripts/run_exp13.py` | **생성** — 메인 실험 스크립트 |
| `data/exp13/contextual_chunks_{doc_key}.json` | **출력** — 맥락 프리픽스 캐시 |
| `data/exp13/vectordb_ctx_{doc_key}/` | **출력** — Contextual VDB |
| `data/experiments/exp13_metrics.csv` | **출력** — 문항별 결과 |

### 3.3 참조 파일 (수정 없음)

| 파일 | 용도 |
|------|------|
| `scripts/run_exp12.py` | 스크립트 구조 템플릿 |
| `src/bidflow/parsing/table_chunker.py` | `chunk_v4_hybrid()` — 현재 파서 |
| `src/bidflow/retrieval/hybrid_search.py` | HybridRetriever 구조 참조 |
| `src/bidflow/retrieval/rerank.py` | rerank() 함수 재사용 |
| `data/exp10e/vectordb_c500_doc_*` | 기존 VDB (원본 청크 로드용) |
| `data/experiments/golden_testset_multi.csv` | 30문항 테스트셋 |

## 4. 의존성 확인

```
# 이미 설치됨
langchain, langchain-openai, langchain-chroma, sentence-transformers, openai

# 새로 필요
kiwipiepy  # pip install kiwipiepy (한국어 형태소 분석)
```

## 5. 성공 기준

| 지표 | 현재 (multi_query) | 목표 | 의미 |
|------|-------------------|------|------|
| Overall kw_v3 | 0.900 | **> 0.92** | +2pp 이상 개선 |
| doc_D kw_v3 | ~0.74 | **> 0.85** | 치명적 실패 해소 |
| Q25 (보안) kw_v3 | 0.211 | **> 0.7** | 핵심 실패 케이스 복구 |
| hard 질문 kw_v3 | ~0.72 | **> 0.80** | 구조적 질문 개선 |

## 6. 리스크 및 대안

| 리스크 | 완화 방안 |
|--------|----------|
| Context prefix가 너무 길어 청크 크기 초과 | prefix를 2문장(~100자) 이내로 제한 |
| LLM이 잘못된 맥락 생성 | temperature=0, 문서 전체 텍스트 참조 |
| Kiwi 설치 실패 | Stage 1에서 ctx_basic만으로도 효과 측정 가능 |
| API 비용 초과 | Contextual prefix 캐싱으로 1회만 호출 |

## 7. 참고: Contextual Retrieval 검증 논문/사례

- Anthropic "Contextual Retrieval" (2024.09): BM25+Contextual embedding으로 retrieval 실패율 49% 감소
- 핵심 인사이트: 단순히 청크 앞에 맥락을 추가하는 것만으로도 임베딩 품질과 BM25 매칭이 동시 개선
- 우리 시스템은 이미 Hybrid(BM25+Vector)+Reranker를 사용하므로, Contextual prefix 추가의 효과가 두 검색기 모두에서 나타날 것
