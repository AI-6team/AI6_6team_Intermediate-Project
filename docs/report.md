# RFP RAG System - 프로젝트 보고서

**작성자**: AI 6팀 김보윤  

## 0. 이전 과정, 향후 계획

기존의 Langchain LCEL기반이 아닌 Retriver로 구현을 했었는데, 구현이 복잡해지고 프로젝트 목표(langchain 기반 RAG 구현)에도 맞지 않는 것으로 판단이 되어서 전체 코드를 바꾸고 있다.
- 향후 계획
작동은 하지만 아직 미흡하고 일부는 아직 제대로 동작되지 않는 부분이 있다. 우선 파이프라인을 완성한 후 수정할 계획이다.

현재 검색된 청크의 점수가 좋지 않은 부분이 많다. 따라서 retriever 이전 부분의 흐름을 검토하고 수정해야 한다.
마찬가지로 retriever가 가져온 정보에서 키워드와 관련된 근거인지 확인하는 절차를 더 보강해야할 것 같다.
+ Langgraph를 도입하고 성능 측정을 시도할 예정이다.




## 1. 프로젝트 개요

RFP(제안요청서) 문서를 대상으로 한 **RAG(Retrieval-Augmented Generation) 기반 질의응답 및 핵심 정보 추출 시스템**이다.
다수의 장문 RFP 문서에서 발주 기관, 사업 개요, 예산, 마감일, 제출 방법, 참가 자격 등 구조화된 정보를 자동으로 추출하고, 입찰 참여 여부에 대한 의사결정을 지원한다.

### 핵심 설계 원칙

- **No Hallucination**: Retriever가 반환한 청크 외 정보를 절대 사용하지 않으며, 모든 출력 필드에 근거 Chunk ID를 포함한다. 정보가 없으면 `"정보 없음"`을 반환한다.
- **Contract-Based Output**: 모든 모듈의 출력은 사전 정의된 JSON Schema(Pydantic 모델)를 따른다.

---

## 2. 시스템 아키텍처

```
[문서 입력 (PDF/DOCX/HWP/HWPX)]
       ↓
[텍스트 추출 & 정제]
       ↓
[토큰 기반 청킹 (600 tokens, 10% overlap)]
       ↓
[LangChain FAISS 벡터 인덱스 구축]

[사용자 질의]
       ↓
[Query Analyzer — 질의 유형 분류 (LCEL)]
       ↓
[Hybrid Retriever — FAISS + BM25 스코어 퓨전]
       ↓
[Generator — 구조화된 정보 추출 (LCEL)]
       ↓
[Decision Engine — 룰 기반 의사결정]
       ↓
[JSON 출력]
```

---

## 3. 기술 스택

| 구분 | 기술 | 용도 |
|------|------|------|
| LLM 프레임워크 | **LangChain** (langchain, langchain-openai, langchain-community) | LCEL 체인, BM25Retriever, FAISS 벡터스토어 통합 |
| LLM API | **OpenAI GPT-4o-mini** | 질의 분류(Analyzer) 및 정보 추출(Generator) |
| 임베딩 | **OpenAI text-embedding-3-small** (1536차원) | 청크 벡터 임베딩 |
| 벡터 DB | **FAISS** (faiss-cpu, LangChain 래퍼) | 벡터 유사도 검색 |
| 키워드 검색 | **BM25** (rank-bm25, LangChain BM25Retriever) | 키워드 기반 검색 |
| 토크나이저 | **tiktoken** (cl100k_base) | 토큰 기반 청킹 |
| 데이터 모델 | **Pydantic v2** | 입출력 계약(schema) 정의 및 검증 |
| 백엔드 API | **FastAPI + Uvicorn** | REST API 서버 |
| 프론트엔드 | **Streamlit** | 웹 기반 사용자 인터페이스 |
| 문서 추출 | **pdfplumber**, **python-docx**, **olefile** | PDF, DOCX, HWP/HWPX 텍스트 추출 |
| 설정 관리 | **PyYAML**, **python-dotenv** | 프롬프트/룰 YAML, 환경 변수 관리 |
| HTTP 클라이언트 | **httpx** | Streamlit → FastAPI 통신 |
| 데이터 처리 | **Pandas**, **NumPy** | CSV 로딩, 수치 처리 |

---

## 4. 디렉토리 구조

```
project-root/
├── main.py                    # CLI 진입점 (ingest / query / status)
├── run.py                     # FastAPI + Streamlit 동시 실행
├── streamlit_app.py           # Streamlit 웹 UI
├── requirements.txt           # Python 의존성
├── .env                       # 환경 변수 (OPENAI_API_KEY)
├── CLAUDE.md                  # 설계 사양서
│
├── configs/
│   ├── settings.py            # 중앙 설정 (경로, 모델, 파라미터)
│   ├── prompts.yaml           # LLM 프롬프트 템플릿
│   └── rules.yaml             # 의사결정 룰 정의
│
├── api/
│   └── server.py              # FastAPI 엔드포인트
│
├── src/
│   ├── models/
│   │   └── schema.py          # Pydantic 데이터 모델
│   ├── database/
│   │   ├── loader.py          # CSV 로딩 및 청킹
│   │   └── vector_store.py    # LangChain FAISS 벡터스토어
│   ├── core/
│   │   ├── analyzer.py        # 질의 유형 분류 (LCEL)
│   │   ├── retriever.py       # 하이브리드 검색 (FAISS + BM25)
│   │   ├── generator.py       # 구조화 정보 추출 (LCEL)
│   │   └── decision.py        # 룰 기반 의사결정
│   └── utils/
│       ├── formatting.py      # 텍스트 정제, 날짜 파싱, 예산 포맷
│       └── extractors.py      # 문서 형식별 텍스트 추출
│
├── scripts/
│   ├── ingest_batch.py        # 배치 인제스트 스크립트
│   └── run_query.py           # 단일 질의 스크립트
│
└── data/
    ├── raw/
    │   ├── data_list.csv      # 문서 메타데이터 원장
    │   └── files/             # 원본 PDF 파일
    ├── input/                 # 청크 JSON 저장
    ├── metadata/              # 문서 메타데이터 JSON 저장
    └── vector_db/             # FAISS 인덱스 파일
```

---

## 5. 모듈별 상세 설명

### 5.1 문서 입력 및 텍스트 추출 (`src/utils/extractors.py`)

4가지 문서 형식에서 텍스트를 추출한다.

| 형식 | 라이브러리 | 추출 방법 |
|------|-----------|-----------|
| PDF | pdfplumber | 페이지별 `extract_text()` |
| DOCX | python-docx | 문단(paragraph) 순회 |
| HWP | olefile | PrvText 스트림 (UTF-16LE) → BodyText 섹션 파싱 fallback |
| HWPX | zipfile + xml.etree | ZIP 내 Contents/section*.xml에서 `<t>` 태그 추출 |

### 5.2 문서 로딩 및 청킹 (`src/database/loader.py`)

- **CSV 로딩**: `data/raw/data_list.csv`에서 한국어 컬럼명을 영문 내부 필드로 매핑
- **Doc ID 생성**: `공고 번호` 기반, 없으면 파일명 MD5 해시
- **토큰 기반 청킹**: tiktoken(`cl100k_base`) 기준 600 토큰 단위, 10% 오버랩
- **Chunk ID 규칙**: `{doc_id}_chunk_{index:02d}`

### 5.3 벡터 스토어 (`src/database/vector_store.py`)

LangChain FAISS를 사용한 벡터 인덱스 관리 모듈이다.

- **임베딩**: `langchain_openai.OpenAIEmbeddings`로 `text-embedding-3-small` 모델 사용
- **인덱스 관리**: `langchain_community.vectorstores.FAISS`로 빌드/추가/검색/저장/로드
- **검색**: `similarity_search_with_score`로 L2 거리 기반 검색, `1/(1+distance)` 변환으로 유사도 반환
- **초기화**: `reset()` 메서드로 인덱스 파일 삭제 및 메모리 초기화
- **메타데이터 호환**: `chunks_meta` (pickle)를 별도 저장하여 기존 코드와 호환

### 5.4 질의 분석기 (`src/core/analyzer.py`)

LangChain LCEL 체인으로 사용자 질의를 분류한다.

- **체인 구성**: `ChatPromptTemplate | ChatOpenAI(.bind(response_format=json)) | JsonOutputParser`
- **질의 유형 분류**:
  - `summary`: RFP 전체 요약
  - `extraction`: 특정 필드 추출 (예산, 마감일 등)
  - `decision`: 입찰 참여 판단 (복합 질의 포함)
- **출력**: `QueryAnalysis` (query_type, required_fields, target_doc_id)
- **프롬프트 안전 처리**: `_prepare_template()` 함수로 YAML 내 JSON 예시의 중괄호를 이스케이프

### 5.5 하이브리드 검색기 (`src/core/retriever.py`)

FAISS 벡터 검색과 LangChain BM25 키워드 검색을 결합한다.

- **벡터 검색**: `VectorStore.search()`로 코사인 유사도 기반 검색
- **BM25 검색**: `langchain_community.retrievers.BM25Retriever`로 키워드 매칭, 랭크 기반 점수(`1/(1+rank)`) 부여
- **스코어 퓨전**: Min-Max 정규화 후 가중 합산 (`0.7 * vector + 0.3 * bm25`)
- **확장 검색**: `top_k * 3`개 후보를 가져온 뒤 퓨전 후 상위 K개 반환
- **문서 필터링**: `target_doc_id` 지정 시 해당 문서의 청크만 반환
- **TOP_K**: 기본값 10

### 5.6 정보 추출 생성기 (`src/core/generator.py`)

LangChain LCEL 체인으로 검색된 청크에서 구조화된 정보를 추출한다.

- **체인 구성**: `ChatPromptTemplate | ChatOpenAI(.bind(response_format=json)) | JsonOutputParser`
- **컨텍스트 포맷**: 각 청크에 `chunk_id`와 `관련도 점수`를 함께 전달하여 LLM이 신뢰도 높은 청크를 우선 참조
- **추출 힌트**: 프롬프트에 필드별 키워드 힌트 포함 (발주 기관, 예산, 마감일 등의 패턴)
- **추출 필드 (6개)**:
  - `issuing_agency`: 발주 기관
  - `project_overview`: 사업 개요
  - `budget`: 예산
  - `deadline`: 마감일
  - `submission_method`: 제출 방법
  - `qualification`: 참가 자격
- **출력**: `RFPSummary` — 각 필드는 `SourcedField(value, source[chunk_ids])`

### 5.7 의사결정 엔진 (`src/core/decision.py`)

YAML로 정의된 룰을 기반으로 입찰 참여 추천을 생성한다. LLM을 사용하지 않는 순수 룰 기반 모듈이다.

- **예산 룰**: 5천만 원 미만 또는 100억 초과 시 주의, 범위 내이면 적정
- **마감일 룰**: 7일 미만 긴급(red), 14일 미만 주의(yellow), 경과 시 보류
- **정보 완성도 룰**: 누락 필드 2개 초과 시 주의, 4개 초과 시 보류
- **신호 집계**: red 존재 → `"참여 보류"`, yellow 존재 → `"검토 필요"`, 모두 green → `"참여 권장"`

### 5.8 데이터 모델 (`src/models/schema.py`)

Pydantic v2 모델로 시스템 전체의 입출력 계약을 정의한다.

| 모델 | 용도 |
|------|------|
| `DocumentMetadata` | 문서 메타데이터 (doc_id, 사업명, 예산, 기관명 등) |
| `Chunk` | 단일 텍스트 청크 (chunk_id, text, token_count, metadata) |
| `QueryAnalysis` | 질의 분류 결과 (query_type, required_fields) |
| `RetrievalResult` | 검색 결과 (chunk_id, text, vector/bm25/final score) |
| `SourcedField` | 근거 추적 필드 (value + source chunk IDs) |
| `RFPSummary` | RFP 요약 (6개 SourcedField) |
| `DecisionSupport` | 의사결정 (recommendation + reason) |
| `RFPOutput` | 최종 출력 (RFPSummary + DecisionSupport) |

---

## 6. 최종 출력 형식

```json
{
  "rfp_summary": {
    "issuing_agency": { "value": "조달청", "source": ["RFP_001_chunk_02"] },
    "project_overview": { "value": "공공 데이터 통합 플랫폼 구축 사업", "source": ["RFP_001_chunk_01"] },
    "budget": { "value": "약 30억 원", "source": ["RFP_001_chunk_05"] },
    "deadline": { "value": "2024-11-15", "source": ["RFP_001_chunk_07"] },
    "submission_method": { "value": "나라장터 전자 제출", "source": ["RFP_001_chunk_08"] },
    "qualification": { "value": "정보시스템 구축 실적 보유 업체", "source": ["RFP_001_chunk_10"] }
  },
  "decision_support": {
    "recommendation": "검토 필요",
    "reason": "예산 규모 적정; 마감까지 12일 (주의); 정보 충분"
  }
}
```

---

## 7. 인터페이스

### 7.1 CLI (`main.py`)

```bash
# 배치 인제스트 (CSV → 청킹 → 임베딩 → FAISS 인덱스)
python main.py ingest

# 질의 실행
python main.py query "이 사업의 예산과 마감일은?"

# 결과를 파일로 저장
python main.py query "입찰 참여해야 하는가?" --output result.json

# 시스템 상태 확인
python main.py status
```

### 7.2 REST API (`api/server.py`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 인덱스 존재 여부, 문서/청크 수 |
| GET | `/api/documents` | 인덱싱된 문서 목록 |
| POST | `/api/upload` | 문서 업로드 및 인덱싱 (multipart) |
| POST | `/api/query` | RAG 파이프라인 질의 실행 |
| DELETE | `/api/reset` | 인덱스 초기화 (벡터 DB + 메타데이터 삭제) |

### 7.3 웹 UI (`streamlit_app.py`)

- **사이드바**: 시스템 상태 표시, 문서 업로드, 인덱싱된 문서 목록, 인덱스 초기화 (체크박스 확인 절차)
- **메인 영역**: 질의 입력 → 의사결정 추천 (색상 배지) + RFP 요약 6필드 + 검색된 청크 + 상세 JSON

### 7.4 동시 실행 (`run.py`)

```bash
python run.py  # FastAPI(8000) + Streamlit(8501) 동시 기동
```

---

## 8. 설정 파라미터

### 모델 설정 (`configs/settings.py`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| EMBEDDING_MODEL | text-embedding-3-small | 임베딩 모델 |
| EMBEDDING_DIMENSION | 1536 | 임베딩 차원 |
| CHAT_MODEL | gpt-4o-mini | LLM 모델 |
| CHAT_TEMPERATURE | 0.0 | 생성 온도 (결정적 출력) |
| CHAT_MAX_TOKENS | 4096 | 최대 생성 토큰 |
| CHUNK_SIZE_TOKENS | 600 | 청크 크기 (토큰) |
| CHUNK_OVERLAP_RATIO | 0.10 | 청크 오버랩 비율 |
| VECTOR_WEIGHT | 0.7 | 벡터 검색 가중치 |
| BM25_WEIGHT | 0.3 | BM25 검색 가중치 |
| TOP_K | 10 | 검색 결과 상위 K개 |

### 의사결정 룰 (`configs/rules.yaml`)

| 룰 | 조건 | 신호 |
|----|------|------|
| 예산 | < 5천만 원 또는 > 100억 원 | yellow |
| 마감일 | < 7일 | red |
| 마감일 | < 14일 | yellow |
| 정보 완성도 | 누락 > 4개 | red |
| 정보 완성도 | 누락 > 2개 | yellow |

---

## 9. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
echo "OPENAI_API_KEY=sk-..." > .env

# 3-A. CLI 실행
python main.py ingest                          # 인덱스 구축
python main.py query "예산이 얼마인가요?"        # 질의

# 3-B. 웹 UI 실행
python run.py                                  # API + UI 동시 기동
# 또는 개별 실행:
uvicorn api.server:app --port 8000             # API만
streamlit run streamlit_app.py --server.port 8501  # UI만
```


