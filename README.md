# BidFlow

한국어 입찰제안요청서(RFP) 분석을 위한 AI 기반 RAG 시스템

## 프로젝트 개요

BidFlow는 공공기관 입찰 RFP 문서(HWP, PDF, DOCX, HWPX)를 업로드하면 자동으로 핵심 정보를 추출하고, 입찰 참여 적합성을 판정하는 시스템입니다.

### 주요 기능

- **다중 포맷 파싱**: HWP(hwp5txt + hwp5html 하이브리드), PDF(PyMuPDF + pdfplumber), DOCX, HWPX 지원
- **하이브리드 검색**: BM25 + Vector (Weighted RRF, alpha=0.7) + CrossEncoder Reranking
- **멀티스텝 추출 (G1~G4)**: 사업 기본정보, 일정, 자격요건, 배점표를 단계별 LLM 체인으로 추출
- **정규식 힌트 주입**: 금액/날짜 패턴을 사전 감지하여 LLM 정확도 향상
- **쿼리 분석기**: 질문 유형(summary/extraction/decision) 자동 분류
- **Front-loading**: 문서 앞부분 청크를 강제 포함하여 기본정보 recall 보장
- **규칙 기반 판정**: 자격요건(면허, 신용등급, 지역제한) + 예산/마감일/정보완전성 검증
- **보안**: 3-Rail 아키텍처 (Input Rail, HWP Deep Scan, PII Masking)
- **Q&A**: RAG 기반 자유 질의응답

## 팀원

| 이름 | 역할 |
|------|------|
| 임창현 | RAG 파이프라인 설계, 13회 실험 최적화, 시스템 통합 |
| 김보윤 | QueryAnalyzer, DecisionEngine, DOCX/HWPX 파서 |
| 김슬기 | Front-loading 전략, 정규식 힌트 주입 |

## 아키텍처

```
User (Streamlit UI)
  │
  ├─ Upload ──→ RFPLoader ──→ Parser (HWP/PDF/DOCX/HWPX)
  │                │              │
  │                ▼              ▼
  │           DocumentStore    VectorStore (ChromaDB)
  │                               │
  ├─ Q&A ────→ RAGChain ──→ HybridRetriever (BM25 + Vector)
  │                │              │
  │                │              ▼
  │                │         Reranker (bge-reranker-v2-m3)
  │                │              │
  │                ▼              ▼
  │           LLM (gpt-5-mini) ← Context
  │
  ├─ Extract ─→ ExtractionPipeline (G1→G2/G3→G4)
  │                │
  │                ▼
  └─ Validate ─→ RuleBasedValidator ──→ 참여 권장/검토 필요/참여 보류
```

## 실험 결과 요약

13회 반복 실험을 통해 최적 설정을 도출했습니다.

| 실험 | 내용 | 결과 (kw_v3) |
|------|------|-------------|
| EXP01~03 | 청킹, 검색, 프롬프트 기초 실험 | 0.72~0.81 |
| EXP04~06 | 테이블 인식, 정규화, 파이프라인 개선 | 0.81~0.83 |
| EXP07~09 | Table-aware RAG, EDA, 일반화 검증 | 0.83~0.86 |
| EXP10 | HWP 파서 V4 하이브리드 + 멀티문서 검증 | 0.896 |
| EXP11 | 프롬프트 엔지니어링 (과도 제약 역효과 확인) | < 0.896 |
| EXP12 | Retrieval 최적화 (multi-query +0.4pp) | **0.900** |
| EXP13 | Contextual Retrieval (한국어 RFP 비효과 확인) | < 0.896 |

**최적 설정**: chunk_size=500, alpha=0.7, pool_size=50, top_k=15, Prompt V2 (table-aware)

## 프로젝트 구조

```
bidflow/
├── configs/               # 설정 파일 (YAML)
├── data/experiments/       # 실험 결과 (CSV, JSON)
├── docs/                  # 문서 (발표 보고서 등)
├── notebooks/             # 실험 노트북 (EXP01~10)
├── scripts/               # 실험 스크립트 (EXP10~13)
├── src/bidflow/
│   ├── apps/ui/           # Streamlit UI (Home, Upload, Matrix, Profile, Decision, Eval)
│   ├── core/              # 설정 관리
│   ├── domain/            # 도메인 모델 (Pydantic)
│   ├── extraction/        # 멀티스텝 추출 (G1~G4 체인, 힌트 감지)
│   ├── ingest/            # 문서 로딩, 저장, 벡터DB 관리
│   ├── parsing/           # 파서 (HWP, PDF, DOCX, HWPX)
│   ├── retrieval/         # RAG 체인, 하이브리드 검색, 리랭킹, 쿼리 분석
│   ├── security/          # 보안 (Input Rail, PII Masking)
│   └── validation/        # 규칙 기반 검증
├── tests/                 # 테스트
├── pyproject.toml         # 의존성
└── bidflow.design.md      # 상세 설계서
```

## 설치 및 실행

### 1. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. 환경 변수

```bash
# .env 파일 생성
OPENAI_API_KEY=your-api-key
LANGFUSE_SECRET_KEY=your-langfuse-key    # (선택)
LANGFUSE_PUBLIC_KEY=your-langfuse-key    # (선택)
```

### 3. 실행

```bash
# Streamlit UI
streamlit run src/bidflow/apps/ui/Home.py

# FastAPI 서버
uvicorn src.bidflow.main:app --reload
```

## 기술 스택

- **LLM**: GPT-5-mini (OpenAI)
- **Embedding**: text-embedding-3-small
- **Reranker**: BAAI/bge-reranker-v2-m3
- **Vector DB**: ChromaDB
- **Framework**: LangChain, FastAPI, Streamlit
- **Observability**: Langfuse
