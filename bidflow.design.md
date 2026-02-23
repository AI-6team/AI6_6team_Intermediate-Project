---
template: design
version: 1.2
description: BidFlow 상세 설계서 - AI 기반 입찰 제안요청서(RFP) 분석 시스템
variables:
  - feature: BidFlow
  - date: 2026-02-03
  - author: 임창현
  - project: AI6_6team_Intermediate-Project
  - version: 1.0.0
---

# BidFlow 상세 설계서 (Detailed Design Document)

> **요약**: 근거 기반 검증(Evidence-based Validation)을 적용한 보안 강화형 지능형 RFP 분석 시스템 설계서입니다.
>
> **프로젝트**: AI6_6team_Intermediate-Project
> **버전**: 1.0.0
> **작성자**: 임창현
> **작성일**: 2026-02-03
> **상태**: 초안 (Draft)
> **기획서**: [AI6기_6팀_중급프로젝트_기획서.md](../AI6기_6팀_중급프로젝트_기획서.md)

---

## 1. 개요 (Overview)

### 1.1 설계 목표 (Design Goals)

*   **근거 기반 검증 (Evidence-Based Validation)**: 모든 추출 및 판정 결과는 반드시 문서 내 구체적 좌표(페이지, 오프셋, 셀 위치)를 근거로 제시해야 합니다.
*   **관심사의 분리 (Separation of Concerns)**: 파싱(구조화), 추출(LLM), 검증(Rule-based)을 명확히 분리하여 테스트 가능성을 확보합니다.
*   **안전성 및 신뢰성 (Safety & Reliability)**: 모호한 정보는 안전하게 "Gray Zone"으로 처리하고, 3-Rail 보안 아키텍처를 구현합니다.
*   **관측 가능성 (Observability)**: Langfuse를 통합하여 RAG 파이프라인의 전체 과정을 추적(Traceability) 가능하게 만듭니다. (Phase 2 고도화)

### 1.2 설계 원칙 (Design Principles)

*   **단일 책임 원칙 (SRP)**: 각 모듈(Parser, Extractor, Validator)은 하나의 역할만 수행합니다.
*   **실패 안전 기본값 (Fail-Safe Defaults)**: 불확실할 경우 GRAY(AMBIGUOUS)를 반환하며, 절대 GREEN/RED를 환각(Hallucination)으로 생성하지 않습니다.
*   **불변성 (Immutability)**: `ExtractionResult`(추출 결과)와 `ValidationResult`(판정 결과)는 불변(Snapshot)이며, `CompanyProfile`(프로필)만 가변입니다.
*   **추적성 (Traceability)**: 모든 산출물은 원본 RFP 문서의 버전(`doc_hash`)으로 추적 가능해야 합니다.

---

## 2. 아키텍처 (Architecture)

### 2.1 컴포넌트 다이어그램 (Component Diagram)

```mermaid
graph TD
    User[사용자 (Streamlit UI)] -->|업로드/조회| API[FastAPI 서버]
    API -->|저장/로드| Storage[로컬 파일시스템 / 벡터 저장소]
    API -->|파싱| Parser[파싱 엔진 (Hybrid)]
    API -->|추출| Extractor[멀티스텝 추출기 (LangChain)]
    API -->|검증| Validator[룰 기반 검증기]
    
    subgraph 데이터 파이프라인
        Parser -->|구조화된 문서| Indexer[이중 인덱서]
        Indexer --> TextIndex[텍스트 인덱스 (Chroma/FAISS)]
        Indexer --> TableIndex[테이블 인덱스]
    end
    
    subgraph 로직 레이어
        Extractor -->|질의| Retriever[하이브리드 검색기]
        Retriever -->|문맥| Extractor
        Extractor -->|Compliance Matrix| Validator
        Validator -->|판정| API
    end
    
    subgraph 관측 가능성
        API -.->|트레이스| Langfuse[Langfuse]
        Extractor -.->|생성 로그| Langfuse
    end
```

### 2.2 데이터 흐름 (Data Flow)

1.  **Ingest (수집)**: 사용자 PDF 업로드 → **[Input Rail]** 파일 무결성/타입 체크 → Parser가 텍스트/표 추출 → `Document` 객체 생성.
2.  **Index (색인)**: `Document` 청킹 → **Indexer** (Input: `Doc`, Output: `Vector/Keyword Index`) → 저장소 적재.
3.  **Extract (추출)**:
    *   **[Process Rail]** 프롬프트 인젝션 감지.
    *   **Retriever** (Input: `Query`, Output: `Top-K Chunks`) → 문맥 제공.
    *   **G1~G4 Multi-step**: 순차적 추출 수행 (G1 결과 → G2/G3/G4 컨텍스트/라우팅 주입).
4.  **Validate (검증)**:
    *   입력: `ComplianceMatrix` + `CompanyProfile`.
    *   로직: Rule Engine 실행 (비교/판정).
    *   출력: `ValidationResult` (불변 스냅샷).
5.  **Serve (제공)**:
    *   **[Output Rail]** 근거 없는 단정(Hallucination) 필터링 및 GRAY 다운그레이드.
    *   UI 결과 표시.

### 2.3 의존성 (Dependencies)

| 컴포넌트 | 의존성 | 목적 |
|---|---|---|
| **Frontend** | Streamlit, Pandas | UI 렌더링, 데이터 시각화 |
| **Backend** | FastAPI, Pydantic | API 엔드포인트, 데이터 검증 |
| **LLM** | LangChain, OpenAI API | 오케스트레이션, 추론 |
| **Vector DB** | ChromaDB (or FAISS) | 시맨틱 검색 |
| **Monitoring** | Langfuse | 트레이싱, 평가 데이터셋 수집 |
| **Parsing** | PyMuPDF, pdfplumber | PDF 텍스트/테이블 추출 |

---

## 3. 데이터 모델 (Data Model)

### 3.1 핵심 엔티티 (Core Entities)

```python
# Pydantic 스키마 정의

class Evidence(BaseModel):
    source_type: Literal["text", "table"]
    page_no: int
    text_snippet: str # 원문 스니펫 (요약본 아님) 또는 셀 텍스트
    # Text Evidence
    chunk_id: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    # Table Evidence
    table_id: Optional[str] = None
    row_idx: Optional[int] = None
    col_idx: Optional[int] = None
    coords: Optional[List[float]] = None # [x0, y0, x1, y1] 하이라이팅 좌표

class ExtractionSlot(BaseModel):
    key: str 
    value: Union[str, int, float, List[str], None]
    status: Literal["FOUND", "NOT_FOUND", "AMBIGUOUS"]
    evidence: List[Evidence]
    integrity_score: float = 1.0 # table_integrity_score와 연동

class ValidationResult(BaseModel):
    slot_key: str
    decision: Literal["GREEN", "RED", "GRAY"]
    reasons: List[str]
    evidence: List[Evidence] # 검증에 사용된 핵심 근거
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] # HIGH 조건: integrity < 0.65 또는 claim-evidence 불일치
    timestamp: datetime = Field(default_factory=datetime.now) # 불변 스냅샷 시점
```

### 3.2 데이터베이스 스키마 (Local JSON/SQLite)

*   **Users**: (MVP: 단일 관리자)
*   **Documents**: `id`, `filename`, `hash`, `upload_date`, `status`, `path`
*   **Profiles**: `id`, `name`, `data (JSON)`
*   **Analyses**: `id`, `doc_id`, `profile_id`, `result_json`, `created_at`

---

## 4. API 명세 (API Specification)

### 4.1 엔드포인트 목록

| 메서드 | 경로 | 설명 | 접근 권한 | 
|---|---|---|---|
| POST | `/api/upload` | RFP PDF 업로드 | Token/Internal |
| POST | `/api/extract/{doc_id}` | 추출 실행 | Token/Internal |
| POST | `/api/validate` | 검증기 실행 (프로필 + 매트릭스) | Token/Internal |
| GET | `/api/documents` | 문서 목록 조회 | Token/Internal |
| GET | `/api/documents/{doc_id}/view` | 뷰어용 파싱 내용 조회 | Token/Internal |

### 4.2 상세 명세 예시

#### `POST /api/extract/{doc_id}`

*   **설명**: 멀티스텝 추출 파이프라인을 실행합니다.
*   **요청**: `{"force_refresh": false}`
*   **응답**:
    ```json
    {
      "task_id": "uuid",
      "status": "processing"
    }
    ```
    (비동기 처리를 권장하나, MVP에서는 30초 이내라면 동기 방식으로 단순화 가능)

#### `POST /api/validate`

*   **설명**: 회사 프로필과 추출된 슬롯 데이터를 기반으로 적격 여부를 판정합니다.
*   **요청**:
    ```json
    {
      "matrix": { ... },     // ComplianceMatrix (Extraction Result)
      "profile": { ... }     // CompanyProfile (Editable)
    }
    ```
*   **응답**:
    ```json
    [
      {
        "slot_key": "license_auth",
        "decision": "RED",
        "reasons": ["보유 면허 '정보통신공사업'이 요구사항 '소프트웨어사업자'와 불일치"],
        "evidence": [...]
      }
    ]
    ```

---

## 5. UI/UX 디자인

### 5.1 화면 레이아웃 (Streamlit)

*   **사이드바**: 내비게이션 (업로드, 대시보드, 프로필, 설정), 업로드 위젯.
*   **메인 영역**:
    *   **상단**: 문서 요약 (사업명, 발주처, D-Day).
    *   **중앙**: Compliance Matrix (테이블 뷰).
        *   행: 요구사항 (예산, 면허, 인력 등).
        *   열: 요구사항 내용, 추출값, 우리 회사 프로필, 상태 (배지).
    *   **우측 (확장형)**: 근거 뷰어 (Evidence Viewer). 
        *   **Text Mode**: 원문 텍스트 하이라이트 (offset 기반).
        *   **Table Mode**: 원본 테이블 이미지 위 Bounding Box 오버레이 (coords 기반).

### 5.2 사용자 흐름 (User Flow)

1.  **업로드**: 사이드바에 PDF 드래그 & 드롭 → 시스템이 파싱 및 테이블 무결성 위험 체크.
2.  **분석**: 추출 자동 시작 → 진행률 표시 (G1...G4).
3.  **검토**: 사용자가 초기 "신호등" 확인 (GREEN/RED/GRAY).
4.  **수정**: "부적격(RED)" 항목 클릭 → 프로필 업데이트 ("이 면허 실제로는 보유함") → 시스템 즉시 재검증 → GREEN으로 변경.
 
---

## 6. 프로젝트 구조 (Clean Architecture)

```
bidflow/
├── .env
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/                # 설정 (YAML)
│   └── dev.yaml
├── data/                   # 데이터 저장소
│   ├── raw/
│   ├── processed/
│   └── vectordb/
├── experiments/            # [Phase 2] 실험 및 평가
├── src/
│   ├── bidflow/
│   │   ├── __init__.py
│   │   ├── main.py         # 진입점 (FastAPI)
│   │   ├── core/           # 설정, 로깅, 에러 처리
│   │   ├── domain/         # 비즈니스 로직 (슬롯, 규칙)
│   │   │   ├── models.py
│   │   │   └── rules.py
│   │   ├── ingest/         # 파싱 및 로딩
│   │   │   └── pdf_parser.py
│   │   ├── extraction/     # LangChain 파이프라인
│   │   │   ├── pipeline.py # G1~G4 오케스트레이션
│   │   │   ├── chains.py   # 개별 체인
│   │   │   └── prompts/    # 프롬프트 템플릿 (Markdown)
│   │   ├── validation/     # 검증 엔진
│   │   │   └── engine.py
│   │   ├── api/            # API 라우터
│   │   │   └── routers/
│   │   └── ui/             # Streamlit 앱
│   │       ├── Home.py
│   │       └── pages/
└── tests/
    ├── unit/
    └── integration/
```

---

## 7. 구현 로드맵 및 체크리스트

### 7.1 설정 및 기초 (현재 단계)
- [x] Git 레포지토리 및 폴더 구조 초기화
- [x] 가상환경 생성 및 의존성 설치 (`requirements.txt` / `pyproject.toml`)
- [ ] `configs/dev.yaml` 및 `.env` 설정
- [ ] Langfuse 설정

### 7.2 Phase 1: Ingest & Parse
- [ ] `PDFParser` 구현 (텍스트 + 테이블)
- [ ] `DocumentStore` 구현 (로컬 파일시스템)
- [ ] `VectorStore` (Chroma/FAISS) 수집 구현

### 7.3 Phase 1: Extract & Validate
- [ ] `Slots` 및 `Pydantic Models` 정의
- [ ] `Multi-step Extraction Chain` 구현
- [ ] `RuleValidator` 구현
- [ ] `Langfuse` 트레이싱 연동 (Extraction Chain 단위)

### 7.4 Phase 1: UI & Integration
- [ ] Streamlit 레이아웃 구성
- [ ] UI와 백엔드 API 연동
- [ ] 근거 하이라이팅(Overlay) 구현

---

## 8. 개발 환경 설정 가이드

### 8.1 필수 조건
*   Python 3.10 이상
*   Git
*   OpenAI API Key
*   Langfuse Public/Secret Keys

### 8.2 설치 방법
```bash
# 1. 클론 및 이동
git clone ...
cd bidflow

# 2. 가상환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -e .

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 입력
```
