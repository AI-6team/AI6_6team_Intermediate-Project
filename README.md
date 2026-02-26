# BidFlow

한국어 입찰 제안요청서(RFP) 분석을 위한 보안 강화형 RAG 시스템

> 마지막 업데이트: 2026-02-26  
> 기준 코드: `src/bidflow` + `frontend`

## 1) 프로젝트 요약

BidFlow는 RFP 문서를 업로드하면:

1. 문서를 파싱/청킹/인덱싱하고
2. G1~G4 슬롯(기본정보/일정/자격/배점)을 추출한 뒤
3. 회사 프로필과 규칙 기반으로 적격성을 판정하고
4. 팀 단위로 의견(댓글/답글)과 판정 결과를 공유하는 시스템입니다.

핵심은 `LLM 추출`과 `규칙 기반 판정`을 분리해, 성능뿐 아니라 재현성과 통제 가능성을 확보한 점입니다.

### 기술 스택

| 영역 | 기술 |
|---|---|
| Backend | Python 3.x, FastAPI, SQLite(WAL) / PostgreSQL, LangChain |
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Axios |
| Vector DB | ChromaDB (text-embedding-3-small) |
| LLM | gpt-5-mini (추출), BAAI/bge-reranker-v2-m3 (리랭킹) |
| 평가 | RAGAS 0.4.3 (Faithfulness, Context Recall) |
| 관측 | Langfuse (추출/검색 체인 추적) |

## 2) 문제 정의

입찰 실무에서는 마감일, 자격요건, 보안/계약 조건 한 줄을 놓치면 탈락으로 이어질 수 있습니다.  
따라서 본 프로젝트는 단순 요약이 아니라 아래를 목표로 설계되었습니다.

- 누락 위험 감소: 긴 문서에서 핵심 슬롯을 구조화
- 검증 가능한 판단: 근거(Evidence) 기반 추출 + 룰 엔진 판정
- 사내 적용 전제: 인증/권한/데이터 분리/보안 레일 적용

## 3) 시스템 아키텍처

```text
Next.js Frontend
  ├─ /dashboard       문서 업로드(1개=단건, 2개 이상=배치 UI)
  ├─ /analysis        G1~G4 추출 실행/조회
  ├─ /validation      회사 프로필 기반 규칙 검증
  └─ /team            팀 문서/판정/코멘트 협업
          │
          ▼
FastAPI Backend (bidflow.main:app)
  ├─ /auth/*          로그인/회원가입/프로필(팀장 권한)
  ├─ /api/v1/ingest/* 업로드/문서 조회
  ├─ /api/v1/extract/* 추출 실행/조회
  ├─ /api/v1/validate 검증 실행
  └─ /api/v1/team/*   팀 문서/판정/코멘트
          │
          ▼
Data Layer
  ├─ Relational DB (SQLite 기본, PostgreSQL 전환 지원)
  ├─ File Storage (data/accounts/{user_id}/...)
  └─ Chroma Vector DB (tenant/user ACL metadata)
```

## 4) 기능별 구현 상세

### 4.1 문서 업로드/파싱/인덱싱

- FastAPI 업로드 엔드포인트: `POST /api/v1/ingest/upload`
- 구현 경로:
  - `src/bidflow/api/routers/ingest.py`
  - `src/bidflow/ingest/service.py`
  - `src/bidflow/ingest/storage.py`
- 동작:
  1. 파일 임시 저장
  2. 확장자 분기 파싱 (`.pdf`, `.hwp`)
  3. `DocumentStore`로 JSON/메타 저장
  4. `VectorStoreManager`로 Chroma 인덱싱

참고:
- 코드베이스에는 DOCX/HWPX 파서(`src/bidflow/parsing/docx_parser.py`, `hwpx_parser.py`)와 `RFPLoader` 기반 확장 경로도 포함되어 있습니다.
- 현재 Next.js 기본 업로드 UI와 `/api/v1/ingest/upload` 경로는 `.pdf`, `.hwp`를 기준으로 동작합니다.

### 4.2 파싱 전략

- PDF: `pdfplumber` 기반 텍스트/테이블 추출 (`src/bidflow/parsing/pdf_parser.py`)
- HWP:
  - 1차 `hwp5txt`
  - 실패 시 `olefile` fallback (`src/bidflow/parsing/hwp_parser.py`)
  - 필요 시 deep scan으로 숨은 스트림 검사
- HWP HTML/테이블 강화:
  - `hwp5html` + BeautifulSoup 기반 테이블 구조화
  - `col_path` 직렬화, rowspan/colspan 해소
  - `TableAwareChunker`로 텍스트/테이블 분리 청킹

### 4.3 저장소/멀티테넌시

- 핵심 모듈: `src/bidflow/ingest/storage.py`
- 구성:
  - `StorageRegistry`: 사용자/공유/팀 경로 계산
  - `DocumentStore`: 문서/추출결과/프로필/세션 저장
  - `VectorStoreManager`: 사용자별 Chroma 컬렉션 관리
- 데이터 경계:
  - 파일: `data/accounts/{user_id}/raw|processed|vectordb`
  - 팀 공유: `data/shared/teams/{team_name}/...`
  - 벡터 메타: `tenant_id`, `user_id`, `group_id`, `access_level`

### 4.4 검색/RAG

- 핵심 모듈:
  - `src/bidflow/retrieval/hybrid_search.py`
  - `src/bidflow/retrieval/rag_chain.py`
  - `src/bidflow/retrieval/structure_aware.py`
  - `src/bidflow/retrieval/rerank.py`
- 구현 방식:
  - BM25 + Vector 검색
  - Weighted RRF로 병합
  - 선택적 Cross-Encoder rerank
  - 문서 범위 필터(doc_hash) + ACL 필터(tenant/user/group)
  - Structure-aware (TOC 감지 + chapter prefix)
  - 힌트 주입(정규식), front-loading, answer postprocess 전략 지원
- 최종 확정 설정 (EXP22 기준):
  - chunk=500, overlap=50, hybrid alpha=0.7, top_k=20, pool_size=50
  - Reranker: BAAI/bge-reranker-v2-m3, Embedding: text-embedding-3-small
  - Prompt V5 (8규칙), SC 5회 (temp 0.0~0.5), stability_v1 후처리
  - 운영 선택: first_deterministic (temp=0.0), oracle gap 2.33pp

> 설정값별 확정 근거는 [종합 보고서 > 4.7 최종 확정 파이프라인 설정](docs/최종_종합보고서.md#47-최종-확정-파이프라인-설정-exp21-p1--exp22-기준) 참고

### 4.5 G1~G4 멀티스텝 추출

- 파이프라인: `src/bidflow/extraction/pipeline.py`
- 체인:
  - G1 기본정보 (`project_name`, `issuer`, `period`, `budget`)
  - G2 일정 (`submission_deadline`, `briefing_date`, `qna_period`)
  - G3 자격 (`required_licenses`, `region_restriction`, `financial_credit`, `restrictions`)
  - G4 배점표(`items`)
- 특징:
  - `with_structured_output` 기반 슬롯 강제 구조화
  - 실패 시 `_empty_slot` fallback 제공
  - 추출 결과를 `doc_hash_result.json` + SQLite에 동시 저장

### 4.6 규칙 기반 검증/판정

- 검증 모듈: `src/bidflow/validation/validator.py`
- 규칙:
  - 면허/신용등급/지역 제한
  - 예산 규모/마감 긴급도/정보 완전성
  - 규칙값은 `configs/decision_rules.yaml`에서 로드
- 종합 판정:
  - RED 존재 시 보류(리스크 우선)
  - GRAY만 존재 시 검토 필요
  - 전부 GREEN이면 참여 권장

### 4.7 인증/권한/팀 협업

- 인증/사용자: `src/bidflow/api/main.py`, `src/bidflow/api/deps.py`
- DB: `src/bidflow/db/crud.py`, `src/bidflow/db/database.py`
- 구현 포인트:
  - 회원가입/로그인/JWT 기반 인증
  - bcrypt 비밀번호 해싱(legacy SHA-256 로그인 시 자동 마이그레이션)
  - 역할(`member`, `leader`) 분리
  - 팀 프로필 1개 정책(팀장 업데이트, 팀원 fallback 동기화)
  - 팀 문서 집계, 문서별 판정 요약, 댓글/답글/삭제 API
  - in-memory API rate limiting (429 + Retry-After)

### 4.8 프론트엔드 UX 구현

- 위치: `frontend/app/*`, `frontend/components/*`, `frontend/lib/api.ts`
- 주요 페이지:

| 페이지 | 경로 | 핵심 기능 |
|---|---|---|
| 홈 | `/` | 로그인/회원가입 폼, 팀/역할 선택, 기능 소개 |
| 대시보드 | `/dashboard` | 드래그앤드롭 업로드, 배치(2개+) 자동 전환, 진행률 바, 문서 목록/상태 테이블 |
| 분석 | `/analysis` | G1~G4 탭 기반 추출 결과 뷰어, 비동기 폴링(3초), JSON 다운로드 |
| 검증 | `/validation` | 회사 프로필 조회 + 자동 규칙 검증, 색상 코딩(GREEN/RED/GRAY), 근거 표시 |
| 팀 | `/team` | 팀원별 문서 목록, 판정 배지(GO/NO-GO/REVIEW), 댓글/답글 스레딩 |
| 프로필 | `/profile` | 회사 프로필 편집(팀장 전용), 면허/지역/로고 관리, 팀원은 읽기 전용 |

- 공유 컴포넌트:
  - `Sidebar`: 좌측 네비게이션, 다크모드 토글, 접힘/펼침 상태
  - `UserHeader`: 상단 사용자 정보(이름/팀/역할), 로그아웃
  - `CommentSection`: 문서별 팀 댓글/답글, 작성/삭제, 타임스탬프
  - `Modal`: 인증 확인 등 범용 알림 모달

- UX 특징:
  - 다크모드 지원 (localStorage 영속)
  - localStorage 기반 결과 캐싱 (페이지 이동 시 데이터 유지)
  - 역할 기반 UI 분기 (Leader: 프로필 편집 가능, Member: 읽기 전용)
  - 실시간 추출 폴링 + 로딩 상태 시각화

- API 연동: `frontend/lib/api.ts`에서 `NEXT_PUBLIC_API_BASE_URL` 기준 통합, 전 요청 Bearer 토큰 자동 첨부

### 4.9 보안 구현

- 문서: `src/bidflow/security/README.md`
- 적용 레이어:
  - Input Rail: 인젝션 패턴 차단 (`security/rails/input_rail.py`)
  - PII Filter: 입력/출력 마스킹 및 탐지 (`security/pii_filter.py`)
  - Tool Gate: allowlist + SSRF 방어 (`security/tool_gate.py`)
  - HWP Deep Scan: 숨은 스트림 스캔 (`parsing/hwp_parser.py`)
  - 보안 로그: `logs/security.log`, `logs/audit.log` 회전 기록

> 보안 설계 원칙 및 PII 패턴 상세는 [종합 보고서 > 6. 보안 설계 및 적용 상태](docs/최종_종합보고서.md#6-보안-설계-및-적용-상태) 참고

## 5) API 엔드포인트 요약

| 영역 | 메서드 | 경로 | 설명 |
|---|---|---|---|
| Auth | POST | `/auth/register` | 회원가입 |
| Auth | POST | `/auth/login` | 로그인 토큰 발급 |
| Auth | GET/PUT | `/auth/me` | 내 정보 조회/수정(수정은 leader) |
| Ingest | POST | `/api/v1/ingest/upload` | 문서 업로드 + 파싱 + 인덱싱 |
| Ingest | GET | `/api/v1/ingest/documents` | 문서 목록 |
| Ingest | GET | `/api/v1/ingest/documents/{doc_id}/view` | 파싱된 문서 상세 조회 |
| Extraction | POST | `/api/v1/extract/{doc_id}` | G1~G4 추출 실행 |
| Extraction | GET | `/api/v1/extract/{doc_id}` | 추출 결과 조회 |
| Validation | POST | `/api/v1/validate` | 규칙 검증 실행 |
| Team | GET | `/api/v1/team/members` | 팀원 목록 |
| Team | GET | `/api/v1/team/documents` | 팀 문서 목록 |
| Team | GET | `/api/v1/team/decision/{doc_hash}` | 문서 판정 요약 |
| Comments | GET | `/api/v1/comments/{doc_hash}` | 문서별 댓글 조회 (팀 스코프) |
| Comments | POST | `/api/v1/comments` | 댓글 생성 |
| Comments | DELETE | `/api/v1/comments/{comment_id}` | 댓글 삭제 (작성자만) |
| Replies | POST | `/api/v1/comments/{comment_id}/replies` | 답글 생성 |
| Replies | DELETE | `/api/v1/replies/{reply_id}` | 답글 삭제 (작성자만) |
| Health | GET | `/health` | 서버 상태 확인 |

## 6) 실험 결과 요약 (EXP01~22)

기준 문서:

- `docs/planning/HISTORY_v2_execution.md` (최종 업데이트: 2026-02-25)
- `docs/planning/EXP21_phase_stability_execution.md`
- `docs/planning/EXP22_llmjudge_execution.md`

실험 여정 요약:

- **EXP01~03 (baseline 수립)**: 청킹/검색/프롬프트 최적화. chunk=500, hybrid alpha=0.5, 한국어 zero-shot 프롬프트 확정. 한국어 프롬프트 전환만으로 Faithfulness +10pp 향상
- **EXP04~07 (단일문서 고도화)**: verbatim 추출(+22pp), elbow 컷오프(+5.5pp), 테이블 구조화 등 개별 기법 기여 분리. 3-run 반복 검증 프로토콜 도입
- **EXP08~09 (일반화 검증)**: 100문서 EDA(96 HWP + 4 PDF) 후 다문서 검증에서 worst-group KW 0.258로 폭락 → 과적합 확인, 다문서 기준으로 전환
- **EXP10~18 (정밀 개선)**: V4_hybrid 파서, 오답 유형 분해(gen vs retrieval), self-consistency, metric v5 도입으로 단계적 개선

핵심 결과:

- EXP21(P1)에서 성능+안정성 동시 확보: `overall 0.9968`, 3-run gate pass `100%`
- EXP22에서 non-oracle 평가 체계 확정:
  - `kw_v5` 3-run mean `0.9742` (stdev `0.0104`)
  - Faithfulness `0.9402 ± 0.0045`
  - Context Recall `0.9778 ± 0.0019`
- 결론: 고점 성능뿐 아니라 실운영 기준(비-GT 의존) 신뢰성 검증 완료

주요 지표 설명:

- `kw_v5`: GT 키워드가 응답에 포함되는 비율. 정규화(조사 제거, 동의어 치환, 어미 제거 등) 후 3단계 매칭. v2→v3→v4→v5로 진화하며 한국어 RFP 특성에 맞춰 정교화
- `Gate`: split별(dev≥0.99, holdout/sealed≥0.95) kw_v5 평균 임계치 통과 여부
- `Oracle/Non-oracle`: SC 5회 생성 후 GT 기반 최선 선택(oracle) vs temp=0.0 고정 선택(non-oracle). oracle gap=2.33pp
- `Faithfulness/Context Recall`: RAGAS 표준 지표. 응답 근거 충실도 / 컨텍스트 회수율

> 지표 계산식 상세 및 버전별 변경점은 [종합 보고서 > 3.2 지표](docs/최종_종합보고서.md#32-지표) 참고
> 실험별 의사결정 로그(막힘→다음 실험 연결)와 상세 수치는 [종합 보고서 > 4. 실험 여정](docs/최종_종합보고서.md#4-실험-여정-막힘과-다음-실험의-연결) 참고

## 7) 프로젝트 구조

```text
bidflow/
├── configs/                    # base/dev/prod/exp/security 설정 (YAML)
├── data/                       # 로컬 데이터(대부분 gitignore)
├── docs/
│   ├── planning/               # 실험/운영 계획 및 실행 문서
│   └── final_presentation/     # 최종 발표 패키지
├── scripts/                    # 실험/검증 스크립트
├── src/bidflow/
│   ├── api/                    # FastAPI 엔드포인트 (main, deps, routers/)
│   ├── core/                   # 설정 로더, 프로젝트 루트 탐지
│   ├── db/                     # SQLite schema/CRUD (users, documents, comments)
│   ├── domain/                 # Pydantic 도메인 모델 (ComplianceMatrix, Evidence 등)
│   ├── eval/                   # 평가 데이터셋 빌더/리포트 생성
│   ├── extraction/             # G1~G4 추출 파이프라인, 힌트 탐지, 후처리
│   ├── indexing/               # BM25/임베딩 인덱싱
│   ├── ingest/                 # 저장/벡터DB/업로드 처리
│   ├── parsing/                # PDF/HWP/DOCX/HWPX 파서, 테이블 청킹
│   ├── retrieval/              # Hybrid/RAG/Structure-Aware/Rerank
│   ├── security/               # Input·Output·Process Rail, PII Filter, Tool Gate
│   ├── validation/             # 규칙 기반 검증 (validator, engine, rules/)
│   └── launcher.py             # FastAPI+Next.js 통합 실행
├── frontend/
│   ├── app/                    # Next.js 페이지 (dashboard, analysis, validation, team, profile)
│   ├── components/             # 공유 컴포넌트 (Sidebar, UserHeader, CommentSection, Modal)
│   └── lib/                    # API 클라이언트 (api.ts)
├── tests/security/             # 보안 테스트
└── pyproject.toml
```

## 8) 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .
cd frontend && npm install
```

## 9) 환경 변수 (`.env`)

```bash
OPENAI_API_KEY=your_openai_key

# 선택
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public

# FastAPI API key 매핑 (key:user,key:user)
BIDFLOW_API_KEYS=sk-local-dev:admin

# Next.js -> FastAPI 주소
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# JWT 인증
BIDFLOW_JWT_SECRET=replace-with-long-random-secret
BIDFLOW_JWT_EXPIRE_MINUTES=480
# 점진 전환 중 구형 토큰 허용이 필요할 때만 1
BIDFLOW_ALLOW_LEGACY_TOKEN=0

# DB (미설정 시 SQLite: data/bidflow.db)
# PostgreSQL 예시:
# BIDFLOW_DATABASE_URL=postgresql://user:password@localhost:5432/bidflow

# API Rate Limiting
BIDFLOW_RATE_LIMIT_ENABLED=1
BIDFLOW_RATE_LIMIT_AUTH_REQUESTS=20
BIDFLOW_RATE_LIMIT_AUTH_WINDOW_SECONDS=60
BIDFLOW_RATE_LIMIT_REQUESTS=240
BIDFLOW_RATE_LIMIT_WINDOW_SECONDS=60

# 런처 포트(선택)
BIDFLOW_API_PORT=8000
BIDFLOW_WEB_PORT=3000
```

## 10) 실행

```bash
# 통합 실행 (권장): FastAPI + Next.js
bidflow-run

# 개별 실행
uvicorn bidflow.main:app --reload
cd frontend && npm run dev
```

## 11) 테스트

```bash
pytest tests/security -q
```

## 12) 참고 문서

- 종합 보고서: `docs/최종_종합보고서.md`
- 실험 전체 이력: `docs/planning/HISTORY_v2_execution.md`
- 단일화 실행 기록: `docs/planning/next_fastapi_unification_execution_2026-02-26.md`
- 보안 아키텍처: `src/bidflow/security/README.md`

## 13) 데이터/보안 정책

- `data/raw` 원본 데이터셋은 GitHub 업로드 금지
- `local_only/`는 개인 메모/핸드오프 보관용으로 Git 제외
- 사용자 문서와 벡터 데이터는 기본적으로 사용자/팀 경계 내에서만 조회

## 14) Roadmap

| 우선순위 | 항목 | 설명 |
|:---:|---|---|
| 1 | 평가 파이프라인 개선 | judge context에 chapter prefix 포함, non-oracle 기준 gate 재정의(0.97) |
| 2 | 인증 고도화 | Refresh Token + 세션 강제 무효화(로그아웃/탈퇴/비밀번호 변경 연동) |
| 3 | 평가 자동화 | 정기 리그레션 + 품질 임계치 알람 |
| 4 | 배포 안정화 | CI/CD 파이프라인, 로그 대시보드, 장애 대응 절차 |
| 5 | 보안 확장 | Process/Output Rail 전면 적용, Redis 기반 분산 rate limiting |
| 6 | DB 운영 튜닝 | PostgreSQL 인덱스/커넥션 풀/백업·복구 리허설 |

> 상세 한계 분석 및 근거는 [종합 보고서 > 9. 한계와 다음 단계](docs/최종_종합보고서.md#9-한계와-다음-단계) 참고

## 15) Contributors

| 이름 | 역할 |
|---|---|
| 임창현 | RAG 파이프라인 설계/실험 운영, 최종 성능·안정화 의사결정 주도 |
| 김보윤 | QueryAnalyzer/DecisionEngine, 인증/팀 워크스페이스/프로필 정책 |
| 김슬기 | Front-loading/힌트 주입/구조 인지 검색, 보안 레이어 및 웹 연동 고도화 |
