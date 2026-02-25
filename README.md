# BidFlow

한국어 입찰제안요청서(RFP) 분석을 위한 보안 강화형 RAG 시스템

## 프로젝트 개요

BidFlow는 공공기관 입찰 RFP 문서(HWP/PDF/DOCX/HWPX)를 업로드하면 핵심 정보를 자동 추출하고, 회사 프로필과 비교해 입찰 적격성(Go/No-Go)을 판단합니다.

## 주요 기능

- **다중 포맷 파싱**: HWP(hwp5txt+hwp5html), PDF(PyMuPDF+pdfplumber), DOCX, HWPX
- **업로드 UI 통합**: 문서 1개 업로드 시 단일 분석, 2개 이상 업로드 시 다문서 일괄 분석 자동 전환
- **멀티스텝 추출(G1~G4)**: 기본정보, 일정/제출, 자격/결격, 배점표 단계 추출
- **하이브리드 검색**: BM25 + Vector(Chroma) + Rerank(Weighted RRF)
- **RAG 보강 전략**: 정규식 힌트 주입, Front-loading, Structure-aware 검색
- **규칙 기반 검증**: 면허/신용/지역/마감 등 규칙 기반 판단
- **인증/권한**: 회원가입/로그인, 역할(member/leader), 팀 단위 권한 분리
- **팀 워크스페이스**: 팀 문서 합산 조회, 코멘트/답글, 문서별 판정 공유
- **멀티테넌시**: 사용자별 스토리지/벡터DB/DB 레코드 분리
- **보안**: Input/Output Rail, HWP Deep Scan, PII 필터, Tool Gate(SSRF 방어)

## 아키텍처

```text
User (Streamlit UI)
  │
  ├─ Auth/Login/Register (SQLite users, bcrypt, cookie)
  │
  ├─ Upload(단일/다문서 자동 전환)
  │    ├─ RFPLoader -> Parser(HWP/PDF/DOCX/HWPX)
  │    ├─ DocumentStore(SQLite + file storage)
  │    └─ VectorStoreManager(Chroma, user/tenant isolated)
  │
  ├─ ExtractionPipeline (G1 -> G2/G3 -> G4)
  │
  ├─ Retrieval/RAG
  │    ├─ HybridRetriever(BM25 + Vector + RRF + Rerank)
  │    └─ RAGChain(Structure-aware + Postprocess)
  │
  ├─ Validation (RuleBasedValidator)
  │
  └─ Team Workspace
       ├─ team docs aggregation
       └─ comments/replies (SQLite)
```

## 실험 결과 종합 (최신화)

기준 문서: `docs/planning/HISTORY_v2_execution.md` (최종 업데이트: 2026-02-25)

### 지표 정의

- `kw_v3/v4/v5`: 정답 키워드 매칭 기반 핵심 지표(0~1). 버전이 올라갈수록 정규화/표현 유연 매칭이 강화됩니다.
- `kw_v5b`: `kw_v5`에서 공백/괄호/슬래시 변형 매칭을 추가 보정한 지표입니다.
- `Perfect`: 문항 단위 완전 정답(`kw=1.0`) 개수입니다.
- `Gate`: split별 통과 기준(Dev `>=0.99`, Holdout `>=0.95`, Sealed `>=0.95`)입니다.
- `Oracle`/`Non-oracle`: Oracle은 GT 의존 선택(연구용 상한), Non-oracle은 GT 비의존 선택(실운영 시나리오)입니다.
- `Faithfulness`(RAGAS): 생성 답변이 검색 근거에 충실한지(환각 억제) 지표입니다.
- `Context Recall`(RAGAS): 필요한 근거가 검색 컨텍스트에 포함되었는지 지표입니다.
- `Stdev`: 동일 설정 반복 실행 시 점수 변동성(재현성) 지표입니다.

### EXP01~EXP13 요약

| 구간 | 핵심 내용 | 결과 요약 |
|------|-----------|-----------|
| EXP01~03 | 청킹/검색/프롬프트 기초 | baseline 확립 |
| EXP04~06 | 테이블/정규화/파이프라인 개선 | 성능 안정화 |
| EXP07~10 | table-aware + 파서 고도화 | 성능 상승 |
| EXP11~13 | 프롬프트/컨텍스트 실험 | 효과/한계 확인 |

### EXP14~EXP20 요약 (무엇을 했는지 + 결과)

| 실험 | 무엇을 검증했는가 | 핵심 결과 |
|------|------------------|-----------|
| EXP14 | 오답 진단(실패 유형 분해) | imperfect 11건 분석, `gen_failure 6 / partial_retrieval 5` |
| EXP15 | Generation 품질 개선 | `kw_v3=0.9258` |
| EXP16 | 메트릭 v4 + SC 5-shot 검증 | `kw_v4=0.9534` |
| EXP17 | 메트릭 v5 + 0.99 목표 도전 | `kw_v5=0.9547` (0.99 미달) |
| EXP18 | GT 정제 + targeted prompt | `kw_v5=0.9851`, `28/30 perfect` |
| EXP19 | 0.99 달성 + 과적합 검증 | Dev `kw_v5=0.9952`, Holdout raw `0.8821`(과적합 신호) |
| EXP20(D9) | metric `v5b` 도입 | Overall `0.9799`, holdout/sealed gate 통과 |
| EXP20v2(D10) | 평가 후처리로 gate 보완 | Overall `0.9874`, dev/holdout/sealed 3-gate 통과 |
| EXP20v2(D10-R) | 동일 설정 3-run 재현성 점검 | overall gate pass-rate `33.3%` (불안정 확인) |

### EXP21 상세: 일반화 성능 + 재현성 안정화

EXP21은 D10-R 불안정을 줄이기 위해 안정화 백로그(P1~P5)를 분리 실험했습니다.

- `P1`(`answer_postprocess=stability_v1`)이 최종 채택안
- D10 대비 Holdout +3.84pp, Overall +0.95pp 개선
- `P1` 단일 실행: Overall `0.9968`, Dev `1.0000`, Holdout `0.9933`, Sealed `0.9909`, Perfect `48/50`
- `P1-R` 3-run 재현성: Dev/Holdout/Sealed/Overall gate 모두 `3/3 (100%)`
- 결론: 성능뿐 아니라 반복 실행 안정성까지 충족

### EXP22 상세: 평가 신뢰성 강화 (Non-oracle + 다차원 지표)

EXP22는 "점수가 좋아 보이는가"가 아니라 "실운영 기준으로 믿을 수 있는가"를 검증한 단계입니다.

- Oracle 누수 제거: `selection_mode=first_deterministic` (GT 비의존 선택)
- 다차원 평가 추가: `kw_v5` + RAGAS(`Faithfulness`, `Context Recall`)
- 3-run 반복으로 변동성 계측

| 항목 | 결과 |
|------|------|
| Non-oracle `kw_v5` (3-run mean) | `0.9742` |
| `kw_v5` stdev | `0.0104` |
| Oracle `kw_v5` mean | `0.9974` |
| Oracle gap mean | `2.33pp` |
| Faithfulness (mean ± stdev) | `0.9402 ± 0.0045` |
| Context Recall (mean ± stdev) | `0.9778 ± 0.0019` |
| Gate 패턴 일관성 | 3-run 모두 `Dev FAIL / Holdout PASS / Sealed PASS` |

해석:
- Non-oracle 기준에서도 높은 점수와 낮은 변동성을 확인했습니다.
- RAGAS를 추가해 근거 충실도/검색 충실도까지 함께 측정했습니다.
- Gate 패턴이 3-run에서 동일하게 재현되어, 실패/성공 조건이 우연이 아닌 구조적 현상임을 확인했습니다.

## 프로젝트 구조

```text
bidflow/
├── configs/                    # base/dev/prod/exp 설정
├── data/                       # 로컬 데이터(대부분 gitignore)
├── docs/
│   └── planning/               # 실험/운영 계획 및 실행 문서
├── scripts/                    # 실험/검증 실행 스크립트
├── src/bidflow/
│   ├── api/                    # FastAPI API
│   ├── apps/ui/                # Streamlit UI
│   ├── db/                     # SQLite schema/CRUD
│   ├── core/                   # config/util
│   ├── domain/                 # pydantic 모델
│   ├── extraction/             # G1~G4, batch pipeline
│   ├── ingest/                 # loader/storage/vector 관리
│   ├── parsing/                # 파일 파서
│   ├── retrieval/              # RAG/hybrid/rerank/prompt
│   ├── security/               # PII/tool_gate/rails
│   └── validation/             # rule validator
├── tests/
│   └── security/               # 보안 테스트
├── local_only/                 # 로컬 보관(HANDOFF/프롬프트 등, git 제외)
└── pyproject.toml
```

## 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .
```

## 환경 변수 (.env)

```bash
OPENAI_API_KEY=your_openai_key

# 선택
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public

# FastAPI API key 매핑 (key:user,key:user)
BIDFLOW_API_KEYS=sk-local-dev:admin

# Streamlit authenticator cookie key
BIDFLOW_COOKIE_KEY=replace_with_secure_random
```

## 실행

```bash
# 통합 실행 (권장): FastAPI + Streamlit
bidflow-run

# 개별 실행
uvicorn bidflow.main:app --reload
streamlit run src/bidflow/apps/ui/Home.py
```

## 테스트

```bash
pytest tests/security -q
```

## 데이터/보안 정책

- `data/raw` 원본 데이터셋은 GitHub 업로드 금지
- `HANDOFF`/프롬프트/개인 메모는 `local_only/`에 보관하고 Git 제외
