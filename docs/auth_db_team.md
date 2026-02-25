# BidFlow — 인증 · 세션 · DB · 팀 워크스페이스 구현 정리


---

## 목차

1. [전체 구조 개요](#1-전체-구조-개요)
2. [로그인 / 인증 (auth.py)](#2-로그인--인증-authpy)
3. [세션 관리 (session.py)](#3-세션-관리-sessionpy)
4. [SQLite DB 계층 (db/)](#4-sqlite-db-계층-db)
5. [Team Workspace](#5-team-workspace)
6. [데이터 흐름 요약](#6-데이터-흐름-요약)
7. [파일 위치 빠른 참조](#7-파일-위치-빠른-참조)

---

## 1. 전체 구조 개요

```
사용자 요청
    │
    ▼
Home.py / pages/*.py
    │  require_login() ──→ auth.py ──→ SQLite users 테이블
    │  init_app_session() ──→ session.py ──→ SQLite sessions/profiles 테이블
    │
    ▼
기능 페이지 (Upload, Matrix, Profile, Team Workspace …)
    │
    ▼
DocumentStore (storage.py) ──→ SQLite documents / extraction_results 테이블
Team Workspace (team.py)   ──→ SQLite comments / replies 테이블
```

**외부 의존성**

| 라이브러리 | 용도 |
|-----------|------|
| `streamlit-authenticator` | 로그인 폼 · 쿠키 기반 세션 유지 |
| `bcrypt` | 비밀번호 해시 생성·검증 |
| `sqlite3` (표준 라이브러리) | 모든 구조화 데이터 저장 |
| `python-dotenv` | `.env`에서 환경변수 로드 |

---

## 2. 로그인 / 인증 (auth.py)

**파일**: `src/bidflow/apps/ui/auth.py`

### 2-1. 구조

이전에는 `configs/users.yaml`에서 직접 읽고 쓰던 방식을 **SQLite `users` 테이블**로 교체했습니다.
`streamlit-authenticator`는 dict 형식의 credentials를 받으므로, DB에서 읽은 뒤 동일한 형식으로 변환해 호환성을 유지합니다.

```
SQLite users 테이블
    │
    ▼  crud.get_credentials_dict()
{"usernames": {"alice": {"name":…, "email":…, "password": <bcrypt hash>}}}
    │
    ▼
stauth.Authenticate(credentials, cookie_name, cookie_key, expiry_days)
```

### 2-2. 환경변수

| 변수명 | 설명 | 위치 |
|--------|------|------|
| `BIDFLOW_COOKIE_KEY` | 쿠키 서명 키 (무작위 문자열) | `.env` |
| `BIDFLOW_API_KEYS` | FastAPI X-API-Key 목록 | `.env` |

### 2-3. 주요 함수

| 함수 | 설명 |
|------|------|
| `load_authenticator()` | DB에서 credentials를 읽어 `Authenticate` 객체 반환 |
| `require_login()` | 모든 페이지 첫 줄에서 호출. 미인증 시 `st.stop()`. 인증 시 `username` 반환 |
| `register_form()` | 회원가입 폼. 성공 시 bcrypt 해시를 DB에 저장 |
| `get_user_info(username)` | name, email, team 등 사용자 정보 반환 |
| `get_user_team(username)` | 소속 팀명 반환 (없으면 빈 문자열) |
| `get_team_members(team_name)` | 같은 팀 사용자 목록 반환 |
| `deactivate_account(username, delete_data)` | 계정 삭제. `delete_data=True`이면 파일 공간도 삭제 |

### 2-4. 로그인 흐름

```
1. Home.py 진입
2. authentication_status가 없으면 → 로그인/회원가입 탭 표시
3. authenticator.login(location="sidebar") 호출
4. streamlit-authenticator가 쿠키 확인 또는 폼 입력 검증
5. 인증 성공 → st.session_state["authentication_status"] = True
                st.session_state["username"] = "alice"
                st.session_state["name"] = "Alice"
6. require_login() → username 반환
```

### 2-5. 회원가입 흐름

```
1. 폼 입력 (username, name, email, password, team)
2. 유효성 검사 (정규식, 길이, 중복 확인)
3. bcrypt.hashpw(password) → DB upsert_user()
4. StorageRegistry.ensure_spaces(username) → 사용자 디렉토리 생성
5. 팀 입력 시 ensure_team_spaces(team) → 팀 디렉토리 생성
```

---

## 3. 세션 관리 (session.py)

**파일**: `src/bidflow/apps/ui/session.py`

### 3-1. 역할

Streamlit의 `st.session_state`는 브라우저 탭을 닫으면 사라집니다.
`init_app_session(user_id)`는 앱 재접속 시 DB에서 이전 상태를 복원합니다.

```python
# 모든 기능 페이지 공통 호출 패턴
user_id = require_login()
init_app_session(user_id)
```

### 3-2. 복원 항목

| 항목 | DB 테이블 | session_state 키 |
|------|-----------|-----------------|
| 마지막 작업 문서 해시 | `sessions.current_doc_hash` | `current_doc_hash` |
| 추출 결과 (Matrix) | `extraction_results` | `extraction_results` |
| 회사 프로필 | `profiles` | `company_profile` |

### 3-3. 저장 시점

| 이벤트 | 저장 함수 |
|--------|-----------|
| 문서 업로드 완료 | `store.save_session_state({"current_doc_hash": hash})` |
| 추출 완료 | `store.save_extraction_result(doc_hash, result)` |
| 프로필 수정 | `store.save_profile(profile)` |

### 3-4. 세션 복원 로직

```python
def init_app_session(user_id):
    store = DocumentStore(user_id=user_id)

    # 1. 세션 복원
    if "current_doc_hash" not in st.session_state:
        saved = store.load_session_state()           # SQLite sessions 조회
        if saved and store.load_document(saved["current_doc_hash"]):
            st.session_state["current_doc_hash"] = saved["current_doc_hash"]
            result = store.load_extraction_result(...)
            if result:
                st.session_state["extraction_results"] = result

    # 2. 프로필 복원
    if "company_profile" not in st.session_state:
        data = store.load_profile()                  # SQLite profiles 조회
        st.session_state["company_profile"] = CompanyProfile(**data) if data else <default>
```

---

## 4. SQLite DB 계층 (db/)

**디렉토리**: `src/bidflow/db/`

### 4-1. 파일 구성

| 파일 | 역할 |
|------|------|
| `database.py` | 연결 생성 · WAL 모드 · 스키마 초기화 (`init_db`) |
| `crud.py` | 7개 테이블 CRUD 함수 전체 |

**DB 위치**: `data/bidflow.db` (gitignore 적용)

### 4-2. 테이블 스키마

```sql
-- 사용자 계정
CREATE TABLE users (
    username      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,      -- bcrypt 해시
    team          TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now'))
);

-- RFP 문서 (메타데이터 + 청크 전체)
CREATE TABLE documents (
    doc_hash     TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_json TEXT NOT NULL,       -- RFPDocument.model_dump() JSON
    upload_date  TEXT,
    PRIMARY KEY (doc_hash, user_id)
);

-- Compliance Matrix 추출 결과
CREATE TABLE extraction_results (
    doc_hash    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (doc_hash, user_id)
);

-- 회사 프로필
CREATE TABLE profiles (
    user_id      TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL
);

-- 세션 상태 (마지막 문서 등)
CREATE TABLE sessions (
    user_id          TEXT PRIMARY KEY,
    current_doc_hash TEXT,
    extra_json       TEXT
);

-- 팀 코멘트
CREATE TABLE comments (
    id          TEXT PRIMARY KEY,     -- UUID
    team_name   TEXT NOT NULL,
    doc_hash    TEXT NOT NULL,
    author      TEXT NOT NULL,
    author_name TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX idx_comments_team_doc ON comments(team_name, doc_hash);

-- 코멘트 답글
CREATE TABLE replies (
    id          TEXT PRIMARY KEY,
    comment_id  TEXT NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    author      TEXT NOT NULL,
    author_name TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
```

### 4-3. WAL 모드와 동시성

```python
conn.execute("PRAGMA journal_mode=WAL")   # 읽기·쓰기 동시 허용
conn.execute("PRAGMA foreign_keys=ON")    # ON DELETE CASCADE 활성화
```

- 기존 JSON 방식은 read-modify-write 중 **Race condition** 발생 가능
- WAL 모드: 여러 읽기 + 1개 쓰기를 동시에 허용, 쓰기는 자동 직렬화
- `with conn:` 트랜잭션으로 부분 쓰기 오류 시 자동 롤백

### 4-4. 주요 CRUD 함수 목록

**users**

| 함수 | 설명 |
|------|------|
| `upsert_user(username, name, email, password_hash, team)` | 삽입 또는 갱신 |
| `get_user(username)` | 단일 사용자 조회 |
| `delete_user(username)` | 삭제 |
| `get_team_members(team_name)` | 팀원 목록 |
| `get_credentials_dict()` | streamlit-authenticator 호환 dict 반환 |

**documents / extraction_results**

| 함수 | 설명 |
|------|------|
| `upsert_document(...)` | 문서 저장 |
| `get_document(doc_hash, user_id)` | 문서 조회 (content 포함) |
| `list_documents(user_id)` | 목록 조회 (content 제외, 빠름) |
| `upsert_extraction(doc_hash, user_id, result)` | 추출 결과 저장 |
| `get_extraction(doc_hash, user_id)` | 추출 결과 조회 |

**comments / replies**

| 함수 | 설명 |
|------|------|
| `add_comment(team_name, doc_hash, author, author_name, text)` | 코멘트 추가 |
| `get_comments(team_name, doc_hash)` | 코멘트 + 답글 목록 |
| `delete_comment(comment_id, requester)` | 본인 코멘트 삭제 |
| `add_reply(comment_id, author, author_name, text)` | 답글 추가 |
| `delete_reply(reply_id, requester)` | 본인 답글 삭제 |

### 4-5. DB 초기화

앱 시작 시 `Home.py`에서 1회 호출합니다.

```python
from bidflow.db.database import init_db
init_db()   # 테이블이 없으면 생성, 있으면 스킵
```

### 4-6. 마이그레이션

기존 JSON/YAML 데이터를 SQLite로 일괄 이전합니다.

```bash
python scripts/migrate_to_sqlite.py
```

| 원본 | 대상 테이블 |
|------|------------|
| `configs/users.yaml` | `users` |
| `data/accounts/{uid}/processed/{hash}.json` | `documents` |
| `data/accounts/{uid}/processed/{hash}_result.json` | `extraction_results` |
| `data/accounts/{uid}/profile.json` | `profiles` |
| `data/accounts/{uid}/session.json` | `sessions` |
| `data/shared/teams/{team}/comments/{hash}.json` | `comments` + `replies` |

---

## 5. Team Workspace

### 5-1. 관련 파일

| 파일 | 역할 |
|------|------|
| `src/bidflow/apps/ui/team.py` | 팀 유틸리티 함수 모음 |
| `src/bidflow/apps/ui/pages/6_Team_Workspace.py` | Streamlit 페이지 |

### 5-2. 진입 조건

```python
user_id = require_login()          # 로그인 필수
team_name = get_user_team(user_id) # 팀 소속 필수 → 없으면 st.stop()
team_members = get_team_members(team_name)
```

### 5-3. 화면 구성

```
┌──────────────────────────────────────────────────────────┐
│ Team Workspace                                           │
│ 팀: team_ai  |  팀원: Alice, Bob, Carol                  │
├──────────────────────────────────────────────────────────┤
│ 안건 선택 ▼  [ RFP_2026_공공.pdf (by Alice, 2026-02-20)] │
├─────────────────────┬────────────────────────────────────┤
│ 안건 정보           │ 판정 결과                          │
│ 파일명: …           │ ⚠ 조건부 입찰 가능                │
│ 업로더: Alice       │ RED:2  GRAY:3  GREEN:5             │
│ 날짜: 2026-02-20    │                                    │
├──────────────────────────────────────────────────────────┤
│ 팀 코멘트                                                │
│ [코멘트 작성 폼]                                         │
│                                                          │
│ ┌──────────────────────────────────────────┐             │
│ │ Alice • 2026-02-20 14:30                 │  🗑(본인)  │
│ │ "납품 실적 기준이 너무 높습니다."         │             │
│ │   ↳ Bob • 14:45  "동의합니다."           │  🗑(본인)  │
│ │   [답글 달기]                            │             │
│ └──────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

### 5-4. 주요 함수 (team.py)

| 함수 | 설명 |
|------|------|
| `get_team_documents(team_members)` | 팀원 전체 문서 목록 합산 (업로더 정보 포함) |
| `get_decision_summary(member_username, doc_hash)` | 특정 팀원의 판정 결과 요약 (signal, RED/GRAY/GREEN 수) |
| `load_comments(team_name, doc_hash)` | 코멘트 + 답글 목록 |
| `add_comment(team_name, doc_hash, author, author_name, text)` | 코멘트 등록 |
| `add_reply(team_name, doc_hash, comment_id, author, author_name, text)` | 답글 등록 |
| `delete_comment(team_name, doc_hash, comment_id, requester)` | 본인 코멘트 삭제 |
| `delete_reply(team_name, doc_hash, comment_id, reply_id, requester)` | 본인 답글 삭제 |

### 5-5. 코멘트 데이터 구조

```python
# get_comments() 반환 형식
[
    {
        "id": "uuid",
        "author": "alice",
        "author_name": "Alice",
        "text": "납품 실적 기준이 너무 높습니다.",
        "created_at": "2026-02-20T14:30:00",
        "replies": [
            {
                "id": "uuid",
                "author": "bob",
                "author_name": "Bob",
                "text": "동의합니다.",
                "created_at": "2026-02-20T14:45:00"
            }
        ]
    }
]
```

### 5-6. 권한 규칙

- **삭제**: 본인(`author == user_id`)만 가능. DB 레벨에서도 `WHERE author = ?`로 강제
- **코멘트 삭제 시 답글 자동 삭제**: `ON DELETE CASCADE` (FK 제약)
- **조회**: 같은 팀 소속이면 전원 열람 가능

---

## 6. 데이터 흐름 요약

```
회원가입
  입력 → 유효성 검사 → bcrypt 해시
       → crud.upsert_user() → users 테이블
       → StorageRegistry.ensure_spaces() → 파일 디렉토리 생성

로그인
  폼 입력 or 쿠키
       → crud.get_credentials_dict() → stauth.Authenticate 검증
       → 성공: session_state["authentication_status"] = True

페이지 진입
  require_login() → username
  init_app_session(username)
       → crud.get_session() → session_state["current_doc_hash"]
       → crud.get_extraction() → session_state["extraction_results"]
       → crud.get_profile() → session_state["company_profile"]

문서 업로드 / 추출
  DocumentStore.save_document() → crud.upsert_document()
  DocumentStore.save_extraction_result() → crud.upsert_extraction()
  DocumentStore.save_session_state() → crud.upsert_session()

팀 코멘트
  add_comment() → crud.add_comment() → comments 테이블
  add_reply() → crud.add_reply() → replies 테이블
  delete_comment() → crud.delete_comment() + CASCADE → replies 자동 삭제
```

---

## 7. 파일 위치 빠른 참조

```
src/bidflow/
├── db/
│   ├── __init__.py
│   ├── database.py          # 연결 · WAL · init_db()
│   └── crud.py              # 7개 테이블 CRUD
├── apps/ui/
│   ├── Home.py              # init_db() 호출 · 로그인/회원가입 진입점
│   ├── auth.py              # require_login · register_form · 계정 관리
│   ├── session.py           # init_app_session (세션·프로필 복원)
│   ├── team.py              # 팀 유틸리티 (코멘트 CRUD · 문서 합산)
│   └── pages/
│       └── 6_Team_Workspace.py   # 팀 워크스페이스 Streamlit 페이지
├── ingest/
│   └── storage.py           # DocumentStore (SQLite 백엔드)
configs/
└── users.yaml               # (마이그레이션 후 레거시, gitignore 적용)
data/
└── bidflow.db               # SQLite DB (gitignore 적용)
scripts/
└── migrate_to_sqlite.py     # JSON/YAML → SQLite 일괄 이전
```
