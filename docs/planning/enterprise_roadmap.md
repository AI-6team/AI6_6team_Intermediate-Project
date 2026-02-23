# BidFlow 사내 RAG 제품 고도화 로드맵 (Enterprise Roadmap)

본 문서는 현재의 단일 사용자(All-in-one local) MVP 모델을 **다중 사용자(Multi-tenancy) 지원 사내 엔터프라이즈 서비스**로 확장하기 위한 기술적 로드맵입니다.

---

## 1. 제품 비전 (Vision)

**"개인화된 입찰 비서 (Personalized Bidding Assistant)"**
*   **개인화**: 내 프로젝트, 내 관심 공고, 내가 분석한 RFP 관리.
*   **협업**: 팀원과 분석 결과 공유, 댓글 및 피드백.
*   **보안**: 부서/직급별 접근 권한(RBAC) 제어.

## 2. 핵심 기능 고도화 (Feature Upgrades)

### 2.1 인증 및 권한 관리 (Authentication & Authorization)
현재: 없음 (누구나 접근 가능)
**목표: 사용자 식별 및 접근 제어**

1.  **로그인 구현**:
    *   Phase 1: 간이 로그인 (Streamlit Authenticator) - `users.yaml` 기반 비밀번호 인증.
    *   Phase 2: SSO 연동 (Google Workspace, LDAP, AD) - 사내 계정 통합.
2.  **세션 관리**:
    *   로그인 상태 유지 및 자동 로그아웃.

### 2.2 멀티 테넌시 데이터 구조 (Multi-tenancy Data)
현재: `data/processed/`에 모든 파일 저장, `storage.py`가 전역 공유.
**목표: 사용자별 데이터 격리**

1.  **스토리지 구조 변경**:
    *   `data/{user_id}/processed/`: 개인별 분석 파일 격리.
    *   `data/shared/`: 팀 공유/공개 파일 저장소.
2.  **DB 도입 검토**:
    *   파일(JSON) 기반 관리의 한계(검색/조회/동시성) 극복을 위해 **SQLite** 또는 **PostgreSQL** 도입.
    *   User, RFP, Extraction 테이블 설계.

### 2.3 개인화 프로필 (Profile Management)
현재: 전역 1개 프로필 (`data/profile.json`).
**목표: 다중 프로필 및 버전 관리**

1.  **N개 프로필 지원**: "본사 기준", "지사 기준", "컨소시엄 기준" 등 다양한 프로필 생성 및 저장.
2.  **분석 시 선택**: RFP 분석/판정 시점에 '어떤 프로필로 판정할지' 선택 가능하게 변경.

---

## 3. 아키텍처 변경안 (Architecture Changes)

### 3.1 Backend Separation (API 분리)
*   현재: Streamlit 내부 로직과 비즈니스 로직 혼재.
*   **변경**:
    *   **FastAPI** 서버를 별도로 띄워 핵심 로직(추출/검증)을 API화.
    *   Streamlit은 순수 **Frontend** 역할만 수행 (API 호출 및 시각화).
    *   이점: 모바일 앱, 사내 메신저 봇 등 다른 클라이언트 확장 용이.

### 3.2 Security Hardening
*   **Audit Log**: 누가 언제 어떤 문서를 분석했는지 감사 로그 기록.
*   **PII Masking**: 업로드된 문서에서 개인정보(주민번호, 전화번호) 자동 마스킹 후 저장.

---

## 4. 단계별 실행 계획 (Phased Execution)

### Phase 4-1: Authentication (2주)
*   `Streamlit-Authenticator` 라이브러리 도입.
*   로그인 화면 구현 및 세션 제어 적용.
*   `Home.py` 진입 전 로그인 강제.

### Phase 4-2: User Isolation (2주)
*   `DocumentStore` 클래스 리팩토링: `user_id` 파라미터 필수화.
*   파일 저장 경로에 `user_id` 디렉토리 구조 적용.
*   UI에서 "내 문서함" 구현.

### Phase 4-3: Database & API (4주+)
*   API 서버 분리 (FastAPI).
*   DB 스키마 설계 및 마이그레이션.
*   협업 기능(공유) 추가.

---

## 5. 결론
이 로드맵은 BidFlow를 단순한 "도구"에서 "플랫폼"으로 진화시키는 과정입니다.
우선 **Phase 4-1 (간이 로그인)**부터 시작하여 최소한의 보안을 확보하는 것을 권장합니다.
