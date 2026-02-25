# Security Architecture

이 프로젝트는 멀티테넌트 환경에서 안전하게 RAG 기반 문서 분석 서비스를 운영하기 위한 보안 구조를 구현합니다.

## 🎯 Core Objectives

- **테넌트 간 데이터 완전 격리** (Isolation)
- **업로드 문서 보안 검증** (File Security)
- **개인정보(PII) 보호** (Privacy)
- **프롬프트 인젝션 및 LLM 보안 가드레일** (Guardrails)
- **SSRF 및 외부 공격 차단** (Network Security)
- **감사 로그 기반 추적 가능성 확보** (Auditability)

---

## 🛡️ Security Features

### 1. Tenant Isolation & Access Control
멀티 고객 환경에서 데이터 혼입을 원천 차단하기 위해 테넌트 단위 격리를 적용했습니다.

- **Storage Isolation**: 각 테넌트의 문서는 물리적으로 독립된 저장소에 저장됩니다.
  - Path: `data/{tenant_id}/`
- **Vector DB Metadata**: 문서 인덱싱 시 `tenant_id`, `user_id`, `group_id`가 자동 주입됩니다.
- **Retrieval ACL**: 검색 시 ACL 필터가 강제 적용되어 타 테넌트 데이터 접근을 차단합니다.
- **Tenant Data Purge**: 테넌트 데이터 삭제 시 파일 시스템, 벡터 DB 인덱스, 메타데이터가 동시에 제거됩니다.

### 2. File Upload Security
외부 문서 업로드 단계에서 다층 보안 검사를 수행합니다.

- **Extension Whitelist**:
  - ✅ 허용: `.pdf`, `.hwp`, `.docx`, `.hwpx`
  - 🚫 차단: `.exe`, `.sh`, `.bat` 등 실행 파일
- **Magic Number Check**: 파일 헤더를 검사하여 확장자 위변조를 탐지합니다.
- **Size Limit**: 기본 50MB 제한.
- **Parsing Isolation**: 파싱은 별도 프로세스에서 수행되며, 300초 타임아웃(Time-boxing)을 적용하여 DoS를 방지합니다.

### 3. Data Sanitization & Integrity
- **Text Cleaning**: Mojibake(깨진 문자) 제거 및 바이너리 노이즈 필터링.
- **Validation**: 의미 없는 텍스트(10자 이하 청크)는 저장하지 않습니다.
- **Single Document Policy**: 새 문서 업로드 시 기존 테넌트 데이터를 퍼지(Purge)하여 검색 결과 혼입을 방지합니다.

### 4. PII Protection (Privacy)
중앙화된 필터 시스템(`src/bidflow/security/pii_filter.py`)을 통해 개인정보를 보호합니다.

- **Detection & Masking**:
  - 주민등록번호, 외국인등록번호, 여권번호
  - 전화번호, 이메일, IP 주소
- **Scope**: 모든 파싱 및 입력 단계에서 자동 적용됩니다.

### 5. Prompt Injection Defense
RAG 프롬프트 구조를 재설계하여 외부 지시 실행을 방지합니다.

- **Structured Prompt**: `<context>`, `<hints>` 태그를 사용하여 데이터와 지시를 분리합니다.
- **System Rules**: 외부 문서의 지시보다 시스템 정책을 우선하도록 강제합니다.

### 6. Input / Output Guardrails
LLM의 입출력 양단에 보안 레일을 적용했습니다.

- **Input Rail**: 사용자 질문 검증 (길이 제한, PII 마스킹, 인젝션 패턴 검사)
- **Output Rail**: 모델 응답 검증 (주민번호, 금융정보 등 민감 데이터 유출 차단)

### 7. Tool Execution Security & SSRF
도구 실행 전 보안 게이트(`ToolExecutionGate`)를 통과해야 합니다.

- **Validation**: 검색 파라미터 및 허용 도메인 검사.
- **SSRF Protection**: 내부망 IP(`127.0.0.1`, `10.*`, `192.168.*` 등) 및 마스킹된 호스트 접근을 차단합니다.

### 8. Audit & Security Logging
모든 주요 이벤트는 감사 로그로 기록되며, `gzip`으로 자동 로테이션됩니다.

- **Log Files**: `security.log` (보안 위협), `audit.log` (사용 이력)
- **Fields**: Tenant, User, IP, Referenced Documents, Page Numbers

---

## 🧪 Verification & Testing

보안 기능 검증을 위한 테스트 스위트를 포함하고 있습니다.

### Integration Tests (`tests/test_isolation_security.py`)
총 12개 보안 항목을 통합 테스트합니다.
- 파일 보안, 테넌트 격리, ACL 필터
- PII 마스킹, Prompt Injection 방어
- SSRF 차단, 감사 로그 기록

### Unit Tests (`tests/test_pii_filter.py`)
- 주민번호, 외국인번호, IP, 여권번호 등 패턴 탐지 정확도 검증

---

## 📜 Security Design Principles

1. **Isolation First**: 테넌트 간 데이터는 절대 공유되지 않는다.
2. **Retrieval-Level Security**: 권한 검사는 LLM이 아닌 검색 단계에서 수행한다.
3. **Defense in Depth**: 업로드 → 파싱 → 검색 → 응답 모든 단계에서 보안 검증을 수행한다.
4. **Auditability**: 모든 중요한 이벤트는 추적 가능해야 한다.