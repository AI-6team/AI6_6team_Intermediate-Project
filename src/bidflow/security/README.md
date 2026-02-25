# BidFlow Security Architecture

BidFlow는 멀티테넌트 RAG 서비스를 전제로, 문서 업로드부터 검색/응답까지 보안을 단계적으로 적용합니다.

## Core Objectives

- 사용자/테넌트 데이터 격리
- 업로드 파일 안전성 검증
- 개인정보(PII) 보호
- 프롬프트 인젝션/도구 오남용 차단
- 감사 가능한 로그 체계 확보

## Implemented Controls

### 1) Storage & Tenant Isolation

- 사용자 저장소 분리: `data/accounts/{user_id}/...`
- 공유 저장소 분리: `data/shared/...`
- 벡터 메타데이터에 `tenant_id`, `user_id`, `group_id`, `access_level` 주입
- 검색 단계 ACL 필터 강제 적용(테넌트/사용자 범위)

### 2) Upload/File Security

- 확장자 화이트리스트: `.pdf`, `.hwp`, `.docx`, `.hwpx`
- 실행 파일/스크립트 차단
- Magic Number 검사로 확장자 위변조 탐지
- 파일 크기 제한 및 파싱 타임아웃 적용

### 3) Prompt Injection & Guardrails

- 입력 단계 검사: `InputRail`
- 출력 단계 검사: PII 유출 탐지 후 차단
- 구조화된 컨텍스트 처리와 시스템 규칙 우선 적용
- 도구 호출 전 `ToolExecutionGate` 검증

### 4) Privacy (PII)

- 중앙 PII 필터: `src/bidflow/security/pii_filter.py`
- 주민번호/연락처/이메일/IP 등 탐지 및 마스킹
- 업로드 본문, 질의, 응답 단계 전반에 적용

### 5) Audit Logging

- `security.log`, `audit.log` 로테이션 기록
- 위협 이벤트와 정상 응답 이벤트를 구분 기록

## Tests

현재 저장소에서 실행 중인 보안 테스트는 다음 두 파일입니다.

- `tests/security/test_pii_filter.py`
- `tests/security/test_tool_gate.py`

실행:

```bash
pytest tests/security -q
```

## Security Principles

1. Isolation First: 사용자/테넌트 데이터는 검색 단계에서 먼저 분리한다.
2. Defense in Depth: 업로드, 파싱, 검색, 응답 단계별 방어를 중첩한다.
3. Fail Closed: 검증 실패 시 허용이 아니라 차단을 기본 동작으로 둔다.
4. Auditability: 위협/응답 로그를 남겨 사후 추적 가능하게 한다.
