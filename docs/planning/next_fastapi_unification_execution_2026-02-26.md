# Next+FastAPI 단일화 실행 결과 (2026-02-26)

## 결과 요약
- 목표 1. Next+FastAPI 단일화: 완료
- 목표 2. 팀 공유 프로필 1개 정책: 완료
- 목표 3. 라우트 통합: 완료
- 추가 결과: 실행 가이드/환경변수 문서 갱신

## 반영 내용

### A. 실행 단일화
- `src/bidflow/launcher.py`
  - 기존 `FastAPI + Streamlit` 동시 실행에서 `FastAPI + Next.js` 동시 실행으로 변경
  - Windows/Unix 프로세스 그룹 종료 처리 보강
  - `.env` 기반 포트/URL 변수 적용
    - `BIDFLOW_API_PORT`, `BIDFLOW_WEB_PORT`
    - `NEXT_PUBLIC_API_BASE_URL` 기본값 자동 주입
- `README.md`
  - 실행 섹션을 `bidflow-run`(FastAPI+Next.js) 기준으로 정리
  - 환경변수 예시를 Next.js 연동 기준으로 갱신
- `frontend/README.md`
  - `NEXT_PUBLIC_API_BASE_URL` 설정 안내 추가

### B. 팀 공유 프로필 1개 정책 (백엔드)
- `src/bidflow/api/main.py`
  - 팀 공유 프로필 로드/적용 helper 추가
  - `/auth/me` PUT:
    - 팀장 권한 제한 유지
    - 팀 이름 변경 차단
    - 팀 단위 프로필(`team_name`) 저장/갱신
  - `/auth/me` GET:
    - 팀 공유 프로필 기준 면허 정보 응답
  - `/api/v1/team/documents`:
    - 팀 문서 중복 제거 로직 적용
  - `/api/v1/team/decision/{doc_hash}`:
    - 팀 공유 프로필 기준 판정
- `src/bidflow/db/crud.py`
  - `update_team_licenses(team, licenses)` 추가 (팀원 fallback 동기화)
- `src/bidflow/api/routers/validate.py`
  - 검증 시 팀 공유 프로필 우선 사용, 없으면 사용자 fallback

### C. 라우트/프론트 연동 통합
- `src/bidflow/main.py`
  - `bidflow.main:app` -> `create_app()` 기반 단일 엔트리포인트 고정
- `frontend/lib/api.ts`
  - `API_BASE`를 `NEXT_PUBLIC_API_BASE_URL` 기반으로 표준화
  - `apiUrl()` 헬퍼 추가
  - API 호출 경로 통일
- 다음 파일의 하드코딩 API 주소 제거:
  - `frontend/components/UserHeader.tsx`
  - `frontend/components/CommentSection.tsx`
  - `frontend/app/page.tsx`
  - `frontend/app/analysis/page.tsx`
  - `frontend/app/validation/page.tsx`

## 검증 결과

### 1) 백엔드 정적 컴파일
- 명령:
  - `python -m compileall src/bidflow`
- 결과:
  - 성공 (오류 없음)

### 2) 라우트 스모크 체크
- 명령:
  - `from bidflow.main import app` 후 필수 라우트 존재 여부 검사
- 결과:
  - `route_count=22`
  - `missing=[]` (필수 경로 누락 없음)

### 3) 보안 테스트 회귀
- 명령:
  - `pytest tests/security -q`
- 결과:
  - `9 passed in 0.03s`

### 4) 프론트 lint
- 명령:
  - `npm run lint`
- 결과:
  - 실패 (`34 errors`, `4 warnings`)
  - 주요 원인: 기존 코드베이스의 `no-explicit-any`, `react-hooks/set-state-in-effect`, `react-hooks/static-components` 규칙 위반
  - 이번 작업 범위에서 라우트 통합은 반영 완료했으나, 프론트 전역 lint debt는 별도 정리 필요

## 잔여 이슈 / 후속 권고
1. 프론트 lint debt 정리 전담 작업 필요
2. E2E 시나리오(로그인→업로드→분석→검증→팀워크스페이스) 자동화 테스트 추가 권장
3. `src/bidflow/apps/ui/*`(Streamlit legacy) 정리 여부 결정 필요
