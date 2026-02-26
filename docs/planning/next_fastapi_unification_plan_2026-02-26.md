# Next+FastAPI 단일화 실행 계획 (2026-02-26)

## 목표
1. Next.js + FastAPI를 하나의 실행 동선으로 단일화한다.
2. 팀 단위 회사 프로필을 `팀별 1개`로 통합한다.
3. 프론트/백엔드 라우트를 하나의 기준(`/auth`, `/api/v1/*`)으로 정렬한다.

## 범위
- 포함:
  - 백엔드 엔트리포인트/라우터 통합 정리
  - 팀 공유 프로필 정책 반영
  - 프론트 API 호출 경로 통합
  - 실행/환경변수 문서 정리
- 제외:
  - 프론트 UI 리디자인
  - 기존 대규모 lint debt 전체 해소
  - 실운영 배포 스크립트(CI/CD) 구축

## 작업 스트림
### 1) Next+FastAPI 실행 단일화
- `bidflow.main:app`을 백엔드 단일 엔트리포인트로 고정
- `bidflow-run` 런처를 `uvicorn + npm run dev` 동시 실행으로 변경
- `.env` 기반 포트/호스트 변수 표준화
  - `BIDFLOW_API_PORT`
  - `BIDFLOW_WEB_PORT`
  - `NEXT_PUBLIC_API_BASE_URL`

### 2) 팀 공유 프로필 1개 정책
- `/auth/me` 업데이트를 팀장(`leader`) 전용으로 제한
- 팀 프로필 저장 키를 `team_name` 단위로 강제
- 팀원 사용자 레코드 `licenses` fallback 동기화

### 3) 라우트 통합
- 백엔드 라우터 집합을 `create_app()`에 통합
  - `/auth/*`
  - `/api/v1/ingest/*`
  - `/api/v1/extract/*`
  - `/api/v1/validate`
  - `/api/v1/team/*`
- 프론트 API 호출을 공통 base URL 헬퍼로 통일
  - 하드코딩 `http://localhost:8000` 제거

## 검증 계획
1. 백엔드 정적 검증: `python -m compileall src/bidflow`
2. 라우트 스모크 검증: 앱 라우트 목록에서 필수 경로 존재 확인
3. 보안 테스트 회귀: `pytest tests/security -q`
4. 프론트 품질 게이트: `npm run lint`

## 완료 기준 (Definition of Done)
- `bidflow-run` 실행 시 FastAPI + Next.js 동시 기동 가능
- 프론트 코드에서 API 하드코딩 URL 제거
- 팀 공유 프로필 1개 정책이 `/auth/me`, `/validate`, `/team/decision` 경로에서 일관 반영
- 계획/결과 문서가 `docs/planning`에 남아 있을 것
