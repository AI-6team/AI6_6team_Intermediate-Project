# 팀 협업 가이드 (Git Workflow)

## 브랜치 전략
이 프로젝트는 **Git Flow**를 변형하여 사용합니다.

1.  **main**: 최종 제출용 (Final Release). 언제나 실행 가능한 상태여야 합니다.
2.  **dev**: 개발 통합용 (Integration). 모든 팀원의 코드가 여기서 합쳐집니다.
3.  **개인 브랜치 (feature/이름)**: 각자 작업하는 공간입니다.

## 팀원 작업 순서

### 1. 프로젝트 가져오기 (최초 1회)
```bash
git clone https://github.com/AI-6team/AI6_6team_Intermediate-Project.git
cd bidflow
```

### 2. 본인 브랜치 생성하기
작업은 반드시 `dev` 브랜치를 기준으로 본인의 브랜치를 따서 시작해야 합니다.

```bash
# dev 브랜치로 이동 및 최신화
git checkout dev
git pull origin dev

# 본인 브랜치 생성 (예: feature/chulsu)
git checkout -b feature/본인이름
```

### 3. 작업 및 커밋
```bash
# 변경사항 스테이징
git add .

# 커밋 (메시지는 명확하게)
git commit -m "Feat: 파싱 로직 구현 완료"
```

### 4. 작업 내용 올리기 (Push)
```bash
# 본인 브랜치를 원격 저장소에 올림
git push origin feature/본인이름
```

### 5. 병합 요청 (Pull Request / PR)
1. GitHub 웹페이지 접속
2. **Pull Requests** 탭 클릭
3. **New pull request** 클릭
4. **base: dev** <- **compare: feature/본인이름** 설정 (중요: `main`이 아니라 `dev`로 보내야 합니다!)
5. PR 생성 및 팀원 리뷰 요청

## 주의사항
- `main` 브랜치에는 직접 `push` 하면 안됩니다.
- 충돌(Conflict)이 발생하면 `dev` 브랜치를 본인 브랜치로 `merge`하여 로컬에서 해결 후 다시 `push` 해야 합니다.
