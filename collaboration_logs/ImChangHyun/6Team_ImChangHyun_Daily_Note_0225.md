# Daily 협업일지

### [1] 오늘 날짜 / 이름 / 팀명

- 날짜: 2026-02-25
- 이름: 임창현
- 팀명: 6팀

---

### [2] 오늘 맡은 역할 및 구체적인 작업 내용

> 오늘 당신이 맡았던 역할은 무엇이었고, 어떤 작업을 수행했나요?

✍️ 답변:

```
RAG 평가 체계 최종 확정 (EXP22) 및 인증/팀워크스페이스 기능 통합.

[EXP22: 최종 평가 체계 확정]
1. Oracle 누수 제거: SC 5-shot에서 GT 기반 best 선택 → first_deterministic(temp=0.0) 전환
   - 실운영에서는 정답(GT)이 없으므로 Oracle 의존을 제거하는 것이 필수
2. RAGAS 다차원 평가 도입: kw_v5 단일 지표 → + Faithfulness + Context Recall
3. 3-run 재현성 검증 실행:
   - kw_v5 mean=0.9742 (stdev=1.04pp)
   - Faithfulness mean=0.940 (stdev=0.45pp)
   - Context Recall mean=0.978 (stdev=0.19pp)
   - Gate 패턴 3-run 100% 일관 (dev FAIL, holdout/sealed PASS)
4. Mismatch 수동 검수 (3건):
   - 2건: chapter prefix 미전달로 인한 judge false negative (답변 자체는 정확)
   - 1건: temp=0.0 shot의 실질 SC 선택 손실

[인증/팀워크스페이스 통합 커밋]
- DB 레이어: crud.py (443줄), database.py (141줄)
- Streamlit 인증 UI: auth.py, session.py, team.py
- 팀 워크스페이스 페이지: 6_Team_Workspace.py
- 배치 파이프라인, 컬렉션 매니저 등 백엔드 모듈 통합
- configs 표준화 (base.yaml, exp_reproduce.yaml)

[팀 회의 브리핑 문서 작성]
- MEETING_BRIEF_RAG_EXP01_22.md: EXP01~22 전체 흐름을 팀원용으로 요약
```

---

### [3] 오늘 작업 완료도 체크 (하나만 체크)

> 진척 상황을 정량적으로 표시하고, 간단한 근거도 작성하세요.

- [ ]  🔴 0% (시작 못함)
- [ ]  🟠 25% (시작은 했지만 진척 없음)
- [ ]  🟡 50% (진행 중, 절반 이하)
- [ ]  🔵 75% (거의 완료됨)
- [o]  🟢 100% (완료 및 점검까지 완료)

📌 간단한 근거:

```
EXP22로 실험 파이프라인이 완전히 마무리됨.
Oracle 누수 제거 + RAGAS 다차원 평가 + 3-run 재현성 검증 + mismatch 수동 검수까지 완료.
이로써 22회에 걸친 실험 시리즈가 종결되고, 최종 평가 체계가 확정됨.
인증/팀워크스페이스 기능도 함께 통합 커밋하여 백엔드 기능이 완성 단계에 진입.
```

---

### [4] 오늘 협업 중 제안하거나 피드백한 내용이 있다면?

> 오늘 회의나 메시지에서 당신이 제안하거나 팀에 피드백한 내용은 무엇인가요?

✍️ 답변:

```
- EXP01~22 전체 실험 결과를 팀원이 이해할 수 있도록 브리핑 문서 작성하여 공유
- Oracle gap mean=2.33pp라는 수치를 제시하며,
  "first_deterministic이 거의 oracle 수준이므로 실운영에서도 신뢰할 수 있다"고 설명
- RAGAS 평가에서 Faithfulness=0.940, Context Recall=0.978이 의미하는 바를
  팀원들에게 직관적으로 설명 (94%의 답변이 검색된 문서에 충실, 97.8%의 관련 정보가 검색됨)
```

---

### [5] 오늘 분석/실험 중 얻은 인사이트나 발견한 문제점은?

> EDA, 모델 실험 중 유의미한 점이나 오류가 있었다면 자유롭게 작성하세요.

✍️ 답변:

```
핵심 인사이트:
1. Oracle gap mean=2.33pp → temp=0.0 첫 shot이 Oracle과 거의 동등한 성능
   → 실운영에서 SC 없이 단일 shot만으로도 충분히 높은 품질 가능
2. kw_v5 stdev=1.04pp (3-run) → 허용 가능한 변동 범위
   RAGAS는 stdev<0.5pp로 극히 안정적 → 평가 체계의 신뢰도 확보
3. Mismatch 근본 원인: prepare_judge_contexts가 chapter prefix를 미포함
   → 답변 자체는 정확하나 judge가 false negative를 내는 구조적 문제
   → 이는 평가 도구의 한계이지 파이프라인의 한계가 아님
4. doc_E/평가방식: temp=0.0의 구조적 한계 (3/3 run 동일 kw=0.455)
   → 특정 유형의 질문에서 temp=0.0이 최적이 아닌 경우 존재

문제점:
- ragas 0.4.3에서 gpt-5-mini의 temperature 제한 이슈 발견
  → FixedTempChatOpenAI 커스텀 래퍼로 해결
```

---

### [6] 일정 지연이나 협업 중 어려웠던 점이 있다면?

> 자기 업무 외에도 전체 일정이나 팀 내 협업에서 생긴 문제를 공유해 주세요.

✍️ 답변:

```
RAGAS 라이브러리(0.4.3)에서 gpt-5-mini 모델의 temperature 파라미터를 지원하지 않는 이슈가 있었음.
FixedTempChatOpenAI라는 커스텀 래퍼를 만들어 해결했으나, 외부 라이브러리와 최신 모델 간의
호환성 문제는 예상 외의 시간을 소모하게 함.
```

---

### [7] 오늘 발표 준비나 커뮤니케이션에서 기여한 부분은?

> 슬라이드 제작, 발표 연습, 질문 정리 등 발표와 관련된 활동을 썼다면 기록하세요.

✍️ 답변:

```
MEETING_BRIEF_RAG_EXP01_22.md 작성: EXP01~22 전체 흐름, 핵심 성과, 기술 의사결정을
팀원들이 발표 준비에 활용할 수 있도록 체계적으로 정리.
```

---

### [8] 내일 목표 / 할 일

> 구체적인 개인 업무나 팀 목표 기반 계획을 간단히 적어주세요.

✍️ 답변:

```
- Next.js 프론트엔드 통합 (김슬기 주도, 내가 review 및 백엔드 API 연동 지원)
- 최종 보고서 차트 및 README 정비
- FastAPI + Next.js 단일 런처 구현
```
