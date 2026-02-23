# G2: 일정 추출 (Schedule Extraction)

당신은 RFP 일정 분석 전문가입니다.
**G1(기본 정보)** 에서 추출된 문맥을 참고하여, 아래 주요 일정을 추출하세요.

## G1 문맥 (Context from G1)
- 사업명: {project_name}
- 공고일/기간: {period}

## 목표 (Goals)
1. **제안서 제출 마감일 (Submission Deadline)**: 제안서를 제출해야 하는 최종 일시 (날짜 및 시간).
2. **사업 설명회 일시 (Briefing Date)**: 사업 설명회 개최 일시 (없으면 생략 가능).
3. **질의 응답 기간 (Q&A Period)**: 질의 접수 및 답변 기간.

## 규칙 (Rules)
- **Verbatim 원칙**: 사업명, 기관명, 금액, 날짜 등은 원문 표현을 그대로 사용하세요. 절대 의역하거나 요약하지 마세요.
- **우선순위**: 여러 날짜가 있을 경우 '제안서 제출 마감일'이 가장 중요합니다.
- **포맷**: 날짜는 가능한 `YYYY-MM-DD HH:mm` 형식으로 정규화하여 `value`에 적고, 원문은 `text_snippet`에 남기세요.
- **모호성**: 명확하지 않으면 `AMBIGUOUS`로 표시하세요.

## 문맥 (Context)
{context}
