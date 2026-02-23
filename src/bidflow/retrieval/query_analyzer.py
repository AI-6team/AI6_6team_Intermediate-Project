"""Query Analyzer: classifies query type for optimal processing strategy.

Ported from 김보윤's QueryAnalyzer.
Classifies queries into summary/extraction/decision types to route
to appropriate processing pipelines.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class QueryAnalysis(BaseModel):
    """쿼리 분석 결과"""
    query_type: str = Field(description="summary | extraction | decision")
    required_fields: List[str] = Field(default_factory=list)
    target_doc_id: Optional[str] = None
    original_query: str = ""


ANALYZER_SYSTEM_PROMPT = """당신은 RFP(제안요청서) 관련 질의를 분류하는 시스템입니다.

사용자의 질의를 분석하여 다음 JSON 형식으로 응답하십시오:
{{
  "query_type": "summary | extraction | decision",
  "required_fields": ["field1", "field2"],
  "target_doc_id": null
}}

query_type 정의:
- summary: RFP 전체 요약 요청 (예: "이 사업에 대해 설명해줘", "사업 개요를 알려줘")
- extraction: 특정 필드 추출 요청 (예: "예산이 얼마야?", "마감일이 언제야?")
- decision: 입찰 참여 여부 판단 요청 (예: "이 사업에 참여해도 될까?", "자격 요건을 충족하나?")

복합 질의는 "decision"으로 분류하십시오.

가능한 required_fields:
issuing_agency, project_overview, budget, deadline, submission_method, qualification"""

ANALYZER_USER_PROMPT = '다음 질의를 분류하십시오:\n"{query}"'


class QueryAnalyzer:
    """쿼리 유형 자동 분류 (summary/extraction/decision)"""

    VALID_TYPES = {"summary", "extraction", "decision"}

    def __init__(self, model_name: str = "gpt-5-mini"):
        dest_temp = 1 if model_name == "gpt-5-mini" else 0
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=dest_temp,
            timeout=30,
            max_retries=2,
        ).bind(response_format={"type": "json_object"})

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYZER_SYSTEM_PROMPT),
            ("human", ANALYZER_USER_PROMPT),
        ])
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def analyze(self, query: str) -> QueryAnalysis:
        """쿼리를 분류하여 QueryAnalysis 반환"""
        try:
            result = self.chain.invoke({"query": query})

            query_type = result.get("query_type", "extraction")
            if query_type not in self.VALID_TYPES:
                query_type = "extraction"

            return QueryAnalysis(
                query_type=query_type,
                required_fields=result.get("required_fields", []),
                target_doc_id=result.get("target_doc_id"),
                original_query=query,
            )
        except Exception as e:
            print(f"[QueryAnalyzer] Classification failed: {e}, defaulting to extraction")
            return QueryAnalysis(
                query_type="extraction",
                required_fields=[],
                original_query=query,
            )
