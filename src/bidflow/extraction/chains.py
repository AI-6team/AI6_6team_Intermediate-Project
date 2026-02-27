from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langfuse import observe
import os

from bidflow.domain.models import Evidence, ExtractionSlot

class ExtractionChain:
    """
    기본 추출 체인 클래스
    """
    def __init__(self, model_name: str = "gpt-5-mini"): # Scenario B: gpt-5-mini
        # gpt-5-mini는 reasoning 모델로 temperature=1 필요
        temp = 1 if model_name == "gpt-5-mini" else 0
        self.llm = ChatOpenAI(model=model_name, temperature=temp)

    def _load_prompt(self, prompt_path: str) -> str:
        # 현재 파일 기준 상대 경로로 변환
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, prompt_path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

# --- G1: 기본 정보 ---

class G1Result(BaseModel):
    project_name: ExtractionSlot
    issuer: ExtractionSlot
    period: ExtractionSlot
    budget: ExtractionSlot

class G1Chain(ExtractionChain):
    @observe(name="G1_Basic_Info")
    def run(self, context_text: str) -> Optional[G1Result]:
        prompt_text = self._load_prompt("prompts/g1_basic.md")

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "위 문맥에서 요청된 정보를 추출하여 JSON으로 응답해주세요.")
        ])

        llm_structured = self.llm.with_structured_output(G1Result)
        chain = prompt | llm_structured
        try:
            return chain.invoke({"context": context_text})
        except Exception as e:
            print(f"❌ Extraction Failed (G1): {e}")
            import traceback
            traceback.print_exc()
            return None

# --- G2: 일정 ---

class G2Result(BaseModel):
    submission_deadline: ExtractionSlot
    briefing_date: ExtractionSlot
    qna_period: ExtractionSlot

class G2Chain(ExtractionChain):
    @observe(name="G2_Schedule")
    def run(self, context_text: str, project_name: str, period: str) -> Optional[G2Result]:
        prompt_text = self._load_prompt("prompts/g2_schedule.md")

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "위 문맥에서 요청된 일정 정보를 추출하여 JSON으로 응답해주세요.")
        ])

        llm_structured = self.llm.with_structured_output(G2Result)
        chain = prompt | llm_structured
        try:
            return chain.invoke({
                "context": context_text,
                "project_name": project_name,
                "period": period,
            })
        except Exception as e:
            print(f"❌ Extraction Failed (G2): {e}")
            import traceback
            traceback.print_exc()
            return None

# --- G3: 자격 요건 ---

class G3Result(BaseModel):
    required_licenses: ExtractionSlot
    region_restriction: ExtractionSlot
    financial_credit: ExtractionSlot
    restrictions: ExtractionSlot

class G3Chain(ExtractionChain):
    @observe(name="G3_Qualification")
    def run(self, context_text: str, project_name: str, issuer: str) -> Optional[G3Result]:
        prompt_text = self._load_prompt("prompts/g3_qual.md")

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "위 문맥에서 요청된 자격 요건 정보를 추출하여 JSON으로 응답해주세요.")
        ])

        llm_structured = self.llm.with_structured_output(G3Result)
        chain = prompt | llm_structured
        try:
            return chain.invoke({
                "context": context_text,
                "project_name": project_name,
                "issuer": issuer,
            })
        except Exception as e:
            print(f"❌ Extraction Failed (G3): {e}")
            import traceback
            traceback.print_exc()
            return None

# --- G4: 배점표 ---

class ScoredItem(BaseModel):
    category: str
    item: str
    score: float

class G4Result(BaseModel):
    items: List[ScoredItem]

class G4Chain(ExtractionChain):
    @observe(name="G4_Score")
    def run(self, context_text: str) -> Optional[G4Result]:
        prompt_text = self._load_prompt("prompts/g4_score.md")

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "위 문맥에서 평가 항목과 배점을 추출하여 JSON으로 응답해주세요.")
        ])

        llm_structured = self.llm.with_structured_output(G4Result)
        chain = prompt | llm_structured
        try:
            return chain.invoke({"context": context_text})
        except Exception as e:
            print(f"❌ Extraction Failed (G4): {e}")
            import traceback
            traceback.print_exc()
            return None
