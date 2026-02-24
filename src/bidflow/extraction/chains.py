from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langfuse import observe
import os

from bidflow.domain.models import Evidence, ExtractionSlot


def _get_llm_model() -> str:
    """dev.yaml의 model.llm 값을 읽어 반환합니다."""
    try:
        from bidflow.core.config import get_config
        cfg = get_config("dev")
        return (cfg.model.llm if cfg.model and cfg.model.llm else None) or "gpt-5-mini"
    except Exception:
        return "gpt-5-mini"


class ExtractionChain:
    """기본 추출 체인 클래스"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = _get_llm_model()
        # streaming=True는 PydanticOutputParser와 호환되지 않아 비활성화
        self.llm = ChatOpenAI(model=model_name, temperature=0, streaming=False)

    def _load_prompt(self, prompt_path: str) -> str:
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
    def run(self, context_text: str) -> G1Result:
        prompt_text = self._load_prompt("prompts/g1_basic.md")
        parser = PydanticOutputParser(pydantic_object=G1Result)

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "Context: {context}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser
        try:
            return chain.invoke({
                "context": context_text,
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as e:
            print(f"❌ Extraction Failed (G1): {e}")
            raw_chain = prompt | self.llm
            raw_res = raw_chain.invoke({
                "context": context_text,
                "format_instructions": parser.get_format_instructions()
            })
            print(f"🔍 Raw LLM Output: {raw_res.content}")
            raise RuntimeError(f"G1 추출 실패: {e}") from e


# --- G2: 일정 ---

class G2Result(BaseModel):
    submission_deadline: ExtractionSlot
    briefing_date: ExtractionSlot
    qna_period: ExtractionSlot


class G2Chain(ExtractionChain):
    @observe(name="G2_Schedule")
    def run(self, context_text: str, project_name: str, period: str) -> G2Result:
        prompt_text = self._load_prompt("prompts/g2_schedule.md")
        parser = PydanticOutputParser(pydantic_object=G2Result)

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "Context: {context}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser
        return chain.invoke({
            "context": context_text,
            "project_name": project_name,
            "period": period,
            "format_instructions": parser.get_format_instructions()
        })


# --- G3: 자격 요건 ---

class G3Result(BaseModel):
    required_licenses: ExtractionSlot
    region_restriction: ExtractionSlot
    financial_credit: ExtractionSlot
    restrictions: ExtractionSlot


class G3Chain(ExtractionChain):
    @observe(name="G3_Qualification")
    def run(self, context_text: str, project_name: str, issuer: str) -> G3Result:
        prompt_text = self._load_prompt("prompts/g3_qual.md")
        parser = PydanticOutputParser(pydantic_object=G3Result)

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "Context: {context}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser
        return chain.invoke({
            "context": context_text,
            "project_name": project_name,
            "issuer": issuer,
            "format_instructions": parser.get_format_instructions()
        })


# --- G4: 배점표 ---

class ScoredItem(BaseModel):
    category: str
    item: str
    score: float


class G4Result(BaseModel):
    items: List[ScoredItem]


class G4Chain(ExtractionChain):
    @observe(name="G4_Score")
    def run(self, context_text: str) -> G4Result:
        prompt_text = self._load_prompt("prompts/g4_score.md")
        parser = PydanticOutputParser(pydantic_object=G4Result)

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "Context: {context}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser
        return chain.invoke({
            "context": context_text,
            "format_instructions": parser.get_format_instructions()
        })
