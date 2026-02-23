from typing import Dict, Any
from bidflow.extraction.chains import G1Chain, G2Chain, G3Chain, G4Chain
from bidflow.extraction.hint_detector import HintDetector
from bidflow.ingest.storage import VectorStoreManager
from bidflow.retrieval.hybrid_search import HybridRetriever
from bidflow.domain.models import ComplianceMatrix, ExtractionSlot
from bidflow.security.rails.input_rail import InputRail
from bidflow.security.masking import PIIMasker
from langfuse import observe

class ExtractionPipeline:
    """
    RFP 추출 파이프라인 (G1 -> G2/G3 -> G4)
    """
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.retriever = HybridRetriever(self.vector_manager)
        self.g1_chain = G1Chain()
        self.g2_chain = G2Chain()
        self.g3_chain = G3Chain()
        self.g4_chain = G4Chain()
        self.input_rail = InputRail()
        self.pii_masker = PIIMasker()
        self.hint_detector = HintDetector()

    @observe(name="Extraction_Pipeline")
    def run(self, doc_hash: str) -> Dict[str, Any]:
        """
        문서 해시를 받아 전체 생성을 수행하고 결과를 반환합니다.
        """
        # 1. 문맥 검색 (Retrieval) — HybridRetriever + Reranker
        docs = self.retriever.invoke("사업 개요 및 제안 요청 사항")
        context_text = "\n\n".join([d.page_content for d in docs])

        # --- Security Rail Application ---
        # 1. Prompt Injection 검사
        self.input_rail.check(context_text)
        
        # 2. PII Masking
        context_text = self.pii_masker.mask(context_text)
        # ---------------------------------

        # 3. 정규식 힌트 감지 (김슬기 전략)
        hints = self.hint_detector.format_hints(context_text)
        if hints:
            print(f"[HintDetector] {hints}")
            context_with_hints = f"{hints}\n\n{context_text}"
        else:
            context_with_hints = context_text

        results = {}

        # 4. G1: 기본 정보 (힌트 포함 컨텍스트 사용)
        print(f"--- G1 추출 시작 ({doc_hash}) ---")
        g1_result = self.g1_chain.run(context_with_hints)
        results["g1"] = g1_result.model_dump()
        
        project_name = g1_result.project_name.value
        # 기간 정보 추출 (문자열로 변환 필요)
        period_val = str(g1_result.period.value)
        issuer_val = str(g1_result.issuer.value)

        # 3. G2: 일정 (G1 의존)
        print("--- G2 추출 시작 ---")
        g2_result = self.g2_chain.run(context_text, project_name, period_val)
        results["g2"] = g2_result.model_dump()

        # 4. G3: 자격 (G1 의존)
        print("--- G3 추출 시작 ---")
        g3_result = self.g3_chain.run(context_text, project_name, issuer_val)
        results["g3"] = g3_result.model_dump()

        # 5. G4: 배점 (테이블 위주지만 텍스트도 참고)
        # 배점표는 "평가 항목 및 배점" 쿼리로 다시 검색하는 것이 좋음
        print("--- G4 추출 시작 ---")
        docs_g4 = self.retriever.invoke("제안서 평가 항목 및 배점 기준표")
        context_g4 = "\n\n".join([d.page_content for d in docs_g4])
        
        g4_result = self.g4_chain.run(context_g4)
        results["g4"] = g4_result.model_dump()

        print("--- 추출 완료 ---")
        return results
