import pandas as pd
from typing import List
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.documents import Document

from ragas import evaluate
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.testset.transforms import Transforms
from ragas.testset.persona import Persona
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.extractors import EmbeddingExtractor, KeyphrasesExtractor, NERExtractor, HeadlinesExtractor

class FixedTempChatOpenAI(ChatOpenAI):
    """
    Force temperature=1 for models that do not support other values (e.g. o1/o3 series)
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["temperature"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["temperature"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

class RagasRunner:
    """
    RAGAS 기반의 평가 파이프라인 Runner (v0.2+ API 대응)
    """
    def __init__(self):
        # LangChain 모델을 Ragas Wrapper로 감싸서 사용
        # Ragas 내부 동작의 안정성을 위해 gpt-5-mini 사용
        # [Fix] O-series 모델은 temperature=1 강제 (0.01 등 미지원 에러 방지)
        base_llm = FixedTempChatOpenAI(model="gpt-5-mini", timeout=180, max_retries=3)
        base_embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")

        self.llm = LangchainLLMWrapper(base_llm)
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    def generate_testset(self, docs: List[Document], test_size: int = 3) -> pd.DataFrame:
        """
        문서에서 합성 테스트셋(질문-정답)을 생성합니다.
        비용/시간 절약을 위해 기본 3개만 생성합니다.

        Args:
            docs: LangChain Document 리스트
            test_size: 생성할 테스트 케이스 수

        Returns:
            question, ground_truth 등이 포함된 DataFrame
        """
        # 1. 도메인 특화 페르소나 수동 정의 (자동 생성 실패 방지)
        personas = [
            Persona(
                name="입찰 실무자 (Bid Manager)",
                role_description="RFP에서 자격요건, 제출서류, 평가기준과 일정/예산을 빠르게 확인하려는 담당자. 핵심적인 제약사항과 마감 기한에 민감함.",
            ),
            Persona(
                name="기술 제안서 작성자 (Technical Writer)",
                role_description="요구사항을 기술적으로 분석하고, 시스템 아키텍처와 기능 명세를 구체화하려는 엔지니어. 기술적 세부 사항과 평가 항목 매핑에 관심이 많음.",
            ),
             Persona(
                name="사업 책임자 (Project Director)",
                role_description="사업의 전략적 중요성과 기대 효과, 예산 적정성을 검토하는 의사결정권자. 전반적인 사업 범위와 목표에 집중함.",
            ),
        ]

        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            persona_list=personas,  # 수동 페르소나 주입
        )

        # 2. Transforms 강화 (KG 풍부화)
        # HeadlineSplitter는 HeadlinesExtractor가 먼저 수행되어야 함
        transforms = [
            HeadlinesExtractor(llm=self.llm),
            HeadlineSplitter(),
            EmbeddingExtractor(embedding_model=self.embeddings),
            # [Fix] gpt-5-mini(o3-mini)가 strict JSON parsing에 실패하는 경우가 있어 복잡한 추출기는 제외함
            # KeyphrasesExtractor(llm=self.llm),
            # NERExtractor(llm=self.llm),
        ]

        testset = generator.generate_with_langchain_docs(
            docs,
            testset_size=test_size,
            transforms=transforms
        )

        return testset.to_pandas()

    def run_eval(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """
        생성된 테스트셋에 대해 평가를 수행합니다.
        이제 RAGChain을 통해 실제 답변과 검색된 문맥을 생성하여 평가합니다.

        Args:
            dataset_df: generate_testset에서 반환된 DataFrame

        Returns:
            평가 점수가 포함된 DataFrame
        """
        from bidflow.retrieval.rag_chain import RAGChain
        
        # RAG 체인 초기화
        rag_chain = RAGChain(model_name="gpt-5-mini")
        
        # 실제 답변 생성 (RAG)
        answers = []
        contexts = []
        
        print(f"Evaluating {len(dataset_df)} test cases...")
        
        for idx, row in dataset_df.iterrows():
            question = row["question"] if "question" in row else row.get("user_input", "")
            if not question:
                answers.append("N/A")
                contexts.append([])
                continue
                
            try:
                result = rag_chain.invoke(question)
                answers.append(result["answer"])
                contexts.append(result["retrieved_contexts"])
            except Exception as e:
                print(f"Error evaluating Q: {question[:30]}... -> {e}")
                answers.append("Error")
                contexts.append([])

        # 평가용 데이터셋 구성
        eval_dict = {
            "user_input": dataset_df["question"].tolist() if "question" in dataset_df.columns else dataset_df["user_input"].tolist(),
            "response": answers,
            "retrieved_contexts": contexts,
        }

        # ground_truth가 있으면 추가 (일부 메트릭에서 사용)
        if "ground_truth" in dataset_df.columns:
            eval_dict["reference"] = dataset_df["ground_truth"].tolist()
        elif "reference" in dataset_df.columns: # v0.2 fallback
             eval_dict["reference"] = dataset_df["reference"].tolist()

        hf_dataset = Dataset.from_dict(eval_dict)

        hf_dataset = Dataset.from_dict(eval_dict)

        from ragas.metrics import ContextRecall, ContextPrecision

        # 평가 메트릭 설정 (v0.2+ 클래스 기반)
        metrics = [
            Faithfulness(llm=self.llm),
            ResponseRelevancy(llm=self.llm, embeddings=self.embeddings),
            ContextRecall(llm=self.llm),
            ContextPrecision(llm=self.llm),
        ]

        # 평가 실행
        results = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,  # 에러 시 NaN 반환
        )

        return results.to_pandas()
