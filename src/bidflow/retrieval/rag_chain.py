from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from bidflow.retrieval.hybrid_search import HybridRetriever
from bidflow.extraction.hint_detector import HintDetector

class RAGChain:
    """
    RAG (Retrieval-Augmented Generation) 체인
    QueryAnalyzer(김보윤), HintDetector(김슬기), Front-loading(김슬기) 통합
    """
    def __init__(self, retriever=None, model_name: str = "gpt-5-mini", use_query_analyzer: bool = False):
        dest_temp = 0
        if model_name == "gpt-5-mini":
            dest_temp = 1 # [Fix] Reasoning model requires temp=1

        self.llm = ChatOpenAI(model=model_name, temperature=dest_temp, timeout=60, max_retries=2)
        self.hint_detector = HintDetector()

        # QueryAnalyzer (optional, 김보윤)
        self.query_analyzer = None
        if use_query_analyzer:
            from bidflow.retrieval.query_analyzer import QueryAnalyzer
            self.query_analyzer = QueryAnalyzer(model_name=model_name)
            print("[RAGChain] QueryAnalyzer enabled")

        if retriever is None:
            self.retriever = HybridRetriever()
        else:
            self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_template(
            "아래 문맥(Context)만을 근거로 질문에 답하세요.\n"
            "반드시 원문에 있는 사업명, 기관명, 금액, 날짜 등의 표현을 그대로(Verbatim) 사용하세요.\n"
            "문맥에 답이 없으면 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
            "{hints}\n"
            "## 문맥 (Context)\n{context}\n\n"
            "## 질문\n{question}\n\n"
            "## 답변\n"
        )
        
    def invoke(self, question: str, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        질문에 대한 답변과 검색된 문서를 반환합니다.

        Args:
            question: 사용자 질문
            doc_ids: 특정 문서 ID 목록 (Front-loading에 사용)

        Returns:
            {
                "answer": str,
                "retrieved_contexts": List[str]
            }
        """
        # 0. Query Analysis (optional, 김보윤)
        query_type = None
        if self.query_analyzer:
            analysis = self.query_analyzer.analyze(question)
            query_type = analysis.query_type
            print(f"[QueryAnalyzer] type={query_type}, fields={analysis.required_fields}")

        # 1. Retrieve
        docs = self.retriever.invoke(question)

        # 2. Front-loading: doc_ids 지정 시 앞부분 청크 강제 포함
        if doc_ids:
            head_docs = self._get_front_loaded_docs(doc_ids)
            if head_docs:
                seen = set(doc.page_content for doc in head_docs)
                merged = list(head_docs)
                for doc in docs:
                    if doc.page_content not in seen:
                        merged.append(doc)
                docs = merged

        context_text = "\n\n".join([doc.page_content for doc in docs])

        # 3. 정규식 힌트 감지
        hints = self.hint_detector.format_hints(context_text)

        # 4. Generate
        answer = self.prompt.pipe(self.llm).pipe(StrOutputParser()).invoke({
            "context": context_text,
            "question": question,
            "hints": hints,
        })

        result = {
            "answer": answer,
            "retrieved_contexts": [doc.page_content for doc in docs],
        }
        if query_type:
            result["query_type"] = query_type
        return result

    def _get_front_loaded_docs(self, doc_ids: List[str], max_chunk_idx: int = 15) -> List[Document]:
        """문서의 앞부분 청크를 가져옴 (Front-loading 전략, 김슬기)"""
        try:
            from bidflow.ingest.storage import VectorStoreManager
            vsm = VectorStoreManager()
            result = vsm.vector_db.get(
                where={"doc_hash": {"$in": doc_ids}},
                include=["metadatas", "documents"]
            )
            if not result or not result["documents"]:
                return []

            head_docs = []
            for text, meta in zip(result["documents"], result["metadatas"]):
                # chunk_index 또는 page_no 기반 필터링
                chunk_idx = meta.get("chunk_index", meta.get("page_no", 999))
                if isinstance(chunk_idx, int) and chunk_idx <= max_chunk_idx:
                    head_docs.append(Document(page_content=text, metadata=meta))

            head_docs.sort(key=lambda x: x.metadata.get("chunk_index", x.metadata.get("page_no", 0)))
            if head_docs:
                print(f"[Front-loading] {len(head_docs)}개 앞부분 청크 포함")
            return head_docs
        except Exception as e:
            print(f"[Front-loading] Failed: {e}")
            return []
