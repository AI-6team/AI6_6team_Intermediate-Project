from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from bidflow.ingest.storage import VectorStoreManager
from bidflow.core.config import get_config

class HybridRetriever(BaseRetriever):
    """
    키워드 검색(BM25)과 벡터 검색(VectorStore)을 결합한 하이브리드 검색기입니다.
    EnsembleRetriever가 없는 환경을 대비해 RRF(Reciprocal Rank Fusion)를 직접 구현했습니다.
    Config에서 hybrid_alpha, rerank 설정을 자동으로 읽어 적용합니다.
    """


    vector_retriever: Any = None
    bm25_retriever: Any = None
    weights: List[float] = [0.5, 0.5]
    top_k: int = 15
    pool_size: int = 50
    use_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    tenant_id: str = "default"
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    acl_filter: Optional[Dict[str, Any]] = None

    def __init__(self, vector_store_manager: VectorStoreManager = None, top_k: int = None, weights: List[float] = None, tenant_id: str = "default", user_id: str = None, group_id: str = None, doc_ids: List[str] = None, **kwargs):
        """
        Args:
            vector_store_manager: 미리 초기화된 VectorStoreManager 인스턴스 (없으면 새로 생성)
            top_k: 최종 반환할 문서 수 (None이면 config에서 읽음)
            weights: [BM25 가중치, 벡터 검색 가중치] (None이면 config에서 alpha로 계산)
        """
        # Config 읽기
        cfg = get_config()
        retrieval_cfg = cfg.retrieval or {}

        # top_k: 인자 > config > 기본값 15
        if top_k is None:
            top_k = retrieval_cfg.get("top_k", 15) if isinstance(retrieval_cfg, dict) else 15

        # weights: 인자 > config alpha > 기본값 [0.3, 0.7]
        if weights is None:
            alpha = retrieval_cfg.get("hybrid_alpha", 0.7) if isinstance(retrieval_cfg, dict) else 0.7
            weights = [round(1 - alpha, 2), round(alpha, 2)]  # [BM25, Vector]

        # Rerank 설정
        use_rerank = retrieval_cfg.get("rerank", False) if isinstance(retrieval_cfg, dict) else False
        pool_size = retrieval_cfg.get("rerank_pool", 50) if isinstance(retrieval_cfg, dict) else 50
        rerank_model = retrieval_cfg.get("rerank_model", "BAAI/bge-reranker-v2-m3") if isinstance(retrieval_cfg, dict) else "BAAI/bge-reranker-v2-m3"

        # ACL Filter
        conditions = [{"tenant_id": tenant_id}]
        if user_id:
            conditions.append({"user_id": user_id})
        if group_id:
            conditions.append({"group_id": group_id})

        if len(conditions) > 1:
            acl_filter = {"$and": conditions}
        else:
            acl_filter = conditions[0]

        # rerank 사용 시 후보군을 pool_size만큼 가져옴
        search_k = pool_size if use_rerank else top_k

        # 1. 컴포넌트 초기화
        if vector_store_manager is None:
            vector_store_manager = VectorStoreManager()

        vector_retriever = vector_store_manager.get_retriever(search_kwargs={"k": search_k, "filter": acl_filter})

        # BM25 초기화 (테넌트 문서만 로드)
        all_docs = []
        try:
             # Fetch only tenant documents for BM25
             result = vector_store_manager.vector_db.get(where=acl_filter)
             if result and result["documents"]:
                for i, text in enumerate(result["documents"]):
                    meta = result["metadatas"][i] if result["metadatas"] else {}
                    all_docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"Warning: Failed to fetch docs for BM25: {e}")

        if not all_docs:
             bm25_retriever = None
        else:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = search_k

        # 2. Pydantic 초기화 (super().__init__에 필드 전달)
        super().__init__(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=top_k,
            weights=weights,
            pool_size=pool_size,
            use_rerank=use_rerank,
            rerank_model=rerank_model,
            tenant_id=tenant_id,
            user_id=user_id,
            group_id=group_id,
            doc_ids=doc_ids,
            acl_filter=acl_filter,
            **kwargs
        )

        print(f"[HybridRetriever] weights={self.weights}, top_k={self.top_k}, rerank={self.use_rerank}, pool={self.pool_size}")

    def set_doc_ids(self, doc_ids: List[str]):
        """검색 대상을 특정 문서 ID 목록으로 제한합니다."""
        self.doc_ids = doc_ids

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        BM25와 Vector 검색 결과를 RRF로 결합하여 반환
        rerank 설정 시 pool_size만큼 후보를 가져온 뒤 Cross-encoder로 재정렬
        """
        search_k = self.pool_size if self.use_rerank else self.top_k

        # 1. 각 검색기 실행
        bm25_docs = []
        if self.bm25_retriever:
            try:
                self.bm25_retriever.k = search_k * 2  # 후보군을 더 많이 가져옴
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"BM25 Error: {e}")

        try:
            # Vector 검색 시 doc_ids 필터 적용 (Query-time filtering)
            current_filter = self.acl_filter.copy() if self.acl_filter else {}
            
            if self.doc_ids:
                doc_filter = {"doc_hash": {"$in": self.doc_ids}}
                
                if current_filter:
                    if "$and" in current_filter:
                        # 기존 $and 조건에 추가
                        new_conditions = current_filter["$and"] + [doc_filter]
                        final_filter = {"$and": new_conditions}
                    else:
                        # 기존 단일 조건과 $and 결합
                        final_filter = {"$and": [current_filter, doc_filter]}
                else:
                    final_filter = doc_filter
            else:
                final_filter = current_filter

            self.vector_retriever.search_kwargs["k"] = search_k * 2
            self.vector_retriever.search_kwargs["filter"] = final_filter
            vector_docs = self.vector_retriever.invoke(query)
        except Exception as e:
            print(f"Vector Error: {e}")
            vector_docs = []

        # [Filter] doc_ids가 설정된 경우 해당 문서만 필터링
        if self.doc_ids:
            bm25_docs = [d for d in bm25_docs if d.metadata.get("doc_hash") in self.doc_ids]

        # 2. RRF (Reciprocal Rank Fusion) with Weights
        # rerank 사용 시 pool_size만큼, 아니면 top_k만큼 RRF 결과 가져옴
        rrf_top = self.pool_size if self.use_rerank else self.top_k
        merged = self._rrf_merge(bm25_docs, vector_docs, k=60, limit=rrf_top)

        # 3. Reranker 적용 (설정된 경우)
        if self.use_rerank and len(merged) > 0:
            from bidflow.retrieval.rerank import rerank
            merged = rerank(query, merged, top_k=self.top_k, model_name=self.rerank_model)

        return merged

    def _rrf_merge(self, list1: List[Document], list2: List[Document], k=60, limit: int = None) -> List[Document]:
        """
        두 문서 리스트를 Weighted RRF 알고리즘으로 병합
        Score = weight * (1 / (rank + k))
        list1: BM25 Results (weights[0])
        list2: Vector Results (weights[1])
        """
        if limit is None:
            limit = self.top_k

        scores = defaultdict(float)
        doc_map = {}

        w_bm25 = self.weights[0] if len(self.weights) > 0 else 0.5
        w_vector = self.weights[1] if len(self.weights) > 1 else 0.5

        # BM25 점수 합산
        for rank, doc in enumerate(list1):
            scores[doc.page_content] += w_bm25 * (1 / (rank + k))
            doc_map[doc.page_content] = doc

        # Vector 점수 합산
        for rank, doc in enumerate(list2):
            scores[doc.page_content] += w_vector * (1 / (rank + k))
            if doc.page_content not in doc_map:
                doc_map[doc.page_content] = doc

        # 점수순 정렬
        sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [doc_map[content] for content in sorted_contents[:limit]]
