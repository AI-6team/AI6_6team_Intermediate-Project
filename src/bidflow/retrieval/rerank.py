from typing import List, Optional
from langchain_core.documents import Document

_reranker_instance = None


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """싱글턴 패턴으로 CrossEncoder 모델을 로드합니다."""
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading model: {model_name}")
        _reranker_instance = CrossEncoder(model_name)
        print("[Reranker] Model loaded successfully")
    return _reranker_instance


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 15,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> List[Document]:
    """
    Cross-encoder reranker로 문서를 재정렬하여 상위 top_k개를 반환합니다.

    Args:
        query: 검색 쿼리
        docs: 후보 문서 리스트
        top_k: 반환할 최종 문서 수
        model_name: Cross-encoder 모델명

    Returns:
        재정렬된 상위 top_k 문서 리스트
    """
    if not docs:
        return docs

    model = _get_reranker(model_name)

    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )

    result = [doc for doc, score in scored_docs[:top_k]]
    print(f"[Reranker] pool={len(docs)} → top={len(result)}")
    return result
