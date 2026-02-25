import os
from typing import List, Optional
from langchain_core.documents import Document

_reranker_instance = None
_reranker_device = "cpu"
_reranker_batch_size = 16


def _get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """싱글턴 패턴으로 CrossEncoder 모델을 로드합니다."""
    global _reranker_instance, _reranker_device, _reranker_batch_size
    if _reranker_instance is None:
        import torch
        from sentence_transformers import CrossEncoder

        max_length = int(os.getenv("BIDFLOW_RERANK_MAX_LENGTH", "256"))
        _reranker_batch_size = int(os.getenv("BIDFLOW_RERANK_BATCH_SIZE", "16"))
        require_gpu = os.getenv("BIDFLOW_RERANK_REQUIRE_GPU", "1") == "1"

        if torch.cuda.is_available():
            _reranker_device = "cuda:0"
            frac = float(os.getenv("BIDFLOW_RERANK_GPU_FRACTION", "0.8"))
            # VRAM 상한을 명시해 shared memory(시스템 메모리) 전이를 줄입니다.
            try:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
            except Exception as e:
                print(f"[Reranker] WARN: set_per_process_memory_fraction failed: {e}")
            model_kwargs = {"dtype": torch.float16}
        else:
            _reranker_device = "cpu"
            if require_gpu:
                raise RuntimeError(
                    "CUDA is not available. Set BIDFLOW_RERANK_REQUIRE_GPU=0 to allow CPU fallback."
                )
            model_kwargs = None

        print(f"[Reranker] Loading model: {model_name}")
        _reranker_instance = CrossEncoder(
            model_name,
            max_length=max_length,
            device=_reranker_device,
            model_kwargs=model_kwargs,
        )
        print(
            f"[Reranker] Model loaded successfully "
            f"(device={_reranker_device}, max_length={max_length}, batch_size={_reranker_batch_size})"
        )
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
    scores = model.predict(pairs, batch_size=_reranker_batch_size, device=_reranker_device)

    scored_docs = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )

    result = [doc for doc, score in scored_docs[:top_k]]
    print(f"[Reranker] pool={len(docs)} → top={len(result)}")
    return result
