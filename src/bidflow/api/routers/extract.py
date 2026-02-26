from fastapi import APIRouter, Depends, HTTPException

from bidflow.api.deps import get_current_user
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.ingest.storage import DocumentStore, StorageRegistry

router = APIRouter()


@router.post("/extract/{doc_id}")
def extract_document(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    문서에 대해 G1~G4 멀티스텝 추출을 실행합니다.

    동기(def) 엔드포인트로 실행해 FastAPI threadpool에서 처리되도록 하여,
    장시간 추출 중에도 /auth/me, /team/* 등 다른 요청이 응답되게 합니다.
    """
    user_id = current_user.get("username") if isinstance(current_user, dict) else str(current_user)
    registry = StorageRegistry()
    store = DocumentStore(user_id=user_id, registry=registry)
    doc = store.load_document(doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    try:
        pipeline = ExtractionPipeline(user_id=user_id)
        results = pipeline.run(doc.doc_hash)

        # 추출 결과를 DB에 저장
        store.save_extraction_result(doc.doc_hash, results)

        return {
            "status": "success",
            "doc_id": doc_id,
            "user_id": user_id,
            "data": results,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"추출 실패: {str(e)}")


@router.get("/extract/{doc_id}")
def get_extraction_result(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """저장된 추출 결과를 조회합니다."""
    user_id = current_user.get("username") if isinstance(current_user, dict) else str(current_user)
    registry = StorageRegistry()
    store = DocumentStore(user_id=user_id, registry=registry)

    result = store.load_extraction_result(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="추출 결과가 없습니다. 먼저 분석을 실행해주세요.")

    return {
        "status": "success",
        "doc_id": doc_id,
        "user_id": user_id,
        "data": result,
    }
