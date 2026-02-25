from fastapi import APIRouter, Depends, HTTPException
from bidflow.api.deps import get_current_user
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.ingest.storage import DocumentStore, StorageRegistry

router = APIRouter()


@router.post("/extract/{doc_id}")
async def extract_document(
    doc_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    문서에 대해 G1~G4 멀티스텝 추출을 실행합니다.
    요청별로 user_id에 맞는 Pipeline / Store를 생성합니다.
    """
    registry = StorageRegistry()
    store = DocumentStore(user_id=user_id, registry=registry)
    doc = store.load_document(doc_id)

    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    try:
        pipeline = ExtractionPipeline(user_id=user_id)
        results = pipeline.run(doc.doc_hash)

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
