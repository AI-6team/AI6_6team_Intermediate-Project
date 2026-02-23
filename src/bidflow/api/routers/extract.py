from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from bidflow.api.deps import get_current_user
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.ingest.storage import DocumentStore

router = APIRouter()

pipeline = ExtractionPipeline()
store = DocumentStore()

@router.post("/extract/{doc_id}", dependencies=[Depends(get_current_user)])
# async def extract_document(doc_id: str, background_tasks: BackgroundTasks):
async def extract_document(doc_id: str):
    """
    문서에 대해 G1~G4 멀티스텝 추출을 실행합니다.
    (MVP: 동기 실행으로 결과를 바로 반환함)
    """
    doc = store.load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    try:
        # 동기 실행
        results = pipeline.run(doc.doc_hash)
        
        # 결과 저장 로직이 필요하다면 여기에 추가 (현재는 반환만)
        # MVP: 결과를 문서 객체나 별도 Analysis 객체에 저장해야 하지만, 
        # 우선 JSON 응답으로 반환하여 UI에서 확인하게 함.
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "data": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"추출 실패: {str(e)}")
