from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from bidflow.api.deps import get_current_user
from bidflow.ingest.storage import DocumentStore
from bidflow.ingest.service import IngestService

router = APIRouter()

# 의존성 주입을 위한 함수
def get_ingest_service():
    return IngestService()

@router.post("/upload")
async def upload_rfp(
    file: UploadFile = File(...),
    service: IngestService = Depends(get_ingest_service),
    user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith(('.pdf', '.hwp')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    try:
        user_id = user.get("username", "")
        
        doc = await service.process_upload(file, user_id)
        
        return {
            "status": "success",
            "doc_id": doc.id,
            "doc_hash": doc.doc_hash,
            "filename": doc.filename,
            "chunk_count": len(doc.chunks),
            "table_count": len(doc.tables),
            "user_id": user_id,
            "message": "파일 업로드 및 파싱이 완료되었습니다."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
def list_documents(user: dict = Depends(get_current_user)):
    user_id = user.get("username", "")
    store = DocumentStore(user_id=user_id)
    return store.list_documents()

@router.get("/documents/{doc_id}/view")
def view_document(doc_id: str, user: dict = Depends(get_current_user)):
    user_id = user.get("username", "")
    store = DocumentStore(user_id=user_id)
    doc = store.load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc
