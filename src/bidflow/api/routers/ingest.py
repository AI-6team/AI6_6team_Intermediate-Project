from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from .ingest_service import IngestService
from bidflow.domain.schemas import UploadResponse
from bidflow.api.deps import get_current_user
from bidflow.ingest.storage import DocumentStore

router = APIRouter()

# 의존성 주입을 위한 함수
def get_ingest_service():
    return IngestService()

@router.post("/upload", response_model=UploadResponse)
async def upload_rfp(
    file: UploadFile = File(...),
    service: IngestService = Depends(get_ingest_service),
    user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith(('.pdf', '.hwp')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    try:
        # user dict에서 username 추출
        doc = await service.process_upload(file, user["username"])
        return UploadResponse(
            doc_id=doc.id,
            filename=doc.filename,
            status=doc.status,
            message="파일 업로드 및 파싱이 완료되었습니다."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
def list_documents(user: dict = Depends(get_current_user)):
    store = DocumentStore(user_id=user["username"])
    return store.list_documents()

@router.get("/documents/{doc_id}/view")
def view_document(doc_id: str, user: dict = Depends(get_current_user)):
    store = DocumentStore(user_id=user["username"])
    doc = store.load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc