from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from typing import List
import shutil
import os
import tempfile
from bidflow.api.deps import get_current_user
from bidflow.ingest.pdf_parser import PDFParser
from bidflow.ingest.storage import DocumentStore, VectorStoreManager
from bidflow.domain.models import RFPDocument

router = APIRouter()

# Initialize services (singleton-ish for MVP)
parser = PDFParser()
store = DocumentStore()
vector_manager = VectorStoreManager()

@router.post("/upload", response_model=dict, dependencies=[Depends(get_current_user)])
async def upload_rfp(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    RFP PDF를 업로드하고 파싱하여 메타데이터를 저장하며, VectorDB에 수집합니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원됩니다.")

    # 1. 업로드된 파일을 임시 저장 (또는 raw 폴더에 직접 저장)
    # 안전한 해시 계산 및 파싱을 위해 먼저 임시 파일로 저장합니다.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {str(e)}")

    # 2. 파싱 (MVP를 위해 동기 처리, 비동기 가능)
    try:
        doc: RFPDocument = parser.parse(tmp_path)
        # 원본 파일명으로 수정 (임시 파일명 대신)
        doc.filename = file.filename
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 파싱 실패: {str(e)}")
        print(traceback.format_exc())
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"파싱 실패: {str(e)}")
    
    # 3. 문서 저장소에 저장 (raw 파일을 영구 저장소로 이동/복사 처리 포함)
    # 참고: parser.parse는 tmp_path를 file_path로 사용합니다.
    # DocumentStore.save_document는 doc.file_path에서 raw 폴더로 복사를 시도합니다.
    # 영구적인 raw 위치를 유지하려면 doc.file_path를 업데이트하거나, save_document가 복사를 처리하도록 둡니다.
    # 여기서는 save_document가 raw 복사를 처리하도록 합니다.
    
    saved_path = store.save_document(doc)
    
    # 4. VectorDB 수집 (Background Task 권장하나, MVP 피드백 루프를 위해 동기 처리)
    # background_tasks.add_task(vector_manager.ingest_document, doc) 
    try:
        vector_manager.ingest_document(doc)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] VectorDB 수집 실패: {str(e)}")
        print(traceback.format_exc())
        # 수집 실패해도 문서는 저장되었으므로 경고만 발생
        print(f"[WARNING] 문서는 저장되었으나 VectorDB 수집에 실패했습니다.")
    
    # 임시 파일 정리
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return {
        "status": "success",
        "doc_id": doc.id,
        "doc_hash": doc.doc_hash,
        "filename": doc.filename,
        "chunk_count": len(doc.chunks),
        "table_count": len(doc.tables)
    }

@router.get("/documents", dependencies=[Depends(get_current_user)])
def list_documents():
    """
    모든 처리된 문서 목록을 조회합니다.
    """
    # store.list_documents()는 이제 리스트[Dict]를 반환함
    docs = store.list_documents()
    return docs

@router.get("/documents/{doc_id}/view", dependencies=[Depends(get_current_user)])
def view_document(doc_id: str):
    """
    문서의 전체 파싱 내용을 조회합니다.
    """
    doc = store.load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc
