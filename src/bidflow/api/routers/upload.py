from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import shutil
import os
import tempfile
from bidflow.api.deps import get_current_user
from bidflow.ingest.pdf_parser import PDFParser
from bidflow.ingest.storage import DocumentStore, VectorStoreManager, StorageRegistry
from bidflow.domain.models import RFPDocument

router = APIRouter()

# 파서는 상태 없음이므로 모듈 레벨 유지
parser = PDFParser()


@router.post("/upload", response_model=dict)
async def upload_rfp(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """
    RFP PDF를 업로드하고 파싱하여 메타데이터를 저장하며, VectorDB에 수집합니다.
    요청별로 user_id에 맞는 DocumentStore / VectorStoreManager를 생성합니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원됩니다.")

    registry = StorageRegistry()
    store = DocumentStore(user_id=user_id, registry=registry)
    vector_manager = VectorStoreManager(user_id=user_id, registry=registry)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {str(e)}")

    try:
        doc: RFPDocument = parser.parse(tmp_path)
        doc.filename = file.filename
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 파싱 실패: {str(e)}")
        print(traceback.format_exc())
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"파싱 실패: {str(e)}")

    saved_path = store.save_document(doc)

    try:
        vector_manager.ingest_document(doc)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] VectorDB 수집 실패: {str(e)}")
        print(traceback.format_exc())
        print(f"[WARNING] 문서는 저장되었으나 VectorDB 수집에 실패했습니다.")

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return {
        "status": "success",
        "doc_id": doc.id,
        "doc_hash": doc.doc_hash,
        "filename": doc.filename,
        "chunk_count": len(doc.chunks),
        "table_count": len(doc.tables),
        "user_id": user_id,
    }


@router.get("/documents")
def list_documents(user_id: str = Depends(get_current_user)):
    """사용자의 처리된 문서 목록을 조회합니다."""
    store = DocumentStore(user_id=user_id)
    return store.list_documents()


@router.get("/documents/{doc_id}/view")
def view_document(doc_id: str, user_id: str = Depends(get_current_user)):
    """문서의 전체 파싱 내용을 조회합니다."""
    store = DocumentStore(user_id=user_id)
    doc = store.load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return doc
