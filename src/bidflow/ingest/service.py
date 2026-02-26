import os
import shutil
import tempfile
import asyncio
import hashlib
from fastapi import UploadFile
from bidflow.domain.models import RFPDocument
from bidflow.ingest.pdf_parser import PDFParser
from bidflow.ingest.storage import DocumentStore, VectorStoreManager, StorageRegistry

try:
    from bidflow.parsing.hwp_parser import HWPParser
except ImportError:
    HWPParser = None

class IngestService:
    def __init__(self):
        # 파서는 상태가 없다면 한 번만 초기화해서 재사용
        self.pdf_parser = PDFParser()
        self.hwp_parser = HWPParser() if HWPParser else None

    def _calculate_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def process_upload(self, file: UploadFile, user_id: str) -> RFPDocument:
        """
        업로드된 파일을 임시 저장하고, 파싱 및 DB 저장을 수행합니다.
        """
        # 1. 임시 파일로 저장
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # 2. 파일 파싱 (확장자 분기)
            loop = asyncio.get_running_loop()
            
            if suffix == '.pdf':
                doc = await loop.run_in_executor(None, self.pdf_parser.parse, tmp_path)
            elif suffix == '.hwp':
                if not self.hwp_parser:
                    raise ValueError("HWP parsing is not supported (HWPParser module not found).")
                chunks = await loop.run_in_executor(None, self.hwp_parser.parse, tmp_path)
                
                doc_hash = self._calculate_hash(tmp_path)
                doc = RFPDocument(
                    id=doc_hash,
                    filename=file.filename,
                    file_path=tmp_path,
                    doc_hash=doc_hash,
                    chunks=chunks,
                    tables=[],
                    status="READY"
                )
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")

            doc.filename = file.filename

            # 3. 저장소 레지스트리 및 매니저 초기화
            registry = StorageRegistry()
            store = DocumentStore(user_id=user_id, registry=registry)
            vector_manager = VectorStoreManager(user_id=user_id, registry=registry)

            # 4. 메타데이터 저장 (JSON 등)
            store.save_document(doc)

            # 5. 벡터 DB 인덱싱 (실패해도 메타데이터 저장은 유지하거나, 필요시 롤백 로직 추가)
            try:
                vector_manager.ingest_document(doc)
            except Exception as e:
                print(f"[WARNING] VectorDB ingestion failed: {e}")

            return doc
        finally:
            # 6. 임시 파일 정리
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
