import os
import shutil
import tempfile
import asyncio
import hashlib
from fastapi import UploadFile
from bidflow.domain.models import RFPDocument
from bidflow.parsing.pdf_parser import PDFParser

try:
    from bidflow.parsing.hwp_parser import HWPParser
except ImportError:
    HWPParser = None

class IngestService:
    def __init__(self):
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
        모든 블로킹 I/O는 run_in_executor로 실행하여 이벤트 루프 차단을 방지합니다.
        """
        # 1. 임시 파일로 저장
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # 2. 파일 파싱 (확장자 분기) — executor에서 실행
            loop = asyncio.get_running_loop()

            if suffix == '.pdf':
                chunks = await loop.run_in_executor(
                    None, self.pdf_parser.parse, tmp_path
                )
                tables = await loop.run_in_executor(
                    None, self.pdf_parser.extract_tables, tmp_path
                )

                doc_hash = self._calculate_hash(tmp_path)
                doc = RFPDocument(
                    id=doc_hash,
                    filename=file.filename,
                    file_path=tmp_path,
                    doc_hash=doc_hash,
                    chunks=chunks,
                    tables=tables,
                    status="READY"
                )
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

            # 3. 저장 및 벡터 인덱싱 — executor에서 실행 (이벤트 루프 차단 방지)
            def _save_and_ingest():
                from bidflow.ingest.storage import DocumentStore, VectorStoreManager, StorageRegistry
                registry = StorageRegistry()
                store = DocumentStore(user_id=user_id, registry=registry)
                store.save_document(doc)

                try:
                    vector_manager = VectorStoreManager(user_id=user_id, registry=registry)
                    vector_manager.ingest_document(doc)
                except Exception as e:
                    print(f"[WARNING] VectorDB ingestion failed: {e}")

            await loop.run_in_executor(None, _save_and_ingest)

            return doc
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
