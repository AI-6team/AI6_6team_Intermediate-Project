import os
import asyncio
from functools import partial
from fastapi import UploadFile
from bidflow.domain.models import RFPDocument
from bidflow.ingest.loader import RFPLoader
from bidflow.ingest.storage import DocumentStore

class IngestService:
    def __init__(self, upload_dir: str = "data/raw"):
        # RFPLoader가 내부적으로 저장소를 관리하므로 upload_dir은 사용하지 않음
        pass

    async def process_upload(self, file: UploadFile, user_id: str) -> RFPDocument:
        """
        RFPLoader를 사용하여 파일을 처리합니다. (PDF, HWP 등 지원)
        """
        loader = RFPLoader(user_id=user_id)
        loop = asyncio.get_running_loop()
        
        # RFPLoader.process_file은 동기 함수이므로 executor에서 실행
        doc_hash = await loop.run_in_executor(
            None,
            partial(loader.process_file, file.file, file.filename, user_id=user_id)
        )
        
        store = DocumentStore(user_id=user_id)
        return store.load_document(doc_hash)
