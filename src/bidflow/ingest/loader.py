import hashlib
import os
import shutil
from typing import Optional
from bidflow.domain.models import RFPDocument, ParsedChunk
from bidflow.parsing.pdf_parser import PDFParser
from bidflow.parsing.hwp_parser import HWPParser
from bidflow.parsing.docx_parser import DOCXParser
from bidflow.parsing.hwpx_parser import HWPXParser
from bidflow.ingest.storage import DocumentStore, VectorStoreManager
from bidflow.security.rails.input_rail import InputRail, SecurityException

class RFPLoader:
    """
    파일 업로드 -> 파싱 -> 저장 -> 벡터DB 적재를 담당하는 파사드
    """
    def __init__(self):
        from bidflow.core.config import get_config
        self.config = get_config("dev") # Default to dev for MVP
        self.doc_store = DocumentStore()
        self.vec_manager = VectorStoreManager()
        self.pdf_parser = PDFParser()
        self.hwp_parser = HWPParser()
        self.docx_parser = DOCXParser()
        self.hwpx_parser = HWPXParser()
        self.input_rail = InputRail()

    def process_file(self, file_obj, filename: str, chunk_size: int = None, chunk_overlap: int = None, table_strategy: str = None) -> str:
        """
        Streamlit UploadedFile 객체를 받아 처리합니다.
        반환값: 생성된 doc_hash
        """
        # Config Defaults
        chunk_size = chunk_size or self.config.parsing.chunk_size or 500
        chunk_overlap = chunk_overlap or self.config.parsing.chunk_overlap or 50
        table_strategy = table_strategy or self.config.parsing.table_strategy or "text"

        # 1. 파일 임시 저장 & 해시 계산
        temp_path = os.path.join("data/raw", filename)
        os.makedirs("data/raw", exist_ok=True)
        
        content = file_obj.read()
        doc_hash = hashlib.md5(content).hexdigest()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 2. 파싱 (확장자 분기)
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            # table_strategy 전달
            chunks = self.pdf_parser.parse(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, table_strategy=table_strategy)
            tables = self.pdf_parser.extract_tables(temp_path)
        elif ext == ".hwp":
            # [Security] HWP Deep Scan (Forensic)
            print("🔒 Performing Deep Scan on HWP structure...")
            if self.hwp_parser.deep_scan(temp_path, self.input_rail.patterns):
                raise SecurityException("Banned pattern detected in HWP hidden stream (Deep Scan)")

            chunks = self.hwp_parser.parse(temp_path)
            tables = [] # HWP 표 미지원
        elif ext == ".docx":
            chunks = self.docx_parser.parse(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            tables = self.docx_parser.extract_tables(temp_path)
        elif ext == ".hwpx":
            chunks = self.hwpx_parser.parse(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            tables = self.hwpx_parser.extract_tables(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        # [Security] 인젝션 전체 스캔 (Ingest 단계 방어)
        print(f"🔒 Scanning {len(chunks)} chunks for Prompt Injection...")
        for i, chunk in enumerate(chunks):
            # Debug: 스캔 중인 텍스트 확인
            print(f"[Chunk {i}] Preview: {chunk.text[:50]}...")
            self.input_rail.check(chunk.text)
            
        # 3. RFPDocument 생성
        doc = RFPDocument(
            id=doc_hash,
            filename=filename,
            file_path=temp_path,
            doc_hash=doc_hash,
            chunks=chunks,
            tables=tables,
            status="READY"
        )
        
        # 4. 저장 (JSON + Chroma)
        self.doc_store.save_document(doc)
        self.vec_manager.ingest_document(doc)
        
        print(f"Processed {filename} -> {doc_hash} (Chunks: {len(chunks)})")
        return doc_hash
