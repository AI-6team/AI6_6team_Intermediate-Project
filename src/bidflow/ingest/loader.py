import hashlib
import re
import os
import shutil
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

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

    def process_file(self, file_obj, filename: str, chunk_size: int = None, chunk_overlap: int = None, table_strategy: str = None, tenant_id: str = "default", user_id: str = "system", group_id: str = "general", max_file_size_mb: int = 50) -> str:
        """
        Streamlit UploadedFile 객체를 받아 처리합니다.
        반환값: 생성된 doc_hash
        """
        # 0. 확장자 체크 (Fail Fast)
        ext = os.path.splitext(filename)[1].lower()
        
        # [Security] 실행 파일 및 스크립트 차단 (Blocklist)
        blocked_exts = [".exe", ".sh", ".bat", ".cmd", ".js", ".vbs", ".py", ".php", ".jsp", ".asp", ".dll", ".bin"]
        if ext in blocked_exts:
             raise ValueError(f"보안 정책 위반: 실행 파일 또는 스크립트({ext})는 업로드할 수 없습니다.")

        supported_exts = [".pdf", ".hwp", ".docx", ".hwpx"]
        if ext not in supported_exts:
             raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}. (지원 형식: {', '.join(supported_exts)})")

        # Config Defaults
        chunk_size = chunk_size or self.config.parsing.chunk_size or 500
        chunk_overlap = chunk_overlap or self.config.parsing.chunk_overlap or 50
        table_strategy = table_strategy or self.config.parsing.table_strategy or "text"

        # 0. 파일 크기 제한 확인
        limit_bytes = max_file_size_mb * 1024 * 1024
        if hasattr(file_obj, "size") and file_obj.size > limit_bytes:
             raise ValueError(f"File size ({file_obj.size} bytes) exceeds limit of {max_file_size_mb} MB")

        # 1. 파일 임시 저장 & 해시 계산
        # Tenant Isolation: Use tenant specific temp path
        temp_dir = os.path.join("data", tenant_id, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        
        content = file_obj.read()
        if len(content) > limit_bytes:
             raise ValueError(f"File size ({len(content)} bytes) exceeds limit of {max_file_size_mb} MB")

        # [Security] Magic Number Check (Extension Spoofing Detection)
        magic_signatures = {
            ".pdf": [b"%PDF"],
            ".hwp": [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"], # OLE Compound File
            ".docx": [b"\x50\x4B\x03\x04"], # ZIP Header
            ".hwpx": [b"\x50\x4B\x03\x04"], # ZIP Header
        }
        
        if ext in magic_signatures:
            is_valid_signature = any(content.startswith(sig) for sig in magic_signatures[ext])
            
            if not is_valid_signature:
                raise ValueError(f"보안 경고: 파일 내용이 확장자({ext})와 일치하지 않습니다. (Magic Number Mismatch)")

        doc_hash = hashlib.md5(content).hexdigest()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 2. 파싱 (확장자 분기)
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
            
        # [Sanitization] 텍스트 정제 (Mojibake 및 바이너리 노이즈 제거)
        valid_chunks = []
        for chunk in chunks:
            cleaned_text = self._sanitize_text(chunk.text)
            masked_text = self._mask_pii(cleaned_text)
            
            if cleaned_text != masked_text:
                print(f"🔒 [PII Masking] Detected & Masked personal info in Chunk {chunk.chunk_id} (Page {chunk.page_no})")
            
            chunk.text = masked_text
            
            # [Validation] 너무 짧거나 의미 없는 청크 삭제 (최소 10자)
            if len(chunk.text.strip()) >= 10:
                valid_chunks.append(chunk)
        chunks = valid_chunks
            
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
        self.doc_store.save_document(doc, tenant_id=tenant_id)
        self.vec_manager.ingest_document(doc, tenant_id=tenant_id, user_id=user_id, group_id=group_id)
        
        print(f"Processed {filename} -> {doc_hash} (Chunks: {len(chunks)})")
        return doc_hash

    def purge_tenant(self, tenant_id: str):
        """
        특정 테넌트의 모든 데이터(파일 + 벡터)를 삭제합니다.
        """
        print(f"Purging data for tenant: {tenant_id}...")
        self.doc_store.purge_tenant_data(tenant_id)
        self.vec_manager.delete_tenant_data(tenant_id)
        print(f"Purge complete for tenant: {tenant_id}")

    def _sanitize_text(self, text: str) -> str:
        """
        바이너리 데이터가 텍스트로 잘못 파싱된 경우 발생하는 노이즈(Mojibake)를 제거합니다.
        허용 범위: 한글, 영문, 숫자, 공통 특수문자, 한자
        """
        # 1. 제어 문자 제거 (줄바꿈, 탭 제외)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
        
        # 2. 허용된 유니코드 범위 외 문자 제거
        # \u0020-\u007E: ASCII (영문, 숫자, 기본 특수문자)
        # \uAC00-\uD7A3: 한글 소리마디
        # \u00A0-\u00FF: Latin-1 Supplement (통화기호, 분수 등)
        # \u1100-\u11FF: 한글 자모
        # \u3130-\u318F: 한글 호환 자모
        # \u4E00-\u9FFF: CJK 통합 한자 (RFP 특성상 유지)
        # \u3000-\u303F: CJK 기호 및 구두점
        # \u2000-\u20CF: 일반 구두점, 통화기호
        # \u2100-\u21FF: 문자형 기호, 화살표
        # \u2500-\u257F: 상자 그리기 (표 등)
        # \uFF00-\uFFEF: 전각/반각 기호
        pattern = r"[^\u0009\u000A\u0020-\u007E\uAC00-\uD7A3\u00A0-\u00FF\u1100-\u11FF\u3130-\u318F\u4E00-\u9FFF\u3000-\u303F\u2000-\u20CF\u2100-\u21FF\u2500-\u257F\uFF00-\uFFEF]+"
        
        cleaned = re.sub(pattern, "", text)
        return cleaned

    def _mask_pii(self, text: str) -> str:
        """
        텍스트에서 개인정보(주민등록번호, 전화번호, 이메일)를 마스킹합니다.
        """
        # 주민등록번호: 6자리-7자리 -> 뒷자리 마스킹
        text = re.sub(r'(\d{6})[-]\d{7}', r'\1-*******', text)
        
        # 휴대전화번호: 010-xxxx-xxxx -> 뒷자리 마스킹
        text = re.sub(r'(01[016789][-.\s]?\d{3,4})[-.\s]?\d{4}', r'\1-****', text)
        
        # 이메일: id@domain -> id@****
        text = re.sub(r'([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', r'\1@****', text)
        
        return text
