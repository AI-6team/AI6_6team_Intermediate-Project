import concurrent.futures
import hashlib
import os
import re
from typing import Optional
from bidflow.domain.models import RFPDocument
from bidflow.parsing.pdf_parser import PDFParser
from bidflow.parsing.hwp_parser import HWPParser
from bidflow.parsing.docx_parser import DOCXParser
from bidflow.parsing.hwpx_parser import HWPXParser
from bidflow.ingest.storage import DocumentStore, VectorStoreManager, StorageRegistry
from bidflow.security.rails.input_rail import InputRail, SecurityException
from bidflow.security.pii_filter import PIIFilter

class RFPLoader:
    """
    파일 업로드 -> 파싱 -> 저장 -> 벡터DB 적재를 담당하는 파사드
    """
    def __init__(self, user_id: str = "global"):
        from bidflow.core.config import get_config

        self.config = get_config("dev") # Default to dev for MVP
        self.user_id = user_id or "global"
        self.default_tenant_id = self.user_id if self.user_id != "global" else "default"
        self.default_group_id = "general"

        registry = StorageRegistry(self.config)
        self.doc_store = DocumentStore(user_id=self.user_id, registry=registry)
        self.vec_manager = VectorStoreManager(user_id=self.user_id, registry=registry)
        self.pdf_parser = PDFParser()
        self.hwp_parser = HWPParser()
        self.docx_parser = DOCXParser()
        self.hwpx_parser = HWPXParser()
        self.input_rail = InputRail()
        self.pii_filter = PIIFilter()

    def process_file(
        self,
        file_obj,
        filename: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        table_strategy: str = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        max_file_size_mb: int = 50,
        parsing_timeout: int = 300,
        single_document_policy: bool = True,
    ) -> str:
        """
        Streamlit UploadedFile 객체를 받아 처리합니다.
        반환값: 생성된 doc_hash
        """
        effective_tenant_id = tenant_id or self.default_tenant_id
        effective_user_id = user_id or (self.user_id if self.user_id != "global" else "system")
        effective_group_id = group_id or self.default_group_id

        # [Policy] 단일 문서 모드: 새 업로드 전 기존 테넌트 데이터 제거
        if single_document_policy:
            self.purge_tenant(effective_tenant_id)

        filename = os.path.basename(filename)
        ext = os.path.splitext(filename)[1].lower()

        # [Security] 실행 파일/스크립트 차단
        blocked_exts = {".exe", ".sh", ".bat", ".cmd", ".js", ".vbs", ".py", ".php", ".jsp", ".asp", ".dll", ".bin"}
        if ext in blocked_exts:
            raise ValueError(f"보안 정책 위반: 실행 파일 또는 스크립트({ext})는 업로드할 수 없습니다.")

        supported_exts = {".pdf", ".hwp", ".docx", ".hwpx"}
        if ext not in supported_exts:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}. (지원 형식: {', '.join(sorted(supported_exts))})")

        # Config Defaults
        chunk_size = chunk_size or self.config.parsing.chunk_size or 500
        chunk_overlap = chunk_overlap or self.config.parsing.chunk_overlap or 50
        table_strategy = table_strategy or self.config.parsing.table_strategy or "text"

        # [Security] 파일 크기 제한
        limit_bytes = max_file_size_mb * 1024 * 1024
        if hasattr(file_obj, "size") and file_obj.size > limit_bytes:
            raise ValueError(f"File size ({file_obj.size} bytes) exceeds limit of {max_file_size_mb} MB")

        # 1. 파일 임시 저장 & 해시 계산
        temp_dir = os.path.join("data", effective_tenant_id, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        content = file_obj.read()
        if len(content) > limit_bytes:
            raise ValueError(f"File size ({len(content)} bytes) exceeds limit of {max_file_size_mb} MB")

        # [Security] Magic Number Check (확장자 위변조 탐지)
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

        # 2. 파싱 (확장자 분기 + 타임아웃)
        def _execute_parser():
            if ext == ".pdf":
                parsed_chunks = self.pdf_parser.parse(
                    temp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    table_strategy=table_strategy,
                )
                parsed_tables = self.pdf_parser.extract_tables(temp_path)
                return parsed_chunks, parsed_tables
            if ext == ".hwp":
                print("🔒 Performing Deep Scan on HWP structure...")
                if self.hwp_parser.deep_scan(temp_path, self.input_rail.patterns):
                    raise SecurityException("Banned pattern detected in HWP hidden stream (Deep Scan)")
                return self.hwp_parser.parse(temp_path), []
            if ext == ".docx":
                parsed_chunks = self.docx_parser.parse(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                parsed_tables = self.docx_parser.extract_tables(temp_path)
                return parsed_chunks, parsed_tables
            if ext == ".hwpx":
                parsed_chunks = self.hwpx_parser.parse(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                parsed_tables = self.hwpx_parser.extract_tables(temp_path)
                return parsed_chunks, parsed_tables
            return [], []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_parser)
                chunks, tables = future.result(timeout=parsing_timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"보안 경고: 파일 파싱 시간이 초과되었습니다 ({parsing_timeout}초). 복잡하거나 악의적인 파일일 수 있습니다."
            ) from exc

        # [Security] 텍스트 정제 + PII 마스킹 + 유효 길이 필터
        valid_chunks = []
        for chunk in chunks:
            cleaned_text = self._sanitize_text(chunk.text)
            masked_text = self._mask_pii(cleaned_text)
            if cleaned_text != masked_text:
                print(f"🔒 [PII Masking] Detected & Masked personal info in Chunk {chunk.chunk_id} (Page {chunk.page_no})")
            chunk.text = masked_text
            if len(chunk.text.strip()) >= 10:
                valid_chunks.append(chunk)
        chunks = valid_chunks

        if not chunks:
            raise ValueError(f"문서 파싱 실패: 유효한 텍스트를 추출할 수 없습니다. (파일명: {filename})")

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
        self.doc_store.save_document(doc, tenant_id=effective_tenant_id)
        self.vec_manager.ingest_document(
            doc,
            tenant_id=effective_tenant_id,
            user_id=effective_user_id,
            group_id=effective_group_id,
        )

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
        바이너리/깨진 문자 노이즈를 정리합니다.
        허용 범위: 한글, 영문, 숫자, 공통 특수문자, CJK 기호 일부
        """
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
        pattern = (
            r"[^\u0009\u000A\u0020-\u007E\uAC00-\uD7A3\u00A0-\u00FF"
            r"\u1100-\u11FF\u3130-\u318F\u3000-\u303F\u2000-\u20CF"
            r"\u2100-\u21FF\u2500-\u257F\uFF00-\uFFEF]+"
        )
        return re.sub(pattern, "", text)

    def _mask_pii(self, text: str) -> str:
        """
        중앙화된 PIIFilter를 통해 텍스트 내 PII를 마스킹합니다.
        """
        return self.pii_filter.sanitize(text)
