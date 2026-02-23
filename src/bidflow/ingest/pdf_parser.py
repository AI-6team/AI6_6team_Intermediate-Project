import fitz  # PyMuPDF
import pdfplumber
import hashlib
from typing import List, Tuple, Optional
from datetime import datetime
import os

from bidflow.domain.models import RFPDocument, ParsedChunk, ParsedTable

class PDFParser:
    """
    PDF 문서를 파싱하여 텍스트와 표를 추출합니다.
    PyMuPDF (fitz)를 사용하여 텍스트와 좌표를 추출합니다.
    pdfplumber를 사용하여 신뢰할 수 있는 표 추출을 수행합니다.
    """

    def __init__(self):
        pass

    def parse(self, file_path: str) -> RFPDocument:
        """
        PDF 파일 파싱을 위한 진입점입니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 1. 문서 해시 생성
        doc_hash = self._calculate_file_hash(file_path)
        filename = os.path.basename(file_path)
        
        # 2. 텍스트 및 표 추출
        chunks, tables = self._extract_content(file_path)

        # 3. 문서 객체 생성
        doc = RFPDocument(
            id=doc_hash, # 현재는 해시를 ID로 사용
            filename=filename,
            file_path=file_path,
            doc_hash=doc_hash,
            chunks=chunks,
            tables=tables,
            status="READY"
        )
        return doc

    def _calculate_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _extract_content(self, file_path: str) -> Tuple[List[ParsedChunk], List[ParsedTable]]:
        chunks = []
        tables = []
        
        # Open with pdfplumber for tables
        with pdfplumber.open(file_path) as pl_pdf:
            # Open with fitz for text
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                # 비즈니스 로직을 위해 1부터 시작하는 페이지 번호 사용
                display_page_num = page_num + 1
                
                # 표 우선 추출 (pdfplumber 사용)
                pl_page = pl_pdf.pages[page_num]
                extracted_tables = pl_page.extract_tables()
                
                page_tables = []
                for idx, table_data in enumerate(extracted_tables):
                    if not table_data:
                        continue
                    
                    # None 값 정리
                    cleaned_rows = [
                        [cell if cell is not None else "" for cell in row]
                        for row in table_data
                    ]
                    
                    table_obj = ParsedTable(
                        table_id=f"doc_{display_page_num}_tbl_{idx}",
                        page_no=display_page_num,
                        rows=cleaned_rows,
                        metadata={"source_tool": "pdfplumber"}
                    )
                    page_tables.append(table_obj)
                
                tables.extend(page_tables)

                # 텍스트 추출 (fitz 사용)
                # 이상적으로는 표 내부 텍스트를 제외하여 중복을 피해야 함.
                # MVP에서는 모든 텍스트 블록을 추출함. 
                # 개선점: 텍스트 추출 전 표 영역 마스킹.
                
                text_blocks = page.get_text("blocks")
                # 블록 형식: (x0, y0, x1, y1, "text", block_no, block_type)
                
                for b_idx, block in enumerate(text_blocks):
                    # block_type 0은 텍스트, 1은 이미지
                    if block[6] == 0: 
                        text_content = block[4].strip()
                        if not text_content:
                            continue
                            
                        chunk = ParsedChunk(
                            chunk_id=f"doc_{display_page_num}_blk_{b_idx}",
                            text=text_content,
                            page_no=display_page_num,
                            metadata={
                                "bbox": [block[0], block[1], block[2], block[3]],
                                "block_type": "text"
                            }
                        )
                        chunks.append(chunk)

            doc.close()
            
        return chunks, tables

# 사용 예시 (직접 실행 시)
if __name__ == "__main__":
    # 테스트용 파일이 없으면 더미 경로 지정
    test_pdf = "test_rfp.pdf"
    if not os.path.exists(test_pdf):
        print(f"테스트를 위해 {test_pdf} 경로에 PDF 파일을 제공해주세요.")
    else:
        parser = PDFParser()
        result = parser.parse(test_pdf)
        print(f"텍스트 청크 {len(result.chunks)}개와 테이블 {len(result.tables)}개를 파싱했습니다.")
        if result.tables:
            print("첫 번째 테이블 샘플:", result.tables[0].rows[:2])
