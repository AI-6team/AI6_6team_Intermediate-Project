import os
import pdfplumber
from typing import List
from bidflow.domain.models import ParsedChunk, ParsedTable

class PDFParser:
    """
    pdfplumber를 사용한 기본 PDF 파서
    """
    def parse(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, table_strategy: str = "text") -> List[ParsedChunk]:
        """
        table_strategy: 
        - "text": 기본 텍스트 추출 (pdfplumber default)
        - "layout": 레이아웃 보존 (pdfplumber layout=True, 표 행 보존에 유리)
        - "markdown": (Future) 표를 Markdown으로 변환하여 별도 청킹
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            page_map = [] # (start_idx, end_idx, page_no)

            # 1. 문서 전체 텍스트 병합 (Cross-page split을 위해)
            for i, page in enumerate(pdf.pages):
                # 전략에 따른 추출 방식 변경
                if table_strategy == "layout":
                    page_text = page.extract_text(layout=True)
                else:
                    page_text = page.extract_text()
                    
                if page_text:
                    start = len(full_text)
                    full_text += page_text + "\n\n"
                    end = len(full_text)
                    page_map.append((start, end, i + 1))
            
            if not full_text:
                return []

            # 2. Text Splitter 적용
            from bidflow.parsing.preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()
            
            # [Conservative Normalization] 분할 전 정규화
            normalized_text = preprocessor.normalize(full_text)
            
            split_texts = text_splitter.split_text(normalized_text)
            
            # 3. Chunk Mapping
            for idx, text in enumerate(split_texts):
                chunks.append(ParsedChunk(
                    chunk_id=f"chunk_{idx}",
                    text=text,
                    page_no=1, # 추후 page_map 기반 매핑 로직 고도화 필요
                    metadata={
                        "source": "pdf",
                        "filename": os.path.basename(file_path),
                        "file_path": file_path,
                        "chunk_size": chunk_size,
                        "table_strategy": table_strategy
                    }
                ))
        return chunks

    def extract_tables(self, file_path: str) -> List[ParsedTable]:
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                for j, table_data in enumerate(extracted):
                    # None 값 제거 및 문자열 변환
                    clean_rows = []
                    for row in table_data:
                        clean_rows.append([str(cell) if cell is not None else "" for cell in row])
                    
                    if clean_rows:
                        tables.append(ParsedTable(
                            table_id=f"p{i+1}_t{j+1}",
                            page_no=i+1,
                            rows=clean_rows,
                            metadata={
                                "source": "pdf",
                                "filename": os.path.basename(file_path),
                                "page": i + 1,
                            }
                        ))
        return tables
