"""
Table-aware Chunker — Phase CE

텍스트와 테이블을 독립적으로 chunking하여 LangChain Document로 변환.
테이블은 col_path 직렬화를 적용하여 검색 정밀도를 높임.
"""
import os
import re
from typing import List, Dict, Literal
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from bidflow.parsing.hwp_html_parser import HWPHtmlParser, TableBlock
from bidflow.parsing.preprocessor import TextPreprocessor


TableMode = Literal["flat", "colpath"]


class TableAwareChunker:
    """
    HWP 문서를 텍스트 청크 + 테이블 청크로 분리 생성

    Modes:
        - V1 (html_basic): 전체 텍스트를 하나로 추출, flat chunking
        - V2 (html_table): 텍스트/테이블 분리, col_path 직렬화
        - V3 (html_full): V2 + col_path + 테이블 개별 문서
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        table_mode: TableMode = "colpath",
        separate_tables: bool = True,
        max_table_chunk_size: int = 1500,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_mode = table_mode
        self.separate_tables = separate_tables
        self.max_table_chunk_size = max_table_chunk_size
        self.parser = HWPHtmlParser()
        self.preprocessor = TextPreprocessor()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_v1_basic(self, file_path: str) -> List[Document]:
        """V1: hwp5html 전체 텍스트 → flat chunking (hwp5txt 대체)"""
        text = self.parser.extract_text_basic(file_path)
        if not text:
            return []

        text = self.preprocessor.normalize(text)
        chunks = self.splitter.split_text(text)

        filename = os.path.basename(file_path)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": "hwp_html",
                    "mode": "v1_basic",
                    "filename": filename,
                    "chunk_type": "text",
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def chunk_v2_table(self, file_path: str) -> List[Document]:
        """V2: 텍스트/테이블 분리 + col_path 직렬화"""
        non_table_text, table_blocks = self.parser.extract_with_tables(file_path)
        if not non_table_text and not table_blocks:
            return []

        filename = os.path.basename(file_path)
        docs = []

        # 1) 비테이블 텍스트 chunking
        if non_table_text:
            normalized = self.preprocessor.normalize(non_table_text)
            text_chunks = self.splitter.split_text(normalized)
            for i, chunk in enumerate(text_chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "hwp_html",
                        "mode": "v2_table",
                        "filename": filename,
                        "chunk_type": "text",
                        "chunk_index": i,
                    },
                ))

        # 2) 테이블 chunking
        table_docs = self._chunk_tables(table_blocks, filename, "v2_table")
        docs.extend(table_docs)

        return docs

    def chunk_v3_full(self, file_path: str) -> List[Document]:
        """V3: V2 + 테이블 개별 문서 + 메타데이터 강화"""
        non_table_text, table_blocks = self.parser.extract_with_tables(file_path)
        if not non_table_text and not table_blocks:
            return []

        filename = os.path.basename(file_path)
        docs = []

        # 1) 비테이블 텍스트 chunking
        if non_table_text:
            normalized = self.preprocessor.normalize(non_table_text)
            text_chunks = self.splitter.split_text(normalized)
            for i, chunk in enumerate(text_chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "hwp_html",
                        "mode": "v3_full",
                        "filename": filename,
                        "chunk_type": "text",
                        "chunk_index": i,
                    },
                ))

        # 2) 테이블: col_path 직렬화 + 개별 문서
        table_docs = self._chunk_tables(table_blocks, filename, "v3_full")
        docs.extend(table_docs)

        return docs

    def chunk_v4_hybrid(self, file_path: str) -> List[Document]:
        """
        V4 (hybrid): hwp5txt 텍스트 + hwp5html 테이블-only 합치기

        hwp5txt가 텍스트에서 우수하고, hwp5html이 테이블에서 우수한 점을 결합.
        - 텍스트 청크: hwp5txt로 추출 → self.splitter(chunk_size 반영)로 분할
        - 테이블 청크: hwp5html로 추출 (테이블 구조 보존, flat 직렬화)
        """
        import subprocess

        filename = os.path.basename(file_path)
        docs = []

        # 1) hwp5txt로 raw 텍스트 추출 후 self.splitter로 분할 (chunk_size 반영)
        try:
            result = subprocess.run(
                ["hwp5txt", file_path],
                capture_output=True, text=True, timeout=60, encoding='utf-8'
            )
            raw_text = result.stdout if result.returncode == 0 else ""
        except Exception:
            raw_text = ""

        if raw_text.strip():
            print(f"[HWPParser] hwp5txt succeeded, extracted {len(raw_text)} chars")
            normalized = self.preprocessor.normalize(raw_text)
            text_splits = self.splitter.split_text(normalized)
            for i, text in enumerate(text_splits):
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": "hwp5txt",
                        "filename": filename,
                        "file_path": file_path,
                        "mode": "v4_hybrid",
                        "chunk_type": "text",
                        "chunk_index": i,
                    },
                ))

        # 2) hwp5html로 테이블-only 청크 생성
        _, table_blocks = self.parser.extract_with_tables(file_path)
        if table_blocks:
            table_docs = self._chunk_tables(table_blocks, filename, "v4_hybrid")
            docs.extend(table_docs)

        return docs

    def chunk_v5_hybrid_smart(self, file_path: str) -> List[Document]:
        """
        V5 (hybrid_smart): V4 + 테이블 품질 필터

        V4와 동일하게 hwp5txt 텍스트 + hwp5html 테이블을 사용하되,
        레이아웃/서식 테이블을 필터링하여 노이즈 제거.
        """
        from bidflow.parsing.hwp_parser import HWPParser

        filename = os.path.basename(file_path)
        docs = []

        # 1) hwp5txt로 텍스트 청크 생성 (V0 방식)
        hwp_parser = HWPParser()
        text_chunks = hwp_parser.parse(file_path)
        for i, chunk in enumerate(text_chunks):
            docs.append(Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "source": "hwp5txt",
                    "mode": "v5_hybrid_smart",
                    "chunk_type": "text",
                    "chunk_index": i,
                },
            ))

        # 2) hwp5html 테이블 추출 + 품질 필터
        _, table_blocks = self.parser.extract_with_tables(file_path)
        if table_blocks:
            from bidflow.parsing.hwp_html_parser import HWPHtmlParser
            filtered = [b for b in table_blocks if HWPHtmlParser.is_data_table(b)]
            if filtered:
                table_docs = self._chunk_tables(filtered, filename, "v5_hybrid_smart")
                docs.extend(table_docs)

        return docs

    def _chunk_tables(
        self, table_blocks: List[TableBlock], filename: str, mode: str
    ) -> List[Document]:
        """테이블 블록들을 Document 리스트로 변환"""
        docs = []
        table_chunk_idx = 0

        for block in table_blocks:
            # 직렬화 방식 선택
            if self.table_mode == "colpath" and block.col_paths:
                table_text = block.to_text_colpath()
            else:
                table_text = block.to_text_flat()

            if not table_text.strip():
                continue

            table_text = self.preprocessor.normalize(table_text)

            # 큰 테이블은 분할
            if len(table_text) > self.max_table_chunk_size:
                sub_chunks = self._split_table_text(table_text)
            else:
                sub_chunks = [table_text]

            for sub_idx, chunk in enumerate(sub_chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "hwp_html",
                        "mode": mode,
                        "filename": filename,
                        "chunk_type": "table",
                        "table_idx": block.table_idx,
                        "table_caption": block.caption[:50] if block.caption else "",
                        "table_rows": block.n_rows,
                        "table_cols": block.n_cols,
                        "has_colpath": bool(block.col_paths),
                        "chunk_index": table_chunk_idx,
                        "sub_chunk": sub_idx if len(sub_chunks) > 1 else None,
                    },
                ))
                table_chunk_idx += 1

        return docs

    def _split_table_text(self, text: str) -> List[str]:
        """큰 테이블 텍스트를 행 단위로 분할"""
        lines = text.split('\n')
        chunks = []
        current = []
        current_len = 0
        header = ""

        # 첫 줄이 [표] 로 시작하면 header로 보존
        if lines and lines[0].startswith('[표]'):
            header = lines[0]
            lines = lines[1:]

        for line in lines:
            line_len = len(line)
            if current_len + line_len > self.max_table_chunk_size and current:
                chunk_text = (header + "\n" if header else "") + "\n".join(current)
                chunks.append(chunk_text)
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len + 1

        if current:
            chunk_text = (header + "\n" if header else "") + "\n".join(current)
            chunks.append(chunk_text)

        return chunks
