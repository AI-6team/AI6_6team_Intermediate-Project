"""HWPX (ZIP-based XML) document parser.

Ported from 김보윤's RFP_RAG_project extractors.py.
HWPX is the newer HWP format stored as a ZIP archive with XML sections.
"""
import os
import zipfile
from typing import List
from xml.etree import ElementTree as ET

from bidflow.domain.models import ParsedChunk, ParsedTable


class HWPXParser:
    """HWPX 텍스트 추출 파서 (ZIP+XML 기반)"""

    def parse(
        self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> List[ParsedChunk]:
        text_content = self._extract_text(file_path)
        if not text_content:
            print("[HWPXParser] No text extracted")
            return []

        from bidflow.parsing.preprocessor import TextPreprocessor
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        preprocessor = TextPreprocessor()
        normalized = preprocessor.normalize(text_content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        split_texts = splitter.split_text(normalized)

        chunks = []
        for i, text in enumerate(split_texts):
            chunks.append(
                ParsedChunk(
                    chunk_id=f"hwpx_chunk_{i}",
                    text=text,
                    page_no=1,
                    metadata={
                        "source": "hwpx",
                        "filename": os.path.basename(file_path),
                        "file_path": file_path,
                        "chunk_index": i,
                    },
                )
            )
        return chunks

    def extract_tables(self, file_path: str) -> List[ParsedTable]:
        # HWPX XML에서 테이블 추출은 복잡하므로 MVP에서는 미지원
        return []

    def _extract_text(self, file_path: str) -> str:
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                text_parts: list[str] = []
                section_files = sorted(
                    n
                    for n in zf.namelist()
                    if n.startswith("Contents/")
                    and n.endswith(".xml")
                    and "section" in n.lower()
                )

                for name in section_files:
                    tree = ET.parse(zf.open(name))
                    root = tree.getroot()
                    for elem in root.iter():
                        if elem.tag.endswith("}t") or elem.tag == "t":
                            if elem.text and elem.text.strip():
                                text_parts.append(elem.text.strip())

                return "\n".join(text_parts)
        except zipfile.BadZipFile as e:
            print(f"[HWPXParser] Bad ZIP format: {e}")
            return ""
        except Exception as e:
            print(f"[HWPXParser] Extraction failed: {e}")
            return ""
