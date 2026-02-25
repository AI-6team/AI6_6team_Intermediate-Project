"""Structure-Aware Retrieval: TOC 감지 + Chapter Prefix 주입.

평가 스크립트(scripts/run_exp19_phase_d_eval.py L968-1042)에서 추출.
EXP19 D7에서 검증: Q23(+60pp), Q24(+53pp) 개선.
"""
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document


def detect_toc_text(vdb) -> Optional[str]:
    """VectorDB에서 목차(TOC) 텍스트를 감지하여 반환.

    청크 내용에서 장번호 패턴을 스캔하고, 스코어링으로 최적 목차를 선택합니다.

    Args:
        vdb: ChromaDB 인스턴스 (vector_db 속성)

    Returns:
        목차 텍스트 또는 None
    """
    result = vdb.get(include=["documents", "metadatas"])
    if not result or not result["documents"]:
        return None

    candidates = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        chunk_idx = meta.get("chunk_index", 999)
        chunk_type = meta.get("chunk_type", "text")

        ch_count = len(re.findall(r"\d+\.\s+[가-힣]", text))
        sec_count = len(re.findall(r"[가-하]\.\s+[가-힣]", text))

        if ch_count >= 3:
            score = ch_count * 2 + sec_count
            if chunk_type == "table":
                score += 10
            if chunk_idx <= 10:
                score += 5
            candidates.append((score, text))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def build_chunk_chapter_map(vdb) -> Dict[int, str]:
    """순차 스캔으로 chunk_index -> 장 제목 매핑 생성.

    Args:
        vdb: ChromaDB 인스턴스 (vector_db 속성)

    Returns:
        {chunk_index: "N. 장제목"} 딕셔너리
    """
    result = vdb.get(include=["documents", "metadatas"])
    if not result or not result["documents"]:
        return {}

    chunks = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        chunks.append((meta.get("chunk_index", 0), text))
    chunks.sort()

    chapter_map = {}
    current_chapter = None
    header_re = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+([가-힣][가-힣\s\(\)·\-/]+?)(?:\s*\n|$)")

    for idx, text in chunks:
        matches = list(header_re.finditer(text[:300]))
        if matches:
            num = int(matches[0].group(1))
            title = matches[0].group(2).strip()[:30]
            if 1 <= num <= 20 and len(title) <= 30:
                current_chapter = f"{num}. {title}"

        if current_chapter:
            chapter_map[idx] = current_chapter

    return chapter_map


def build_enhanced_context(
    docs: List[Document],
    toc_text: Optional[str] = None,
    chunk_chapter_map: Optional[Dict[int, str]] = None,
) -> str:
    """목차 선행 + 청크별 [N. 장제목] 접두사로 컨텍스트 강화.

    Args:
        docs: 검색된 Document 리스트
        toc_text: detect_toc_text()로 얻은 목차 텍스트
        chunk_chapter_map: build_chunk_chapter_map()으로 얻은 매핑

    Returns:
        강화된 컨텍스트 문자열
    """
    parts = []

    if toc_text:
        parts.append(f"[문서 목차 (Table of Contents)]\n{toc_text}")
        parts.append("─" * 40)

    for doc in docs:
        chunk_text = doc.page_content
        if chunk_chapter_map:
            chunk_idx = doc.metadata.get("chunk_index")
            chapter = chunk_chapter_map.get(chunk_idx) if chunk_idx is not None else None
            if chapter:
                chunk_text = f"[{chapter}]\n{chunk_text}"
        parts.append(chunk_text)

    return "\n\n".join(parts)
