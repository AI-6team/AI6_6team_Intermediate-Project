"""Test TableAwareChunker on all 5 documents."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bidflow.parsing.table_chunker import TableAwareChunker

DOCS = {
    "doc_A": "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
    "doc_B": "data/raw/files/한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
    "doc_C": "data/raw/files/국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
    "doc_D": "data/raw/files/한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
    "doc_E": "data/raw/files/재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
}

chunker_v1 = TableAwareChunker(chunk_size=500, chunk_overlap=50, table_mode="flat")
chunker_v2 = TableAwareChunker(chunk_size=500, chunk_overlap=50, table_mode="colpath")

for doc_key, path in DOCS.items():
    print(f"\n{'='*60}")
    print(f"[{doc_key}] {os.path.basename(path)}")

    # V1
    docs_v1 = chunker_v1.chunk_v1_basic(path)
    print(f"  V1 (basic): {len(docs_v1)} chunks, total chars: {sum(len(d.page_content) for d in docs_v1):,}")

    # V2
    docs_v2 = chunker_v2.chunk_v2_table(path)
    text_chunks = [d for d in docs_v2 if d.metadata.get('chunk_type') == 'text']
    table_chunks = [d for d in docs_v2 if d.metadata.get('chunk_type') == 'table']
    print(f"  V2 (table): {len(docs_v2)} chunks (text: {len(text_chunks)}, table: {len(table_chunks)})")
    print(f"    Total chars: {sum(len(d.page_content) for d in docs_v2):,}")

    # V3
    docs_v3 = chunker_v2.chunk_v3_full(path)
    text_chunks3 = [d for d in docs_v3 if d.metadata.get('chunk_type') == 'text']
    table_chunks3 = [d for d in docs_v3 if d.metadata.get('chunk_type') == 'table']
    print(f"  V3 (full):  {len(docs_v3)} chunks (text: {len(text_chunks3)}, table: {len(table_chunks3)})")

    # Show sample col_path table chunk
    if table_chunks:
        sample = table_chunks[0]
        print(f"\n  --- Sample table chunk (V2) ---")
        print(f"  Caption: {sample.metadata.get('table_caption', '')}")
        print(f"  Has col_path: {sample.metadata.get('has_colpath', False)}")
        content_preview = sample.page_content[:200]
        print(f"  Content: {content_preview}")

print(f"\n{'='*60}")
print("Chunker test complete.")
