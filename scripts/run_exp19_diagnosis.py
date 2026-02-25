"""
EXP19 Step 1: Q1/Q7 정밀 진단 (API 비용 0 — VDB 텍스트 검색만)

Q1 (doc_A, kw_v5=0.720): "SSF", "수협은행", "미연동" 등 7개 missing 키워드가 VDB 청크에 존재하는지
Q7 (doc_D, kw_v5=0.833): "제안서 평가 기준" 텍스트가 VDB 청크에 존재하는지 + retrieval top-15에 포함되는지

실행: cd bidflow && python -X utf8 scripts/run_exp19_diagnosis.py
"""
import os, sys, re, warnings
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

EMBEDDING_SMALL = 'text-embedding-3-small'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'


def load_vdb(doc_key):
    embed = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    vdb_path = str(VDB_BASE / f'vectordb_c500_{doc_key}')
    vdb = Chroma(persist_directory=vdb_path, embedding_function=embed, collection_name='bidflow_rfp')
    return vdb


def get_all_chunks(vdb):
    result = vdb.get()
    chunks = []
    if result and result['documents']:
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            chunks.append({'text': text, 'meta': meta, 'idx': i})
    return chunks


def search_keyword_in_chunks(chunks, keyword):
    """청크에서 키워드를 포함하는 청크 인덱스와 위치 반환"""
    found = []
    kw_lower = keyword.lower()
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        if kw_lower in text_lower:
            # 키워드 주변 컨텍스트 추출
            pos = text_lower.index(kw_lower)
            start = max(0, pos - 50)
            end = min(len(chunk['text']), pos + len(keyword) + 50)
            context = chunk['text'][start:end]
            found.append({
                'idx': chunk['idx'],
                'context': context,
                'chunk_len': len(chunk['text']),
            })
    return found


def main():
    print("=" * 70)
    print("EXP19 Step 1: Q1/Q7 정밀 진단")
    print("=" * 70)

    # ================================================================
    # Q1 진단: doc_A VDB에서 missing 키워드 존재 여부
    # ================================================================
    print(f"\n{'#' * 60}")
    print("# Q1 진단: doc_A VDB 청크에서 missing 키워드 검색")
    print(f"{'#' * 60}")

    q1_missing_keywords = [
        "SSF", "회계", "수협은행", "내부", "미연동", "불필요한", "행정업무"
    ]
    # GT의 핵심 구절
    q1_key_phrases = [
        "SSF(회계)",
        "수협은행",
        "내부 시스템 간 미연동",
        "불필요한 행정업무 과다 발생",
    ]

    vdb_a = load_vdb('doc_A')
    chunks_a = get_all_chunks(vdb_a)
    print(f"\n  doc_A: {len(chunks_a)} chunks loaded")

    print(f"\n  --- 개별 키워드 검색 ---")
    for kw in q1_missing_keywords:
        found = search_keyword_in_chunks(chunks_a, kw)
        status = f"FOUND in {len(found)} chunks" if found else "NOT FOUND"
        print(f"  '{kw}': {status}")
        if found:
            for f in found[:2]:  # 최대 2개만 표시
                print(f"    chunk[{f['idx']}] ({f['chunk_len']}ch): ...{f['context']}...")

    print(f"\n  --- 핵심 구절 검색 ---")
    for phrase in q1_key_phrases:
        found = search_keyword_in_chunks(chunks_a, phrase)
        status = f"FOUND in {len(found)} chunks" if found else "NOT FOUND"
        print(f"  '{phrase}': {status}")
        if found:
            for f in found[:2]:
                print(f"    chunk[{f['idx']}] ({f['chunk_len']}ch): ...{f['context']}...")

    # ================================================================
    # Q7 진단: doc_D VDB에서 "제안서 평가 기준" 존재 여부 + retrieval 확인
    # ================================================================
    print(f"\n{'#' * 60}")
    print("# Q7 진단: doc_D VDB 청크에서 '제안서 평가 기준' 검색")
    print(f"{'#' * 60}")

    q7_keywords = [
        "제안서 평가 기준",
        "평가 기준",
        "평가기준",
        "다. 제안서 평가 기준",
        "다. 제안서 평가",
    ]

    vdb_d = load_vdb('doc_D')
    chunks_d = get_all_chunks(vdb_d)
    print(f"\n  doc_D: {len(chunks_d)} chunks loaded")

    print(f"\n  --- 키워드 검색 ---")
    for kw in q7_keywords:
        found = search_keyword_in_chunks(chunks_d, kw)
        status = f"FOUND in {len(found)} chunks" if found else "NOT FOUND"
        print(f"  '{kw}': {status}")
        if found:
            for f in found[:3]:
                print(f"    chunk[{f['idx']}] ({f['chunk_len']}ch): ...{f['context']}...")

    # Retrieval 테스트: Q7 질문으로 실제 retrieval 수행
    print(f"\n  --- Q7 Retrieval 테스트 (pool=50, top_k=15) ---")

    from bidflow.retrieval.rerank import rerank

    q7_question = "예약발매시스템 ISMP의 제안서 평가방법은 어떤 장에서 다루고 있는가?"

    # BM25 + Vector RRF merge
    all_docs = [Document(page_content=c['text'], metadata=c['meta']) for c in chunks_d]
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 100

    vector_retriever = vdb_d.as_retriever(search_kwargs={'k': 100})

    bm25_docs = bm25.invoke(q7_question)
    vector_docs = vector_retriever.invoke(q7_question)

    # RRF merge
    scores = defaultdict(float)
    doc_map = {}
    k = 60
    for rank, doc in enumerate(bm25_docs):
        scores[doc.page_content] += 0.3 * (1 / (rank + k))
        doc_map[doc.page_content] = doc
    for rank, doc in enumerate(vector_docs):
        scores[doc.page_content] += 0.7 * (1 / (rank + k))
        if doc.page_content not in doc_map:
            doc_map[doc.page_content] = doc

    sorted_c = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    pool_docs = [doc_map[c] for c in sorted_c[:50]]

    # Rerank
    reranked = rerank(q7_question, pool_docs, top_k=15, model_name='BAAI/bge-reranker-v2-m3')

    print(f"  Retrieved {len(reranked)} docs")
    context_text = '\n\n'.join([doc.page_content for doc in reranked])

    # Check if "제안서 평가 기준" is in the context
    target_phrases = ["제안서 평가 기준", "평가 기준", "다. 제안서"]
    for phrase in target_phrases:
        if phrase.lower() in context_text.lower():
            print(f"  Context에 '{phrase}' 포함: YES")
        else:
            print(f"  Context에 '{phrase}' 포함: NO")

    # 각 retrieved chunk에서 "평가 기준" 포함 여부
    print(f"\n  --- Retrieved top-15 chunks 중 '평가 기준' 포함 여부 ---")
    for i, doc in enumerate(reranked):
        has_keyword = "평가 기준" in doc.page_content or "평가기준" in doc.page_content
        marker = " *** HAS 평가기준" if has_keyword else ""
        snippet = doc.page_content[:100].replace('\n', ' ')
        print(f"  [{i+1}] {snippet}...{marker}")

    # ================================================================
    # 진단 결론
    # ================================================================
    print(f"\n{'=' * 70}")
    print("진단 결론")
    print(f"{'=' * 70}")

    # Q1 결론
    ssf_found = len(search_keyword_in_chunks(chunks_a, "SSF")) > 0
    bank_found = len(search_keyword_in_chunks(chunks_a, "수협은행")) > 0
    undoc_found = len(search_keyword_in_chunks(chunks_a, "미연동")) > 0

    if ssf_found and bank_found and undoc_found:
        print("  Q1: 키워드가 VDB에 존재 → Retrieval ranking 문제")
        print("       → retriever pool/alpha 조정 또는 GT v3 수정 필요")
    elif not ssf_found or not bank_found or not undoc_found:
        missing_from_vdb = []
        if not ssf_found: missing_from_vdb.append("SSF")
        if not bank_found: missing_from_vdb.append("수협은행")
        if not undoc_found: missing_from_vdb.append("미연동")
        print(f"  Q1: VDB에 미존재 키워드: {missing_from_vdb} → Parsing/Chunking gap")
        print("       → GT v3 수정 권장 (파싱 불가한 세부 정보 제거)")

    # Q7 결론
    eval_criteria_in_context = "평가 기준" in context_text or "평가기준" in context_text
    if eval_criteria_in_context:
        print("  Q7: '평가 기준'이 retrieved context에 포함 → Generation 문제")
        print("       → Targeted prompt로 하위 절 제목 나열 유도")
    else:
        eval_criteria_in_vdb = len(search_keyword_in_chunks(chunks_d, "평가 기준")) > 0
        if eval_criteria_in_vdb:
            print("  Q7: '평가 기준'이 VDB에는 존재하나 top-15에 미포함 → Retrieval ranking 문제")
            print("       → pool_size 증가 또는 targeted query")
        else:
            print("  Q7: '평가 기준'이 VDB에도 미존재 → Parsing gap")

    print(f"\n{'=' * 70}")
    print("EXP19 Step 1 진단 완료")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
