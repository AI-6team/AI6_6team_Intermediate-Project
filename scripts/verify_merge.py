"""
병합 검증 스크립트: 기존 ref_v2 baseline과 동일한 성능 유지 확인
- 30Q golden testset 사용
- ref_v2 파라미터: alpha=0.7, pool_size=50, top_k=15
- 기준: kw_v3 >= 0.896
"""
import os, sys, time, re, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not found'

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from typing import List, Any

# ── Constants ──
EMBEDDING_SMALL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-5-mini'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'
CSV_PATH = 'data/experiments/verify_merge_results.csv'

DOC_CONFIGS = {
    "doc_A": {"source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp", "doc_type": "text_only"},
    "doc_B": {"source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp", "doc_type": "table_simple"},
    "doc_C": {"source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp", "doc_type": "table_complex"},
    "doc_D": {"source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp", "doc_type": "mixed"},
    "doc_E": {"source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp", "doc_type": "hwp_representative"},
}
SOURCE_TO_KEY = {v["source_doc"]: k for k, v in DOC_CONFIGS.items()}

# ── kw_v3 metric (EXP12와 동일) ──
SYNONYM_MAP = {
    '정보전략계획': 'ismp', 'ismp 수립': 'ismp', '정보화전략계획': 'ismp',
    '통합로그인': 'sso', '단일 로그인': 'sso', '싱글사인온': 'sso',
    '간편인증': '간편인증', '간편 인증': '간편인증',
    '2차인증': '2차인증', '2차 인증': '2차인증', '추가인증': '2차인증',
    'project manager': 'pm', '사업관리자': 'pm', '사업책임자': 'pm',
    '프로젝트 매니저': 'pm', 'project leader': 'pl', '프로젝트 리더': 'pl',
    'quality assurance': 'qa', '품질관리': 'qa', '품질보증': 'qa',
    '하자보수': '하자보수', '하자 보수': '하자보수',
    '발주처': '발주기관', '발주 기관': '발주기관',
}
PARTICLES_RE = re.compile(
    r'(은|는|이|가|을|를|의|에|에서|으로|로|와|과|이며|이고|에게|한테|부터|까지|도|만|이라|인|에는|에도)$'
)
ROMAN_MAP = {'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3', 'ⅳ': '4', 'ⅴ': '5',
             'ⅵ': '6', 'ⅶ': '7', 'ⅷ': '8', 'ⅸ': '9', 'ⅹ': '10',
             'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
             'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10'}

def normalize_answer_v2(text):
    if not isinstance(text, str): return str(text).strip().lower()
    t = text.strip().lower()
    t = re.sub(r'[\u00b7\u2027\u2022\u2219]', ' ', t)
    t = re.sub(r'[\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f]', '', t)
    t = re.sub(r'[-\u2013\u2014]', ' ', t)
    t = re.sub(r'(\d),(?=\d{3})', r'\1', t)
    t = re.sub(r'(\d+)\s*(%|퍼센트|percent)', r'\1%', t)
    t = re.sub(r'(\d+)\s*원', r'\1원', t)
    t = re.sub(r'(\d+)\s*억\s*원', r'\1억원', t)
    t = re.sub(r'(\d+)\s*만\s*원', r'\1만원', t)
    t = t.replace('v.a.t', 'vat').replace('vat 포함', 'vat포함')
    for orig, norm in SYNONYM_MAP.items(): t = t.replace(orig.lower(), norm)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def normalize_answer_v3(text):
    t = normalize_answer_v2(text)
    for roman, arabic in ROMAN_MAP.items(): t = t.replace(roman.lower(), arabic)
    t = t.replace('￦', '₩')
    t = re.sub(r'([가-힣a-z0-9])\(', r'\1 (', t)
    t = re.sub(r'\)([가-힣a-z])', r') \1', t)
    words = t.split()
    cleaned = []
    for w in words:
        w = w.rstrip('.,;:!?')
        if not w: continue
        stripped = PARTICLES_RE.sub('', w)
        cleaned.append(stripped if stripped else w)
    return ' '.join(cleaned)

def keyword_accuracy_v3(answer, ground_truth):
    ans_norm = normalize_answer_v3(answer)
    gt_norm = normalize_answer_v3(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words: return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)

# ── Retriever (ref_v2 설정 그대로) ──
class VerifyRetriever(BaseRetriever):
    vector_retriever: Any = None
    bm25_retriever: Any = None
    weights: List[float] = [0.3, 0.7]  # alpha=0.7
    top_k: int = 15
    pool_size: int = 50
    use_rerank: bool = True

    def _get_relevant_documents(self, query, *, run_manager):
        search_k = self.pool_size
        try:
            self.bm25_retriever.k = search_k * 2
            bm25_docs = self.bm25_retriever.invoke(query)
        except: bm25_docs = []
        try:
            self.vector_retriever.search_kwargs['k'] = search_k * 2
            vector_docs = self.vector_retriever.invoke(query)
        except: vector_docs = []
        merged = self._rrf_merge(bm25_docs, vector_docs, k=60, limit=self.pool_size)
        if self.use_rerank and merged:
            from bidflow.retrieval.rerank import rerank
            merged = rerank(query, merged, top_k=self.top_k, model_name='BAAI/bge-reranker-v2-m3')
        return merged

    def _rrf_merge(self, list1, list2, k=60, limit=50):
        scores = defaultdict(float)
        doc_map = {}
        w_bm25, w_vec = self.weights
        for rank, doc in enumerate(list1):
            scores[doc.page_content] += w_bm25 * (1 / (rank + k))
            doc_map[doc.page_content] = doc
        for rank, doc in enumerate(list2):
            scores[doc.page_content] += w_vec * (1 / (rank + k))
            if doc.page_content not in doc_map: doc_map[doc.page_content] = doc
        sorted_c = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[c] for c in sorted_c[:limit]]


def build_retriever(vdb):
    vector_retriever = vdb.as_retriever(search_kwargs={'k': 100})
    result = vdb.get()
    all_docs = []
    if result and result['documents']:
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            all_docs.append(Document(page_content=text, metadata=meta))
    bm25 = BM25Retriever.from_documents(all_docs) if all_docs else BM25Retriever.from_documents([Document(page_content='empty')])
    bm25.k = 100
    return VerifyRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25)


PROMPT_V2 = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n'
    '문맥에는 일반 텍스트와 테이블 데이터가 포함될 수 있습니다.\n'
    '테이블에서 추출된 정보(금액, 기간, 수량, 비율 등)가 있다면 우선적으로 활용하세요.\n'
    '답변 시 원문의 사업명, 기관명, 금액, 날짜, 숫자 등을 정확히 그대로(Verbatim) 인용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)


def main():
    print(f"\n{'='*70}")
    print(f"MERGE VERIFICATION: ref_v2 baseline 재현 테스트")
    print(f"기준: kw_v3 >= 0.896 (ref_v2 baseline)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # 1. Testset 로드
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions")

    # 2. VDB 로드
    embeddings = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    doc_vdbs = {}
    for doc_key in DOC_CONFIGS:
        persist_dir = str(VDB_BASE / f'vectordb_c500_{doc_key}')
        if not os.path.exists(persist_dir):
            print(f"  WARNING: VDB not found: {persist_dir}")
            continue
        vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name='bidflow_rfp')
        doc_vdbs[doc_key] = vdb
        print(f"  {doc_key}: {vdb._collection.count()} chunks loaded")

    # 3. Retriever 빌드
    doc_retrievers = {}
    for doc_key, vdb in doc_vdbs.items():
        doc_retrievers[doc_key] = build_retriever(vdb)

    # 4. LLM + RAG 실행
    llm = ChatOpenAI(model=LLM_MODEL, temperature=1, timeout=60, max_retries=2)
    prompt = ChatPromptTemplate.from_template(PROMPT_V2)
    chain = prompt | llm | StrOutputParser()

    results = []
    errors = []
    start_time = time.time()

    for idx, row in testset.iterrows():
        question = row['question']
        ground_truth = row['ground_truth']
        source_doc = row['source_doc']
        doc_key = SOURCE_TO_KEY.get(source_doc)

        if doc_key is None or doc_key not in doc_retrievers:
            print(f"  [{idx+1}/30] SKIP: no retriever for {source_doc[:30]}...")
            continue

        retriever = doc_retrievers[doc_key]

        try:
            t0 = time.time()
            docs = retriever.invoke(question)
            retrieval_time = time.time() - t0

            context_text = '\n\n'.join([doc.page_content for doc in docs])

            t1 = time.time()
            answer = chain.invoke({'context': context_text, 'question': question})
            gen_time = time.time() - t1

            kw3 = keyword_accuracy_v3(answer, ground_truth)

            results.append({
                'doc_key': doc_key,
                'question': question,
                'ground_truth': ground_truth,
                'answer': answer,
                'kw_v3': kw3,
                'n_retrieved': len(docs),
                'retrieval_time': retrieval_time,
                'generation_time': gen_time,
            })

            status = "OK" if kw3 >= 0.8 else "LOW"
            print(f"  [{idx+1}/30] kw_v3={kw3:.3f} [{status}] doc={doc_key} t={retrieval_time+gen_time:.1f}s | {question[:40]}...")

        except Exception as e:
            errors.append({'question': question[:50], 'error': str(e)})
            results.append({
                'doc_key': doc_key, 'question': question, 'ground_truth': ground_truth,
                'answer': 'ERROR', 'kw_v3': 0.0, 'n_retrieved': 0,
                'retrieval_time': 0, 'generation_time': 0,
            })
            print(f"  [{idx+1}/30] ERROR: {e}")

        # 증분 저장
        pd.DataFrame(results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    total_time = time.time() - start_time
    df = pd.DataFrame(results)

    # 5. 결과 분석
    print(f"\n{'='*70}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*70}")

    overall_kw3 = df['kw_v3'].mean()
    print(f"\n  Overall kw_v3: {overall_kw3:.4f}")
    print(f"  Baseline:      0.8961")
    print(f"  Delta:         {overall_kw3 - 0.8961:+.4f}")
    print(f"  Perfect (1.0): {(df['kw_v3'] == 1.0).sum()}/30")
    print(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Errors:        {len(errors)}")

    # Doc별
    print(f"\n  Doc별 kw_v3:")
    doc_scores = df.groupby('doc_key')['kw_v3'].mean()
    for dk, score in doc_scores.items():
        print(f"    {dk}: {score:.4f}")

    # 판정
    if overall_kw3 >= 0.896:
        print(f"\n  ✅ PASS: kw_v3={overall_kw3:.4f} >= 0.896 (baseline)")
        print(f"  병합 후 성능 유지 확인!")
    else:
        print(f"\n  ❌ FAIL: kw_v3={overall_kw3:.4f} < 0.896 (baseline)")
        print(f"  성능 하락 발생 — 원인 분석 필요")

    print(f"\n  Saved: {CSV_PATH}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
