"""
EXP14: 오답 진단 — Retrieval Failure vs Generation Failure 분류

11개 imperfect 문항(kw_v3 < 1.0)에 대해:
1. Retrieval만 실행하여 15개 chunk를 가져옴
2. ground_truth 키워드가 chunks 안에 있는지 확인
3. "retrieval failure" vs "generation failure" 분류
4. 진단 결과를 CSV로 저장

실행: cd bidflow && python -X utf8 scripts/run_exp14_diagnosis.py
"""
import os, sys, re, json, time, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not found'

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict

# ── Constants ──
EMBEDDING_SMALL = 'text-embedding-3-small'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'
CSV_PATH = 'data/experiments/exp14_diagnosis.csv'
DETAIL_PATH = 'data/experiments/exp14_diagnosis_detail.json'

DOC_CONFIGS = {
    "doc_A": {
        "source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
    },
    "doc_B": {
        "source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
    },
    "doc_C": {
        "source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
    },
    "doc_D": {
        "source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
    },
    "doc_E": {
        "source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
    },
}
SOURCE_TO_KEY = {v["source_doc"]: k for k, v in DOC_CONFIGS.items()}


# ── Normalization (from run_exp12.py) ──
SYNONYM_MAP = {
    '정보전략계획': 'ismp', 'ismp 수립': 'ismp', '정보화전략계획': 'ismp',
    '통합로그인': 'sso', '단일 로그인': 'sso', '싱글사인온': 'sso',
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


def normalize_v2(text):
    if not isinstance(text, str):
        return str(text).strip().lower()
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
    for orig, norm in SYNONYM_MAP.items():
        t = t.replace(orig.lower(), norm)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def normalize_v3(text):
    t = normalize_v2(text)
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)
    t = t.replace('￦', '₩')
    t = re.sub(r'([가-힣a-z0-9])\(', r'\1 (', t)
    t = re.sub(r'\)([가-힣a-z])', r') \1', t)
    words = t.split()
    cleaned = []
    for w in words:
        w = w.rstrip('.,;:!?')
        if not w:
            continue
        stripped = PARTICLES_RE.sub('', w)
        cleaned.append(stripped if stripped else w)
    return ' '.join(cleaned)


def keyword_accuracy_v3(answer, ground_truth):
    ans_norm = normalize_v3(answer)
    gt_norm = normalize_v3(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


def extract_gt_keywords(ground_truth):
    """정답에서 핵심 키워드 추출 (정규화 후 2글자 이상)"""
    gt_norm = normalize_v3(ground_truth)
    words = [w for w in gt_norm.split() if len(w) > 1]
    return words


def check_keywords_in_context(gt_keywords, context_text):
    """키워드가 context에 있는지 확인, 개별 매칭 정보 반환"""
    ctx_norm = normalize_v3(context_text)
    results = []
    for kw in gt_keywords:
        found = kw in ctx_norm
        results.append({'keyword': kw, 'found_in_context': found})
    found_count = sum(1 for r in results if r['found_in_context'])
    total = len(results)
    coverage = found_count / total if total > 0 else 0.0
    return coverage, found_count, total, results


# ── Retriever (simplified from run_exp12.py) ──
def build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50):
    """VDB로부터 retriever 생성 (rerank 포함)"""
    vector_retriever = vdb.as_retriever(search_kwargs={'k': pool_size * 2})
    result = vdb.get()
    all_docs = []
    if result and result['documents']:
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            all_docs.append(Document(page_content=text, metadata=meta))
    bm25 = BM25Retriever.from_documents(all_docs) if all_docs else BM25Retriever.from_documents([Document(page_content='empty')])
    bm25.k = pool_size * 2
    return SimpleRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        weights=[round(1 - alpha, 2), round(alpha, 2)],
        top_k=top_k,
        pool_size=pool_size,
    )


class SimpleRetriever:
    """Lightweight retriever without Pydantic BaseRetriever overhead"""
    def __init__(self, vector_retriever, bm25_retriever, weights, top_k, pool_size):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights
        self.top_k = top_k
        self.pool_size = pool_size
        self.rerank_model = 'BAAI/bge-reranker-v2-m3'

    def retrieve(self, query):
        """Retrieve + RRF merge + rerank, return list of Documents"""
        search_k = self.pool_size
        try:
            self.bm25_retriever.k = search_k * 2
            bm25_docs = self.bm25_retriever.invoke(query)
        except:
            bm25_docs = []
        try:
            self.vector_retriever.search_kwargs['k'] = search_k * 2
            vector_docs = self.vector_retriever.invoke(query)
        except:
            vector_docs = []

        merged = self._rrf_merge(bm25_docs, vector_docs, k=60, limit=self.pool_size)

        # Rerank
        if merged:
            from bidflow.retrieval.rerank import rerank
            merged = rerank(query, merged, top_k=self.top_k, model_name=self.rerank_model)

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
            if doc.page_content not in doc_map:
                doc_map[doc.page_content] = doc
        sorted_c = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[c] for c in sorted_c[:limit]]


def main():
    print(f"\n{'='*70}")
    print(f"EXP14: 오답 진단 (Retrieval vs Generation Failure)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP 1: baseline 결과에서 imperfect 문항 추출 ──
    print(f"\n[STEP 1] Baseline 결과 로드...")
    exp12 = pd.read_csv('data/experiments/exp12_metrics.csv')
    ref = exp12[exp12['config'] == 'ref_v2'].copy()
    imperfect = ref[ref['kw_v3'] < 1.0].reset_index(drop=True)
    print(f"  Total: {len(ref)}, Perfect: {(ref['kw_v3']==1.0).sum()}, Imperfect: {len(imperfect)}")

    # ── STEP 2: VDB 로드 ──
    print(f"\n[STEP 2] VDB 로드...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    doc_retrievers = {}
    for doc_key in DOC_CONFIGS:
        persist_dir = str(VDB_BASE / f'vectordb_c500_{doc_key}')
        if not os.path.exists(persist_dir):
            print(f"  WARNING: VDB not found: {persist_dir}")
            continue
        vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings,
                     collection_name='bidflow_rfp')
        doc_retrievers[doc_key] = build_retriever(vdb)
        print(f"  {doc_key}: {vdb._collection.count()} chunks, retriever ready")

    # ── STEP 3: 진단 실행 ──
    print(f"\n[STEP 3] 진단 실행 ({len(imperfect)}개 문항)...")
    print(f"{'='*70}")

    diagnosis_results = []
    detail_results = []

    for i, row in imperfect.iterrows():
        doc_key = row['doc_key']
        question = row['question']
        ground_truth = row['ground_truth']
        answer = row['answer']
        kw_v3 = row['kw_v3']
        q_type = row.get('q_type', '')
        category = row.get('category', '')
        difficulty = row.get('difficulty', '')

        print(f"\n  Q{i+1}/{len(imperfect)}: [{doc_key}] kw_v3={kw_v3:.3f}")
        print(f"  질문: {question}")

        if doc_key not in doc_retrievers:
            print(f"  SKIP: No retriever for {doc_key}")
            continue

        # Retrieval 실행
        retriever = doc_retrievers[doc_key]
        t0 = time.time()
        docs = retriever.retrieve(question)
        retrieval_time = time.time() - t0
        n_retrieved = len(docs)

        # Context 구성
        context_text = "\n\n".join([doc.page_content for doc in docs])
        context_length = len(context_text)

        # GT 키워드 추출 및 context 매칭
        gt_keywords = extract_gt_keywords(ground_truth)
        coverage, found_count, total_kw, kw_details = check_keywords_in_context(
            gt_keywords, context_text
        )

        # 진단 분류
        # coverage >= 0.8: generation failure (정답이 context에 있는데 LLM이 못 뽑음)
        # coverage < 0.5: retrieval failure (정답이 context에 없음)
        # 0.5 <= coverage < 0.8: partial retrieval failure
        if coverage >= 0.8:
            diagnosis = "generation_failure"
        elif coverage < 0.5:
            diagnosis = "retrieval_failure"
        else:
            diagnosis = "partial_retrieval"

        # 추가 분석: answer의 kw_v3와 context의 coverage 비교
        # context에 있는 키워드 기준으로 answer가 얼마나 커버하는지
        answer_coverage = keyword_accuracy_v3(answer, ground_truth)
        context_has_but_answer_missing = []
        answer_has = []
        neither_has = []
        ans_norm = normalize_v3(str(answer))
        for kd in kw_details:
            kw = kd['keyword']
            in_ctx = kd['found_in_context']
            in_ans = kw in ans_norm
            if in_ctx and not in_ans:
                context_has_but_answer_missing.append(kw)
            elif in_ctx and in_ans:
                answer_has.append(kw)
            elif not in_ctx:
                neither_has.append(kw)

        print(f"  Retrieved: {n_retrieved} chunks ({context_length} chars), time={retrieval_time:.1f}s")
        print(f"  GT keywords: {total_kw}개, context 내 발견: {found_count}/{total_kw} ({coverage:.1%})")
        print(f"  진단: {diagnosis}")
        if context_has_but_answer_missing:
            print(f"  Context에 있으나 답변에 누락된 키워드: {context_has_but_answer_missing[:5]}")
        if neither_has:
            print(f"  Context에도 없는 키워드: {neither_has[:5]}")

        diagnosis_results.append({
            'q_idx': i + 1,
            'doc_key': doc_key,
            'question': question,
            'ground_truth': ground_truth,
            'answer': answer,
            'kw_v3': kw_v3,
            'q_type': q_type,
            'category': category,
            'difficulty': difficulty,
            'n_retrieved': n_retrieved,
            'retrieval_time': round(retrieval_time, 2),
            'context_length': context_length,
            'gt_keyword_count': total_kw,
            'gt_keywords_in_context': found_count,
            'context_coverage': round(coverage, 4),
            'diagnosis': diagnosis,
            'context_has_answer_missing': ';'.join(context_has_but_answer_missing),
            'neither_has': ';'.join(neither_has),
        })

        detail_results.append({
            'q_idx': i + 1,
            'doc_key': doc_key,
            'question': question,
            'ground_truth': ground_truth,
            'answer': str(answer)[:500],
            'kw_v3': kw_v3,
            'diagnosis': diagnosis,
            'context_coverage': round(coverage, 4),
            'keyword_details': kw_details,
            'context_has_but_answer_missing': context_has_but_answer_missing,
            'neither_has': neither_has,
            'chunks_preview': [doc.page_content[:200] for doc in docs[:5]],
        })

    # ── STEP 4: 결과 저장 및 요약 ──
    print(f"\n{'='*70}")
    print(f"[STEP 4] 결과 요약")
    print(f"{'='*70}")

    diag_df = pd.DataFrame(diagnosis_results)
    diag_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    with open(DETAIL_PATH, 'w', encoding='utf-8') as f:
        json.dump(detail_results, f, ensure_ascii=False, indent=2)

    # 요약 테이블
    print(f"\n{'='*70}")
    print(f"  진단 결과 요약 테이블")
    print(f"{'='*70}")
    print(f"{'문항':>4} | {'doc':>6} | {'kw_v3':>6} | {'유형':>10} | {'ctx_cover':>10} | {'진단':>20}")
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")
    for _, r in diag_df.iterrows():
        print(f"Q{r['q_idx']:>2}  | {r['doc_key']:>6} | {r['kw_v3']:.3f}  | {r['q_type']:>10} | "
              f"{r['context_coverage']:.1%}       | {r['diagnosis']}")

    # 진단 분포
    print(f"\n진단 분포:")
    for diag_type, count in diag_df['diagnosis'].value_counts().items():
        avg_kw = diag_df[diag_df['diagnosis'] == diag_type]['kw_v3'].mean()
        avg_cover = diag_df[diag_df['diagnosis'] == diag_type]['context_coverage'].mean()
        print(f"  {diag_type}: {count}개 (avg kw_v3={avg_kw:.3f}, avg ctx_coverage={avg_cover:.1%})")

    # doc별 분포
    print(f"\n문서별 분포:")
    for doc_key in sorted(diag_df['doc_key'].unique()):
        doc_data = diag_df[diag_df['doc_key'] == doc_key]
        diag_counts = doc_data['diagnosis'].value_counts().to_dict()
        print(f"  {doc_key}: {len(doc_data)}개 — {diag_counts}")

    # 전략 추천
    gen_failures = len(diag_df[diag_df['diagnosis'] == 'generation_failure'])
    ret_failures = len(diag_df[diag_df['diagnosis'] == 'retrieval_failure'])
    partial = len(diag_df[diag_df['diagnosis'] == 'partial_retrieval'])

    print(f"\n{'='*70}")
    print(f"  전략 추천")
    print(f"{'='*70}")
    print(f"  Generation failures: {gen_failures}")
    print(f"  Retrieval failures:  {ret_failures}")
    print(f"  Partial retrieval:   {partial}")

    if gen_failures >= ret_failures:
        print(f"\n  → Generation 개선 (EXP16) 우선 추천")
        print(f"    - Self-Consistency (3회 생성 → 다수결)")
        print(f"    - Question-type별 프롬프트 분기")
        print(f"    - Chain-of-Thought 추론")
    else:
        print(f"\n  → Retrieval 개선 (EXP15) 우선 추천")
        print(f"    - HyDE (가상 답변 생성 → 검색)")
        print(f"    - Query Decomposition")
        print(f"    - doc_D 특화 chunking")

    print(f"\n  Saved: {CSV_PATH}")
    print(f"  Saved: {DETAIL_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP14 진단 완료")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
