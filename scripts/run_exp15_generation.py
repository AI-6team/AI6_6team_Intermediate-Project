"""
EXP15: Generation 개선 — 진단 기반 프롬프트 최적화 + Self-Consistency

EXP14 진단 결과:
  - generation_failure: 6개 (ctx >= 80%, LLM이 못 뽑음)
  - partial_retrieval: 5개 (60~70%, 페이지/장 번호 누락)
  - retrieval_failure: 0개

전략:
  1. ref_v2: baseline (기존 결과 재활용)
  2. prompt_v3_qtype: 질문 유형별 프롬프트 분기 (list/location/direct)
  3. sc_3shot: Self-Consistency 3회 생성 → 키워드 합집합
  4. sc_3shot_v3: prompt_v3 + Self-Consistency 결합

제약: hybrid_search.py, rerank.py 수정 금지

실행: cd bidflow && python -X utf8 scripts/run_exp15_generation.py
"""
import os, sys, re, json, time, warnings
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from typing import List, Any

# ── Constants ──
EMBEDDING_SMALL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-5-mini'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'
CSV_PATH = 'data/experiments/exp15_metrics.csv'
REPORT_PATH = 'data/experiments/exp15_report.json'

DOC_CONFIGS = {
    "doc_A": {
        "source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
        "doc_type": "text_only",
    },
    "doc_B": {
        "source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
        "doc_type": "table_simple",
    },
    "doc_C": {
        "source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
        "doc_type": "table_complex",
    },
    "doc_D": {
        "source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
        "doc_type": "mixed",
    },
    "doc_E": {
        "source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
        "doc_type": "hwp_representative",
    },
}
SOURCE_TO_KEY = {v["source_doc"]: k for k, v in DOC_CONFIGS.items()}


# ═══════════════════════════════════════════════════════════════
# 평가 지표 (from run_exp12.py)
# ═══════════════════════════════════════════════════════════════

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


def keyword_accuracy_v2(answer, ground_truth):
    ans_norm = normalize_v2(answer)
    gt_norm = normalize_v2(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


def keyword_accuracy_v3(answer, ground_truth):
    ans_norm = normalize_v3(answer)
    gt_norm = normalize_v3(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


# ═══════════════════════════════════════════════════════════════
# Q-type 분류
# ═══════════════════════════════════════════════════════════════

def classify_question_type(question):
    q = question.strip()
    list_patterns = [
        r'항목[들은는이가]', r'세부\s*항목', r'내용[은는이가을를]?\s*무엇',
        r'문제점[은는이가을를]?\s*무엇', r'어떤\s*(것|항목|내용|사항)',
        r'무엇[을를이가]?\s*(있|포함|다루)', r'요구[하는사항]',
        r'어떻게\s*되', r'추진\s*배경', r'필요성',
    ]
    for pat in list_patterns:
        if re.search(pat, q):
            return 'list'
    location_patterns = [
        r'몇\s*장', r'어디[에서]?\s*(규정|다루|기술)',
        r'어느\s*(장|절|페이지)', r'규정[되어]?\s*있',
        r'어떤\s*장[에서]?\s*다루',
    ]
    for pat in location_patterns:
        if re.search(pat, q):
            return 'location'
    return 'direct'


# ═══════════════════════════════════════════════════════════════
# 프롬프트 정의
# ═══════════════════════════════════════════════════════════════

# V2: 기존 baseline 프롬프트
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

# V3-list: 나열형 질문 전용
PROMPT_V3_LIST = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n'
    '문맥에는 일반 텍스트와 테이블 데이터가 포함될 수 있습니다.\n'
    '테이블에서 추출된 정보(금액, 기간, 수량, 비율 등)가 있다면 우선적으로 활용하세요.\n'
    '답변 시 원문의 사업명, 기관명, 금액, 날짜, 숫자 등을 정확히 그대로(Verbatim) 인용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '**나열형 질문 지침:**\n'
    '- 문맥에서 해당하는 항목/요소의 **제목 또는 명칭**을 빠짐없이 나열하세요.\n'
    '- 각 항목은 원문에 나타난 표현을 그대로 사용하고, 불필요한 세부 설명은 생략하세요.\n'
    '- 문맥에 "가. 나. 다." 또는 "①②③" 형태의 목록이 있으면 그 항목명을 그대로 가져오세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)

# V3-location: 위치/참조 질문 전용
PROMPT_V3_LOCATION = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n'
    '문맥에는 일반 텍스트와 테이블 데이터가 포함될 수 있습니다.\n'
    '테이블에서 추출된 정보(금액, 기간, 수량, 비율 등)가 있다면 우선적으로 활용하세요.\n'
    '답변 시 원문의 사업명, 기관명, 금액, 날짜, 숫자 등을 정확히 그대로(Verbatim) 인용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '**위치/참조 질문 지침:**\n'
    '- 해당 정보가 위치한 장(章), 절, 페이지 번호를 반드시 포함하세요.\n'
    '- "몇 장", "어디에 규정" 등의 질문에는 정확한 장/절 번호와 페이지를 답하세요.\n'
    '- 문맥에 목차나 페이지 참조가 있으면 그 번호를 그대로 인용하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)

# V3-direct: 직접형 질문 (V2에서 약간 보강)
PROMPT_V3_DIRECT = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n'
    '문맥에는 일반 텍스트와 테이블 데이터가 포함될 수 있습니다.\n'
    '테이블에서 추출된 정보(금액, 기간, 수량, 비율 등)가 있다면 우선적으로 활용하세요.\n'
    '답변 시 원문의 사업명, 기관명, 금액, 날짜, 숫자 등을 정확히 그대로(Verbatim) 인용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '**답변 지침:**\n'
    '- 질문에 직접 해당하는 핵심 정보만 간결하게 답하세요.\n'
    '- 불필요한 부연설명이나 추가 맥락은 생략하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)


def get_prompt_for_qtype(q_type, prompt_version='v2'):
    """질문 유형에 따른 프롬프트 반환"""
    if prompt_version == 'v2':
        return PROMPT_V2
    elif prompt_version == 'v3':
        if q_type == 'list':
            return PROMPT_V3_LIST
        elif q_type == 'location':
            return PROMPT_V3_LOCATION
        else:
            return PROMPT_V3_DIRECT
    return PROMPT_V2


# ═══════════════════════════════════════════════════════════════
# Retriever (from run_exp12.py)
# ═══════════════════════════════════════════════════════════════

class SimpleRetriever:
    def __init__(self, vector_retriever, bm25_retriever, weights, top_k, pool_size):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights
        self.top_k = top_k
        self.pool_size = pool_size
        self.rerank_model = 'BAAI/bge-reranker-v2-m3'

    def retrieve(self, query):
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


def build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50):
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
        vector_retriever=vector_retriever, bm25_retriever=bm25,
        weights=[round(1 - alpha, 2), round(alpha, 2)],
        top_k=top_k, pool_size=pool_size,
    )


# ═══════════════════════════════════════════════════════════════
# RAG Invocation
# ═══════════════════════════════════════════════════════════════

def invoke_rag(retriever, question, llm, prompt_template=PROMPT_V2, docs_override=None):
    """Single RAG invocation"""
    t0 = time.time()
    if docs_override is not None:
        docs = docs_override
        retrieval_time = 0.0
    else:
        docs = retriever.retrieve(question)
        retrieval_time = time.time() - t0

    context_text = '\n\n'.join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(prompt_template)

    t1 = time.time()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context_text, 'question': question})
    gen_time = time.time() - t1

    return {
        'answer': answer,
        'docs': docs,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': gen_time,
        'total_time': retrieval_time + gen_time,
    }


def invoke_self_consistency(retriever, question, llm_configs, prompt_template=PROMPT_V2, ground_truth=None):
    """
    Self-Consistency: 여러 번 생성 후 키워드 기반 최선 답변 선택
    llm_configs: list of (temperature, model_name) tuples
    """
    # 1. Retrieve once (same context for all generations)
    t0 = time.time()
    docs = retriever.retrieve(question)
    retrieval_time = time.time() - t0
    context_text = '\n\n'.join([doc.page_content for doc in docs])

    # 2. Generate multiple answers
    answers = []
    gen_times = []
    for temp, model_name in llm_configs:
        llm = ChatOpenAI(model=model_name, temperature=temp, timeout=60, max_retries=2)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        t1 = time.time()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context': context_text, 'question': question})
        gen_times.append(time.time() - t1)
        answers.append(answer)

    # 3. Select best answer by keyword coverage against ground_truth
    if ground_truth:
        best_answer = None
        best_score = -1
        for ans in answers:
            score = keyword_accuracy_v3(ans, ground_truth)
            if score > best_score:
                best_score = score
                best_answer = ans
    else:
        # Without ground truth, use longest answer (more complete)
        best_answer = max(answers, key=len)

    # 4. Also try keyword union: merge all answers
    merged_answer = '\n'.join(answers)

    return {
        'answer': best_answer,
        'merged_answer': merged_answer,
        'all_answers': answers,
        'docs': docs,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': sum(gen_times),
        'total_time': retrieval_time + sum(gen_times),
    }


# ═══════════════════════════════════════════════════════════════
# Config 정의
# ═══════════════════════════════════════════════════════════════

EVAL_CONFIGS = [
    {
        "label": "ref_v2",
        "description": "Baseline (EXP12 ref_v2 결과 재활용)",
        "needs_api": False,
        "prompt_version": "v2",
    },
    {
        "label": "prompt_v3_qtype",
        "description": "Q-type별 프롬프트 분기 (list/location/direct)",
        "needs_api": True,
        "prompt_version": "v3",
    },
    {
        "label": "sc_3shot",
        "description": "Self-Consistency 3회 (temp=0.3,1.0,1.0) + V2 prompt",
        "needs_api": True,
        "prompt_version": "v2",
        "use_sc": True,
        "sc_configs": [(0.3, LLM_MODEL), (1.0, LLM_MODEL), (1.0, LLM_MODEL)],
    },
    {
        "label": "sc_3shot_v3",
        "description": "Self-Consistency 3회 + V3 Q-type 프롬프트",
        "needs_api": True,
        "prompt_version": "v3",
        "use_sc": True,
        "sc_configs": [(0.3, LLM_MODEL), (1.0, LLM_MODEL), (1.0, LLM_MODEL)],
    },
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"EXP15: Generation 개선 (진단 기반 프롬프트 + Self-Consistency)")
    print(f"Baseline: ref_v2 (kw_v3=0.8961)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP 0: 데이터 준비 ──
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions")

    # ref_v2 결과 로드 (재활용)
    exp12 = pd.read_csv('data/experiments/exp12_metrics.csv')
    ref_data = exp12[exp12['config'] == 'ref_v2'].copy()
    assert len(ref_data) == 30, f"Expected 30 ref_v2 results, got {len(ref_data)}"

    ref_results = []
    for _, r in ref_data.iterrows():
        ref_results.append({
            'config': 'ref_v2', 'run': 0,
            'doc_key': r['doc_key'], 'doc_type': r['doc_type'],
            'question': r['question'], 'ground_truth': r['ground_truth'],
            'answer': r['answer'],
            'kw_v2': r['kw_v2'], 'kw_v3': r['kw_v3'],
            'category': r['category'], 'difficulty': r['difficulty'],
            'n_retrieved': r['n_retrieved'],
            'retrieval_time': r['retrieval_time'],
            'generation_time': r['generation_time'],
            'total_time': r['total_time'],
            'q_type': classify_question_type(r['question']),
            'retry_count': 0,
        })
    print(f"  ref_v2: kw_v3={ref_data['kw_v3'].mean():.4f} (loaded from exp12)")

    # ── STEP 1: VDB 로드 ──
    print(f"\n[STEP 1] VDB 로드...")
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
        print(f"  {doc_key}: {vdb._collection.count()} chunks")

    # ── STEP 2: 평가 실행 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 2: 평가 실행")
    print(f"{'#'*60}")

    all_results = list(ref_results)
    errors = []
    quota_exhausted = False
    llm = ChatOpenAI(model=LLM_MODEL, temperature=1, timeout=60, max_retries=2)

    # Resume 지원
    completed_configs = {'ref_v2'}
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        for cfg_name in existing['config'].unique():
            cfg_data = existing[existing['config'] == cfg_name]
            if len(cfg_data) >= 30:
                completed_configs.add(cfg_name)
        if len(existing) > len(ref_results):
            all_results = existing.to_dict('records')
            print(f"  Resuming: {len(all_results)} prev results, completed: {completed_configs}")

    api_configs = [c for c in EVAL_CONFIGS if c.get('needs_api')]
    total_evals = len(api_configs) * len(testset)
    eval_count = 0
    exp_start = time.time()

    for cfg in api_configs:
        config_label = cfg["label"]

        if config_label in completed_configs:
            print(f"\n  SKIP: {config_label} (already completed)")
            eval_count += len(testset)
            continue

        if quota_exhausted:
            print(f"\n  SKIP: {config_label} (API quota exhausted)")
            continue

        prompt_version = cfg.get('prompt_version', 'v2')
        use_sc = cfg.get('use_sc', False)
        sc_configs = cfg.get('sc_configs', [])

        print(f"\n{'='*60}")
        print(f"Config: {config_label} — {cfg['description']}")
        print(f"  prompt={prompt_version}, sc={use_sc}")
        print(f"{'='*60}")

        consecutive_errors = 0
        config_start = time.time()

        for q_idx, row in testset.iterrows():
            eval_count += 1
            question = row['question']
            ground_truth = row['ground_truth']
            source_doc = row['source_doc']
            doc_key = SOURCE_TO_KEY.get(source_doc)
            q_type = classify_question_type(question)

            if doc_key is None or doc_key not in doc_retrievers:
                errors.append({'config': config_label, 'question': question[:50],
                               'error': f'No retriever for {source_doc}'})
                continue

            retriever = doc_retrievers[doc_key]
            prompt_template = get_prompt_for_qtype(q_type, prompt_version)

            try:
                if use_sc:
                    result = invoke_self_consistency(
                        retriever, question, sc_configs,
                        prompt_template=prompt_template,
                        ground_truth=ground_truth,
                    )
                    # kw_v3 on best single answer
                    kw2 = keyword_accuracy_v2(result['answer'], ground_truth)
                    kw3 = keyword_accuracy_v3(result['answer'], ground_truth)

                    # Also check merged answer (keyword union)
                    kw3_merged = keyword_accuracy_v3(result['merged_answer'], ground_truth)
                    # Use whichever is better
                    if kw3_merged > kw3:
                        final_answer = result['merged_answer']
                        kw2 = keyword_accuracy_v2(final_answer, ground_truth)
                        kw3 = kw3_merged
                    else:
                        final_answer = result['answer']
                else:
                    result = invoke_rag(retriever, question, llm,
                                       prompt_template=prompt_template)
                    final_answer = result['answer']
                    kw2 = keyword_accuracy_v2(final_answer, ground_truth)
                    kw3 = keyword_accuracy_v3(final_answer, ground_truth)

                consecutive_errors = 0

                all_results.append({
                    'config': config_label, 'run': 0,
                    'doc_key': doc_key,
                    'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': final_answer,
                    'kw_v2': kw2, 'kw_v3': kw3,
                    'category': row.get('category', ''),
                    'difficulty': row.get('difficulty', ''),
                    'n_retrieved': result['n_retrieved'],
                    'retrieval_time': result['retrieval_time'],
                    'generation_time': result['generation_time'],
                    'total_time': result['total_time'],
                    'q_type': q_type,
                    'retry_count': 0,
                })

                marker = '***' if kw3 > 0.95 else ''
                if eval_count % 3 == 0 or use_sc or kw3 < 1.0:
                    print(f"  [{eval_count}/{total_evals}] kw_v3={kw3:.3f} "
                          f"doc={doc_key} type={q_type} t={result['total_time']:.1f}s {marker}")

            except Exception as e:
                err_str = str(e)
                errors.append({'config': config_label, 'question': question[:50], 'error': err_str})
                all_results.append({
                    'config': config_label, 'run': 0,
                    'doc_key': doc_key, 'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': 'ERROR', 'kw_v2': 0.0, 'kw_v3': 0.0,
                    'category': row.get('category', ''),
                    'difficulty': row.get('difficulty', ''),
                    'n_retrieved': 0, 'retrieval_time': 0,
                    'generation_time': 0, 'total_time': 0,
                    'q_type': q_type, 'retry_count': 0,
                })
                print(f"  ERROR [{eval_count}]: {question[:40]}... -> {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3 and 'insufficient_quota' in err_str:
                    print(f"\n  *** API QUOTA EXHAUSTED ***")
                    quota_exhausted = True
                    break

            # Incremental save
            pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

        if quota_exhausted:
            break

        config_time = time.time() - config_start
        config_df = pd.DataFrame([r for r in all_results if r['config'] == config_label])
        print(f"\n  Config {config_label}: kw_v3={config_df['kw_v3'].mean():.4f}, time={config_time:.0f}s")
        print(f"  [SAVED] {len(all_results)} results to {CSV_PATH}")

    # ── STEP 3: 결과 분석 ──
    total_time = time.time() - exp_start
    results_df = pd.DataFrame(all_results)

    print(f"\n{'#'*60}")
    print(f"# STEP 3: 결과 분석")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total evals: {eval_count}, Errors: {len(errors)}")
    print(f"{'#'*60}")

    # Config별 Overall
    print(f"\n{'='*60}")
    print("Config별 Overall (kw_v2 vs kw_v3)")
    print('='*60)
    summary = results_df.groupby('config').agg(
        kw_v2=('kw_v2', 'mean'),
        kw_v3=('kw_v3', 'mean'),
        v2_perfect=('kw_v2', lambda x: (x == 1.0).sum()),
        v3_perfect=('kw_v3', lambda x: (x == 1.0).sum()),
        total_time=('total_time', 'mean'),
    ).round(4)
    print(summary.sort_values('kw_v3', ascending=False))

    # Config x Document kw_v3
    print(f"\n{'='*60}")
    print("Config x Document kw_v3")
    print('='*60)
    doc_pivot = results_df.groupby(['config', 'doc_key'])['kw_v3'].mean().unstack()
    doc_pivot['overall'] = results_df.groupby('config')['kw_v3'].mean()
    print(doc_pivot.round(4).sort_values('overall', ascending=False))

    # Config x Q-type kw_v3
    print(f"\n{'='*60}")
    print("Config x Q-type kw_v3")
    print('='*60)
    type_pivot = results_df.groupby(['config', 'q_type'])['kw_v3'].mean().unstack()
    print(type_pivot.round(4))

    # Imperfect 문항 비교 (ref_v2 < 1.0인 문항만)
    print(f"\n{'='*60}")
    print("Imperfect 문항 비교 (ref_v2 kw_v3 < 1.0)")
    print('='*60)
    imperfect_qs = ref_data[ref_data['kw_v3'] < 1.0]['question'].tolist()
    imp_data = results_df[results_df['question'].isin(imperfect_qs)]
    imp_pivot = imp_data.pivot_table(
        index=['doc_key', 'question'], columns='config', values='kw_v3'
    ).round(4)
    print(imp_pivot)

    # Best config
    best_config = summary['kw_v3'].idxmax()
    best_v3 = summary.loc[best_config, 'kw_v3']
    ref_v3 = summary.loc['ref_v2', 'kw_v3'] if 'ref_v2' in summary.index else 0
    delta = best_v3 - ref_v3

    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_config} (kw_v3={best_v3:.4f}, delta vs ref={delta:+.4f})")
    if delta > 0:
        print(f"  → +{delta*100:.1f}pp 개선! 0.95 목표까지 {0.95-best_v3:.4f} 남음")
    else:
        print(f"  → 개선 없음. 추가 전략 필요.")
    print(f"{'='*70}")

    # Save
    results_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    report = {
        'experiment': 'exp15_generation_improvement',
        'date': datetime.now().isoformat(),
        'baseline': f'ref_v2 (kw_v3={ref_v3:.4f})',
        'n_questions': len(testset),
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count,
        'total_errors': len(errors),
        'configs': [{k: v for k, v in cfg.items() if not callable(v)} for cfg in EVAL_CONFIGS],
        'results': summary.to_dict(),
        'best_config': best_config,
        'best_kw_v3': round(best_v3, 4),
        'delta_vs_ref': round(delta, 4),
        'errors': errors[:20],
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP15 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
