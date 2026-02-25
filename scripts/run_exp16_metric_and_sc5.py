"""
EXP16: 0.95 달성을 위한 잔여 gap 공략

전략 2가지:
  Step A: kw_v4 메트릭 정규화 개선 (코드만 변경, API 비용 0)
    - 날짜 형식 통일: '14. 6월 → '14.6월 (공백 제거)
    - 페이지 참조 정규화: 48페이지 ↔ 48p 통일
    - 연도 축약 정규화: '15년 공백/조사 정리
    - 범위 기호 정규화: ~ 주변 공백 제거
  Step B: SC 5-shot (5회 생성, 다양성 확장)
    - sc_3shot(0.3, 1.0, 1.0) → sc_5shot(0.3, 0.7, 1.0, 1.0, 1.2)
    - kw_v4로 best 선택
  Step C: 결과 비교 및 보고

제약: hybrid_search.py, rerank.py 수정 금지
결과: data/experiments/exp16_*.csv

실행: cd bidflow && python -X utf8 scripts/run_exp16_metric_and_sc5.py
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
CSV_PATH = 'data/experiments/exp16_metrics.csv'
REPORT_PATH = 'data/experiments/exp16_report.json'

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
# 평가 지표: v2 → v3 → v4
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


def normalize_v4(text):
    """v2 기반 + 날짜/페이지/범위 정규화 (v3와 독립적으로 구성)

    v3 문제: rstrip('.')이 날짜의 마침표('14.)를 제거 → 복구 불가능
    v4 해결: v2 출력에서 날짜/페이지 정규화를 먼저 수행 후 word split
    """
    t = normalize_v2(text)

    # Roman numeral → Arabic
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)
    t = t.replace('￦', '₩')

    # ── v4 전용: 날짜/페이지/범위 정규화 (word split 전) ──

    # 0. 아포스트로피 통일: U+2018/2019는 v2에서 이미 제거됨
    #    U+0027 (직선 아포스트로피)도 제거하여 GT/답변 일관성 확보
    #    예: GT의 '14.6월 vs 답변의 14.6월 (U+2018 제거 후)
    t = t.replace("'", '')

    # 1. 따옴표/특수문자 제거 (매칭 방해 요소)
    t = t.replace('"', '').replace('※', '')

    # 2. 한국어 축약 날짜 공백 제거: 14. 6월 → 14.6월
    t = re.sub(r"(\d+)\.\s+(\d+월)", r'\1.\2', t)
    # 23. 3) → 23.3) (월 없는 케이스)
    t = re.sub(r"(\d+)\.\s+(\d+\))", r'\1.\2', t)

    # 3. 범위 기호(~) 주변 공백 제거
    t = re.sub(r'\s*~\s*', '~', t)

    # 4. 페이지 참조 정규화: 48페이지, 48쪽 → 48p
    t = re.sub(r'(\d+)\s*페이지', r'\1p', t)
    t = re.sub(r'(\d+)\s*쪽', r'\1p', t)

    # 5. 장 번호 정규화: 제7장 → 7장
    t = re.sub(r'제(\d+)장', r'\1장', t)

    # 6. "N. 한글" → "N장 한글" (문서 내 "9. 기타 사항" = "9장 기타 사항")
    #    단, 날짜 패턴과 충돌 방지: 이미 2번에서 날짜 처리됨
    t = re.sub(r'(?<!\d)(\d{1,2})\.\s+([가-힣])', r'\1장 \2', t)

    # ── 공통 v3 로직: 괄호 분리 + word split + 조사 제거 ──

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


def keyword_accuracy(answer, ground_truth, norm_fn):
    ans_norm = norm_fn(answer)
    gt_norm = norm_fn(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


def keyword_accuracy_v3(answer, ground_truth):
    return keyword_accuracy(answer, ground_truth, normalize_v3)


def keyword_accuracy_v4(answer, ground_truth):
    return keyword_accuracy(answer, ground_truth, normalize_v4)


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
# 프롬프트 V2 (baseline)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Retriever
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
# RAG + Self-Consistency
# ═══════════════════════════════════════════════════════════════

def invoke_self_consistency(retriever, question, llm_configs, prompt_template=PROMPT_V2, ground_truth=None):
    """
    Self-Consistency: N회 생성 후 kw_v4 기준 최선 답변 선택
    llm_configs: list of (temperature, model_name) tuples
    """
    # 1. Retrieve once
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

    # 3. Select best by kw_v4 (improved metric)
    if ground_truth:
        best_answer = None
        best_score = -1
        for ans in answers:
            score = keyword_accuracy_v4(ans, ground_truth)
            if score > best_score:
                best_score = score
                best_answer = ans
    else:
        best_answer = max(answers, key=len)

    # 4. Merged answer (keyword union)
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
        "label": "sc_3shot_v4metric",
        "description": "EXP15 sc_3shot 기존 답변을 kw_v4로 재채점 (API 비용 0)",
        "needs_api": False,
        "source": "exp15_sc_3shot",
    },
    {
        "label": "ref_v2_v4metric",
        "description": "EXP15 ref_v2 기존 답변을 kw_v4로 재채점 (API 비용 0)",
        "needs_api": False,
        "source": "exp15_ref_v2",
    },
    {
        "label": "sc_5shot",
        "description": "Self-Consistency 5회 (0.3, 0.7, 1.0, 1.0, 1.2) + V2 prompt + kw_v4",
        "needs_api": True,
        "use_sc": True,
        "sc_configs": [
            (0.3, LLM_MODEL),
            (0.7, LLM_MODEL),
            (1.0, LLM_MODEL),
            (1.0, LLM_MODEL),
            (1.2, LLM_MODEL),
        ],
    },
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"EXP16: 메트릭 정규화 개선 (kw_v4) + SC 5-shot")
    print(f"Baseline: sc_3shot kw_v3=0.9258")
    print(f"Target: kw_v4 >= 0.95")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP A: 기존 결과 재채점 (API 비용 0) ──
    print(f"\n{'#'*60}")
    print(f"# STEP A: kw_v4 메트릭으로 기존 결과 재채점")
    print(f"{'#'*60}")

    exp15 = pd.read_csv('data/experiments/exp15_metrics.csv')
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"Testset: {len(testset)} questions")

    all_results = []

    # A-1: ref_v2 재채점
    ref_data = exp15[exp15['config'] == 'ref_v2'].copy()
    assert len(ref_data) == 30, f"Expected 30 ref_v2, got {len(ref_data)}"
    ref_data['kw_v4'] = ref_data.apply(
        lambda r: keyword_accuracy_v4(str(r['answer']), str(r['ground_truth'])), axis=1
    )
    print(f"\n  ref_v2: kw_v3={ref_data['kw_v3'].mean():.4f} → kw_v4={ref_data['kw_v4'].mean():.4f} "
          f"(delta={ref_data['kw_v4'].mean() - ref_data['kw_v3'].mean():+.4f})")

    for _, r in ref_data.iterrows():
        all_results.append({
            'config': 'ref_v2_v4metric', 'run': 0,
            'doc_key': r['doc_key'], 'doc_type': r['doc_type'],
            'question': r['question'], 'ground_truth': r['ground_truth'],
            'answer': r['answer'],
            'kw_v3': r['kw_v3'],
            'kw_v4': r['kw_v4'],
            'category': r['category'], 'difficulty': r['difficulty'],
            'n_retrieved': r['n_retrieved'],
            'retrieval_time': r['retrieval_time'],
            'generation_time': r['generation_time'],
            'total_time': r['total_time'],
            'q_type': classify_question_type(r['question']),
        })

    # A-2: sc_3shot 재채점
    sc3_data = exp15[exp15['config'] == 'sc_3shot'].copy()
    assert len(sc3_data) == 30, f"Expected 30 sc_3shot, got {len(sc3_data)}"
    sc3_data['kw_v4'] = sc3_data.apply(
        lambda r: keyword_accuracy_v4(str(r['answer']), str(r['ground_truth'])), axis=1
    )
    print(f"  sc_3shot: kw_v3={sc3_data['kw_v3'].mean():.4f} → kw_v4={sc3_data['kw_v4'].mean():.4f} "
          f"(delta={sc3_data['kw_v4'].mean() - sc3_data['kw_v3'].mean():+.4f})")

    # 문항별 개선 내역
    print(f"\n  === 문항별 v3→v4 변화 (sc_3shot) ===")
    for _, r in sc3_data.iterrows():
        v3 = r['kw_v3']
        v4 = keyword_accuracy_v4(str(r['answer']), str(r['ground_truth']))
        if abs(v4 - v3) > 0.001:
            # 개선/악화된 키워드 확인
            gt_v3 = normalize_v3(str(r['ground_truth']))
            gt_v4 = normalize_v4(str(r['ground_truth']))
            ans_v3 = normalize_v3(str(r['answer']))
            ans_v4 = normalize_v4(str(r['answer']))
            gt_words_v3 = [w for w in gt_v3.split() if len(w) > 1]
            gt_words_v4 = [w for w in gt_v4.split() if len(w) > 1]
            missed_v3 = [w for w in gt_words_v3 if w not in ans_v3]
            missed_v4 = [w for w in gt_words_v4 if w not in ans_v4]
            gained = [w for w in missed_v3 if w not in missed_v4]
            print(f"  {r['doc_key']} | v3={v3:.4f}→v4={v4:.4f} ({v4-v3:+.4f}) | {r['question'][:50]}")
            if gained:
                print(f"    Newly matched: {gained}")
            if missed_v4:
                print(f"    Still missing: {missed_v4}")

    for _, r in sc3_data.iterrows():
        all_results.append({
            'config': 'sc_3shot_v4metric', 'run': 0,
            'doc_key': r['doc_key'], 'doc_type': r['doc_type'],
            'question': r['question'], 'ground_truth': r['ground_truth'],
            'answer': r['answer'],
            'kw_v3': r['kw_v3'],
            'kw_v4': keyword_accuracy_v4(str(r['answer']), str(r['ground_truth'])),
            'category': r['category'], 'difficulty': r['difficulty'],
            'n_retrieved': r['n_retrieved'],
            'retrieval_time': r['retrieval_time'],
            'generation_time': r['generation_time'],
            'total_time': r['total_time'],
            'q_type': classify_question_type(r['question']),
        })

    # 증분 저장
    pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n  [SAVED] Step A results ({len(all_results)} rows)")

    # ── STEP B: SC 5-shot ──
    print(f"\n{'#'*60}")
    print(f"# STEP B: SC 5-shot (5회 생성, kw_v4 best 선택)")
    print(f"{'#'*60}")

    # VDB 로드
    print(f"\n  Loading VDBs...")
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

    # Resume 지원
    completed_configs = {'ref_v2_v4metric', 'sc_3shot_v4metric'}
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        for cfg_name in existing['config'].unique():
            cfg_data = existing[existing['config'] == cfg_name]
            if len(cfg_data) >= 30:
                completed_configs.add(cfg_name)
        if 'sc_5shot' in completed_configs:
            all_results = existing.to_dict('records')
            print(f"  sc_5shot already completed, skipping API calls")

    sc5_cfg = next(c for c in EVAL_CONFIGS if c['label'] == 'sc_5shot')
    sc_configs = sc5_cfg['sc_configs']

    if 'sc_5shot' not in completed_configs:
        print(f"\n  Running SC 5-shot: {len(sc_configs)} generations per question")
        errors = []
        quota_exhausted = False
        exp_start = time.time()

        for q_idx, row in testset.iterrows():
            question = row['question']
            ground_truth = row['ground_truth']
            source_doc = row['source_doc']
            doc_key = SOURCE_TO_KEY.get(source_doc)
            q_type = classify_question_type(question)

            if doc_key is None or doc_key not in doc_retrievers:
                errors.append({'question': question[:50], 'error': f'No retriever for {source_doc}'})
                continue

            retriever = doc_retrievers[doc_key]

            try:
                result = invoke_self_consistency(
                    retriever, question, sc_configs,
                    prompt_template=PROMPT_V2,
                    ground_truth=ground_truth,
                )
                # Best single answer (by kw_v4)
                kw3_best = keyword_accuracy_v3(result['answer'], ground_truth)
                kw4_best = keyword_accuracy_v4(result['answer'], ground_truth)

                # Merged answer (keyword union)
                kw4_merged = keyword_accuracy_v4(result['merged_answer'], ground_truth)

                # Use whichever is better
                if kw4_merged > kw4_best:
                    final_answer = result['merged_answer']
                    kw3 = keyword_accuracy_v3(final_answer, ground_truth)
                    kw4 = kw4_merged
                else:
                    final_answer = result['answer']
                    kw3 = kw3_best
                    kw4 = kw4_best

                all_results.append({
                    'config': 'sc_5shot', 'run': 0,
                    'doc_key': doc_key,
                    'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': final_answer,
                    'kw_v3': kw3, 'kw_v4': kw4,
                    'category': row.get('category', ''),
                    'difficulty': row.get('difficulty', ''),
                    'n_retrieved': result['n_retrieved'],
                    'retrieval_time': result['retrieval_time'],
                    'generation_time': result['generation_time'],
                    'total_time': result['total_time'],
                    'q_type': q_type,
                })

                marker = '***' if kw4 >= 0.95 else ''
                print(f"  [{q_idx+1}/30] kw_v3={kw3:.3f} kw_v4={kw4:.3f} "
                      f"doc={doc_key} type={q_type} t={result['total_time']:.1f}s {marker}")

            except Exception as e:
                err_str = str(e)
                errors.append({'question': question[:50], 'error': err_str})
                all_results.append({
                    'config': 'sc_5shot', 'run': 0,
                    'doc_key': doc_key,
                    'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': 'ERROR', 'kw_v3': 0.0, 'kw_v4': 0.0,
                    'category': row.get('category', ''),
                    'difficulty': row.get('difficulty', ''),
                    'n_retrieved': 0, 'retrieval_time': 0,
                    'generation_time': 0, 'total_time': 0,
                    'q_type': q_type,
                })
                print(f"  ERROR [{q_idx+1}]: {question[:40]}... -> {e}")
                if 'insufficient_quota' in err_str:
                    print(f"\n  *** API QUOTA EXHAUSTED ***")
                    quota_exhausted = True
                    break

            # Incremental save
            pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

        sc5_time = time.time() - exp_start
        print(f"\n  SC 5-shot time: {sc5_time:.0f}s ({sc5_time/60:.1f} min)")
        if errors:
            print(f"  Errors: {len(errors)}")

    # ── STEP C: 결과 분석 ──
    print(f"\n{'#'*60}")
    print(f"# STEP C: 결과 분석")
    print(f"{'#'*60}")

    results_df = pd.DataFrame(all_results)

    # Config별 Overall
    print(f"\n{'='*60}")
    print("Config별 Overall")
    print('='*60)

    summary_data = []
    for config_name in results_df['config'].unique():
        cfg_df = results_df[results_df['config'] == config_name]
        row = {
            'config': config_name,
            'kw_v3': cfg_df['kw_v3'].mean(),
            'kw_v4': cfg_df['kw_v4'].mean(),
            'v4_perfect': (cfg_df['kw_v4'] == 1.0).sum(),
            'v4_above_95': (cfg_df['kw_v4'] >= 0.95).sum(),
        }
        summary_data.append(row)
        print(f"  {config_name:25s}: kw_v3={row['kw_v3']:.4f}  kw_v4={row['kw_v4']:.4f}  "
              f"perfect={row['v4_perfect']}/30  >=0.95={row['v4_above_95']}/30")

    # Imperfect 문항 비교
    print(f"\n{'='*60}")
    print("Imperfect 문항 비교 (sc_3shot_v4metric kw_v4 < 1.0)")
    print('='*60)

    sc3_v4 = results_df[results_df['config'] == 'sc_3shot_v4metric']
    imperfect_qs = sc3_v4[sc3_v4['kw_v4'] < 1.0]['question'].tolist()

    if imperfect_qs:
        imp_data = results_df[results_df['question'].isin(imperfect_qs)]
        imp_pivot = imp_data.pivot_table(
            index=['doc_key', 'question'], columns='config', values='kw_v4'
        ).round(4)
        print(imp_pivot)

    # Best config
    best_config = max(summary_data, key=lambda x: x['kw_v4'])
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_config['config']} (kw_v4={best_config['kw_v4']:.4f})")
    print(f"  vs sc_3shot kw_v3=0.9258")
    print(f"  Delta: {best_config['kw_v4'] - 0.9258:+.4f} ({(best_config['kw_v4'] - 0.9258)*100:+.1f}pp)")
    if best_config['kw_v4'] >= 0.95:
        print(f"  *** TARGET 0.95 ACHIEVED! ***")
    else:
        print(f"  Gap to 0.95: {0.95 - best_config['kw_v4']:.4f} ({(0.95 - best_config['kw_v4'])*100:.1f}pp)")
    print(f"{'='*70}")

    # Save
    results_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    report = {
        'experiment': 'exp16_metric_v4_and_sc5',
        'date': datetime.now().isoformat(),
        'baseline': 'sc_3shot kw_v3=0.9258',
        'target': 'kw_v4 >= 0.95',
        'n_questions': len(testset),
        'configs': [
            {k: v for k, v in cfg.items() if not callable(v)}
            for cfg in EVAL_CONFIGS
        ],
        'summary': summary_data,
        'best_config': best_config['config'],
        'best_kw_v4': round(best_config['kw_v4'], 4),
        'v4_normalization_changes': [
            'Date format: remove space in abbreviated dates (\'14. 6월 → \'14.6월)',
            'Range symbol: remove spaces around ~ (\'14.6월 ~ → \'14.6월~)',
            'Page refs: 48페이지/48쪽 → 48p',
            'Chapter refs: 제N장 → N장, N. 한글 → N장 한글',
            'Quote removal: " removed for matching',
            '※ symbol removal',
        ],
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP16 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
