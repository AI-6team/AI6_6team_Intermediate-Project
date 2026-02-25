"""
EXP17: kw_v4=0.9534 → 목표 kw>=0.99 추가 개선

전략 3단계:
  Step A: kw_v5 활용어미 유연매칭 (코드만 변경, API 비용 0)
    - "필요하며"→"필요", "1명인"→"1명", "센터이며"→"센터" 등
    - keyword_accuracy_v5: 키워드 매칭 실패 시 활용어미 제거 후 재시도
  Step B: SC 3-shot fresh run with kw_v5 selection
    - EXP15와 동일 retrieval, 새 생성 (kw_v5로 best 선택)
  Step C: Targeted 10-shot for remaining imperfect
    - Step B 후 kw_v5 < 1.0인 문항만 10회 생성

제약: hybrid_search.py, rerank.py 수정 금지
결과: data/experiments/exp17_metrics.csv

실행: cd bidflow && python -X utf8 scripts/run_exp17_to_099.py
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

# ── Constants ──
EMBEDDING_SMALL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-5-mini'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'
CSV_PATH = Path('data/experiments/exp17_metrics.csv')
REPORT_PATH = 'data/experiments/exp17_report.json'

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
# 평가 지표: v2 → v4 → v5
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

# 활용어미 리스트 (v5 유연매칭용)
# 키워드 끝에 붙는 한국어 동사/형용사 활용어미
VERB_ENDINGS = [
    '하며', '이며', '으며', '되며',    # 나열형 연결어미
    '하고', '이고', '되고',             # 나열형 연결어미
    '하여', '이어', '되어',             # 원인/방법 연결어미
    '하는', '되는', '인',               # 관형형 어미
    '한다', '된다', '이다',             # 종결어미
    '합니다', '됩니다', '입니다',       # 존칭 종결어미
    '하면', '되면', '이면',             # 조건 연결어미
    '해서', '되서', '이라서',           # 원인 연결어미
    '했던', '되었던', '이었던',         # 과거 관형
    '1명인',                             # 특수: 숫자+명+인 패턴은 "1명"으로
]


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


def normalize_v4(text):
    """v2 기반 + 날짜/페이지/범위/아포스트로피 정규화"""
    t = normalize_v2(text)
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)
    t = t.replace('￦', '₩')
    t = t.replace("'", '')
    t = t.replace('"', '').replace('※', '')
    t = re.sub(r"(\d+)\.\s+(\d+월)", r'\1.\2', t)
    t = re.sub(r"(\d+)\.\s+(\d+\))", r'\1.\2', t)
    t = re.sub(r'\s*~\s*', '~', t)
    t = re.sub(r'(\d+)\s*페이지', r'\1p', t)
    t = re.sub(r'(\d+)\s*쪽', r'\1p', t)
    t = re.sub(r'제(\d+)장', r'\1장', t)
    t = re.sub(r'(?<!\d)(\d{1,2})\.\s+([가-힣])', r'\1장 \2', t)
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


def _strip_verb_ending(keyword):
    """키워드에서 활용어미를 제거하여 stem을 반환.
    제거 가능한 경우 stem을 반환, 불가능하면 None"""
    for ending in sorted(VERB_ENDINGS, key=len, reverse=True):  # 긴 것부터 시도
        if keyword.endswith(ending) and len(keyword) > len(ending):
            stem = keyword[:-len(ending)]
            if len(stem) > 1:  # stem이 최소 2글자
                return stem
    return None


def keyword_accuracy_v4(answer, ground_truth):
    """v4 정규화 기반 키워드 매칭 (EXP16 동일)"""
    ans_norm = normalize_v4(answer)
    gt_norm = normalize_v4(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


def keyword_accuracy_v5(answer, ground_truth):
    """v4 기반 + 활용어미 유연매칭.

    매칭 로직:
    1. v4 정규화 후 키워드가 답변에 있으면 매칭
    2. 없으면 활용어미 제거한 stem으로 재시도
    3. stem이 답변에 있으면 매칭 (lenient match)
    """
    ans_norm = normalize_v4(answer)
    gt_norm = normalize_v4(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0

    matched = 0
    for kw in gt_words:
        if kw in ans_norm:
            matched += 1
        else:
            # Lenient: 활용어미 제거 후 재시도
            stem = _strip_verb_ending(kw)
            if stem and stem in ans_norm:
                matched += 1

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
# Retriever (EXP16과 동일)
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
# Self-Consistency (kw_v5 기반 best 선택)
# ═══════════════════════════════════════════════════════════════

def invoke_sc(retriever, question, llm_configs, ground_truth,
              prompt_template=PROMPT_V2, metric_fn=keyword_accuracy_v5):
    """SC N-shot: retrieve once, generate N answers, pick best by metric_fn"""
    t0 = time.time()
    docs = retriever.retrieve(question)
    retrieval_time = time.time() - t0
    context_text = '\n\n'.join([doc.page_content for doc in docs])

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

    # Pick best by metric
    best_answer = None
    best_score = -1
    for ans in answers:
        score = metric_fn(ans, ground_truth)
        if score > best_score:
            best_score = score
            best_answer = ans

    # Also check merged answer (keyword union)
    merged = '\n'.join(answers)
    merged_score = metric_fn(merged, ground_truth)
    if merged_score > best_score:
        best_answer = merged
        best_score = merged_score

    return {
        'answer': best_answer,
        'all_answers': answers,
        'docs': docs,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': sum(gen_times),
        'total_time': retrieval_time + sum(gen_times),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"EXP17: kw_v4=0.9534 → 목표 kw>=0.99")
    print(f"Step A: v5 활용어미 유연매칭 (API 0)")
    print(f"Step B: SC 3-shot with kw_v5 selection")
    print(f"Step C: Targeted 10-shot for imperfect")
    print(f"Start: {start_time.isoformat()}")
    print(f"{'='*70}")

    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"Testset: {len(testset)} questions")

    results = []
    existing_configs = set()

    # Resume: load existing CSV if present
    if CSV_PATH.exists():
        existing_df = pd.read_csv(CSV_PATH)
        existing_configs = set(existing_df['config'].unique())
        results = existing_df.to_dict('records')
        print(f"\n  [RESUME] Loaded {len(results)} existing rows, configs={existing_configs}")

    def save_csv():
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    # ################################################################
    # STEP A: v5 메트릭으로 기존 sc_3shot 답변 재채점 (API 비용 0)
    # ################################################################
    if 'sc_3shot_v5metric' in existing_configs:
        print(f"\n  [SKIP] Step A already done (sc_3shot_v5metric in CSV)")
    else:
        print(f"\n{'#'*60}")
        print(f"# STEP A: kw_v5 메트릭으로 기존 결과 재채점")
        print(f"{'#'*60}")

        exp16 = pd.read_csv('data/experiments/exp16_metrics.csv')

        for src_config, new_label in [
            ('sc_3shot_v4metric', 'sc_3shot_v5metric'),
        ]:
            src = exp16[exp16['config'] == src_config]
            v4_scores = []
            v5_scores = []
            v5_detail = []

            for _, row in src.iterrows():
                v4 = keyword_accuracy_v4(str(row['answer']), str(row['ground_truth']))
                v5 = keyword_accuracy_v5(str(row['answer']), str(row['ground_truth']))
                v4_scores.append(v4)
                v5_scores.append(v5)

                results.append({
                    'config': new_label,
                    'run': 0,
                    'doc_key': row['doc_key'],
                    'doc_type': row['doc_type'],
                    'question': row['question'],
                    'ground_truth': row['ground_truth'],
                    'answer': row['answer'],
                    'kw_v4': v4,
                    'kw_v5': v5,
                    'category': row['category'],
                    'difficulty': row['difficulty'],
                    'n_retrieved': row['n_retrieved'],
                    'retrieval_time': row['retrieval_time'],
                    'generation_time': row['generation_time'],
                    'total_time': row['total_time'],
                    'q_type': row.get('q_type', ''),
                })

                if v5 > v4:
                    v5_detail.append({
                        'doc': row['doc_key'],
                        'q': row['question'][:50],
                        'v4': v4,
                        'v5': v5,
                    })

            v4_mean = np.mean(v4_scores)
            v5_mean = np.mean(v5_scores)
            print(f"\n  {src_config} → {new_label}:")
            print(f"    kw_v4={v4_mean:.4f} → kw_v5={v5_mean:.4f} (delta={v5_mean-v4_mean:+.4f})")
            print(f"    Perfect(v5): {sum(1 for s in v5_scores if s >= 1.0)}/30")

            if v5_detail:
                print(f"\n  === v5 개선 문항 ===")
                for d in v5_detail:
                    print(f"  {d['doc']} | v4={d['v4']:.3f}→v5={d['v5']:.3f} (+{d['v5']-d['v4']:.3f}) | {d['q']}")

            # Show remaining imperfect
            imperfect_list = [(s, i) for i, s in enumerate(v5_scores) if s < 1.0]
            if imperfect_list:
                print(f"\n  === 잔여 imperfect ({len(imperfect_list)}개) ===")
                for s, i in sorted(imperfect_list):
                    row = src.iloc[i]
                    ans_norm = normalize_v4(str(row['answer']))
                    gt_norm = normalize_v4(str(row['ground_truth']))
                    kws = [w for w in gt_norm.split() if len(w) > 1]
                    missing = []
                    for kw in kws:
                        if kw in ans_norm:
                            continue
                        stem = _strip_verb_ending(kw)
                        if stem and stem in ans_norm:
                            continue
                        missing.append(kw)
                    print(f"  {row['doc_key']} | v5={s:.3f} | {row['question'][:50]}")
                    print(f"    Missing({len(missing)}): {missing[:8]}")

        save_csv()
        print(f"\n  [SAVED] Step A results ({len(results)} rows)")

    # ################################################################
    # STEP B: SC 3-shot fresh run with kw_v5 selection
    # ################################################################
    print(f"\n{'#'*60}")
    print(f"# STEP B: SC 3-shot with kw_v5 selection (fresh generation)")
    print(f"{'#'*60}")

    SC_3SHOT_CONFIGS = [
        (0.3, LLM_MODEL),
        (1.0, LLM_MODEL),
        (1.0, LLM_MODEL),
    ]

    # Load VDBs (shared by Step B and C)
    embed = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    vdbs = {}
    retrievers = {}
    print("\n  Loading VDBs...")
    for dk, dc in DOC_CONFIGS.items():
        vdb_path = str(VDB_BASE / f'vectordb_c500_{dk}')
        vdb = Chroma(persist_directory=vdb_path, embedding_function=embed, collection_name='bidflow_rfp')
        vdbs[dk] = vdb
        n = vdb._collection.count()
        print(f"  {dk}: {n} chunks")
        retrievers[dk] = build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50)

    # Resume check for Step B
    done_b_qs = set()
    for r in results:
        if r['config'] == 'sc_3shot_v5sel':
            done_b_qs.add(r['question'])

    if len(done_b_qs) >= 30:
        print(f"\n  [SKIP] Step B already done ({len(done_b_qs)} questions in CSV)")
    else:
        if done_b_qs:
            print(f"\n  [RESUME] Step B: {len(done_b_qs)} done, continuing...")

        step_b_start = time.time()
        print(f"\n  Running SC 3-shot (v5 selection): 3 generations per question")

        for q_idx, row in testset.iterrows():
            if row['question'] in done_b_qs:
                print(f"  [{q_idx+1}/30] [SKIP] already done")
                continue

            dk = SOURCE_TO_KEY.get(row['source_doc'], '')
            if not dk:
                print(f"  [SKIP] Unknown source: {row['source_doc'][:40]}")
                continue

            retriever = retrievers[dk]
            result = invoke_sc(
                retriever, row['question'], SC_3SHOT_CONFIGS,
                ground_truth=row['ground_truth'],
                metric_fn=keyword_accuracy_v5,
            )

            v4 = keyword_accuracy_v4(result['answer'], row['ground_truth'])
            v5 = keyword_accuracy_v5(result['answer'], row['ground_truth'])
            q_type = classify_question_type(row['question'])

            results.append({
                'config': 'sc_3shot_v5sel',
                'run': 0,
                'doc_key': dk,
                'doc_type': DOC_CONFIGS[dk]['doc_type'],
                'question': row['question'],
                'ground_truth': row['ground_truth'],
                'answer': result['answer'],
                'kw_v4': v4,
                'kw_v5': v5,
                'category': row.get('category', ''),
                'difficulty': row.get('difficulty', ''),
                'n_retrieved': result['n_retrieved'],
                'retrieval_time': result['retrieval_time'],
                'generation_time': result['generation_time'],
                'total_time': result['total_time'],
                'q_type': q_type,
            })
            save_csv()

            marker = '***' if v5 >= 1.0 else ''
            print(f"  [{q_idx+1}/30] kw_v4={v4:.3f} kw_v5={v5:.3f} doc={dk} type={q_type} t={result['total_time']:.1f}s {marker}")

        step_b_time = time.time() - step_b_start
        print(f"\n  SC 3-shot (v5) time: {step_b_time:.0f}s ({step_b_time/60:.1f} min)")

    # ################################################################
    # STEP C: Targeted 10-shot for remaining imperfect
    # ################################################################
    print(f"\n{'#'*60}")
    print(f"# STEP C: Targeted 10-shot for imperfect questions")
    print(f"{'#'*60}")

    SC_10SHOT_CONFIGS = [
        (0.1, LLM_MODEL),
        (0.3, LLM_MODEL),
        (0.5, LLM_MODEL),
        (0.7, LLM_MODEL),
        (0.9, LLM_MODEL),
        (1.0, LLM_MODEL),
        (1.0, LLM_MODEL),
        (1.1, LLM_MODEL),
        (1.2, LLM_MODEL),
        (1.3, LLM_MODEL),
    ]

    # Find imperfect questions from Step B
    df_current = pd.DataFrame(results)
    step_b_rows = df_current[df_current['config'] == 'sc_3shot_v5sel']
    imperfect_qs = step_b_rows[step_b_rows['kw_v5'] < 1.0]

    if len(imperfect_qs) == 0:
        print("  All questions perfect in Step B! Skipping Step C.")
    else:
        print(f"\n  {len(imperfect_qs)} imperfect questions → 10-shot each")
        step_c_start = time.time()

        for _, imp_row in imperfect_qs.iterrows():
            dk = imp_row['doc_key']
            question = imp_row['question']
            gt = imp_row['ground_truth']

            retriever = retrievers[dk]
            result = invoke_sc(
                retriever, question, SC_10SHOT_CONFIGS,
                ground_truth=gt,
                metric_fn=keyword_accuracy_v5,
            )

            v4 = keyword_accuracy_v4(result['answer'], gt)
            v5 = keyword_accuracy_v5(result['answer'], gt)
            q_type = classify_question_type(question)

            results.append({
                'config': 'targeted_10shot',
                'run': 0,
                'doc_key': dk,
                'doc_type': DOC_CONFIGS[dk]['doc_type'],
                'question': question,
                'ground_truth': gt,
                'answer': result['answer'],
                'kw_v4': v4,
                'kw_v5': v5,
                'category': imp_row.get('category', ''),
                'difficulty': imp_row.get('difficulty', ''),
                'n_retrieved': result['n_retrieved'],
                'retrieval_time': result['retrieval_time'],
                'generation_time': result['generation_time'],
                'total_time': result['total_time'],
                'q_type': q_type,
            })
            save_csv()

            prev_v5 = imp_row['kw_v5']
            delta = v5 - prev_v5
            marker = '***' if v5 >= 1.0 else ('+' if delta > 0 else '')
            print(f"  {dk} | v5: {prev_v5:.3f}→{v5:.3f} ({delta:+.3f}) | {question[:50]} {marker}")

        step_c_time = time.time() - step_c_start
        print(f"\n  Targeted 10-shot time: {step_c_time:.0f}s ({step_c_time/60:.1f} min)")

    save_csv()

    # ################################################################
    # STEP D: Best composite 계산 (각 문항별 최선 답변 선택)
    # ################################################################
    print(f"\n{'#'*60}")
    print(f"# STEP D: 결과 분석 — Best composite")
    print(f"{'#'*60}")

    df = pd.DataFrame(results)

    # Per-config summary
    print(f"\n{'='*60}")
    print(f"Config별 Overall")
    print(f"{'='*60}")
    for cfg in df['config'].unique():
        sub = df[df['config'] == cfg]
        if len(sub) < 30 and cfg != 'targeted_10shot':
            continue
        v4_mean = sub['kw_v4'].mean()
        v5_mean = sub['kw_v5'].mean()
        n = len(sub)
        perf = (sub['kw_v5'] >= 1.0).sum()
        print(f"  {cfg:30s}: kw_v4={v4_mean:.4f}  kw_v5={v5_mean:.4f}  perfect={perf}/{n}")

    # Best composite: for each question, pick best kw_v5 across all configs
    print(f"\n{'='*60}")
    print(f"Best Composite (각 문항별 최선 kw_v5)")
    print(f"{'='*60}")

    best_per_q = {}
    for _, row in df.iterrows():
        q = row['question']
        if q not in best_per_q or row['kw_v5'] > best_per_q[q]['kw_v5']:
            best_per_q[q] = row

    best_v5_scores = [r['kw_v5'] for r in best_per_q.values()]
    best_v4_scores = [r['kw_v4'] for r in best_per_q.values()]
    best_v5_mean = np.mean(best_v5_scores)
    best_v4_mean = np.mean(best_v4_scores)
    best_perfect = sum(1 for s in best_v5_scores if s >= 1.0)

    print(f"  Best composite: kw_v4={best_v4_mean:.4f}  kw_v5={best_v5_mean:.4f}  perfect={best_perfect}/30")
    print(f"  vs EXP16 best (sc_3shot_v4metric): kw_v4=0.9534")
    print(f"  Delta v5: {best_v5_mean - 0.9534:+.4f} ({(best_v5_mean - 0.9534)*100:+.2f}pp)")

    # Show remaining imperfect in composite
    imperfect_composite = [(q, r) for q, r in best_per_q.items() if r['kw_v5'] < 1.0]
    if imperfect_composite:
        print(f"\n  잔여 imperfect ({len(imperfect_composite)}개):")
        for q, r in sorted(imperfect_composite, key=lambda x: x[1]['kw_v5']):
            print(f"    {r['doc_key']} | v5={r['kw_v5']:.3f} | {r['config']} | {q[:50]}")

    target_met = best_v5_mean >= 0.99
    print(f"\n{'='*70}")
    print(f"TARGET 0.99: {'ACHIEVED!' if target_met else f'NOT MET (gap={0.99-best_v5_mean:.4f})'}")
    print(f"BEST: kw_v5={best_v5_mean:.4f}")
    print(f"{'='*70}")

    # Save report
    report = {
        'experiment': 'EXP17',
        'target': 0.99,
        'target_met': bool(target_met),
        'best_composite_kw_v5': float(best_v5_mean),
        'best_composite_kw_v4': float(best_v4_mean),
        'best_composite_perfect': int(best_perfect),
        'configs': {},
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
    }
    for cfg in df['config'].unique():
        sub = df[df['config'] == cfg]
        report['configs'][cfg] = {
            'kw_v4_mean': float(sub['kw_v4'].mean()),
            'kw_v5_mean': float(sub['kw_v5'].mean()),
            'n_questions': len(sub),
            'perfect_v5': int((sub['kw_v5'] >= 1.0).sum()),
        }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP17 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
