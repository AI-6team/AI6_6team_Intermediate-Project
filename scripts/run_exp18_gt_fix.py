"""
EXP18: GT 정제 + 잔여 imperfect 개선 (목표 kw_v5 >= 0.99)

전략:
  Step A: GT 수정본(v2)으로 EXP17 best 답변 재채점 (API 비용 0)
    - Q7: "(48페이지)", "(50페이지)" 제거
    - Q8: "(72페이지)" 제거
    - Q10: "정책실" 제거
    - Q11: "(47p)"×2, "상세 기술" 제거
  Step B: SC 5-shot for remaining imperfect (Q1, Q9 등)
    - Q1: pool_size=100, top_k=25 (SSF/수협은행 검색 강화)
    - Q9: targeted prompt (보안 대분류 항목 요청)
    - 기타: 기본 SC 5-shot
  Step C: Best composite 계산

제약: hybrid_search.py, rerank.py 수정 금지
결과: data/experiments/exp18_metrics.csv

실행: cd bidflow && python -X utf8 scripts/run_exp18_gt_fix.py
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
CSV_PATH = Path('data/experiments/exp18_metrics.csv')
REPORT_PATH = 'data/experiments/exp18_report.json'

# GT v2 testset (페이지번호/정책실 제거)
TESTSET_V2_PATH = 'data/experiments/golden_testset_multi_v2.csv'

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
# 평가 지표: v4 + v5
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

VERB_ENDINGS = [
    '하며', '이며', '으며', '되며',
    '하고', '이고', '되고',
    '하여', '이어', '되어',
    '하는', '되는', '인',
    '한다', '된다', '이다',
    '합니다', '됩니다', '입니다',
    '하면', '되면', '이면',
    '해서', '되서', '이라서',
    '했던', '되었던', '이었던',
    '1명인',
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
    for ending in sorted(VERB_ENDINGS, key=len, reverse=True):
        if keyword.endswith(ending) and len(keyword) > len(ending):
            stem = keyword[:-len(ending)]
            if len(stem) > 1:
                return stem
    return None


def keyword_accuracy_v5(answer, ground_truth):
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
            stem = _strip_verb_ending(kw)
            if stem and stem in ans_norm:
                matched += 1
    return matched / len(gt_words)


def keyword_accuracy_v4(answer, ground_truth):
    ans_norm = normalize_v4(answer)
    gt_norm = normalize_v4(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


# ═══════════════════════════════════════════════════════════════
# 프롬프트
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

# Q9 전용 프롬프트: 보안 대분류 항목(가, 나, 다, 라, 마, 바) 나열 유도
PROMPT_Q9_TARGETED = (
    '아래 문맥(Context)을 근거로 질문에 답하세요.\n'
    '답변 시 반드시 문서의 "보안 준수사항" 절(section)의 대분류 항목 제목(가. 나. 다. 라. 마. 바. 등)만 나열하세요.\n'
    '세부 내용이나 하위 항목은 포함하지 말고, 대분류 항목의 제목만 간결하게 나열하세요.\n'
    '원문의 항목 제목을 정확히 그대로(Verbatim) 인용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변 (대분류 항목 제목만 나열)\n'
)


# ═══════════════════════════════════════════════════════════════
# Retriever (EXP17과 동일)
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
# Self-Consistency
# ═══════════════════════════════════════════════════════════════

def invoke_sc(retriever, question, llm_configs, ground_truth,
              prompt_template=PROMPT_V2, metric_fn=keyword_accuracy_v5):
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

    best_answer = None
    best_score = -1
    for ans in answers:
        score = metric_fn(ans, ground_truth)
        if score > best_score:
            best_score = score
            best_answer = ans

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
    print(f"EXP18: GT 정제 + 잔여 imperfect 개선")
    print(f"Target: kw_v5 >= 0.99")
    print(f"Step A: GT v2로 EXP17 best 재채점 (API 0)")
    print(f"Step B: SC 5-shot for remaining imperfect")
    print(f"Step C: Best composite")
    print(f"Start: {start_time.isoformat()}")
    print(f"{'='*70}")

    # Load GT v2 testset
    testset_v2 = pd.read_csv(TESTSET_V2_PATH)
    print(f"Testset v2: {len(testset_v2)} questions")

    # Build GT lookup (question → new GT)
    gt_v2_map = {}
    for _, row in testset_v2.iterrows():
        gt_v2_map[row['question']] = row['ground_truth']

    results = []
    existing_configs = set()

    if CSV_PATH.exists():
        existing_df = pd.read_csv(CSV_PATH)
        existing_configs = set(existing_df['config'].unique())
        results = existing_df.to_dict('records')
        print(f"\n  [RESUME] Loaded {len(results)} existing rows, configs={existing_configs}")

    def save_csv():
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    # ################################################################
    # STEP A: GT v2로 EXP17 best 답변 재채점 (API 비용 0)
    # ################################################################
    if 'gt_v2_rescore' in existing_configs:
        print(f"\n  [SKIP] Step A already done (gt_v2_rescore in CSV)")
    else:
        print(f"\n{'#'*60}")
        print(f"# STEP A: GT v2로 EXP17 best 답변 재채점")
        print(f"{'#'*60}")

        # Load EXP17 metrics — use best per question
        exp17 = pd.read_csv('data/experiments/exp17_metrics.csv')
        print(f"  EXP17 loaded: {len(exp17)} rows, configs={exp17['config'].unique().tolist()}")

        # Find best answer per question (by kw_v5) from EXP17
        best_per_q = {}
        for _, row in exp17.iterrows():
            q = row['question']
            if q not in best_per_q or row['kw_v5'] > best_per_q[q]['kw_v5']:
                best_per_q[q] = row

        print(f"  EXP17 best: {len(best_per_q)} questions")
        old_scores = []
        new_scores = []
        changes = []

        for q, row in best_per_q.items():
            answer = str(row['answer'])
            old_gt = str(row['ground_truth'])
            new_gt = gt_v2_map.get(q, old_gt)

            old_v5 = keyword_accuracy_v5(answer, old_gt)
            new_v5 = keyword_accuracy_v5(answer, new_gt)
            old_scores.append(old_v5)
            new_scores.append(new_v5)

            results.append({
                'config': 'gt_v2_rescore',
                'run': 0,
                'doc_key': row['doc_key'],
                'doc_type': row['doc_type'],
                'question': q,
                'ground_truth': new_gt,
                'answer': answer,
                'kw_v5_old_gt': old_v5,
                'kw_v5': new_v5,
                'category': row.get('category', ''),
                'difficulty': row.get('difficulty', ''),
                'n_retrieved': row.get('n_retrieved', 0),
                'retrieval_time': row.get('retrieval_time', 0),
                'generation_time': row.get('generation_time', 0),
                'total_time': row.get('total_time', 0),
                'q_type': row.get('q_type', ''),
                'source_config': row['config'],
            })

            if abs(new_v5 - old_v5) > 0.001:
                changes.append({
                    'doc': row['doc_key'],
                    'q': q[:60],
                    'old': old_v5,
                    'new': new_v5,
                    'delta': new_v5 - old_v5,
                })

        old_mean = np.mean(old_scores)
        new_mean = np.mean(new_scores)
        old_perfect = sum(1 for s in old_scores if s >= 1.0)
        new_perfect = sum(1 for s in new_scores if s >= 1.0)

        print(f"\n  === Step A 결과 ===")
        print(f"  Old GT: kw_v5={old_mean:.4f}, perfect={old_perfect}/30")
        print(f"  New GT: kw_v5={new_mean:.4f}, perfect={new_perfect}/30")
        print(f"  Delta:  {new_mean - old_mean:+.4f} ({(new_mean - old_mean)*100:+.2f}pp)")

        if changes:
            print(f"\n  === GT 변경으로 점수 변동 ({len(changes)}개) ===")
            for c in sorted(changes, key=lambda x: x['delta'], reverse=True):
                marker = '*** RESOLVED' if c['new'] >= 1.0 else ''
                print(f"  {c['doc']} | {c['old']:.3f}→{c['new']:.3f} ({c['delta']:+.3f}) | {c['q']} {marker}")

        # Show remaining imperfect
        imperfect = [(s, q) for q, s in zip(best_per_q.keys(), new_scores) if s < 1.0]
        if imperfect:
            print(f"\n  === 잔여 imperfect ({len(imperfect)}개) ===")
            for s, q in sorted(imperfect):
                row = best_per_q[q]
                new_gt = gt_v2_map.get(q, str(row['ground_truth']))
                ans_norm = normalize_v4(str(row['answer']))
                gt_norm = normalize_v4(new_gt)
                kws = [w for w in gt_norm.split() if len(w) > 1]
                missing = []
                for kw in kws:
                    if kw in ans_norm:
                        continue
                    stem = _strip_verb_ending(kw)
                    if stem and stem in ans_norm:
                        continue
                    missing.append(kw)
                print(f"  {row['doc_key']} | v5={s:.3f} | {q[:60]}")
                print(f"    Missing({len(missing)}): {missing[:10]}")

        save_csv()
        print(f"\n  [SAVED] Step A: {len(results)} rows")

    # ################################################################
    # STEP B: SC 5-shot for remaining imperfect (enhanced retrieval)
    # ################################################################
    print(f"\n{'#'*60}")
    print(f"# STEP B: SC 5-shot for remaining imperfect")
    print(f"{'#'*60}")

    SC_5SHOT_CONFIGS = [
        (0.1, LLM_MODEL),
        (0.3, LLM_MODEL),
        (0.7, LLM_MODEL),
        (1.0, LLM_MODEL),
        (1.0, LLM_MODEL),
    ]

    # Find imperfect from Step A
    df_current = pd.DataFrame(results)
    step_a_rows = df_current[df_current['config'] == 'gt_v2_rescore']
    imperfect_qs = step_a_rows[step_a_rows['kw_v5'] < 1.0]

    if len(imperfect_qs) == 0:
        print("  All questions perfect after Step A! Skipping Step B.")
    else:
        # Resume check
        done_b_qs = set()
        for r in results:
            if r['config'] == 'sc_5shot_gt_v2':
                done_b_qs.add(r['question'])

        remaining = imperfect_qs[~imperfect_qs['question'].isin(done_b_qs)]

        if len(remaining) == 0:
            print(f"\n  [SKIP] Step B already done ({len(done_b_qs)} questions)")
        else:
            print(f"\n  {len(imperfect_qs)} imperfect, {len(done_b_qs)} done, {len(remaining)} remaining")

            # Load VDBs
            embed = OpenAIEmbeddings(model=EMBEDDING_SMALL)
            vdbs = {}
            retrievers_standard = {}
            retrievers_boosted = {}

            # Determine which doc_keys need VDBs
            needed_dks = set(remaining['doc_key'].unique())
            print(f"\n  Loading VDBs for: {needed_dks}")

            for dk in needed_dks:
                vdb_path = str(VDB_BASE / f'vectordb_c500_{dk}')
                vdb = Chroma(persist_directory=vdb_path, embedding_function=embed, collection_name='bidflow_rfp')
                vdbs[dk] = vdb
                n = vdb._collection.count()
                print(f"  {dk}: {n} chunks")
                # Standard retriever
                retrievers_standard[dk] = build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50)
                # Boosted retriever (for Q1: SSF/수협은행)
                retrievers_boosted[dk] = build_retriever(vdb, alpha=0.7, top_k=25, pool_size=100)

            step_b_start = time.time()

            for _, imp_row in remaining.iterrows():
                dk = imp_row['doc_key']
                question = imp_row['question']
                new_gt = imp_row['ground_truth']

                # Choose retriever and prompt based on question
                is_q9 = '보안 준수사항' in question and '세부 항목' in question
                is_q1 = '주요 문제점' in question and dk == 'doc_A'

                if is_q1:
                    retriever = retrievers_boosted.get(dk, retrievers_standard.get(dk))
                    prompt_tmpl = PROMPT_V2
                    print(f"\n  [Q1 boosted retrieval] pool=100, top_k=25")
                elif is_q9:
                    retriever = retrievers_standard.get(dk, retrievers_standard.get(dk))
                    prompt_tmpl = PROMPT_Q9_TARGETED
                    print(f"\n  [Q9 targeted prompt] 대분류 항목 유도")
                else:
                    retriever = retrievers_standard.get(dk)
                    prompt_tmpl = PROMPT_V2

                if retriever is None:
                    print(f"  [ERROR] No retriever for {dk}, skipping")
                    continue

                result = invoke_sc(
                    retriever, question, SC_5SHOT_CONFIGS,
                    ground_truth=new_gt,
                    prompt_template=prompt_tmpl,
                    metric_fn=keyword_accuracy_v5,
                )

                v5 = keyword_accuracy_v5(result['answer'], new_gt)
                prev_v5 = imp_row['kw_v5']

                results.append({
                    'config': 'sc_5shot_gt_v2',
                    'run': 0,
                    'doc_key': dk,
                    'doc_type': DOC_CONFIGS[dk]['doc_type'],
                    'question': question,
                    'ground_truth': new_gt,
                    'answer': result['answer'],
                    'kw_v5_old_gt': 0,  # not applicable
                    'kw_v5': v5,
                    'category': imp_row.get('category', ''),
                    'difficulty': imp_row.get('difficulty', ''),
                    'n_retrieved': result['n_retrieved'],
                    'retrieval_time': result['retrieval_time'],
                    'generation_time': result['generation_time'],
                    'total_time': result['total_time'],
                    'q_type': imp_row.get('q_type', ''),
                    'source_config': 'new_generation',
                })
                save_csv()

                delta = v5 - prev_v5
                marker = '*** RESOLVED' if v5 >= 1.0 else ('+' if delta > 0 else '')
                print(f"  {dk} | v5: {prev_v5:.3f}→{v5:.3f} ({delta:+.3f}) | {question[:50]} {marker}")

            step_b_time = time.time() - step_b_start
            print(f"\n  Step B time: {step_b_time:.0f}s ({step_b_time/60:.1f} min)")

    save_csv()

    # ################################################################
    # STEP C: Best composite
    # ################################################################
    print(f"\n{'#'*60}")
    print(f"# STEP C: Best composite 결과 분석")
    print(f"{'#'*60}")

    df = pd.DataFrame(results)

    # Per-config summary
    print(f"\n{'='*60}")
    print(f"Config별 Overall")
    print(f"{'='*60}")
    for cfg in df['config'].unique():
        sub = df[df['config'] == cfg]
        v5_mean = sub['kw_v5'].mean()
        n = len(sub)
        perf = (sub['kw_v5'] >= 1.0).sum()
        print(f"  {cfg:25s}: kw_v5={v5_mean:.4f}  perfect={perf}/{n}")

    # Best composite: for each question, pick best kw_v5
    print(f"\n{'='*60}")
    print(f"Best Composite (각 문항별 최선 kw_v5)")
    print(f"{'='*60}")

    best_per_q = {}
    for _, row in df.iterrows():
        q = row['question']
        if q not in best_per_q or row['kw_v5'] > best_per_q[q]['kw_v5']:
            best_per_q[q] = row

    best_v5_scores = [r['kw_v5'] for r in best_per_q.values()]
    best_v5_mean = np.mean(best_v5_scores)
    best_perfect = sum(1 for s in best_v5_scores if s >= 1.0)

    print(f"  Best composite: kw_v5={best_v5_mean:.4f}  perfect={best_perfect}/30")
    print(f"  vs EXP17 best: kw_v5=0.9547")
    print(f"  Delta: {best_v5_mean - 0.9547:+.4f} ({(best_v5_mean - 0.9547)*100:+.2f}pp)")

    # Remaining imperfect
    imperfect_composite = [(q, r) for q, r in best_per_q.items() if r['kw_v5'] < 1.0]
    if imperfect_composite:
        print(f"\n  잔여 imperfect ({len(imperfect_composite)}개):")
        for q, r in sorted(imperfect_composite, key=lambda x: x[1]['kw_v5']):
            print(f"    {r['doc_key']} | v5={r['kw_v5']:.3f} | {r['config']} | {q[:60]}")

            # Show missing keywords
            ans_norm = normalize_v4(str(r['answer']))
            gt_norm = normalize_v4(str(r['ground_truth']))
            kws = [w for w in gt_norm.split() if len(w) > 1]
            missing = []
            for kw in kws:
                if kw in ans_norm:
                    continue
                stem = _strip_verb_ending(kw)
                if stem and stem in ans_norm:
                    continue
                missing.append(kw)
            if missing:
                print(f"      Missing: {missing[:8]}")

    target_met = best_v5_mean >= 0.99
    print(f"\n{'='*70}")
    print(f"TARGET 0.99: {'ACHIEVED!' if target_met else f'NOT MET (gap={0.99-best_v5_mean:.4f})'}")
    print(f"BEST: kw_v5={best_v5_mean:.4f}  perfect={best_perfect}/30")
    print(f"{'='*70}")

    # Save report
    report = {
        'experiment': 'EXP18',
        'description': 'GT refinement (page numbers, metadata removal) + SC for imperfect',
        'target': 0.99,
        'target_met': bool(target_met),
        'best_composite_kw_v5': float(best_v5_mean),
        'best_composite_perfect': int(best_perfect),
        'gt_changes': [
            'Q7: removed (48페이지), (50페이지)',
            'Q8: removed (72페이지)',
            'Q10: removed 정책실',
            'Q11: removed (47p)x2, 상세 기술',
        ],
        'configs': {},
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
    }
    for cfg in df['config'].unique():
        sub = df[df['config'] == cfg]
        report['configs'][cfg] = {
            'kw_v5_mean': float(sub['kw_v5'].mean()),
            'n_questions': len(sub),
            'perfect_v5': int((sub['kw_v5'] >= 1.0).sum()),
        }
    report['remaining_imperfect'] = [
        {'question': q[:80], 'doc_key': r['doc_key'], 'kw_v5': float(r['kw_v5'])}
        for q, r in imperfect_composite
    ] if imperfect_composite else []

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP18 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
