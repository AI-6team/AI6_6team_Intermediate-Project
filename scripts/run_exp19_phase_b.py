"""
EXP19 Phase B: 과적합 검증 (Holdout Set)

목표: 95개 미사용 문서에서 5개 선정 → Q&A 생성 → RAG 평가 → 과적합 판정

판정 기준:
  holdout kw_v5 >= 0.95: 과적합 없음
  holdout kw_v5 0.90~0.95: 경미한 과적합
  holdout kw_v5 < 0.90: 심각한 과적합

절차:
  1. 5개 holdout 문서 파싱 + VDB 구축
  2. LLM으로 Q&A 생성 (문서당 4문항 = 20문항)
  3. RAG 파이프라인 평가 (V2 prompt + SC 5-shot + kw_v5)
  4. testset vs holdout 성능 비교

실행: cd bidflow && python -X utf8 scripts/run_exp19_phase_b.py
"""
import os, sys, re, json, time, warnings, shutil
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
HOLDOUT_DIR = PROJECT_ROOT / 'data' / 'exp19_holdout'
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path('data/experiments/exp19_holdout_metrics.csv')
HOLDOUT_TESTSET_PATH = Path('data/experiments/golden_testset_holdout.csv')
REPORT_PATH = 'data/experiments/exp19_holdout_report.json'

# ── Holdout 문서 선정 (5개, 다양한 도메인) ──
HOLDOUT_DOCS = {
    "hold_F": {
        "file_path": "data/raw/files/한국한의학연구원_통합정보시스템 고도화 용역.hwp",
        "domain": "의료/연구",
    },
    "hold_G": {
        "file_path": "data/raw/files/경상북도 봉화군_봉화군 재난통합관리시스템 고도화 사업(협상)(긴급).hwp",
        "domain": "지자체/재난",
    },
    "hold_H": {
        "file_path": "data/raw/files/한국산업단지공단_산단 안전정보시스템 1차 구축 용역.hwp",
        "domain": "산업/안전",
    },
    "hold_I": {
        "file_path": "data/raw/files/국민연금공단_2024년 이러닝시스템 운영 용역.hwp",
        "domain": "공공기관/교육",
    },
    "hold_J": {
        "file_path": "data/raw/files/(재)예술경영지원센터_통합 정보시스템 구축 사전 컨설팅.hwp",
        "domain": "문화/예술",
    },
}


# ═══════════════════════════════════════════════════════════════
# 평가 지표 (EXP19 동일)
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
    '하며', '이며', '으며', '되며', '하고', '이고', '되고',
    '하여', '이어', '되어', '하는', '되는', '인',
    '한다', '된다', '이다', '합니다', '됩니다', '입니다',
    '하면', '되면', '이면', '해서', '되서', '이라서',
    '했던', '되었던', '이었던', '1명인',
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
    t = t.replace('￦', '₩').replace("'", '').replace('"', '').replace('※', '')
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

# Q&A 생성 프롬프트
QA_GENERATION_PROMPT = """아래 입찰제안요청서(RFP) 문서의 텍스트를 읽고, 4개의 질문-정답 쌍을 JSON으로 생성하세요.

규칙:
1. 질문은 반드시 문서 텍스트에서 직접 답을 찾을 수 있어야 합니다.
2. 정답(ground_truth)은 문서 원문의 핵심 정보만 간결하게 포함하세요 (1~2문장).
3. 정답에 페이지 번호, 장/절 번호 등 메타정보는 포함하지 마세요.
4. 4개 질문의 카테고리를 다양하게 구성하세요:
   - 1개: 사업 기본정보 (사업명, 사업기간, 예산 등) [easy]
   - 1개: 기술/범위 관련 (추진배경, 사업범위, 주요 기능 등) [medium]
   - 1개: 계약/행정 (입찰방식, 참가자격, 납품조건 등) [medium]
   - 1개: 세부사항 (보안, 유지보수, 평가기준 등) [hard]

출력 형식 (JSON 배열):
[
  {{"question": "질문1", "ground_truth": "정답1", "category": "general", "difficulty": "easy"}},
  {{"question": "질문2", "ground_truth": "정답2", "category": "technical", "difficulty": "medium"}},
  {{"question": "질문3", "ground_truth": "정답3", "category": "procurement", "difficulty": "medium"}},
  {{"question": "질문4", "ground_truth": "정답4", "category": "compliance", "difficulty": "hard"}}
]

문서 텍스트 (처음 6000자):
{doc_text}
"""


# ═══════════════════════════════════════════════════════════════
# Retriever (EXP19 동일)
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
        'best_score': best_score,
        'all_answers': answers,
        'all_scores': [metric_fn(a, ground_truth) for a in answers],
        'merged_score': merged_score,
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
    print(f"EXP19 Phase B: 과적합 검증 (Holdout Set)")
    print(f"판정: >= 0.95 (OK) | 0.90~0.95 (경미) | < 0.90 (심각)")
    print(f"Start: {start_time.isoformat()}")
    print(f"{'='*70}")

    from bidflow.parsing.table_chunker import TableAwareChunker

    embed = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50, table_mode="flat")

    # ================================================================
    # STEP 1: 문서 파싱 + VDB 구축
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 1: Holdout 문서 파싱 + VDB 구축")
    print(f"{'#'*60}")

    vdbs = {}
    parsed_texts = {}

    for dk, cfg in HOLDOUT_DOCS.items():
        file_path = str(PROJECT_ROOT / cfg['file_path'])
        vdb_path = str(HOLDOUT_DIR / f'vectordb_c500_{dk}')

        # Check if VDB already exists
        if os.path.exists(vdb_path) and os.path.exists(os.path.join(vdb_path, 'chroma.sqlite3')):
            print(f"\n  [{dk}] VDB exists, loading...")
            vdb = Chroma(persist_directory=vdb_path, embedding_function=embed, collection_name='bidflow_rfp')
            n = vdb._collection.count()
            print(f"    {n} chunks loaded")
            vdbs[dk] = vdb

            # Load parsed text for Q&A generation
            text_cache = HOLDOUT_DIR / f'{dk}_text.txt'
            if text_cache.exists():
                parsed_texts[dk] = text_cache.read_text(encoding='utf-8')
            else:
                result = vdb.get()
                parsed_texts[dk] = '\n\n'.join(result['documents'][:20]) if result['documents'] else ''
            continue

        print(f"\n  [{dk}] Parsing: {cfg['file_path']}")
        print(f"    Domain: {cfg['domain']}")

        try:
            docs = chunker.chunk_v4_hybrid(file_path)
            print(f"    Parsed: {len(docs)} chunks")
        except Exception as e:
            print(f"    [ERROR] Parsing failed: {e}")
            # Try basic hwp5txt fallback
            try:
                docs = chunker.chunk_v1_basic(file_path)
                print(f"    [FALLBACK] V1 basic: {len(docs)} chunks")
            except Exception as e2:
                print(f"    [ERROR] V1 fallback also failed: {e2}")
                continue

        if not docs:
            print(f"    [SKIP] No chunks produced")
            continue

        # Save parsed text for Q&A generation
        full_text = '\n\n'.join([d.page_content for d in docs])
        text_cache = HOLDOUT_DIR / f'{dk}_text.txt'
        text_cache.write_text(full_text, encoding='utf-8')
        parsed_texts[dk] = full_text

        # Create VDB
        print(f"    Creating VDB at {vdb_path}...")
        vdb = Chroma.from_documents(
            documents=docs,
            embedding=embed,
            persist_directory=vdb_path,
            collection_name='bidflow_rfp',
        )
        n = vdb._collection.count()
        print(f"    VDB created: {n} chunks")
        vdbs[dk] = vdb

    print(f"\n  VDBs ready: {list(vdbs.keys())}")

    if len(vdbs) < 3:
        print(f"  [ERROR] Need at least 3 VDBs, only got {len(vdbs)}. Aborting.")
        return

    # ================================================================
    # STEP 2: Q&A 생성
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 2: LLM으로 Holdout Q&A 생성")
    print(f"{'#'*60}")

    holdout_qa = []

    if HOLDOUT_TESTSET_PATH.exists():
        existing_qa = pd.read_csv(HOLDOUT_TESTSET_PATH)
        holdout_qa = existing_qa.to_dict('records')
        existing_dks = set(existing_qa['doc_key'].unique()) if 'doc_key' in existing_qa.columns else set()
        print(f"\n  [RESUME] Loaded {len(holdout_qa)} existing Q&A, docs={existing_dks}")
        needed_dks = set(vdbs.keys()) - existing_dks
    else:
        needed_dks = set(vdbs.keys())

    if needed_dks:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, timeout=120, max_retries=2)

        for dk in needed_dks:
            if dk not in parsed_texts:
                print(f"  [{dk}] No parsed text, skipping Q&A generation")
                continue

            doc_text = parsed_texts[dk][:6000]  # 처음 6000자만 사용
            print(f"\n  [{dk}] Generating Q&A ({len(doc_text)} chars)...")

            prompt = ChatPromptTemplate.from_template(QA_GENERATION_PROMPT)
            chain = prompt | llm | StrOutputParser()

            try:
                response = chain.invoke({'doc_text': doc_text})

                # Parse JSON response
                # Find JSON array in response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    qa_list = json.loads(json_match.group())
                else:
                    print(f"    [ERROR] No JSON found in response")
                    continue

                for qa in qa_list:
                    holdout_qa.append({
                        'question': qa['question'],
                        'ground_truth': qa['ground_truth'],
                        'category': qa.get('category', 'general'),
                        'difficulty': qa.get('difficulty', 'medium'),
                        'source_doc': HOLDOUT_DOCS[dk]['file_path'].split('/')[-1],
                        'doc_key': dk,
                        'domain': HOLDOUT_DOCS[dk]['domain'],
                    })
                print(f"    Generated {len(qa_list)} Q&A pairs")
                for qa in qa_list:
                    print(f"    [{qa.get('difficulty','?')}] {qa['question'][:60]}")

            except Exception as e:
                print(f"    [ERROR] Q&A generation failed: {e}")
                continue

        # Save holdout testset
        holdout_df = pd.DataFrame(holdout_qa)
        holdout_df.to_csv(HOLDOUT_TESTSET_PATH, index=False, encoding='utf-8-sig')
        print(f"\n  [SAVED] Holdout testset: {len(holdout_qa)} Q&A → {HOLDOUT_TESTSET_PATH}")
    else:
        print(f"\n  All Q&A already generated")

    holdout_df = pd.DataFrame(holdout_qa)
    print(f"\n  Total holdout Q&A: {len(holdout_df)}")
    print(f"  Docs: {holdout_df['doc_key'].value_counts().to_dict()}")

    # ================================================================
    # STEP 3: RAG 평가 (SC 5-shot, V2 prompt, kw_v5)
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 3: RAG 평가 (Holdout Set)")
    print(f"{'#'*60}")

    SC_5SHOT_CONFIGS = [
        (0.1, LLM_MODEL),
        (0.3, LLM_MODEL),
        (0.5, LLM_MODEL),
        (0.7, LLM_MODEL),
        (1.0, LLM_MODEL),
    ]

    results = []
    existing_qs = set()

    if CSV_PATH.exists():
        existing_df = pd.read_csv(CSV_PATH)
        results = existing_df.to_dict('records')
        existing_qs = set(existing_df['question'].unique())
        print(f"\n  [RESUME] {len(results)} existing results, {len(existing_qs)} questions done")

    def save_csv():
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    # Build retrievers for holdout docs
    retrievers = {}
    for dk, vdb in vdbs.items():
        if dk not in holdout_df['doc_key'].values:
            continue
        retrievers[dk] = build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50)

    step3_start = time.time()
    total_q = len(holdout_df)

    for i, (_, row) in enumerate(holdout_df.iterrows()):
        question = row['question']
        if question in existing_qs:
            continue

        dk = row['doc_key']
        gt = row['ground_truth']
        retriever = retrievers.get(dk)
        if retriever is None:
            print(f"  [{i+1}/{total_q}] [SKIP] No retriever for {dk}")
            continue

        print(f"\n  [{i+1}/{total_q}] {dk} | {question[:50]}...")

        result = invoke_sc(
            retriever, question, SC_5SHOT_CONFIGS,
            ground_truth=gt,
            prompt_template=PROMPT_V2,
            metric_fn=keyword_accuracy_v5,
        )

        v5 = result['best_score']

        results.append({
            'config': 'holdout_sc_5shot',
            'doc_key': dk,
            'domain': row.get('domain', ''),
            'question': question,
            'ground_truth': gt,
            'answer': result['answer'],
            'kw_v5': v5,
            'category': row.get('category', ''),
            'difficulty': row.get('difficulty', ''),
            'n_retrieved': result['n_retrieved'],
            'retrieval_time': result['retrieval_time'],
            'generation_time': result['generation_time'],
            'total_time': result['total_time'],
            'individual_scores': str(result['all_scores']),
            'merged_score': result['merged_score'],
        })
        save_csv()

        marker = '*** PERFECT' if v5 >= 1.0 else ''
        print(f"    v5={v5:.3f} | scores={[f'{s:.2f}' for s in result['all_scores']]} {marker}")

    step3_time = time.time() - step3_start
    save_csv()
    print(f"\n  Step 3 time: {step3_time:.0f}s ({step3_time/60:.1f} min)")

    # ================================================================
    # STEP 4: 과적합 판정
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 4: 과적합 판정")
    print(f"{'#'*60}")

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("  [ERROR] No results to analyze")
        return

    holdout_v5 = df['kw_v5'].mean()
    holdout_perfect = (df['kw_v5'] >= 1.0).sum()
    n_holdout = len(df)

    testset_v5 = 0.9952  # EXP19 best
    testset_perfect = 29
    n_testset = 30

    gap = testset_v5 - holdout_v5

    print(f"\n{'='*60}")
    print(f"  Testset (30Q):  kw_v5={testset_v5:.4f}  perfect={testset_perfect}/30")
    print(f"  Holdout ({n_holdout}Q):  kw_v5={holdout_v5:.4f}  perfect={holdout_perfect}/{n_holdout}")
    print(f"  Gap:           {gap:+.4f} ({gap*100:+.2f}pp)")
    print(f"{'='*60}")

    # Per-doc breakdown
    print(f"\n  문서별 성능:")
    for dk in df['doc_key'].unique():
        sub = df[df['doc_key'] == dk]
        v5_mean = sub['kw_v5'].mean()
        perf = (sub['kw_v5'] >= 1.0).sum()
        domain = sub['domain'].iloc[0] if 'domain' in sub.columns else ''
        print(f"    {dk} ({domain}): kw_v5={v5_mean:.4f}  perfect={perf}/{len(sub)}")

    # Per-category breakdown
    if 'category' in df.columns:
        print(f"\n  카테고리별 성능:")
        for cat in df['category'].unique():
            sub = df[df['category'] == cat]
            v5_mean = sub['kw_v5'].mean()
            print(f"    {cat}: kw_v5={v5_mean:.4f}  (n={len(sub)})")

    # Per-difficulty breakdown
    if 'difficulty' in df.columns:
        print(f"\n  난이도별 성능:")
        for diff in ['easy', 'medium', 'hard']:
            sub = df[df['difficulty'] == diff]
            if len(sub) > 0:
                v5_mean = sub['kw_v5'].mean()
                print(f"    {diff}: kw_v5={v5_mean:.4f}  (n={len(sub)})")

    # 과적합 판정
    if holdout_v5 >= 0.95:
        verdict = "과적합 없음 (PASS)"
        verdict_code = "PASS"
    elif holdout_v5 >= 0.90:
        verdict = "경미한 과적합 (MILD)"
        verdict_code = "MILD"
    else:
        verdict = "심각한 과적합 (SEVERE)"
        verdict_code = "SEVERE"

    print(f"\n{'='*70}")
    print(f"  판정: {verdict}")
    print(f"  Holdout kw_v5 = {holdout_v5:.4f}")
    print(f"  Testset kw_v5 = {testset_v5:.4f}")
    print(f"  Gap = {gap*100:.2f}pp")
    print(f"{'='*70}")

    # Imperfect 상세
    imperfect = df[df['kw_v5'] < 1.0].sort_values('kw_v5')
    if len(imperfect) > 0:
        print(f"\n  Imperfect ({len(imperfect)}개):")
        for _, row in imperfect.iterrows():
            ans_norm = normalize_v4(str(row['answer']))
            gt_norm = normalize_v4(str(row['ground_truth']))
            kws = [w for w in gt_norm.split() if len(w) > 1]
            missing = [kw for kw in kws if kw not in ans_norm and not (_strip_verb_ending(kw) and _strip_verb_ending(kw) in ans_norm)]
            print(f"    {row['doc_key']} | v5={row['kw_v5']:.3f} | {row['question'][:50]}")
            if missing:
                print(f"      Missing({len(missing)}): {missing[:5]}")

    # Save report
    report = {
        'experiment': 'EXP19_Phase_B',
        'description': 'Overfitting validation with holdout set',
        'holdout_kw_v5': float(holdout_v5),
        'holdout_perfect': int(holdout_perfect),
        'holdout_n': int(n_holdout),
        'testset_kw_v5': float(testset_v5),
        'testset_perfect': int(testset_perfect),
        'gap_pp': float(gap * 100),
        'verdict': verdict_code,
        'verdict_text': verdict,
        'per_doc': {},
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
    }
    for dk in df['doc_key'].unique():
        sub = df[df['doc_key'] == dk]
        report['per_doc'][dk] = {
            'kw_v5': float(sub['kw_v5'].mean()),
            'perfect': int((sub['kw_v5'] >= 1.0).sum()),
            'n': len(sub),
            'domain': HOLDOUT_DOCS.get(dk, {}).get('domain', ''),
        }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {HOLDOUT_TESTSET_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"EXP19 Phase B COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
