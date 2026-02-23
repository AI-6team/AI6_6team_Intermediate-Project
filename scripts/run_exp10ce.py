"""
EXP10ce: Phase CE — Parser Switch + Table Gap Resolution
hwp5html 기반 4개 config (V0~V3) 비교 실험

V0: hwp5txt 기존 파서 (baseline, Phase B 재현)
V1: hwp5html basic (전체 텍스트 flat chunking)
V2: hwp5html table (텍스트/테이블 분리 + col_path)
V3: hwp5html full (V2 + 테이블 개별 문서)

실행: cd bidflow && python -u scripts/run_exp10ce.py
"""
import os, sys, time, re, json, warnings, shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not found'

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from typing import List, Any

# ── Constants ──
EXP_DIR = PROJECT_ROOT / 'data' / 'exp10ce'
EXP_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'text-embedding-3-small'
N_RUNS = 3

# ── Document configs ──
DOC_CONFIGS = {
    "doc_A": {
        "name": "수협중앙회 (text_only)",
        "file_path": "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
        "doc_type": "text_only",
        "source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
    },
    "doc_B": {
        "name": "한국교육과정평가원 (table_simple)",
        "file_path": "data/raw/files/한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
        "doc_type": "table_simple",
        "source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
    },
    "doc_C": {
        "name": "국립중앙의료원 (table_complex)",
        "file_path": "data/raw/files/국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
        "doc_type": "table_complex",
        "source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
    },
    "doc_D": {
        "name": "한국철도공사 (mixed)",
        "file_path": "data/raw/files/한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
        "doc_type": "mixed",
        "source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
    },
    "doc_E": {
        "name": "스포츠윤리센터 (hwp_representative)",
        "file_path": "data/raw/files/재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
        "doc_type": "hwp_representative",
        "source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
    },
}
SOURCE_TO_KEY = {v["source_doc"]: k for k, v in DOC_CONFIGS.items()}

# ── kw_v2 helpers (Phase B와 동일) ──
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

def normalize_answer_v2(text):
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

def keyword_accuracy_v2(answer, ground_truth):
    ans_norm = normalize_answer_v2(answer)
    gt_norm = normalize_answer_v2(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)

# ── Retriever (Phase B와 동일) ──
class ExperimentRetriever(BaseRetriever):
    vector_retriever: Any = None
    bm25_retriever: Any = None
    weights: List[float] = [0.3, 0.7]
    top_k: int = 15
    pool_size: int = 50
    use_rerank: bool = True
    rerank_model: str = 'BAAI/bge-reranker-v2-m3'

    def _get_relevant_documents(self, query, *, run_manager):
        search_k = self.pool_size if self.use_rerank else self.top_k
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

        rrf_top = self.pool_size if self.use_rerank else self.top_k
        merged = self._rrf_merge(bm25_docs, vector_docs, k=60, limit=rrf_top)

        if self.use_rerank and merged:
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


def build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50, use_rerank=True):
    vector_retriever = vdb.as_retriever(search_kwargs={'k': pool_size * 2})
    result = vdb.get()
    all_docs = []
    if result and result['documents']:
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            all_docs.append(Document(page_content=text, metadata=meta))
    bm25 = BM25Retriever.from_documents(all_docs) if all_docs else BM25Retriever.from_documents([Document(page_content='empty')])
    bm25.k = pool_size * 2
    return ExperimentRetriever(
        vector_retriever=vector_retriever, bm25_retriever=bm25,
        weights=[round(1 - alpha, 2), round(alpha, 2)],
        top_k=top_k, pool_size=pool_size, use_rerank=use_rerank,
    )


def build_rag_chain(retriever, model_name='gpt-5-mini'):
    temp = 1 if model_name == 'gpt-5-mini' else 0
    llm = ChatOpenAI(model=model_name, temperature=temp, timeout=60, max_retries=2)
    prompt = ChatPromptTemplate.from_template(
        '아래 문맥(Context)만을 근거로 질문에 답하세요.\n'
        '반드시 원문에 있는 사업명, 기관명, 금액, 날짜 등의 표현을 그대로(Verbatim) 사용하세요.\n'
        '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
        '## 문맥 (Context)\n{context}\n\n'
        '## 질문\n{question}\n\n'
        '## 답변\n'
    )

    def invoke_fn(question):
        t0 = time.time()
        docs = retriever.invoke(question)
        retrieval_time = time.time() - t0
        context_text = '\n\n'.join([doc.page_content for doc in docs])
        t1 = time.time()
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context': context_text, 'question': question})
        gen_time = time.time() - t1
        return {
            'answer': answer,
            'retrieved_contexts': [doc.page_content for doc in docs],
            'n_retrieved': len(docs),
            'retrieval_time': retrieval_time,
            'generation_time': gen_time,
            'total_time': retrieval_time + gen_time,
        }
    return invoke_fn


# ══════════════════════════════════════════════════════════
# STEP 1: Indexing — 4 configs × 5 documents = 20 collections
# ══════════════════════════════════════════════════════════

def index_documents():
    """4개 config로 5개 문서를 인덱싱"""
    from bidflow.parsing.hwp_parser import HWPParser
    from bidflow.parsing.table_chunker import TableAwareChunker
    from bidflow.parsing.preprocessor import TextPreprocessor

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    preprocessor = TextPreprocessor()
    hwp_parser = HWPParser()
    chunker = TableAwareChunker(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        table_mode="colpath",
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    config_labels = ["V0_hwp5txt", "V1_html_basic", "V2_html_table", "V3_html_full"]
    index_stats = {}

    for config_label in config_labels:
        print(f"\n{'#'*60}")
        print(f"# Indexing: {config_label}")
        print(f"{'#'*60}")

        for doc_key, doc_info in DOC_CONFIGS.items():
            file_path = doc_info["file_path"]
            persist_dir = str(EXP_DIR / f'vectordb_{config_label}_{doc_key}')

            # 기존 DB 삭제 후 새로 생성
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)

            print(f"\n  [{doc_key}] {doc_info['name']}")

            # Config별 문서 변환
            if config_label == "V0_hwp5txt":
                # 기존 hwp5txt 파서 사용
                chunks = hwp_parser.parse(file_path)
                docs = [
                    Document(
                        page_content=c.text,
                        metadata={**c.metadata, "mode": "v0_hwp5txt", "chunk_type": "text"},
                    )
                    for c in chunks
                ]
            elif config_label == "V1_html_basic":
                docs = chunker.chunk_v1_basic(file_path)
            elif config_label == "V2_html_table":
                docs = chunker.chunk_v2_table(file_path)
            elif config_label == "V3_html_full":
                docs = chunker.chunk_v3_full(file_path)
            else:
                raise ValueError(f"Unknown config: {config_label}")

            if not docs:
                print(f"    WARNING: No documents generated!")
                continue

            # ChromaDB에 임베딩+저장
            vdb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name='bidflow_rfp',
            )

            n_text = sum(1 for d in docs if d.metadata.get('chunk_type') == 'text')
            n_table = sum(1 for d in docs if d.metadata.get('chunk_type') == 'table')
            total_chars = sum(len(d.page_content) for d in docs)

            key = f"{config_label}_{doc_key}"
            index_stats[key] = {
                'config': config_label, 'doc_key': doc_key,
                'n_chunks': len(docs), 'n_text': n_text, 'n_table': n_table,
                'total_chars': total_chars,
            }
            print(f"    → {len(docs)} chunks (text:{n_text}, table:{n_table}), {total_chars:,} chars")

    # 인덱싱 통계 저장
    with open(str(EXP_DIR / 'index_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(index_stats, f, ensure_ascii=False, indent=2)
    print(f"\nIndex stats saved: {EXP_DIR}/index_stats.json")

    return index_stats


# ══════════════════════════════════════════════════════════
# STEP 2: Evaluation — 4 configs × 3 runs × 30 questions
# ══════════════════════════════════════════════════════════

def run_evaluation():
    """4개 config로 평가 실행"""
    print(f"\n{'='*70}")
    print(f"EXP10ce: Phase CE Evaluation")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # Load testset
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions from {testset['source_doc'].nunique()} documents")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    config_labels = ["V0_hwp5txt", "V1_html_basic", "V2_html_table", "V3_html_full"]

    # Retrieval params (Phase B best: alpha=0.7, rerank, pool=50, top_k=15)
    RETRIEVAL_PARAMS = {'alpha': 0.7, 'top_k': 15, 'pool_size': 50, 'use_rerank': True}

    all_results = []
    errors = []
    total_evals = len(config_labels) * N_RUNS * len(testset)
    eval_count = 0
    exp_start = time.time()

    for config_label in config_labels:
        print(f"\n{'#'*70}")
        print(f"# Config: {config_label}")
        print(f"{'#'*70}")

        # Load per-document VDBs for this config
        doc_vdbs = {}
        for doc_key in DOC_CONFIGS:
            persist_dir = str(EXP_DIR / f'vectordb_{config_label}_{doc_key}')
            if not os.path.exists(persist_dir):
                print(f"  WARNING: VDB not found: {persist_dir}")
                continue
            vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name='bidflow_rfp')
            doc_vdbs[doc_key] = vdb
            print(f"  {doc_key}: {vdb._collection.count()} chunks loaded")

        # Build per-doc chains
        doc_chains = {}
        for doc_key in doc_vdbs:
            retriever = build_retriever(doc_vdbs[doc_key], **RETRIEVAL_PARAMS)
            doc_chains[doc_key] = build_rag_chain(retriever)

        for run_idx in range(N_RUNS):
            run_start = time.time()
            print(f"\n--- Run {run_idx + 1}/{N_RUNS} ---")

            for q_idx, row in testset.iterrows():
                eval_count += 1
                question = row['question']
                ground_truth = row['ground_truth']
                source_doc = row['source_doc']
                doc_key = SOURCE_TO_KEY.get(source_doc)

                if doc_key is None or doc_key not in doc_chains:
                    errors.append({'config': config_label, 'run': run_idx, 'question': question[:50], 'error': f'No chain for {source_doc}'})
                    continue

                try:
                    result = doc_chains[doc_key](question)
                    kw = keyword_accuracy_v2(result['answer'], ground_truth)

                    all_results.append({
                        'config': config_label, 'run': run_idx,
                        'doc_key': doc_key, 'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                        'question': question, 'ground_truth': ground_truth,
                        'answer': result['answer'],
                        'kw_v2': kw,
                        'category': row.get('category', ''),
                        'difficulty': row.get('difficulty', ''),
                        'n_retrieved': result['n_retrieved'],
                        'retrieval_time': result['retrieval_time'],
                        'generation_time': result['generation_time'],
                        'total_time': result['total_time'],
                        'timeout': result['total_time'] > 120,
                    })

                    if eval_count % 5 == 0:
                        elapsed = time.time() - exp_start
                        eta = (elapsed / eval_count) * (total_evals - eval_count)
                        print(f"  [{eval_count}/{total_evals}] kw={kw:.2f} t={result['total_time']:.1f}s elapsed={elapsed:.0f}s ETA={eta:.0f}s")

                except Exception as e:
                    errors.append({'config': config_label, 'run': run_idx, 'question': question[:50], 'error': str(e)})
                    all_results.append({
                        'config': config_label, 'run': run_idx,
                        'doc_key': doc_key, 'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                        'question': question, 'ground_truth': ground_truth,
                        'answer': 'ERROR', 'kw_v2': 0.0,
                        'category': row.get('category', ''), 'difficulty': row.get('difficulty', ''),
                        'n_retrieved': 0, 'retrieval_time': 0, 'generation_time': 0,
                        'total_time': 0, 'timeout': False,
                    })
                    print(f"  ERROR [{eval_count}/{total_evals}]: {question[:40]}... -> {e}")

            print(f"  Run {run_idx + 1} done in {time.time() - run_start:.0f}s")

    total_time = time.time() - exp_start

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # ── Results summary ──
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total evals: {eval_count}, Errors: {len(errors)}")
    print(f"{'='*70}")

    # Config summary
    print("\nConfig Overall Mean (3-run avg):")
    config_summary = results_df.groupby('config').agg(
        kw_v2_mean=('kw_v2', 'mean'),
        kw_v2_std=('kw_v2', 'std'),
        total_time_mean=('total_time', 'mean'),
        total_time_p95=('total_time', lambda x: np.percentile(x, 95)),
        timeout_rate=('timeout', 'mean'),
    ).round(4)
    print(config_summary)

    # Per-document
    print("\nConfig x Document kw_v2:")
    doc_pivot = results_df.groupby(['config', 'doc_key'])['kw_v2'].mean().unstack()
    doc_pivot['macro_mean'] = doc_pivot.mean(axis=1)
    print(doc_pivot.round(4))

    # text vs table gap
    print("\nText vs Table Gap:")
    for cfg_name in config_labels:
        cfg_data = results_df[results_df['config'] == cfg_name]
        if cfg_data.empty:
            continue
        text_kw = cfg_data[cfg_data['doc_type'] == 'text_only']['kw_v2'].mean()
        table_kw = cfg_data[cfg_data['doc_type'] != 'text_only']['kw_v2'].mean()
        gap = text_kw - table_kw if not (np.isnan(text_kw) or np.isnan(table_kw)) else 0
        print(f"  {cfg_name}: text={text_kw:.4f}, table={table_kw:.4f}, gap={gap:.4f}")

    # Difficulty
    print("\nDifficulty breakdown:")
    diff_pivot = results_df.groupby(['config', 'difficulty'])['kw_v2'].mean().unstack()
    print(diff_pivot.round(4))

    # Worst group
    print("\nWorst group per config:")
    for cfg_name in config_labels:
        cfg_data = results_df[results_df['config'] == cfg_name]
        if cfg_data.empty:
            continue
        doc_means = cfg_data.groupby('doc_key')['kw_v2'].mean()
        worst = doc_means.idxmin()
        print(f"  {cfg_name}: {worst}={doc_means[worst]:.4f}")

    # ── Quality floor check ──
    QUALITY_FLOOR = {'kw_v2': 0.50}
    WORST_GROUP_FLOOR = {'kw_v2': 0.35}
    OPS_CEILING = {'timeout_rate': 0.10, 'p95_latency_sec': 120}

    # Phase CE targets
    TARGET = {'overall_kw_v2': 0.80, 'table_kw_v2': 0.80, 'text_table_gap': 0.10}

    report = {}
    print(f"\n{'='*70}")
    print("Quality Floor + Phase CE Target Check")
    print(f"{'='*70}")
    for cfg_name in config_labels:
        cd = results_df[results_df['config'] == cfg_name]
        if cd.empty:
            continue
        kw_mean = cd['kw_v2'].mean()
        kw_std = cd['kw_v2'].std()
        doc_means = cd.groupby('doc_key')['kw_v2'].mean()
        worst_kw = doc_means.min()
        worst_doc = doc_means.idxmin()
        macro_kw = doc_means.mean()
        to_rate = cd['timeout'].mean()
        p95_lat = np.percentile(cd['total_time'], 95)

        text_kw = cd[cd['doc_type'] == 'text_only']['kw_v2'].mean()
        table_kw = cd[cd['doc_type'] != 'text_only']['kw_v2'].mean()
        text_table_gap = abs(text_kw - table_kw) if not (np.isnan(text_kw) or np.isnan(table_kw)) else 999

        pass_kw = kw_mean >= QUALITY_FLOOR['kw_v2']
        pass_worst = worst_kw >= WORST_GROUP_FLOOR['kw_v2']
        pass_ops = to_rate <= OPS_CEILING['timeout_rate'] and p95_lat <= OPS_CEILING['p95_latency_sec']
        pass_target_overall = kw_mean >= TARGET['overall_kw_v2']
        pass_target_table = table_kw >= TARGET['table_kw_v2']
        pass_target_gap = text_table_gap <= TARGET['text_table_gap']

        report[cfg_name] = {
            'kw_v2_overall': round(kw_mean, 4), 'kw_v2_std': round(kw_std, 4),
            'kw_v2_macro': round(macro_kw, 4), 'kw_v2_worst': round(worst_kw, 4),
            'worst_doc': worst_doc,
            'text_kw_v2': round(text_kw, 4) if not np.isnan(text_kw) else None,
            'table_kw_v2': round(table_kw, 4) if not np.isnan(table_kw) else None,
            'text_table_gap': round(text_table_gap, 4),
            'timeout_rate': round(to_rate, 4), 'p95_latency_sec': round(p95_lat, 1),
            'pass_floor_kw': pass_kw, 'pass_floor_worst': pass_worst, 'pass_ops': pass_ops,
            'pass_target_overall': pass_target_overall,
            'pass_target_table': pass_target_table,
            'pass_target_gap': pass_target_gap,
        }
        floor_status = 'PASS' if (pass_kw and pass_worst and pass_ops) else 'FAIL'
        target_status = 'PASS' if (pass_target_overall and pass_target_table and pass_target_gap) else 'FAIL'
        print(f"\n  {cfg_name}: Floor={floor_status}, Target={target_status}")
        print(f"    kw_v2={kw_mean:.4f} (floor={QUALITY_FLOOR['kw_v2']}) {'OK' if pass_kw else 'FAIL'}")
        print(f"    worst={worst_doc}={worst_kw:.4f} (floor={WORST_GROUP_FLOOR['kw_v2']}) {'OK' if pass_worst else 'FAIL'}")
        print(f"    text={text_kw:.4f}, table={table_kw:.4f}, gap={text_table_gap:.4f} (target≤{TARGET['text_table_gap']}) {'OK' if pass_target_gap else 'FAIL'}")
        print(f"    target_overall≥{TARGET['overall_kw_v2']}: {'OK' if pass_target_overall else 'FAIL'}")
        print(f"    timeout={to_rate:.4f}, p95={p95_lat:.1f}s {'OK' if pass_ops else 'FAIL'}")

    # ── Save results ──
    results_df.to_csv('data/experiments/exp10ce_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved: data/experiments/exp10ce_metrics.csv")

    exp_report = {
        'experiment': 'exp10ce_parser_switch',
        'phase': 'CE',
        'date': datetime.now().isoformat(),
        'testset': 'golden_testset_multi.csv',
        'n_questions': len(testset),
        'n_documents': len(DOC_CONFIGS),
        'n_runs': N_RUNS,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'embedding_model': EMBEDDING_MODEL,
        'llm_model': 'gpt-5-mini',
        'retrieval_params': RETRIEVAL_PARAMS,
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count,
        'total_errors': len(errors),
        'config_labels': config_labels,
        'results': report,
        'config_summary': config_summary.to_dict(),
        'doc_pivot': doc_pivot.to_dict(),
        'targets': TARGET,
        'errors': errors[:20],
    }
    with open('data/experiments/exp10ce_report.json', 'w', encoding='utf-8') as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: data/experiments/exp10ce_report.json")

    if errors:
        with open(str(EXP_DIR / 'exp10ce_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Saved: {EXP_DIR}/exp10ce_errors.json")

    print(f"\n{'='*70}")
    print(f"EXP10ce COMPLETE")
    print(f"End: {datetime.now().isoformat()}")
    print(f"{'='*70}")


def main():
    print(f"\n{'='*70}")
    print(f"EXP10ce: Phase CE — Parser Switch + Table Gap Resolution")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # Step 1: Indexing
    print(f"\n{'▶'*20} STEP 1: INDEXING {'◀'*20}")
    index_start = time.time()
    index_stats = index_documents()
    index_time = time.time() - index_start
    print(f"\nIndexing complete in {index_time:.0f}s ({index_time/60:.1f} min)")

    # Step 2: Evaluation
    print(f"\n{'▶'*20} STEP 2: EVALUATION {'◀'*20}")
    run_evaluation()


if __name__ == '__main__':
    main()
