"""
EXP10e: Phase E — Retrieval Pipeline Re-optimization (Multi-Document)

단일 문서 과적합된 EXP01-09 파라미터를 5문서 테스트셋에서 재최적화.
V4_hybrid 파서 고정, chunk_size / pool_size / prompt 3차원 스윕.

실행: cd bidflow && python -u scripts/run_exp10e.py
"""
import os, sys, time, re, json, warnings, shutil
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
EXP_DIR = PROJECT_ROOT / 'data' / 'exp10e'
EXP_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL = 'text-embedding-3-small'
N_RUNS = 1  # reduced from 3 to conserve API quota

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

# ── Experiment configs ──
# 3차원 스윕: chunk_size × pool_size × prompt
PROMPT_V1 = (
    '아래 문맥(Context)만을 근거로 질문에 답하세요.\n'
    '반드시 원문에 있는 사업명, 기관명, 금액, 날짜 등의 표현을 그대로(Verbatim) 사용하세요.\n'
    '문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)

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

EVAL_CONFIGS = [
    {"label": "c300",      "chunk_size": 300, "chunk_overlap": 30,  "pool_size": 50,  "prompt": PROMPT_V1, "alpha": 0.7, "top_k": 15},
    {"label": "c500",      "chunk_size": 500, "chunk_overlap": 50,  "pool_size": 50,  "prompt": PROMPT_V1, "alpha": 0.7, "top_k": 15},
    {"label": "c800",      "chunk_size": 800, "chunk_overlap": 80,  "pool_size": 50,  "prompt": PROMPT_V1, "alpha": 0.7, "top_k": 15},
    {"label": "c500_p100", "chunk_size": 500, "chunk_overlap": 50,  "pool_size": 100, "prompt": PROMPT_V1, "alpha": 0.7, "top_k": 15},
    {"label": "c500_pv2",  "chunk_size": 500, "chunk_overlap": 50,  "pool_size": 50,  "prompt": PROMPT_V2, "alpha": 0.7, "top_k": 15},
]

# ── kw_v2 helpers ──
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

# ── Retriever ──
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


def build_rag_chain(retriever, prompt_template, model_name='gpt-5-mini'):
    temp = 1 if model_name == 'gpt-5-mini' else 0
    llm = ChatOpenAI(model=model_name, temperature=temp, timeout=60, max_retries=2)
    prompt = ChatPromptTemplate.from_template(prompt_template)
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


def main():
    print(f"\n{'='*70}")
    print(f"EXP10e: Phase E — Retrieval Pipeline Re-optimization")
    print(f"V4_hybrid parser fixed, sweeping chunk_size/pool_size/prompt")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    from bidflow.parsing.table_chunker import TableAwareChunker
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # ── STEP 1: Indexing (per chunk_size) ──
    # Different chunk_sizes need different VDBs
    chunk_sizes_needed = sorted(set(cfg["chunk_size"] for cfg in EVAL_CONFIGS))
    print(f"\nChunk sizes to index: {chunk_sizes_needed}")

    index_stats = {}

    for cs in chunk_sizes_needed:
        overlap = next(cfg["chunk_overlap"] for cfg in EVAL_CONFIGS if cfg["chunk_size"] == cs)
        chunker = TableAwareChunker(
            chunk_size=cs, chunk_overlap=overlap, table_mode="flat",
        )
        cs_label = f"c{cs}"

        print(f"\n{'#'*60}")
        print(f"# Indexing: chunk_size={cs}, overlap={overlap}")
        print(f"{'#'*60}")

        for doc_key, doc_info in DOC_CONFIGS.items():
            file_path = doc_info["file_path"]
            persist_dir = str(EXP_DIR / f'vectordb_{cs_label}_{doc_key}')

            # Skip re-indexing if VDB already exists (resume support)
            if os.path.exists(persist_dir):
                existing_vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name='bidflow_rfp')
                count = existing_vdb._collection.count()
                if count > 0:
                    print(f"\n  [{doc_key}] SKIP (existing VDB: {count} chunks)")
                    index_stats[f"{cs_label}_{doc_key}"] = {'n_chunks': count, 'source': 'cached'}
                    continue
                shutil.rmtree(persist_dir)

            print(f"\n  [{doc_key}] {doc_info['name']}")
            docs = chunker.chunk_v4_hybrid(file_path)

            if not docs:
                print(f"    WARNING: No documents generated!")
                continue

            vdb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name='bidflow_rfp',
            )

            n_text = sum(1 for d in docs if d.metadata.get('chunk_type') == 'text')
            n_table = sum(1 for d in docs if d.metadata.get('chunk_type') == 'table')
            total_chars = sum(len(d.page_content) for d in docs)

            key = f"{cs_label}_{doc_key}"
            index_stats[key] = {
                'chunk_size': cs, 'doc_key': doc_key,
                'n_chunks': len(docs), 'n_text': n_text, 'n_table': n_table,
                'total_chars': total_chars,
            }
            print(f"    → {len(docs)} chunks (text:{n_text}, table:{n_table}), {total_chars:,} chars")

    with open(str(EXP_DIR / 'index_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(index_stats, f, ensure_ascii=False, indent=2)

    # Print indexing summary
    print(f"\n{'='*70}")
    print("INDEXING SUMMARY:")
    for cs in chunk_sizes_needed:
        cs_label = f"c{cs}"
        total_chunks = sum(index_stats[f"{cs_label}_{dk}"]["n_chunks"] for dk in DOC_CONFIGS if f"{cs_label}_{dk}" in index_stats)
        print(f"  chunk_size={cs}: {total_chunks} total chunks across 5 docs")
    print(f"{'='*70}")

    # ── STEP 2: Evaluation ──
    print(f"\n{'='*70}")
    print(f"STEP 2: EVALUATION ({len(EVAL_CONFIGS)} configs × {N_RUNS} runs × 30Q)")
    print(f"{'='*70}")

    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"Testset: {len(testset)} questions")

    all_results = []
    errors = []
    total_evals = len(EVAL_CONFIGS) * N_RUNS * len(testset)
    eval_count = 0
    exp_start = time.time()
    csv_path = 'data/experiments/exp10e_metrics.csv'
    quota_exhausted = False

    # Resume: load existing results if any
    completed_configs = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        for cfg_name in prev['config'].unique():
            cfg_data = prev[prev['config'] == cfg_name]
            if len(cfg_data) >= N_RUNS * len(testset):
                completed_configs.add(cfg_name)
        all_results = prev.to_dict('records')
        print(f"  Resuming: {len(all_results)} previous results, skipping configs: {completed_configs}")

    for cfg in EVAL_CONFIGS:
        config_label = cfg["label"]
        cs = cfg["chunk_size"]
        cs_label = f"c{cs}"
        pool_size = cfg["pool_size"]
        alpha = cfg["alpha"]
        top_k = cfg["top_k"]
        prompt_template = cfg["prompt"]

        if config_label in completed_configs:
            print(f"\n  SKIP: {config_label} (already completed)")
            eval_count += N_RUNS * len(testset)
            continue

        if quota_exhausted:
            print(f"\n  SKIP: {config_label} (API quota exhausted)")
            continue

        print(f"\n{'#'*60}")
        print(f"# Config: {config_label} (chunk={cs}, pool={pool_size}, alpha={alpha})")
        print(f"{'#'*60}")

        # Load VDBs for this chunk_size
        doc_vdbs = {}
        for doc_key in DOC_CONFIGS:
            persist_dir = str(EXP_DIR / f'vectordb_{cs_label}_{doc_key}')
            if not os.path.exists(persist_dir):
                print(f"  WARNING: VDB not found: {persist_dir}")
                continue
            vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name='bidflow_rfp')
            doc_vdbs[doc_key] = vdb
            print(f"  {doc_key}: {vdb._collection.count()} chunks loaded")

        # Build chains with config-specific params
        doc_chains = {}
        for doc_key in doc_vdbs:
            retriever = build_retriever(
                doc_vdbs[doc_key], alpha=alpha, top_k=top_k,
                pool_size=pool_size, use_rerank=True,
            )
            doc_chains[doc_key] = build_rag_chain(retriever, prompt_template)

        consecutive_errors = 0
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
                    consecutive_errors = 0

                    all_results.append({
                        'config': config_label, 'run': run_idx,
                        'doc_key': doc_key, 'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                        'question': question, 'ground_truth': ground_truth,
                        'answer': result['answer'],
                        'kw_v2': kw,
                        'chunk_size': cs, 'pool_size': pool_size, 'alpha': alpha,
                        'category': row.get('category', ''),
                        'difficulty': row.get('difficulty', ''),
                        'n_retrieved': result['n_retrieved'],
                        'retrieval_time': result['retrieval_time'],
                        'generation_time': result['generation_time'],
                        'total_time': result['total_time'],
                        'timeout': result['total_time'] > 120,
                    })

                    if eval_count % 10 == 0:
                        elapsed = time.time() - exp_start
                        eta = (elapsed / eval_count) * (total_evals - eval_count)
                        print(f"  [{eval_count}/{total_evals}] kw={kw:.2f} t={result['total_time']:.1f}s elapsed={elapsed:.0f}s ETA={eta:.0f}s")

                except Exception as e:
                    err_str = str(e)
                    errors.append({'config': config_label, 'run': run_idx, 'question': question[:50], 'error': err_str})
                    all_results.append({
                        'config': config_label, 'run': run_idx,
                        'doc_key': doc_key, 'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                        'question': question, 'ground_truth': ground_truth,
                        'answer': 'ERROR', 'kw_v2': 0.0,
                        'chunk_size': cs, 'pool_size': pool_size, 'alpha': alpha,
                        'category': row.get('category', ''), 'difficulty': row.get('difficulty', ''),
                        'n_retrieved': 0, 'retrieval_time': 0, 'generation_time': 0,
                        'total_time': 0, 'timeout': False,
                    })
                    print(f"  ERROR [{eval_count}/{total_evals}]: {question[:40]}... -> {e}")

                    # Stop early on quota exhaustion (3 consecutive errors)
                    consecutive_errors += 1
                    if consecutive_errors >= 3 and 'insufficient_quota' in err_str:
                        print(f"\n  *** API QUOTA EXHAUSTED — stopping evaluation ***")
                        quota_exhausted = True
                        break

            if quota_exhausted:
                break
            print(f"  Run {run_idx + 1} done in {time.time() - run_start:.0f}s")

        # Incremental save after each config
        pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  [SAVED] {len(all_results)} results to {csv_path}")

    total_time = time.time() - exp_start
    results_df = pd.DataFrame(all_results)

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total evals: {eval_count}, Errors: {len(errors)}")
    print(f"{'='*70}")

    # Overall config comparison
    config_summary = results_df.groupby('config').agg(
        kw_v2_mean=('kw_v2', 'mean'),
        kw_v2_std=('kw_v2', 'std'),
        total_time_mean=('total_time', 'mean'),
    ).round(4)
    print(f"\nConfig Overall (sorted by kw_v2):")
    print(config_summary.sort_values('kw_v2_mean', ascending=False))

    # Per-document breakdown
    doc_pivot = results_df.groupby(['config', 'doc_key'])['kw_v2'].mean().unstack()
    doc_pivot['overall'] = results_df.groupby('config')['kw_v2'].mean()
    print(f"\nConfig × Document kw_v2:")
    print(doc_pivot.round(4).sort_values('overall', ascending=False))

    # Text vs Table Gap per config
    print(f"\nText vs Table Gap:")
    for cfg in EVAL_CONFIGS:
        cfg_data = results_df[results_df['config'] == cfg['label']]
        if cfg_data.empty:
            continue
        text_kw = cfg_data[cfg_data['doc_type'] == 'text_only']['kw_v2'].mean()
        table_kw = cfg_data[cfg_data['doc_type'] != 'text_only']['kw_v2'].mean()
        gap = text_kw - table_kw if not (np.isnan(text_kw) or np.isnan(table_kw)) else 0
        print(f"  {cfg['label']}: text={text_kw:.4f}, table={table_kw:.4f}, gap={gap:+.4f}")

    # Difficulty analysis
    print(f"\nDifficulty × Config kw_v2:")
    diff_pivot = results_df.groupby(['config', 'difficulty'])['kw_v2'].mean().unstack()
    print(diff_pivot.round(4))

    # Best config identification
    best_config = config_summary['kw_v2_mean'].idxmax()
    best_kw = config_summary.loc[best_config, 'kw_v2_mean']
    baseline_kw = config_summary.loc['c500', 'kw_v2_mean'] if 'c500' in config_summary.index else 0
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_config} (kw_v2={best_kw:.4f})")
    print(f"vs c500 baseline: {best_kw - baseline_kw:+.4f}")
    print(f"{'='*70}")

    # ── Save ──
    results_df.to_csv('data/experiments/exp10e_metrics.csv', index=False, encoding='utf-8-sig')
    exp_report = {
        'experiment': 'exp10e_retrieval_reoptimization',
        'phase': 'E',
        'date': datetime.now().isoformat(),
        'n_questions': len(testset), 'n_runs': N_RUNS,
        'embedding_model': EMBEDDING_MODEL,
        'llm_model': 'gpt-5-mini',
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count, 'total_errors': len(errors),
        'configs': [{k: v for k, v in cfg.items() if k != 'prompt'} for cfg in EVAL_CONFIGS],
        'results': {
            'best_config': best_config,
            'best_kw_v2': round(best_kw, 4),
            'baseline_kw_v2': round(baseline_kw, 4),
            'delta': round(best_kw - baseline_kw, 4),
        },
        'config_summary': config_summary.to_dict(),
        'doc_pivot': doc_pivot.to_dict(),
        'index_stats': index_stats,
        'errors': errors[:20],
    }
    with open('data/experiments/exp10e_report.json', 'w', encoding='utf-8') as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)

    if errors:
        with open(str(EXP_DIR / 'exp10e_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: data/experiments/exp10e_metrics.csv")
    print(f"Saved: data/experiments/exp10e_report.json")
    print(f"\n{'='*70}")
    print(f"EXP10e COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
