"""
EXP12: Retrieval 최적화 (Embedding, Reranker Pool, Alpha, Multi-Query)

EXP11에서 프롬프트 엔지니어링 접근이 모두 베이스라인 이하로 확인됨.
→ Retrieval 파라미터 튜닝 + 임베딩 모델 업그레이드로 방향 전환.

Stage 1 (기존 VDB 재사용): — 완료
  - ref_v2:      Phase E 베이스라인 재활용
  - alpha_05:    BM25 가중치 50%로 증가
  - pool_80:     리랭커 후보 80개 (기존 50)
  - pool_80_k20: 후보 80개 + top_k=20
  - multi_query: LLM으로 한국어 쿼리 3개 변형 생성

Stage 2 (KURE-v1 오픈소스 임베딩 재인덱싱):
  - emb_kure:      nlpai-lab/KURE-v1 한국어 특화 임베딩 (1024차원)
  - combined_best: emb_kure + Stage 1 최적 파라미터

실행: cd bidflow && python -u scripts/run_exp12.py
"""
import os, sys, time, re, json, warnings
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
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List, Any

# ── Constants ──
EXP_DIR = PROJECT_ROOT / 'data' / 'exp12'
EXP_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_SMALL = 'text-embedding-3-small'
EMBEDDING_KURE = 'nlpai-lab/KURE-v1'  # 한국어 특화 오픈소스 임베딩 (1024차원)
LLM_MODEL = 'gpt-5-mini'
CSV_PATH = 'data/experiments/exp12_metrics.csv'
VDB_BASE_SMALL = PROJECT_ROOT / 'data' / 'exp10e'
VDB_BASE_KURE = EXP_DIR / 'vectordb_kure'
VDB_BASE_KURE.mkdir(parents=True, exist_ok=True)


# ── LocalEmbeddings: SentenceTransformer 래퍼 (LangChain 호환) ──
class LocalEmbeddings(Embeddings):
    """HuggingFace SentenceTransformer를 LangChain Embeddings 인터페이스로 래핑"""
    def __init__(self, model_name: str, device: str = 'cuda',
                 query_prefix: str = '', doc_prefix: str = '', batch_size: int = 16):
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f'{self.doc_prefix}{t}' for t in texts]
        return self.model.encode(
            prefixed, normalize_embeddings=True, batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        prefixed = f'{self.query_prefix}{text}'
        return self.model.encode(
            [prefixed], normalize_embeddings=True
        )[0].tolist()

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


# ═══════════════════════════════════════════════════════════════
# 평가 지표: kw_v2 + kw_v3 (EXP11과 동일)
# ═══════════════════════════════════════════════════════════════

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


def normalize_answer_v3(text):
    t = normalize_answer_v2(text)
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


def keyword_accuracy(answer, ground_truth, norm_fn):
    ans_norm = norm_fn(answer)
    gt_norm = norm_fn(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = sum(1 for w in gt_words if w in ans_norm)
    return matched / len(gt_words)


def keyword_accuracy_v2(answer, ground_truth):
    return keyword_accuracy(answer, ground_truth, normalize_answer_v2)


def keyword_accuracy_v3(answer, ground_truth):
    return keyword_accuracy(answer, ground_truth, normalize_answer_v3)


# ═══════════════════════════════════════════════════════════════
# Retriever (파라미터 오버라이드 가능)
# ═══════════════════════════════════════════════════════════════

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
    """VDB로부터 파라미터 오버라이드 가능한 retriever 생성"""
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


# ═══════════════════════════════════════════════════════════════
# Multi-Query: 한국어 쿼리 변형 생성
# ═══════════════════════════════════════════════════════════════

MULTI_QUERY_PROMPT = (
    '당신은 한국어 검색 쿼리 최적화 전문가입니다.\n'
    '아래 원본 질문과 동일한 정보를 찾기 위한 다른 표현의 검색 쿼리 3개를 생성하세요.\n\n'
    '규칙:\n'
    '1. 원본 질문의 의도를 정확히 유지하세요.\n'
    '2. 동의어, 유사 표현, 다른 조사/어미를 사용하세요.\n'
    '3. 각 쿼리는 한 줄씩, 번호 없이 작성하세요.\n'
    '4. 반드시 3개만 생성하세요.\n\n'
    '원본 질문: {question}\n\n'
    '변형 쿼리 3개:\n'
)

_multi_query_llm = None

def get_multi_query_llm():
    global _multi_query_llm
    if _multi_query_llm is None:
        _multi_query_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, timeout=30, max_retries=2)
    return _multi_query_llm


def generate_query_variants(question, n=3):
    """LLM으로 한국어 쿼리 변형 생성"""
    llm = get_multi_query_llm()
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({'question': question})
        lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
        # 번호 접두사 제거 (1. 2. 3. 등)
        cleaned = []
        for line in lines:
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if line and line != question:
                cleaned.append(line)
        return cleaned[:n]
    except Exception as e:
        print(f"    [MultiQuery] Error: {e}")
        return []


def multi_query_retrieve(retriever, question, top_k=15):
    """원본 + 변형 쿼리로 검색 후 합산, 중복 제거, 리랭크"""
    variants = generate_query_variants(question)
    all_queries = [question] + variants
    print(f"    [MultiQuery] {len(all_queries)} queries: original + {len(variants)} variants")

    # 각 쿼리로 검색 (리랭크 전 pool_size만큼)
    all_docs_map = {}  # page_content → (doc, max_score)
    for q in all_queries:
        try:
            # 리랭크 없이 RRF까지만 실행하여 pool 확보
            search_k = retriever.pool_size
            retriever.bm25_retriever.k = search_k * 2
            bm25_docs = retriever.bm25_retriever.invoke(q)
            retriever.vector_retriever.search_kwargs['k'] = search_k * 2
            vector_docs = retriever.vector_retriever.invoke(q)
            merged = retriever._rrf_merge(bm25_docs, vector_docs, k=60, limit=search_k)
            for doc in merged:
                if doc.page_content not in all_docs_map:
                    all_docs_map[doc.page_content] = doc
        except Exception as e:
            print(f"    [MultiQuery] Query error: {e}")

    # 중복 제거된 전체 후보
    unique_docs = list(all_docs_map.values())
    print(f"    [MultiQuery] Unique candidates: {len(unique_docs)}")

    # 리랭크로 최종 top_k 선택
    if unique_docs and retriever.use_rerank:
        from bidflow.retrieval.rerank import rerank
        final_docs = rerank(question, unique_docs, top_k=top_k, model_name=retriever.rerank_model)
    else:
        final_docs = unique_docs[:top_k]

    return final_docs


# ═══════════════════════════════════════════════════════════════
# Stage 2: KURE-v1 오픈소스 임베딩 (재인덱싱)
# ═══════════════════════════════════════════════════════════════

_kure_embeddings = None

def get_kure_embeddings():
    """KURE-v1 임베딩 싱글턴"""
    global _kure_embeddings
    if _kure_embeddings is None:
        print(f"    [KURE-v1] Loading model: {EMBEDDING_KURE}...")
        _kure_embeddings = LocalEmbeddings(
            model_name=EMBEDDING_KURE,
            device='cuda',
            query_prefix='',  # KURE-v1은 prefix 불필요
            doc_prefix='',
            batch_size=32,
        )
        print(f"    [KURE-v1] Model loaded successfully")
    return _kure_embeddings


def reindex_with_kure_embedding(doc_key):
    """기존 small VDB에서 청크를 읽어 KURE-v1 임베딩으로 재인덱싱"""
    kure_dir = str(VDB_BASE_KURE / f'vectordb_c500_{doc_key}')
    kure_emb = get_kure_embeddings()

    if os.path.exists(kure_dir):
        # 이미 재인덱싱됨 → 바로 로드
        vdb = Chroma(persist_directory=kure_dir, embedding_function=kure_emb,
                     collection_name='bidflow_rfp')
        count = vdb._collection.count()
        if count > 0:
            print(f"    [Reindex] {doc_key}: 기존 KURE VDB 로드 ({count} chunks)")
            return vdb

    # 기존 small VDB에서 청크 읽기
    small_dir = str(VDB_BASE_SMALL / f'vectordb_c500_{doc_key}')
    small_embeddings = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    small_vdb = Chroma(persist_directory=small_dir, embedding_function=small_embeddings,
                       collection_name='bidflow_rfp')
    result = small_vdb.get()

    if not result or not result['documents']:
        print(f"    [Reindex] WARNING: No chunks in {small_dir}")
        return None

    # LangChain Document로 변환
    lc_docs = []
    for i, text in enumerate(result['documents']):
        meta = result['metadatas'][i] if result['metadatas'] else {}
        lc_docs.append(Document(page_content=text, metadata=meta))

    print(f"    [Reindex] {doc_key}: {len(lc_docs)} chunks → KURE-v1 embedding...")

    # KURE-v1 임베딩으로 새 VDB 생성
    try:
        os.makedirs(kure_dir, exist_ok=True)
        vdb = Chroma.from_documents(
            lc_docs,
            kure_emb,
            persist_directory=kure_dir,
            collection_name='bidflow_rfp',
        )
        print(f"    [Reindex] {doc_key}: Done ({vdb._collection.count()} chunks)")
        return vdb
    except Exception as e:
        print(f"    [Reindex] ERROR for {doc_key}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# RAG Chain
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


def build_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=1, timeout=60, max_retries=2)


def invoke_rag(retriever, question, llm, docs_override=None):
    """RAG 호출. docs_override가 있으면 검색 건너뛰고 직접 전달"""
    t0 = time.time()
    if docs_override is not None:
        docs = docs_override
        retrieval_time = 0.0
    else:
        docs = retriever.invoke(question)
        retrieval_time = time.time() - t0

    context_text = '\n\n'.join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_V2)

    t1 = time.time()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context_text, 'question': question})
    gen_time = time.time() - t1

    return {
        'answer': answer,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': gen_time,
        'total_time': retrieval_time + gen_time,
    }


def invoke_multi_query_rag(retriever, question, llm, top_k=15):
    """Multi-query RAG: 여러 쿼리로 검색 후 합산, 생성"""
    t0 = time.time()
    docs = multi_query_retrieve(retriever, question, top_k=top_k)
    retrieval_time = time.time() - t0

    context_text = '\n\n'.join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_V2)

    t1 = time.time()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context_text, 'question': question})
    gen_time = time.time() - t1

    return {
        'answer': answer,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': gen_time,
        'total_time': retrieval_time + gen_time,
    }


# ═══════════════════════════════════════════════════════════════
# 질문 타입 분류 (분석용)
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
# Config 정의
# ═══════════════════════════════════════════════════════════════

EVAL_CONFIGS = [
    {
        "label": "ref_v2",
        "description": "Phase E c500_pv2 결과 재활용 (베이스라인)",
        "needs_api": False,
        "stage": 1,
    },
    {
        "label": "alpha_05",
        "description": "Stage 1: BM25 가중치 50% (alpha=0.5)",
        "needs_api": True,
        "stage": 1,
        "alpha": 0.5,
        "pool_size": 50,
        "top_k": 15,
    },
    {
        "label": "pool_80",
        "description": "Stage 1: 리랭커 후보 80개 (pool_size=80)",
        "needs_api": True,
        "stage": 1,
        "alpha": 0.7,
        "pool_size": 80,
        "top_k": 15,
    },
    {
        "label": "pool_80_k20",
        "description": "Stage 1: pool=80 + top_k=20",
        "needs_api": True,
        "stage": 1,
        "alpha": 0.7,
        "pool_size": 80,
        "top_k": 20,
    },
    {
        "label": "multi_query",
        "description": "Stage 1: LLM 한국어 쿼리 3변형 + 합산 리랭크",
        "needs_api": True,
        "stage": 1,
        "alpha": 0.7,
        "pool_size": 50,
        "top_k": 15,
        "use_multi_query": True,
    },
    {
        "label": "emb_kure",
        "description": "Stage 2: KURE-v1 한국어 특화 임베딩 (1024차원)",
        "needs_api": True,
        "stage": 2,
        "alpha": 0.7,
        "pool_size": 50,
        "top_k": 15,
        "embedding": EMBEDDING_KURE,
    },
    {
        "label": "combined_best",
        "description": "Stage 2: KURE-v1 + Stage 1 최적 파라미터 (런타임 결정)",
        "needs_api": True,
        "stage": 2,
        "embedding": EMBEDDING_KURE,
        # alpha, pool_size, top_k는 Stage 1 결과에서 결정
    },
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"EXP12: Retrieval 최적화 (Embedding, Pool, Alpha, MultiQuery)")
    print(f"Baseline: ref_v2 (kw_v3=0.8961)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP 0: Phase E 결과 로드 (ref_v2 베이스라인) ──
    print(f"\n{'#'*60}")
    print(f"# STEP 0: Phase E 결과 로드 (ref_v2 베이스라인)")
    print(f"{'#'*60}")

    prev_path = 'data/experiments/exp10e_metrics.csv'
    assert os.path.exists(prev_path), f"Phase E results not found: {prev_path}"
    prev_df = pd.read_csv(prev_path)
    prev_pv2 = prev_df[prev_df['config'] == 'c500_pv2'].copy()
    assert len(prev_pv2) == 30, f"Expected 30 c500_pv2 results, got {len(prev_pv2)}"

    # kw_v3로 재채점
    prev_pv2['kw_v3'] = prev_pv2.apply(
        lambda r: keyword_accuracy_v3(str(r['answer']), str(r['ground_truth'])), axis=1
    )
    print(f"  ref_v2 baseline: kw_v2={prev_pv2['kw_v2'].mean():.4f}, kw_v3={prev_pv2['kw_v3'].mean():.4f}")

    # ref_v2 결과 저장용
    ref_results = []
    for _, r in prev_pv2.iterrows():
        ref_results.append({
            'config': 'ref_v2', 'run': 0,
            'doc_key': r['doc_key'],
            'doc_type': r['doc_type'],
            'question': r['question'],
            'ground_truth': r['ground_truth'],
            'answer': r['answer'],
            'kw_v2': r['kw_v2'],
            'kw_v3': r['kw_v3'],
            'category': r['category'],
            'difficulty': r['difficulty'],
            'n_retrieved': r['n_retrieved'],
            'retrieval_time': r['retrieval_time'],
            'generation_time': r['generation_time'],
            'total_time': r['total_time'],
            'q_type': classify_question_type(r['question']),
            'retry_count': 0,
        })

    # ── STEP 1: Testset 로드 ──
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions")
    q_types = testset['question'].apply(classify_question_type)
    print(f"Question types: {q_types.value_counts().to_dict()}")

    # ── STEP 2: Stage 1 VDB 로드 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 2: Stage 1 VDB 로드 (text-embedding-3-small)")
    print(f"{'#'*60}")

    small_embeddings = OpenAIEmbeddings(model=EMBEDDING_SMALL)
    doc_vdbs_small = {}
    for doc_key in DOC_CONFIGS:
        persist_dir = str(VDB_BASE_SMALL / f'vectordb_c500_{doc_key}')
        if not os.path.exists(persist_dir):
            print(f"  WARNING: VDB not found: {persist_dir}")
            continue
        vdb = Chroma(persist_directory=persist_dir, embedding_function=small_embeddings,
                     collection_name='bidflow_rfp')
        doc_vdbs_small[doc_key] = vdb
        print(f"  {doc_key}: {vdb._collection.count()} chunks")

    # ── STEP 3: 평가 실행 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 3: 평가 실행")
    print(f"{'#'*60}")

    all_results = list(ref_results)
    errors = []
    llm = build_llm()
    quota_exhausted = False

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
            print(f"  Resuming: {len(all_results)} previous results, completed: {completed_configs}")

    # Stage 1 먼저 실행
    stage1_configs = [c for c in EVAL_CONFIGS if c.get('needs_api') and c.get('stage') == 1]
    stage2_configs = [c for c in EVAL_CONFIGS if c.get('needs_api') and c.get('stage') == 2]

    total_evals = (len(stage1_configs) + len(stage2_configs)) * len(testset)
    eval_count = 0
    exp_start = time.time()

    # ────── Stage 1 ──────
    print(f"\n{'='*70}")
    print(f"  STAGE 1: Retrieval 파라미터 튜닝 ({len(stage1_configs)} configs)")
    print(f"{'='*70}")

    for cfg in stage1_configs:
        config_label = cfg["label"]

        if config_label in completed_configs:
            print(f"\n  SKIP: {config_label} (already completed)")
            eval_count += len(testset)
            continue

        if quota_exhausted:
            print(f"\n  SKIP: {config_label} (API quota exhausted)")
            continue

        alpha = cfg.get('alpha', 0.7)
        pool_size = cfg.get('pool_size', 50)
        top_k = cfg.get('top_k', 15)
        use_multi_query = cfg.get('use_multi_query', False)

        print(f"\n{'='*60}")
        print(f"Config: {config_label} — {cfg['description']}")
        print(f"  alpha={alpha}, pool_size={pool_size}, top_k={top_k}, multi_query={use_multi_query}")
        print(f"{'='*60}")

        # 설정별 retriever 빌드
        doc_retrievers = {}
        for doc_key, vdb in doc_vdbs_small.items():
            doc_retrievers[doc_key] = build_retriever(
                vdb, alpha=alpha, top_k=top_k, pool_size=pool_size
            )

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

            try:
                if use_multi_query:
                    result = invoke_multi_query_rag(retriever, question, llm, top_k=top_k)
                else:
                    result = invoke_rag(retriever, question, llm)

                kw2 = keyword_accuracy_v2(result['answer'], ground_truth)
                kw3 = keyword_accuracy_v3(result['answer'], ground_truth)
                consecutive_errors = 0

                all_results.append({
                    'config': config_label, 'run': 0,
                    'doc_key': doc_key,
                    'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': result['answer'],
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

                if eval_count % 5 == 0 or use_multi_query:
                    print(f"  [{eval_count}/{total_evals}] kw_v2={kw2:.3f} kw_v3={kw3:.3f} "
                          f"doc={doc_key} type={q_type} n={result['n_retrieved']} "
                          f"t={result['total_time']:.1f}s")

            except Exception as e:
                err_str = str(e)
                errors.append({'config': config_label, 'question': question[:50], 'error': err_str})
                all_results.append({
                    'config': config_label, 'run': 0,
                    'doc_key': doc_key,
                    'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                    'question': question, 'ground_truth': ground_truth,
                    'answer': 'ERROR', 'kw_v2': 0.0, 'kw_v3': 0.0,
                    'category': row.get('category', ''),
                    'difficulty': row.get('difficulty', ''),
                    'n_retrieved': 0, 'retrieval_time': 0,
                    'generation_time': 0, 'total_time': 0,
                    'q_type': q_type, 'retry_count': 0,
                })
                print(f"  ERROR [{eval_count}/{total_evals}]: {question[:40]}... -> {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3 and 'insufficient_quota' in err_str:
                    print(f"\n  *** API QUOTA EXHAUSTED ***")
                    quota_exhausted = True
                    break

            # 문항별 증분 저장
            pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

        if quota_exhausted:
            break

        config_time = time.time() - config_start
        config_df = pd.DataFrame([r for r in all_results if r['config'] == config_label])
        print(f"  Config {config_label}: kw_v3={config_df['kw_v3'].mean():.4f}, "
              f"time={config_time:.0f}s")
        print(f"  [SAVED] {len(all_results)} results to {CSV_PATH}")

    # ────── Stage 1 결과 분석 → combined_best 파라미터 결정 ──────
    if not quota_exhausted:
        print(f"\n{'='*70}")
        print(f"  Stage 1 결과 요약")
        print(f"{'='*70}")
        stage1_df = pd.DataFrame(all_results)
        stage1_summary = stage1_df.groupby('config')['kw_v3'].mean().sort_values(ascending=False)
        print(stage1_summary.round(4))

        # combined_best에 적용할 최적 파라미터 결정
        # ref_v2 제외하고 Stage 1 API 설정 중 최고 선택
        stage1_api_configs = [c for c in stage1_configs if c['label'] != 'ref_v2']
        best_stage1_label = None
        best_stage1_score = 0
        for cfg in stage1_api_configs:
            label = cfg['label']
            if label in stage1_summary.index:
                score = stage1_summary[label]
                if score > best_stage1_score:
                    best_stage1_score = score
                    best_stage1_label = label

        # ref_v2도 포함하여 비교
        ref_score = stage1_summary.get('ref_v2', 0)
        if ref_score >= best_stage1_score:
            # ref_v2가 최고 → combined_best에 기본 파라미터
            best_params = {'alpha': 0.7, 'pool_size': 50, 'top_k': 15}
            print(f"\n  Best Stage 1: ref_v2 (kw_v3={ref_score:.4f}) → 기본 파라미터 사용")
        else:
            # Stage 1 API config 중 최고 파라미터 사용
            best_cfg = next(c for c in stage1_api_configs if c['label'] == best_stage1_label)
            best_params = {
                'alpha': best_cfg.get('alpha', 0.7),
                'pool_size': best_cfg.get('pool_size', 50),
                'top_k': best_cfg.get('top_k', 15),
            }
            print(f"\n  Best Stage 1: {best_stage1_label} (kw_v3={best_stage1_score:.4f})")

        print(f"  combined_best params: {best_params}")

        # combined_best 설정 업데이트
        for cfg in stage2_configs:
            if cfg['label'] == 'combined_best':
                cfg.update(best_params)
                cfg['description'] = f"Stage 2: KURE-v1 + {best_stage1_label or 'ref_v2'} params"

    # ────── Stage 2 ──────
    if not quota_exhausted:
        print(f"\n{'='*70}")
        print(f"  STAGE 2: KURE-v1 임베딩 모델 ({len(stage2_configs)} configs)")
        print(f"{'='*70}")

        # 재인덱싱
        print(f"\n  Step 2a: KURE-v1 (nlpai-lab/KURE-v1) 재인덱싱...")
        doc_vdbs_kure = {}
        for doc_key in DOC_CONFIGS:
            if doc_key not in doc_vdbs_small:
                continue
            vdb = reindex_with_kure_embedding(doc_key)
            if vdb:
                doc_vdbs_kure[doc_key] = vdb

        if not doc_vdbs_kure:
            print("  ERROR: KURE-v1 embedding VDB 생성 실패")
        else:
            for cfg in stage2_configs:
                config_label = cfg["label"]

                if config_label in completed_configs:
                    print(f"\n  SKIP: {config_label} (already completed)")
                    eval_count += len(testset)
                    continue

                if quota_exhausted:
                    print(f"\n  SKIP: {config_label} (API quota exhausted)")
                    continue

                alpha = cfg.get('alpha', 0.7)
                pool_size = cfg.get('pool_size', 50)
                top_k = cfg.get('top_k', 15)

                print(f"\n{'='*60}")
                print(f"Config: {config_label} — {cfg['description']}")
                print(f"  alpha={alpha}, pool_size={pool_size}, top_k={top_k}, embedding=KURE-v1")
                print(f"{'='*60}")

                # Stage 2 retriever 빌드 (KURE-v1 embedding VDB)
                doc_retrievers = {}
                for doc_key, vdb in doc_vdbs_kure.items():
                    doc_retrievers[doc_key] = build_retriever(
                        vdb, alpha=alpha, top_k=top_k, pool_size=pool_size
                    )

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

                    try:
                        result = invoke_rag(retriever, question, llm)

                        kw2 = keyword_accuracy_v2(result['answer'], ground_truth)
                        kw3 = keyword_accuracy_v3(result['answer'], ground_truth)
                        consecutive_errors = 0

                        all_results.append({
                            'config': config_label, 'run': 0,
                            'doc_key': doc_key,
                            'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                            'question': question, 'ground_truth': ground_truth,
                            'answer': result['answer'],
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

                        if eval_count % 5 == 0:
                            print(f"  [{eval_count}/{total_evals}] kw_v2={kw2:.3f} kw_v3={kw3:.3f} "
                                  f"doc={doc_key} type={q_type} n={result['n_retrieved']} "
                                  f"t={result['total_time']:.1f}s")

                    except Exception as e:
                        err_str = str(e)
                        errors.append({'config': config_label, 'question': question[:50], 'error': err_str})
                        all_results.append({
                            'config': config_label, 'run': 0,
                            'doc_key': doc_key,
                            'doc_type': DOC_CONFIGS[doc_key]['doc_type'],
                            'question': question, 'ground_truth': ground_truth,
                            'answer': 'ERROR', 'kw_v2': 0.0, 'kw_v3': 0.0,
                            'category': row.get('category', ''),
                            'difficulty': row.get('difficulty', ''),
                            'n_retrieved': 0, 'retrieval_time': 0,
                            'generation_time': 0, 'total_time': 0,
                            'q_type': q_type, 'retry_count': 0,
                        })
                        print(f"  ERROR [{eval_count}/{total_evals}]: {question[:40]}... -> {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= 3 and 'insufficient_quota' in err_str:
                            print(f"\n  *** API QUOTA EXHAUSTED ***")
                            quota_exhausted = True
                            break

                    # 문항별 증분 저장
                    pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

                if quota_exhausted:
                    break

                config_time = time.time() - config_start
                config_df = pd.DataFrame([r for r in all_results if r['config'] == config_label])
                print(f"  Config {config_label}: kw_v3={config_df['kw_v3'].mean():.4f}, "
                      f"time={config_time:.0f}s")
                print(f"  [SAVED] {len(all_results)} results to {CSV_PATH}")

    total_time = time.time() - exp_start
    results_df = pd.DataFrame(all_results)

    # ── STEP 4: 결과 분석 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 4: 결과 분석")
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

    # Config x Question Type kw_v3
    print(f"\n{'='*60}")
    print("Config x Question Type kw_v3")
    print('='*60)
    type_pivot = results_df.groupby(['config', 'q_type'])['kw_v3'].mean().unstack()
    print(type_pivot.round(4))

    # Config x Difficulty kw_v3
    print(f"\n{'='*60}")
    print("Config x Difficulty kw_v3")
    print('='*60)
    diff_pivot = results_df.groupby(['config', 'difficulty'])['kw_v3'].mean().unstack()
    print(diff_pivot.round(4))

    # Config x Document (doc_D 특별 분석)
    print(f"\n{'='*60}")
    print("doc_D 문항별 kw_v3 비교 (핵심 실패 영역)")
    print('='*60)
    doc_d_data = results_df[results_df['doc_key'] == 'doc_D']
    if not doc_d_data.empty:
        doc_d_pivot = doc_d_data.pivot_table(
            index='question', columns='config', values='kw_v3'
        ).round(4)
        print(doc_d_pivot)

    # Best config
    best_config = summary['kw_v3'].idxmax()
    best_v3 = summary.loc[best_config, 'kw_v3']
    ref_v3 = summary.loc['ref_v2', 'kw_v3'] if 'ref_v2' in summary.index else 0
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_config} (kw_v3={best_v3:.4f}, delta vs ref={best_v3 - ref_v3:+.4f})")
    print(f"{'='*70}")

    # ── Save ──
    results_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

    exp_report = {
        'experiment': 'exp12_retrieval_optimization',
        'phases': 'Retrieval param tuning + Embedding upgrade',
        'date': datetime.now().isoformat(),
        'baseline': 'ref_v2 (kw_v3=0.8961)',
        'n_questions': len(testset),
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count,
        'total_errors': len(errors),
        'configs': [
            {k: v for k, v in cfg.items() if k != 'embedding_obj'}
            for cfg in EVAL_CONFIGS
        ],
        'results': summary.to_dict(),
        'best_config': best_config,
        'best_kw_v3': round(best_v3, 4),
        'ref_kw_v3': round(ref_v3, 4),
        'delta_vs_ref': round(best_v3 - ref_v3, 4),
        'errors': errors[:20],
    }
    report_path = 'data/experiments/exp12_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)

    if errors:
        with open(str(EXP_DIR / 'exp12_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {report_path}")
    print(f"\n{'='*70}")
    print(f"EXP12 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
