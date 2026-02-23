"""
EXP13: Contextual Retrieval + 한국어 BM25 최적화

EXP12까지 retrieval 파라미터/임베딩 최적화가 한계에 도달 (best kw_v3=0.900).
핵심 실패 원인: 청크에 문서 내 위치/섹션 맥락 정보가 없어 검색 실패.
→ Contextual Retrieval: 각 청크에 LLM으로 문서 맥락 프리픽스 추가.
→ 한국어 BM25: kiwipiepy 형태소 분석으로 BM25 토크나이징 개선.

5개 설정:
  Stage 1:
    - ref_v2:        EXP12 baseline 재활용 (API 호출 불필요)
    - ctx_basic:     Contextual prefix만 적용
    - ctx_bm25_ko:   ctx_basic + Kiwi 한국어 BM25
  Stage 2:
    - ctx_multi_query: ctx_basic + multi_query
    - ctx_full:        ctx_basic + bm25_ko + multi_query

실행: cd bidflow && python -u scripts/run_exp13.py
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
from typing import List, Any

# ── Constants ──
EXP_DIR = PROJECT_ROOT / 'data' / 'exp13'
EXP_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-5-mini'
CSV_PATH = 'data/experiments/exp13_metrics.csv'
VDB_BASE = PROJECT_ROOT / 'data' / 'exp10e'
CTX_VDB_BASE = EXP_DIR  # data/exp13/vectordb_ctx_{doc_key}

DOC_CONFIGS = {
    "doc_A": {
        "name": "수협중앙회",
        "file_path": "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
        "doc_type": "text_only",
        "source_doc": "수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
        "doc_title": "수협중앙회 수산물 사이버직매장 시스템 재구축 ISMP 수립",
    },
    "doc_B": {
        "name": "한국교육과정평가원",
        "file_path": "data/raw/files/한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
        "doc_type": "table_simple",
        "source_doc": "한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
        "doc_title": "국가교육과정정보센터(NCIC) 시스템 운영 및 개선",
    },
    "doc_C": {
        "name": "국립중앙의료원",
        "file_path": "data/raw/files/국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
        "doc_type": "table_complex",
        "source_doc": "국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
        "doc_title": "2024년도 차세대 응급의료 상황관리시스템 구축",
    },
    "doc_D": {
        "name": "한국철도공사",
        "file_path": "data/raw/files/한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
        "doc_type": "mixed",
        "source_doc": "한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
        "doc_title": "예약발매시스템 개량 ISMP 용역",
    },
    "doc_E": {
        "name": "스포츠윤리센터",
        "file_path": "data/raw/files/재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
        "doc_type": "hwp_representative",
        "source_doc": "재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
        "doc_title": "스포츠윤리센터 LMS(학습지원시스템) 기능개선",
    },
}
SOURCE_TO_KEY = {v["source_doc"]: k for k, v in DOC_CONFIGS.items()}


# ═══════════════════════════════════════════════════════════════
# 평가 지표: kw_v2 + kw_v3 (EXP12와 동일)
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
# Contextual Prefix 생성
# ═══════════════════════════════════════════════════════════════

CONTEXT_PROMPT = """다음은 '{doc_title}' 문서의 청크(일부분)입니다.
이 청크가 전체 문서에서 어떤 위치와 맥락에 있는지, 아래 목차를 참고하여 간결하게 설명하세요.

<목차 참고>
{toc_summary}
</목차 참고>

<청크>
{chunk_text}
</청크>

규칙:
1. 반드시 1줄로 작성하세요.
2. 형식: [문서: {{문서명}} | {{장/절}} | {{소주제}}]
3. 장/절 번호와 제목을 정확히 포함하세요.
4. 추측하지 말고, 청크 내용과 목차에 근거하세요.

맥락 설명:"""

_ctx_llm = None

def get_ctx_llm():
    global _ctx_llm
    if _ctx_llm is None:
        _ctx_llm = ChatOpenAI(model=LLM_MODEL, temperature=0, timeout=30, max_retries=2)
    return _ctx_llm


def extract_toc_from_chunks(chunks, max_chars=3000):
    """청크들에서 목차/장절 구조를 추출하여 요약"""
    toc_lines = []
    heading_re = re.compile(
        r'^(제?\s*\d+\s*장|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]+\.?\s|'
        r'\d+\.\s|[가나다라마바사아자차카타파하]\.\s|'
        r'[一二三四五六七八九十]+\s)'
    )
    for chunk in chunks:
        lines = chunk.page_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) > 100:
                continue
            if heading_re.match(line) or line.startswith('<표>'):
                toc_lines.append(line)
    toc_text = '\n'.join(toc_lines[:100])
    if len(toc_text) > max_chars:
        toc_text = toc_text[:max_chars]
    return toc_text if toc_text else "(목차 정보 없음)"


def generate_context_prefix(doc_title, toc_summary, chunk_text, retry=2):
    """단일 청크에 대한 맥락 프리픽스 생성"""
    llm = get_ctx_llm()
    prompt = ChatPromptTemplate.from_template(CONTEXT_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # 청크 텍스트가 너무 길면 앞뒤만 사용
    if len(chunk_text) > 1500:
        chunk_text = chunk_text[:750] + "\n...\n" + chunk_text[-750:]

    for attempt in range(retry + 1):
        try:
            result = chain.invoke({
                'doc_title': doc_title,
                'toc_summary': toc_summary,
                'chunk_text': chunk_text,
            })
            prefix = result.strip()
            # 형식 검증: [문서: ...] 형태인지 확인
            if not prefix.startswith('['):
                prefix = f'[문서: {doc_title} | {prefix}]'
            if not prefix.endswith(']'):
                prefix = prefix + ']'
            return prefix
        except Exception as e:
            if attempt < retry:
                time.sleep(2)
                continue
            print(f"    [CtxPrefix] ERROR: {e}")
            return f'[문서: {doc_title}]'


def generate_contextual_chunks(doc_key, doc_config, original_chunks):
    """문서의 모든 청크에 맥락 프리픽스를 생성하고 캐싱"""
    cache_path = EXP_DIR / f'contextual_chunks_{doc_key}.json'

    # 캐시 확인
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if len(cached) == len(original_chunks):
            print(f"    [CtxPrefix] {doc_key}: 캐시 로드 ({len(cached)} chunks)")
            return cached
        print(f"    [CtxPrefix] {doc_key}: 캐시 크기 불일치 ({len(cached)} vs {len(original_chunks)}), 재생성")

    doc_title = doc_config['doc_title']
    toc_summary = extract_toc_from_chunks(original_chunks)
    print(f"    [CtxPrefix] {doc_key}: 목차 추출 완료 ({len(toc_summary)} chars)")

    contextual_data = []
    batch_size = 10
    total = len(original_chunks)

    for i, chunk in enumerate(original_chunks):
        prefix = generate_context_prefix(doc_title, toc_summary, chunk.page_content)
        contextual_data.append({
            'prefix': prefix,
            'original_text': chunk.page_content,
            'contextual_text': f'{prefix}\n{chunk.page_content}',
            'metadata': chunk.metadata,
        })

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            print(f"    [CtxPrefix] {doc_key}: {i+1}/{total} done")
            # 증분 캐싱
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(contextual_data, f, ensure_ascii=False, indent=1)

    print(f"    [CtxPrefix] {doc_key}: 완료 ({len(contextual_data)} chunks)")
    return contextual_data


def build_contextual_vdb(doc_key, contextual_data, embeddings):
    """Contextual 청크로 새 VDB 생성"""
    ctx_dir = str(CTX_VDB_BASE / f'vectordb_ctx_{doc_key}')

    # 기존 VDB 있으면 로드
    if os.path.exists(ctx_dir):
        vdb = Chroma(persist_directory=ctx_dir, embedding_function=embeddings,
                     collection_name='bidflow_rfp')
        count = vdb._collection.count()
        if count == len(contextual_data):
            print(f"    [CtxVDB] {doc_key}: 기존 VDB 로드 ({count} chunks)")
            return vdb
        print(f"    [CtxVDB] {doc_key}: 크기 불일치 ({count} vs {len(contextual_data)}), 재생성")
        # 삭제 후 재생성
        import shutil
        shutil.rmtree(ctx_dir, ignore_errors=True)

    # 새 VDB 생성
    lc_docs = []
    for item in contextual_data:
        meta = dict(item['metadata'])
        meta['has_context'] = True
        meta['context_prefix'] = item['prefix']
        lc_docs.append(Document(page_content=item['contextual_text'], metadata=meta))

    print(f"    [CtxVDB] {doc_key}: {len(lc_docs)} chunks → 임베딩 + 인덱싱...")
    os.makedirs(ctx_dir, exist_ok=True)
    vdb = Chroma.from_documents(
        lc_docs, embeddings,
        persist_directory=ctx_dir,
        collection_name='bidflow_rfp',
    )
    print(f"    [CtxVDB] {doc_key}: 완료 ({vdb._collection.count()} chunks)")
    return vdb


# ═══════════════════════════════════════════════════════════════
# 한국어 형태소 분석 BM25 (kiwipiepy)
# ═══════════════════════════════════════════════════════════════

_kiwi = None

def get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def korean_tokenizer(text):
    """Kiwi 형태소 분석: 명사, 동사, 형용사, 수식어만 추출"""
    kiwi = get_kiwi()
    tokens = kiwi.tokenize(text)
    # N: 명사류, V: 동사류, MA: 부사, MM: 관형사
    return [t.form for t in tokens
            if t.tag.startswith(('N', 'V', 'MA', 'MM')) and len(t.form) > 1]


# ═══════════════════════════════════════════════════════════════
# Retriever (EXP12와 동일 구조)
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


def build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50, use_rerank=True,
                    preprocess_func=None):
    """VDB로부터 retriever 생성. preprocess_func으로 BM25 토크나이저 커스텀 가능"""
    vector_retriever = vdb.as_retriever(search_kwargs={'k': pool_size * 2})
    result = vdb.get()
    all_docs = []
    if result and result['documents']:
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            all_docs.append(Document(page_content=text, metadata=meta))

    if all_docs:
        if preprocess_func:
            bm25 = BM25Retriever.from_documents(all_docs, preprocess_func=preprocess_func)
        else:
            bm25 = BM25Retriever.from_documents(all_docs)
    else:
        bm25 = BM25Retriever.from_documents([Document(page_content='empty')])
    bm25.k = pool_size * 2

    return ExperimentRetriever(
        vector_retriever=vector_retriever, bm25_retriever=bm25,
        weights=[round(1 - alpha, 2), round(alpha, 2)],
        top_k=top_k, pool_size=pool_size, use_rerank=use_rerank,
    )


# ═══════════════════════════════════════════════════════════════
# Multi-Query (EXP12에서 가져옴)
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
    llm = get_multi_query_llm()
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({'question': question})
        lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
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
    variants = generate_query_variants(question)
    all_queries = [question] + variants
    print(f"    [MultiQuery] {len(all_queries)} queries: original + {len(variants)} variants")

    all_docs_map = {}
    for q in all_queries:
        try:
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

    unique_docs = list(all_docs_map.values())
    print(f"    [MultiQuery] Unique candidates: {len(unique_docs)}")

    if unique_docs and retriever.use_rerank:
        from bidflow.retrieval.rerank import rerank
        final_docs = rerank(question, unique_docs, top_k=top_k, model_name=retriever.rerank_model)
    else:
        final_docs = unique_docs[:top_k]

    return final_docs


# ═══════════════════════════════════════════════════════════════
# RAG Chain (EXP12와 동일)
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
        "description": "EXP12 baseline 재활용",
        "needs_api": False,
        "stage": 0,
    },
    {
        "label": "ctx_basic",
        "description": "Stage 1: Contextual prefix만 적용",
        "needs_api": True,
        "stage": 1,
        "use_ctx_vdb": True,
        "use_bm25_ko": False,
        "use_multi_query": False,
    },
    {
        "label": "ctx_bm25_ko",
        "description": "Stage 1: Contextual prefix + Kiwi 한국어 BM25",
        "needs_api": True,
        "stage": 1,
        "use_ctx_vdb": True,
        "use_bm25_ko": True,
        "use_multi_query": False,
    },
    {
        "label": "ctx_multi_query",
        "description": "Stage 2: Contextual prefix + multi_query",
        "needs_api": True,
        "stage": 2,
        "use_ctx_vdb": True,
        "use_bm25_ko": False,
        "use_multi_query": True,
    },
    {
        "label": "ctx_full",
        "description": "Stage 2: Contextual prefix + bm25_ko + multi_query",
        "needs_api": True,
        "stage": 2,
        "use_ctx_vdb": True,
        "use_bm25_ko": True,
        "use_multi_query": True,
    },
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"EXP13: Contextual Retrieval + 한국어 BM25 최적화")
    print(f"Baseline: ref_v2 (kw_v3=0.900 multi_query)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP 0: Phase E 결과 로드 (ref_v2 베이스라인) ──
    print(f"\n{'#'*60}")
    print(f"# STEP 0: EXP12 ref_v2 결과 로드 (베이스라인)")
    print(f"{'#'*60}")

    prev_path = 'data/experiments/exp12_metrics.csv'
    assert os.path.exists(prev_path), f"EXP12 results not found: {prev_path}"
    prev_df = pd.read_csv(prev_path)
    prev_ref = prev_df[prev_df['config'] == 'ref_v2'].copy()
    assert len(prev_ref) == 30, f"Expected 30 ref_v2 results, got {len(prev_ref)}"

    # kw_v3 재채점
    prev_ref['kw_v3'] = prev_ref.apply(
        lambda r: keyword_accuracy_v3(str(r['answer']), str(r['ground_truth'])), axis=1
    )
    print(f"  ref_v2 baseline: kw_v2={prev_ref['kw_v2'].mean():.4f}, kw_v3={prev_ref['kw_v3'].mean():.4f}")

    ref_results = []
    for _, r in prev_ref.iterrows():
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

    # ── STEP 1: Testset 로드 ──
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions")

    # ── STEP 2: 원본 VDB 로드 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 2: 원본 VDB 로드 (text-embedding-3-small)")
    print(f"{'#'*60}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    doc_vdbs_orig = {}
    doc_chunks_orig = {}  # 원본 청크 (contextual prefix 생성용)
    for doc_key in DOC_CONFIGS:
        persist_dir = str(VDB_BASE / f'vectordb_c500_{doc_key}')
        if not os.path.exists(persist_dir):
            print(f"  WARNING: VDB not found: {persist_dir}")
            continue
        vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings,
                     collection_name='bidflow_rfp')
        doc_vdbs_orig[doc_key] = vdb
        # 원본 청크 로드
        result = vdb.get()
        chunks = []
        for i, text in enumerate(result['documents']):
            meta = result['metadatas'][i] if result['metadatas'] else {}
            chunks.append(Document(page_content=text, metadata=meta))
        doc_chunks_orig[doc_key] = chunks
        print(f"  {doc_key}: {len(chunks)} chunks")

    # ── STEP 3: Contextual Prefix 생성 + Contextual VDB 구축 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 3: Contextual Prefix 생성 + VDB 재인덱싱")
    print(f"{'#'*60}")

    doc_ctx_data = {}  # {doc_key: contextual_data list}
    doc_vdbs_ctx = {}  # {doc_key: Chroma VDB}

    for doc_key, chunks in doc_chunks_orig.items():
        doc_config = DOC_CONFIGS[doc_key]
        print(f"\n  --- {doc_key} ({doc_config['name']}) ---")

        # 1) Contextual prefix 생성 (캐싱 지원)
        ctx_data = generate_contextual_chunks(doc_key, doc_config, chunks)
        doc_ctx_data[doc_key] = ctx_data

        # 2) Contextual VDB 구축 (캐싱 지원)
        ctx_vdb = build_contextual_vdb(doc_key, ctx_data, embeddings)
        doc_vdbs_ctx[doc_key] = ctx_vdb

    print(f"\n  Contextual prefix 생성 + VDB 구축 완료!")

    # ── STEP 4: 평가 실행 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 4: 평가 실행")
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

        use_ctx_vdb = cfg.get('use_ctx_vdb', False)
        use_bm25_ko = cfg.get('use_bm25_ko', False)
        use_multi_query = cfg.get('use_multi_query', False)

        print(f"\n{'='*60}")
        print(f"Config: {config_label} — {cfg['description']}")
        print(f"  ctx_vdb={use_ctx_vdb}, bm25_ko={use_bm25_ko}, multi_query={use_multi_query}")
        print(f"{'='*60}")

        # Retriever 빌드
        doc_retrievers = {}
        vdb_source = doc_vdbs_ctx if use_ctx_vdb else doc_vdbs_orig
        preprocess_fn = korean_tokenizer if use_bm25_ko else None

        for doc_key, vdb in vdb_source.items():
            doc_retrievers[doc_key] = build_retriever(
                vdb, alpha=0.7, top_k=15, pool_size=50,
                preprocess_func=preprocess_fn,
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
                    result = invoke_multi_query_rag(retriever, question, llm, top_k=15)
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
                    'q_type': q_type, 'retry_count': 0,
                })

                print(f"  [{eval_count}/{total_evals}] kw_v2={kw2:.3f} kw_v3={kw3:.3f} "
                      f"doc={doc_key} type={q_type} t={result['total_time']:.1f}s "
                      f"Q: {question[:40]}...")

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
        print(f"\n  Config {config_label}: kw_v3={config_df['kw_v3'].mean():.4f}, time={config_time:.0f}s")
        print(f"  [SAVED] {len(all_results)} results to {CSV_PATH}")

    total_time = time.time() - exp_start
    results_df = pd.DataFrame(all_results)

    # ── STEP 5: 결과 분석 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 5: 결과 분석")
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

    # doc_D 문항별 비교 (핵심 실패 영역)
    print(f"\n{'='*60}")
    print("doc_D 문항별 kw_v3 비교 (핵심 실패 영역)")
    print('='*60)
    doc_d_data = results_df[results_df['doc_key'] == 'doc_D']
    if not doc_d_data.empty:
        doc_d_pivot = doc_d_data.pivot_table(
            index='question', columns='config', values='kw_v3'
        ).round(4)
        print(doc_d_pivot)

    # Q25 보안 문항 특별 분석
    print(f"\n{'='*60}")
    print("Q25 보안 문항 분석 (kw_v3=0.211 → 목표 0.7+)")
    print('='*60)
    q25_data = results_df[results_df['question'].str.contains('보안 준수사항')]
    if not q25_data.empty:
        for _, row in q25_data.iterrows():
            print(f"  [{row['config']}] kw_v3={row['kw_v3']:.3f}")
            print(f"    A: {str(row['answer'])[:150]}...")

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
        'experiment': 'exp13_contextual_retrieval',
        'description': 'Contextual Retrieval + Korean BM25 optimization',
        'date': datetime.now().isoformat(),
        'baseline': f'ref_v2 (kw_v3={ref_v3:.4f})',
        'n_questions': len(testset),
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count,
        'total_errors': len(errors),
        'configs': [{k: v for k, v in cfg.items()} for cfg in EVAL_CONFIGS],
        'results': summary.to_dict(),
        'best_config': best_config,
        'best_kw_v3': round(best_v3, 4),
        'ref_kw_v3': round(ref_v3, 4),
        'delta_vs_ref': round(best_v3 - ref_v3, 4),
        'errors': errors[:20],
    }
    report_path = 'data/experiments/exp13_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)

    if errors:
        with open(str(EXP_DIR / 'exp13_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {report_path}")
    print(f"\n{'='*70}")
    print(f"EXP13 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
