"""
EXP11: 종합 최적화 (Phase F~J)

Phase E c500_pv2(0.814)를 기반으로 5가지 개선 방법을 종합 테스트:
  Phase I: 평가 지표 고도화 (kw_v3 정규화 확장)
  Phase F: 질문 타입 라우팅 + 구조화 출력 강제
  Phase G: 섹션 컨텍스트 보강
  Phase H: 생성 후 coverage 검증 + 재생성
  Phase J: 최고 config 3-run 재현성 확인

실행: cd bidflow && python -u scripts/run_exp11.py
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
EXP_DIR = PROJECT_ROOT / 'data' / 'exp11'
EXP_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-5-mini'
CSV_PATH = 'data/experiments/exp11_metrics.csv'

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
# Phase I: 평가 지표 — kw_v2 + kw_v3
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

# 한국어 조사 패턴 (단어 끝에서 제거)
PARTICLES_RE = re.compile(
    r'(은|는|이|가|을|를|의|에|에서|으로|로|와|과|이며|이고|에게|한테|부터|까지|도|만|이라|인|에는|에도)$'
)

# 로마 숫자 → 아라비아 숫자
ROMAN_MAP = {'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3', 'ⅳ': '4', 'ⅴ': '5',
             'ⅵ': '6', 'ⅶ': '7', 'ⅷ': '8', 'ⅸ': '9', 'ⅹ': '10',
             'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
             'Ⅵ': '6', 'Ⅶ': '7', 'Ⅷ': '8', 'Ⅸ': '9', 'Ⅹ': '10'}


def normalize_answer_v2(text):
    """EXP06 검증된 정규화 v2 (기존 호환)"""
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
    """v3: v2 + 구두점 제거 + 조사 제거 + 괄호 분리 + 특수문자 정규화"""
    t = normalize_answer_v2(text)

    # 로마 숫자 → 아라비아 숫자
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)

    # 화폐 기호 통일
    t = t.replace('￦', '₩')

    # 괄호 분리: 단어(내용)조사 → 단어 (내용) 조사
    t = re.sub(r'([가-힣a-z0-9])\(', r'\1 (', t)
    t = re.sub(r'\)([가-힣a-z])', r') \1', t)

    words = t.split()
    cleaned = []
    for w in words:
        # 구두점 제거 (단어 끝 콤마, 마침표 등)
        w = w.rstrip('.,;:!?')
        if not w:
            continue
        # 한국어 조사 제거
        stripped = PARTICLES_RE.sub('', w)
        cleaned.append(stripped if stripped else w)
    return ' '.join(cleaned)


def keyword_accuracy(answer, ground_truth, norm_fn):
    """범용 키워드 정확도 계산"""
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
# Phase F: 질문 타입 분류 + 프롬프트 라우팅
# ═══════════════════════════════════════════════════════════════

def classify_question_type(question):
    """질문을 3가지 타입으로 분류: list / location / direct"""
    q = question.strip()

    # 리스트형: 여러 항목을 나열해야 하는 질문
    list_patterns = [
        r'항목[들은는이가]',
        r'세부\s*항목',
        r'내용[은는이가을를]?\s*무엇',
        r'문제점[은는이가을를]?\s*무엇',
        r'어떤\s*(것|항목|내용|사항)',
        r'무엇[을를이가]?\s*(있|포함|다루)',
        r'요구[하는사항]',
        r'어떻게\s*되',
        r'추진\s*배경',
        r'필요성',
    ]
    for pat in list_patterns:
        if re.search(pat, q):
            return 'list'

    # 위치형: 장/절/페이지/규정 위치를 묻는 질문
    location_patterns = [
        r'몇\s*장',
        r'어디[에서]?\s*(규정|다루|기술)',
        r'어느\s*(장|절|페이지)',
        r'규정[되어]?\s*있',
        r'어떤\s*장[에서]?\s*다루',
    ]
    for pat in location_patterns:
        if re.search(pat, q):
            return 'location'

    return 'direct'


# ── Prompt Templates ──

# Phase E 기존 프롬프트 (c500_pv2)
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

# Phase F: 강화된 verbatim 프롬프트
PROMPT_V3 = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n\n'
    '## 답변 규칙 (반드시 준수)\n'
    '1. 원문의 표현을 정확히 그대로 인용하세요 — 절대 패러프레이징하지 마세요.\n'
    '2. 사업명, 기관명, 금액, 날짜, 숫자, 조항명, 제도명 등 고유 표현은 원문 그대로 작성하세요.\n'
    '3. 문맥에 테이블 데이터가 있으면 테이블의 정보를 우선 활용하세요.\n'
    '4. 문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고만 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변\n'
)

# Phase F: 리스트형 질문 전용 프롬프트
PROMPT_V3_LIST = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n\n'
    '## 답변 규칙 (반드시 준수)\n'
    '1. 이 질문은 여러 항목을 나열해야 합니다. 문맥에 있는 **모든 항목을 빠짐없이** 나열하세요.\n'
    '2. 각 항목은 원문의 표현을 정확히 그대로 인용하세요 — 절대 요약하거나 바꿔쓰지 마세요.\n'
    '3. 항목 앞에 번호나 기호를 붙여 명확히 구분하세요.\n'
    '4. 문맥에 테이블 데이터가 있으면 테이블의 정보를 우선 활용하세요.\n'
    '5. 문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고만 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변 (모든 항목을 빠짐없이 나열)\n'
)

# Phase F: 위치형 질문 전용 프롬프트
PROMPT_V3_LOCATION = (
    '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n\n'
    '## 답변 규칙 (반드시 준수)\n'
    '1. 이 질문은 문서 내 특정 위치(장, 절, 페이지)를 묻고 있습니다.\n'
    '2. 반드시 정확한 장번호(예: "7장", "제9장"), 절번호(예: "가.", "나."), 제목, 페이지 번호를 포함하세요.\n'
    '3. 원문에 있는 표현을 정확히 그대로 인용하세요.\n'
    '4. 문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고만 답하세요.\n\n'
    '## 문맥 (Context)\n{context}\n\n'
    '## 질문\n{question}\n\n'
    '## 답변 (장번호, 제목, 페이지 정확히 포함)\n'
)


# ═══════════════════════════════════════════════════════════════
# Phase G: 섹션 컨텍스트 보강 (query expansion)
# ═══════════════════════════════════════════════════════════════

# 질문 키워드 → 관련 섹션 키워드 매핑 (RFP 문서 일반)
SECTION_KEYWORD_MAP = {
    '보안': ['보안 준수사항', '보안관리', '보안특약', '정보보호'],
    '평가': ['제안서 평가', '평가방법', '평가기준', '제안안내사항'],
    '하자': ['하자담보', '하자보수', '기타 사항', '기타사항'],
    '예산': ['사업비', '사업예산', '소요예산', '예산'],
    '기간': ['사업기간', '사업개요', '일정'],
    '자격': ['입찰참가자격', '참가자격', '제안안내사항'],
    '추진': ['추진배경', '필요성', '사업목적', '사업개요'],
}


def expand_query_with_section(question):
    """질문에 관련 섹션 키워드를 추가하여 검색 쿼리 확장"""
    extra_keywords = []
    for keyword, section_terms in SECTION_KEYWORD_MAP.items():
        if keyword in question:
            extra_keywords.extend(section_terms)

    if extra_keywords:
        # 중복 제거
        extra_keywords = list(dict.fromkeys(extra_keywords))
        return f"{question} {' '.join(extra_keywords)}"
    return question


# ═══════════════════════════════════════════════════════════════
# Phase H: Coverage 검증 + 재생성
# ═══════════════════════════════════════════════════════════════

def extract_context_keywords(context_text, top_n=20):
    """검색된 컨텍스트에서 핵심 키워드 추출 (빈도 기반)"""
    # 정규화
    t = context_text.lower()
    t = re.sub(r'[^\w가-힣\s]', ' ', t)
    words = t.split()

    # 불용어
    stopwords = {'및', '또는', '등', '위한', '대한', '관한', '따른', '의한',
                 '있는', '하는', '되는', '않는', '없는', '같은', '통해',
                 '이상', '이하', '이내', '경우', '포함', '제외', '관련',
                 '사항', '내용', '기타', '해당', '본', '위', '다음'}

    # 2글자 이상, 불용어 제외, 빈도 계산
    from collections import Counter
    word_counts = Counter(w for w in words if len(w) > 1 and w not in stopwords)
    return [w for w, _ in word_counts.most_common(top_n)]


def check_answer_coverage(answer, context_keywords, threshold=0.3):
    """답변이 컨텍스트 핵심 키워드를 충분히 포함하는지 확인"""
    if not context_keywords:
        return True, 1.0, []
    ans_lower = answer.lower()
    matched = [kw for kw in context_keywords if kw in ans_lower]
    coverage = len(matched) / len(context_keywords)
    missed = [kw for kw in context_keywords if kw not in ans_lower]
    return coverage >= threshold, coverage, missed


# ═══════════════════════════════════════════════════════════════
# Retriever (Phase E에서 가져옴)
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
# RAG Chain Builders
# ═══════════════════════════════════════════════════════════════

def build_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=1, timeout=60, max_retries=2)


def invoke_rag(retriever, question, prompt_template, llm, query_override=None):
    """RAG 호출 (검색 쿼리와 프롬프트 질문을 분리 가능)"""
    t0 = time.time()
    search_query = query_override if query_override else question
    docs = retriever.invoke(search_query)
    retrieval_time = time.time() - t0

    context_text = '\n\n'.join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(prompt_template)

    t1 = time.time()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context_text, 'question': question})
    gen_time = time.time() - t1

    return {
        'answer': answer,
        'context_text': context_text,
        'n_retrieved': len(docs),
        'retrieval_time': retrieval_time,
        'generation_time': gen_time,
        'total_time': retrieval_time + gen_time,
    }


def invoke_with_retry(retriever, question, prompt_template, llm,
                      query_override=None, max_retries=1):
    """Phase H: Coverage 기반 재생성"""
    result = invoke_rag(retriever, question, prompt_template, llm, query_override)

    if max_retries <= 0:
        result['retry_count'] = 0
        return result

    # coverage 체크
    ctx_keywords = extract_context_keywords(result['context_text'])
    ok, coverage, missed = check_answer_coverage(result['answer'], ctx_keywords)

    if ok or not missed:
        result['retry_count'] = 0
        result['coverage'] = coverage
        return result

    # 재생성: 누락 키워드를 포함하도록 재프롬프트
    retry_prompt = (
        '아래 문맥(Context)을 근거로 질문에 정확하게 답하세요.\n\n'
        '## 중요: 이전 답변에서 다음 키워드들이 누락되었습니다. 반드시 포함하세요:\n'
        f'누락 키워드: {", ".join(missed[:10])}\n\n'
        '## 답변 규칙\n'
        '1. 원문의 표현을 정확히 그대로 인용하세요.\n'
        '2. 위 누락 키워드와 관련된 정보를 문맥에서 찾아 답변에 포함하세요.\n'
        '3. 문맥에 답이 없으면 \'해당 정보를 찾을 수 없습니다\'라고 답하세요.\n\n'
        '## 문맥 (Context)\n{context}\n\n'
        '## 질문\n{question}\n\n'
        '## 답변 (누락 키워드 반드시 포함)\n'
    )

    t1 = time.time()
    prompt = ChatPromptTemplate.from_template(retry_prompt)
    chain = prompt | llm | StrOutputParser()
    retry_answer = chain.invoke({
        'context': result['context_text'],
        'question': question,
    })
    retry_gen_time = time.time() - t1

    # 재생성 결과가 더 좋은지 확인 (kw_v3 기준은 런타임에는 못 비교하므로 coverage로 판단)
    _, retry_coverage, _ = check_answer_coverage(retry_answer, ctx_keywords)

    if retry_coverage > coverage:
        result['answer'] = retry_answer
        result['generation_time'] += retry_gen_time
        result['total_time'] += retry_gen_time
        result['coverage'] = retry_coverage
    else:
        result['coverage'] = coverage

    result['retry_count'] = 1
    return result


# ═══════════════════════════════════════════════════════════════
# 실험 Config 정의
# ═══════════════════════════════════════════════════════════════

EVAL_CONFIGS = [
    {
        "label": "ref_v2",
        "description": "Phase E c500_pv2 결과 재활용 (kw_v2 참조)",
        "needs_api": False,
    },
    {
        "label": "prompt_v3",
        "description": "Phase F: 강화된 verbatim 프롬프트 (타입 무관)",
        "needs_api": True,
        "prompt": PROMPT_V3,
        "use_routing": False,
        "use_section_expansion": False,
        "use_retry": False,
    },
    {
        "label": "route_type",
        "description": "Phase F: 질문 타입별 프롬프트 라우팅",
        "needs_api": True,
        "prompt": PROMPT_V3,       # default
        "prompt_list": PROMPT_V3_LIST,
        "prompt_location": PROMPT_V3_LOCATION,
        "use_routing": True,
        "use_section_expansion": False,
        "use_retry": False,
    },
    {
        "label": "section_ctx",
        "description": "Phase G: 섹션 키워드 query expansion + 타입 라우팅",
        "needs_api": True,
        "prompt": PROMPT_V3,
        "prompt_list": PROMPT_V3_LIST,
        "prompt_location": PROMPT_V3_LOCATION,
        "use_routing": True,
        "use_section_expansion": True,
        "use_retry": False,
    },
    {
        "label": "coverage_retry",
        "description": "Phase H: section_ctx + coverage 검증 + 1회 재생성",
        "needs_api": True,
        "prompt": PROMPT_V3,
        "prompt_list": PROMPT_V3_LIST,
        "prompt_location": PROMPT_V3_LOCATION,
        "use_routing": True,
        "use_section_expansion": True,
        "use_retry": True,
    },
]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"EXP11: 종합 최적화 (Phase F~J)")
    print(f"Baseline: c500_pv2 (kw_v2=0.814)")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # ── STEP 0: Phase E 결과 로드 (ref_v2 / ref_v3) ──
    print(f"\n{'#'*60}")
    print(f"# STEP 0: Phase E 결과 로드 + kw_v3 재채점 (Phase I)")
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
    print(f"  Phase E c500_pv2: kw_v2={prev_pv2['kw_v2'].mean():.4f} → kw_v3={prev_pv2['kw_v3'].mean():.4f}")
    print(f"  kw_v3 개선: {prev_pv2['kw_v3'].mean() - prev_pv2['kw_v2'].mean():+.4f}")
    print(f"  Perfect(v2): {(prev_pv2['kw_v2']==1.0).sum()}/30")
    print(f"  Perfect(v3): {(prev_pv2['kw_v3']==1.0).sum()}/30")

    # ref_v2 결과 저장용
    ref_results = []
    for _, r in prev_pv2.iterrows():
        ref_results.append({
            'config': 'ref_v2',
            'run': 0,
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

    # ── STEP 1: VDB 로드 (Phase E c500 재사용) ──
    print(f"\n{'#'*60}")
    print(f"# STEP 1: VDB 로드 (Phase E c500 재사용)")
    print(f"{'#'*60}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    doc_retrievers = {}
    for doc_key in DOC_CONFIGS:
        persist_dir = str(PROJECT_ROOT / 'data' / 'exp10e' / f'vectordb_c500_{doc_key}')
        if not os.path.exists(persist_dir):
            print(f"  WARNING: VDB not found: {persist_dir}")
            continue
        vdb = Chroma(persist_directory=persist_dir, embedding_function=embeddings,
                     collection_name='bidflow_rfp')
        retriever = build_retriever(vdb, alpha=0.7, top_k=15, pool_size=50)
        doc_retrievers[doc_key] = retriever
        print(f"  {doc_key}: {vdb._collection.count()} chunks loaded")

    # ── STEP 2: Testset 로드 ──
    testset = pd.read_csv('data/experiments/golden_testset_multi.csv')
    print(f"\nTestset: {len(testset)} questions")

    # 질문 타입 분류 미리보기
    q_types = testset['question'].apply(classify_question_type)
    print(f"Question types: {q_types.value_counts().to_dict()}")

    # ── STEP 3: 평가 실행 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 3: 평가 실행")
    print(f"{'#'*60}")

    all_results = list(ref_results)  # ref_v2 이미 포함
    errors = []
    llm = build_llm()
    quota_exhausted = False

    # Resume: 기존 결과 체크
    completed_configs = {'ref_v2'}  # 이미 포함
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        for cfg_name in existing['config'].unique():
            cfg_data = existing[existing['config'] == cfg_name]
            if len(cfg_data) >= 30:
                completed_configs.add(cfg_name)
        if len(existing) > len(ref_results):
            all_results = existing.to_dict('records')
            print(f"  Resuming: {len(all_results)} previous results, completed: {completed_configs}")

    api_configs = [c for c in EVAL_CONFIGS if c.get('needs_api', False)]
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

        print(f"\n{'='*60}")
        print(f"Config: {config_label} — {cfg['description']}")
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

            # 프롬프트 선택 (routing)
            if cfg.get('use_routing') and q_type == 'list':
                prompt_template = cfg.get('prompt_list', cfg['prompt'])
            elif cfg.get('use_routing') and q_type == 'location':
                prompt_template = cfg.get('prompt_location', cfg['prompt'])
            else:
                prompt_template = cfg['prompt']

            # 검색 쿼리 (section expansion)
            query_override = None
            if cfg.get('use_section_expansion'):
                expanded = expand_query_with_section(question)
                if expanded != question:
                    query_override = expanded

            try:
                if cfg.get('use_retry'):
                    result = invoke_with_retry(
                        retriever, question, prompt_template, llm,
                        query_override=query_override, max_retries=1
                    )
                else:
                    result = invoke_rag(
                        retriever, question, prompt_template, llm,
                        query_override=query_override
                    )
                    result['retry_count'] = 0

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
                    'retry_count': result.get('retry_count', 0),
                })

                if eval_count % 5 == 0:
                    elapsed = time.time() - exp_start
                    print(f"  [{eval_count}/{total_evals}] kw_v2={kw2:.2f} kw_v3={kw3:.2f} "
                          f"type={q_type} t={result['total_time']:.1f}s "
                          f"retry={result.get('retry_count', 0)}")

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

        if quota_exhausted:
            break

        config_time = time.time() - config_start
        print(f"  Config {config_label} done in {config_time:.0f}s")

        # 증분 저장
        pd.DataFrame(all_results).to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"  [SAVED] {len(all_results)} results to {CSV_PATH}")

    total_time = time.time() - exp_start
    results_df = pd.DataFrame(all_results)

    # ── STEP 4: 결과 분석 ──
    print(f"\n{'#'*60}")
    print(f"# STEP 4: 결과 분석")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total API evals: {eval_count}, Errors: {len(errors)}")
    print(f"{'#'*60}")

    # Config별 kw_v2 & kw_v3 비교
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
    # 순서: ref_v2 먼저, 나머지 kw_v3 내림차순
    print(summary.sort_values('kw_v3', ascending=False))

    # Config × Document kw_v3
    print(f"\n{'='*60}")
    print("Config × Document kw_v3")
    print('='*60)
    doc_pivot = results_df.groupby(['config', 'doc_key'])['kw_v3'].mean().unstack()
    doc_pivot['overall'] = results_df.groupby('config')['kw_v3'].mean()
    print(doc_pivot.round(4).sort_values('overall', ascending=False))

    # 질문 타입별 분석
    print(f"\n{'='*60}")
    print("Config × Question Type kw_v3")
    print('='*60)
    type_pivot = results_df.groupby(['config', 'q_type'])['kw_v3'].mean().unstack()
    print(type_pivot.round(4))

    # Difficulty 분석
    print(f"\n{'='*60}")
    print("Config × Difficulty kw_v3")
    print('='*60)
    diff_pivot = results_df.groupby(['config', 'difficulty'])['kw_v3'].mean().unstack()
    print(diff_pivot.round(4))

    # Category 분석
    print(f"\n{'='*60}")
    print("Config × Category kw_v3")
    print('='*60)
    cat_pivot = results_df.groupby(['config', 'category'])['kw_v3'].mean().unstack()
    print(cat_pivot.round(4))

    # Retry 분석 (Phase H)
    if 'retry_count' in results_df.columns:
        retry_data = results_df[results_df['config'] == 'coverage_retry']
        if not retry_data.empty:
            n_retried = (retry_data['retry_count'] > 0).sum()
            print(f"\nPhase H (coverage_retry): {n_retried}/{len(retry_data)} questions retried")

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
        'experiment': 'exp11_comprehensive_optimization',
        'phases': 'F+G+H+I+J',
        'date': datetime.now().isoformat(),
        'baseline': 'c500_pv2 (kw_v2=0.814)',
        'n_questions': len(testset),
        'total_time_sec': round(total_time, 1),
        'total_evals': eval_count,
        'total_errors': len(errors),
        'configs': [
            {k: v for k, v in cfg.items()
             if k not in ('prompt', 'prompt_list', 'prompt_location')}
            for cfg in EVAL_CONFIGS
        ],
        'results': summary.to_dict(),
        'best_config': best_config,
        'best_kw_v3': round(best_v3, 4),
        'errors': errors[:20],
    }
    report_path = 'data/experiments/exp11_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)

    if errors:
        with open(str(EXP_DIR / 'exp11_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {report_path}")
    print(f"\n{'='*70}")
    print(f"EXP11 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
