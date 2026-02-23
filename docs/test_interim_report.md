# 📄 RAG 시스템 최적화 테스트 중간 보고서

**작성일**: 2026년 2월 11일  
**작성자**: AI 6팀 김슬기  
**프로젝트**: 입찰 공고(RFP) 분석 및 예산 추출 자동화

---

## 1. 개요 (Overview)
본 프로젝트는 공공 입찰 공고(RFP) 문서에서 **"총 사업비(예산)"**와 같은 핵심 정보를 정확하게 추출하는 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 것을 목표로 합니다.  
초기 단계에서 단순 벡터 검색(Vector Search)만으로는 표나 서식에 숨겨진 구체적인 수치를 찾는 데 한계가 있었으며, 이를 극복하기 위해 다양한 검색 및 후처리 기술을 도입하고 실험했습니다.

## 2. 주요 문제점 및 해결 과정 (Challenges & Solutions)

### 🛑 문제점 1: 문서 파싱(HWP/PDF) 프레임워크 최적화
- **현상**: 
  - **HWP**: 구형 HWP 파일이나 특정 인코딩 문서에서 텍스트 깨짐(`얮`, `뀀`) 및 표 데이터 누락. (Windows 종속성 제거 필요)
  - **PDF**: 복잡한 레이아웃(다단, 표)에서 텍스트 순서가 뒤섞이는 문제.
- **해결**: 
  - **HWP**: `olefile` 라이브러리를 사용하여 OLE 구조를 직접 분석하고, `zlib` 압축 해제 및 `UTF-16LE` 디코딩을 수행하는 **Custom Parser** 구현. (Linux 환경 호환성 확보)
  - **PDF**: `pdfplumber`와 `PyMuPDF(fitz)`를 병행 사용하여, 텍스트 추출 속도를 높이고 표 데이터(Table)는 정밀하게 파싱하는 하이브리드 방식 적용.

### 🛑 문제점 2: 검색 정확도 저조 (초기 61%)
- **현상**: "예산은 얼마인가?"라는 질문에 대해, 실제 금액이 적힌 문서 대신 "예산"이라는 단어가 많이 나오는 서식(별지)이나 일반 현황 문서가 검색됨.
- **해결**: 
  - **Hybrid Search 도입**: 키워드 매칭(BM25)과 의미 기반 검색(Vector)을 결합하여 숫자와 고유 명사 검색 능력 강화.
  - **Chunk Size 축소**: 1000자 → 500자로 줄여 정보 밀도를 높임.

### 🛑 문제점 3: 정답 문서가 필터링으로 삭제됨
- **현상**: 노이즈 제거를 위해 도입한 필터링 로직이 "별지 서식"에 포함된 진짜 예산표까지 삭제해버림.
- **해결**: 
  - **스마트 필터링**: 단순히 "서식" 키워드만 보는 것이 아니라, **"금액 패턴(Regex)"**이 포함된 경우 무조건 보존하도록 안전장치 추가.

---

## 3. 기술 스택 (Technology Stack)

| 구분 | 기술 | 용도 |
|------|------|------|
| **LLM 프레임워크** | **LangChain** | RAG 파이프라인 구성, Retriever 통합 |
| **LLM API** | **OpenAI GPT-4o** | 답변 생성 및 추론 (Generation) |
| **임베딩** | **OpenAI text-embedding-3-small** | 텍스트 벡터 변환 (Embedding) |
| **벡터 DB** | **Chroma** | 벡터 저장 및 유사도 검색 (Vector Store) |
| **키워드 검색** | **BM25** (rank-bm25) | 키워드 기반 검색 (Hybrid Search) |
| **리랭킹** | **FlagEmbedding** (BAAI/bge-reranker-v2-m3) | 검색 결과 재순위화 (Reranking) |
| **문서 파싱** | **pdfplumber**, **PyMuPDF**, **olefile** | PDF, HWP 텍스트 및 표 추출 |
| **백엔드 API** | **FastAPI** | REST API 서버 구축 |
| **데이터 분석** | **Pandas**, **Matplotlib**, **Seaborn** | 실험 결과 분석 및 시각화 |
| **설정 관리** | **PyYAML** | 실험 및 환경 설정 관리 |

## 4. 시스템 구성 및 변수 통제 (System Configuration)

이번 실험에서는 **검색 전략(Retrieval Strategy)**과 **청크 크기(Chunk Size)**의 영향을 집중적으로 분석하기 위해, 임베딩 모델 등 기타 변수는 고정했습니다.

| 구성 요소 | 설정 값 | 비고 |
| :--- | :--- | :--- |
| **OS Environment** | **Linux (Ubuntu)** | **(Target)** 운영 서버 환경 (Windows 개발 환경과 호환성 검증 완료) |
| **Embedding Model** | `text-embedding-3-small` | **(Fixed)** 비용 효율성과 성능 균형을 위해 고정. 향후 한국어 특화 모델 비교 예정. |
| **LLM** | `gpt-4o` | **(Fixed)** 추론 및 답변 생성용 |
| **Vector DB** | `Chroma` | 로컬 테스트 및 임베딩 저장 |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | (Case 7, 9 적용) 다국어/한국어 성능 우수 모델 |

## 5. 적용된 핵심 기술 (Key Features)

### 🛠️ 1. Hybrid Search (BM25 + Vector)
- **설명**: 키워드 검색(BM25) 50% + 벡터 검색(Vector) 50% 가중치 적용.
- **효과**: "493,763,000원" 같은 구체적인 수치나 희소한 키워드 검색 성능 대폭 향상.

### 🛠️ 2. Smart Filtering & Regex
- **설명**: 검색된 문서 중 불필요한 서식(재무제표, 실적증명원 등)을 제거하되, 정규식(`\d+(,\d{3})*원`)을 통해 구체적인 금액이 명시된 문서는 보호.
- **효과**: LLM이 불필요한 정보에 현혹되는 것(Hallucination) 방지.

### 🛠️ 3. High Recall Reranking
- **설명**: 1차 검색에서 **50개(Top-50)**의 문서를 넓게 가져온 뒤, 정밀한 **Cross-Encoder 모델(`BAAI/bge-reranker-v2-m3`)**로 재순위화(Reranking)하여 상위 10개를 추출.
- **효과**: 검색 모델이 놓친 정답(20~30위권)을 상위권으로 끌어올림.

---

## 6. 실험 결과 요약 (Experiment Results)

`rag_optimizer.py`를 통해 다양한 파라미터 조합(Grid Search)을 테스트한 결과입니다.

| Case | 설정 (Config) | 정확도 (Recall) | 소요 시간 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1** | Vector Only (Chunk 1000) | 61.0% | 161s | 기준점 (Baseline) |
| **Case 2** | Hybrid (Alpha 0.5) | 64.0% | 166s | 하이브리드 도입으로 소폭 상승 |
| **Case 4** | Chunk 500 + K=6 | 70.0% | 167s | 청크 크기를 줄여 정밀도 향상 |
| **Case 5** | **Chunk 500 + K=10** | **73.0%** | **166s** | **[Best] 속도와 정확도의 균형 최적** |
| **Case 7** | Filtering + Rerank | 66.0% | 370s | 필터링이 과도하게 적용되어 일부 정답 유실 |
| **Case 9** | **High Recall (K=50) + Rerank** | **72.0%** | **551s** | 정확도는 높으나 속도가 느림 (GPU 권장) |

### 📊 상세 분석 (Detailed Analysis)

1. **청크 크기 (Chunk Size): 1000 vs 500**
   - **결과**: 1000자(Case 2, 64%) → 500자(Case 4, 70%)로 줄였을 때 정확도가 **6%p 상승**했습니다.
   - **원인**: 예산 정보는 보통 표나 짧은 문단에 존재합니다. 청크가 크면 불필요한 서식이나 앞뒤 문맥(노이즈)이 섞여 검색 정확도를 떨어뜨리는 것으로 분석됩니다.

2. **검색 방식 (Retrieval Method): Vector vs Hybrid**
   - **결과**: Vector Only(Case 1, 61%) → Hybrid(Case 2, 64%)로 변경 시 **3%p 상승**했습니다.
   - **원인**: "493,763,000원" 같은 구체적인 수치나 "소요예산" 같은 특정 단어는 벡터 유사도보다 키워드 매칭(BM25)이 훨씬 강력하게 잡아냅니다.

3. **검색 개수 (Top-K): 6 vs 10**
   - **결과**: K=6(Case 4, 70%) → K=10(Case 5, 73%)으로 늘렸을 때 **3%p 추가 상승**했습니다.
   - **원인**: 정답 문서가 1~5위가 아닌 7~10위권에 위치하는 경우가 다수 발견되었습니다. LLM의 Context Window가 충분하므로 K를 늘리는 것이 유리합니다.

4. **필터링 (Filtering): On vs Off**
   - **결과**: 필터링 적용 시(Case 6, 68%) 오히려 미적용(Case 5, 73%)보다 성능이 **하락**했습니다.
   - **원인**: "별지 서식" 같은 제목을 가진 문서에 실제 예산표가 포함된 경우가 많아, 필터링 로직이 정답까지 삭제하는 과유불급 현상이 발생했습니다.

5. **리랭킹 (Reranking): On vs Off**
   - **결과**: 리랭킹 적용(Case 9, 72%)이 미적용(Case 5, 73%)과 비슷하거나 소폭 낮았습니다.
   - **원인**: 현재 데이터셋에서는 Hybrid Search(BM25+Vector)가 이미 상위권 문서를 잘 찾고 있어, 리랭커의 이득이 크지 않았습니다. 반면 소요 시간은 3배 이상 증가하여 가성비가 낮습니다.

---

## 7. 최종 결론 및 향후 계획 (Conclusion)

### ✅ 최종 설정 (Best Configuration)
현재 `configs/dev.yaml`에 적용된 최적 설정입니다.
```yaml
CHUNK_SIZE: 500
CHUNK_OVERLAP: 50
RETRIEVER_K: 10
USE_HYBRID: true
HYBRID_ALPHA: 0.5
USE_RERANK: false  # 속도 이슈로 기본은 Off, 필요시 On
```

### 🚀 향후 계획
1.  **GPU 환경 도입**: 리랭킹(Reranking) 모델의 속도 문제를 해결하기 위해 GPU 서버 배포 고려.
2.  **테이블 파싱 고도화**: 현재 텍스트 위주의 파싱을 넘어, 복잡한 표 구조를 HTML/Markdown으로 변환하여 LLM의 이해도 향상.
3.  **사용자 피드백 루프**: 실제 서비스 운영 중 발생하는 오답 케이스를 수집하여 테스트셋(Test Case)에 지속적으로 추가.
4.  **임베딩 모델 고도화 (Optional)**: 현재 `text-embedding-3-small`을 사용 중이나, 향후 한국어 성능이 더 뛰어난 `upstage/solar-embedding-1-large` 등으로 교체 테스트 고려.

