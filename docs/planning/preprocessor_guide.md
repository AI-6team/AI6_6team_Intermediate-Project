# Data Preprocessing Strategy: Conservative Normalization

본 문서는 BidFlow 프로젝트에서 채택한 **데이터 전처리 및 정규화 전략**을 정의합니다.
PDF/HWP 혼용 환경에서 발생하는 검색 누락(Ligature 등)과 구조 왜곡을 방지하기 위해 **"보수적 정규화(Conservative Normalization)"** 원칙을 따릅니다.

---

## 1. 핵심 원칙 (Core Principles)

### 원칙 A. Raw와 Normalized의 분리
*   **Raw Text**: 사용자에게 보여주는 **근거(Evidence)**용. 원문의 표기(원문자, 단위 등)와 레이아웃을 최대한 유지.
*   **Normalized Text**: **검색(Retrieval) 및 인덱싱**용. 검색 매칭률(Recall) 극대화를 위해 표준화.

### 원칙 B. 구조 힌트(Structure Hints) 보존
*   RFP는 문서 구조(조항, 표)가 핵심이므로, **무차별적인 줄바꿈 제거(Line merging)를 금지**합니다.
*   **Paragraph Preservation**: 문단 구분을 의미하는 `\n\n` 등은 유지하되, 과도한 공백(3줄 이상의 개행)만 축소합니다.

### 원칙 C. 로직 통합 (Single Responsibility)
*   각 파서(`PDFParser`, `HWPParser`) 내부에 산재된 제어 문자 제거 로직을 모두 제거하고, **`TextPreprocessor` 모듈로 일원화**합니다.
*   이를 통해 파서 간 실험 변수(정규화 방식 차이)를 통제합니다.

---

## 2. 상세 정규화 규칙 (Normalization Rules)

### (1) Unicode Normalization (NFKC)
*   **목적**: Ligature(`ﬁ` -> `fi`), 전각 문자(Ｐ -> P) 등을 표준화하여 검색 매칭률 향상.
*   **주의**: 원문자(② -> 2)나 단위 기호(㎢ -> km2)가 변환될 수 있음.
    *   *Decision*: 검색용으로는 이득이 크므로 **적용(ON)**하되, 원문 표기가 중요한 경우를 대비해 `raw_text`는 별도 보관을 권장.

### (2) Control Character Removal
*   **목적**: 인쇄 불가능한(non-printable) 바이너리 정크 데이터 제거.
*   **Rule**: `str.isprintable()`이 `False`인 문자를 제거하되, 문맥 유지에 필수적인 **`\n` (개행), `\t` (탭)은 예외적으로 보존**합니다.

### (3) Whitespace & Line Normalization
*   **목적**: 불필요한 공백 노이즈 제거 및 문단 구조 명확화.
*   **Rule**:
    *   `re.sub(r'[^\S\n]+', ' ', text)`: 개행을 제외한 모든 공백(탭, 2칸 공백 등)을 단일 Space로 치환.
    *   `re.sub(r'\n{3,}', '\n\n', text)`: **3개 이상의 연속 개행은 2개로 축소**. (문단 구분 의미는 남기고, 빈 페이지/공백 영역만 제거)

---

## 3. 구현 가이드 (Implementation)

### 3.1 Preprocessor Class (`src/bidflow/parsing/preprocessor.py`)

```python
import unicodedata
import re

class TextPreprocessor:
    """보수적 정규화 전처리기"""
    
    def normalize(self, text: str) -> str:
        # 1. Unicode NFKC (Ligature 해제, 전각->반각)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 제어 문자 제거 (개행/탭 보존)
        text = "".join(c for c in text if c.isprintable() or c in ['\n', '\t'])
        
        # 3. 공백 정규화 (연속 공백 -> 단일, 문단 구분 보존)
        text = re.sub(r'[^\S\n]+', ' ', text)  # 개행 외 공백 정규화
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3개 이상 개행 -> 2개
        
        return text.strip()
```

### 3.2 적용 지점 (Integration Point)
*   **Before**: Parser 내부 (`hwp_parser.py` 등)에서 자체적으로 `clean_text` 수행
*   **After**: Parser는 `raw_text` 추출에 집중 -> `RFPLoader` 또는 Parser의 `parse()` 메서드 마지막 단계에서 `preprocessor.normalize()` 호출

---

## 4. 검증 및 실험 (Validation)

이 전략의 효과를 입증하기 위해 다음 지표를 모니터링합니다.

*   **키워드 매칭률**: Ligature가 포함된 단어(Example: `Office`, `Effective`) 검색 시 Hit Rate 변화.
*   **구조 유지 확인**: 정규화 후에도 Markdown Header Splitter 등이 문단을 정상적으로 인식하는지 확인(`\n\n` 보존 여부).

---

## 5. 최종 운영 원칙 (Final Consensus)

> **"Normalized 텍스트는 검색/인덱싱의 계약(Recall)을 위한 표현이고, Raw 텍스트는 사용자 신뢰/근거 제시(UX)를 위한 표현입니다. 둘은 같은 텍스트가 아니라 역할이 다른 두 개의 뷰(View)로 관리합니다."**

1.  **Storage**: `raw_text`는 반드시 보관하여 UI에 표시합니다.
2.  **Indexing**: `normalized_text`를 사용하여 검색 재현율을 극대화합니다.
3.  **Monitoring**: 정규화 과정에서 제거되거나 변형된 문자(원문자, 제어문자 등)는 로그나 테스트를 통해 지속적으로 모니터링합니다.
