import streamlit as st
from bidflow.utils.fonts import set_korean_font

# 한글 폰트 설정 (Mac/Window/Linux)
set_korean_font()

st.set_page_config(
    page_title="BidFlow",
    page_icon="📑",
    layout="wide"
)

# --- Persistence Load ---
from bidflow.apps.ui.session import init_app_session

init_app_session()

st.title("BidFlow: Intelligent RFP Analysis")
st.markdown("""
### 🚀 보안 강화형 지능형 입찰 분석 시스템
**Don't just Write, Find & Verify.**

BidFlow는 RFP 문서에서 필수/결격 조항을 구조적으로 추출하고, 
회사 프로필과 비교하여 입찰 적격성을 판정하는 보안 중심 RAG 시스템입니다.

---

### Workflow
1. **Upload**: RFP 파일(PDF, HWP)을 업로드하고 분석을 시작하세요.
2. **Matrix**: 추출된 30개 필수 항목(Compliance Matrix)을 확인하세요.
3. **Profile**: 회사의 보유 역량/실적 프로필을 관리하세요.
4. **Decision**: 입찰 Go/No-Go 판정 결과를 근거와 함께 확인하세요.
""")

st.info("👈 사이드바에서 'Upload' 메뉴를 선택하여 시작하세요.")
