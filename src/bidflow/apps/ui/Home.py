import streamlit as st
from bidflow.utils.fonts import set_korean_font
from bidflow.db.database import init_db

init_db()
set_korean_font()

st.set_page_config(
    page_title="BidFlow",
    page_icon="📑",
    layout="wide"
)

from bidflow.apps.ui.auth import load_authenticator, require_login, register_form
from bidflow.apps.ui.session import init_app_session

# ── 미인증 상태: 로그인 / 회원가입 탭 ──────────────────────────────
if not st.session_state.get("authentication_status"):
    st.title("BidFlow")
    st.caption("지능형 입찰 분석 시스템")
    st.divider()

    tab_login, tab_register = st.tabs(["로그인", "회원가입"])

    authenticator, _ = load_authenticator()

    with tab_login:
        authenticator.login(location="main")
        status = st.session_state.get("authentication_status")
        if status is False:
            st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    with tab_register:
        register_form()

    if st.session_state.get("authentication_status"):
        st.rerun()

    st.stop()

# ── 인증 완료: 앱 메인 화면 ─────────────────────────────────────────
user_id = require_login()
init_app_session(user_id=user_id)

st.title("BidFlow: Intelligent RFP Analysis")
st.markdown("""
### 보안 강화형 지능형 입찰 분석 시스템
**Don't just Write, Find & Verify.**

BidFlow는 RFP 문서에서 필수/결격 조항을 구조적으로 추출하고,
회사 프로필과 비교하여 입찰 적격성을 판정하는 보안 중심 RAG 시스템입니다.

---

### Workflow
1. **Upload**: RFP 파일을 1개 또는 여러 개 업로드하면 단일/다문서 모드가 자동 적용됩니다.
2. **Matrix**: 추출된 30개 필수 항목(Compliance Matrix)을 확인하세요.
3. **Profile**: 회사의 보유 역량/실적 프로필을 관리하세요.
4. **Decision**: 입찰 Go/No-Go 판정 결과를 근거와 함께 확인하세요.
""")

st.info("👈 사이드바에서 'Upload' 메뉴를 선택하여 시작하세요.")
