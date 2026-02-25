import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from bidflow.ui.utils import health_check

st.set_page_config(
    page_title="BidFlow - AI RFP Analyzer",
    page_icon="📑",
    layout="wide"
)

# 세션 상태 초기화
if "language" not in st.session_state:
    st.session_state.language = "Korean"  # 기본값 한국어

# 사이드바 설정
with st.sidebar:
    st.title("BidFlow 🚀")
    
    # 언어 선택
    lang = st.radio(
        "Language / 언어",
        options=["Korean", "English"],
        index=0 if st.session_state.language == "Korean" else 1
    )
    st.session_state.language = lang
    
    st.divider()
    
    # 서버 상태 확인
    if health_check():
        st.success("API Server: Online 🟢")
    else:
        st.error("API Server: Offline 🔴")

# 메인 화면
if st.session_state.language == "Korean":
    st.title("BidFlow에 오신 것을 환영합니다! 👋")
    st.markdown("""
    **BidFlow**는 AI 기반의 입찰 제안요청서(RFP) 분석 및 관리 시스템입니다.
    
    ### 주요 기능
    1. **PDF 업로드 & 파싱**: 복잡한 RFP 문서를 자동으로 구조화합니다.
    2. **지능형 추출**: 사업명, 예산, 일정, 배점표 등을 AI가 추출합니다.
    3. **자동 검증**: 회사의 자격 요건 충족 여부를 자동으로 판별합니다(Green/Red/Gray).
    
    👈 **왼쪽 사이드바**에서 메뉴를 선택하여 시작하세요.
    """)
else:
    st.title("Welcome to BidFlow! 👋")
    st.markdown("""
    **BidFlow** is an AI-powered RFP analysis and management system.
    
    ### Key Features
    1. **PDF Upload & Parsing**: Automatically structure complex RFP documents.
    2. **Intelligent Extraction**: AI extracts project name, budget, schedule, scoring criteria, etc.
    3. **Automated Validation**: Automatically determine if your company meets qualification requirements (Green/Red/Gray).
    
    👈 Select a menu from the **Left Sidebar** to get started.
    """)
