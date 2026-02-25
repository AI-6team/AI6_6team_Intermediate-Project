import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import streamlit as st
import json
from bidflow.ui.utils import run_validation, get_documents, run_extraction

st.set_page_config(page_title="Validation - BidFlow", page_icon="✅", layout="wide")

lang = st.session_state.get("language", "Korean")

TEXT = {
    "title": {"Korean": "자격 검증 (Validation)", "English": "Qualification Check"},
    "profile_header": {"Korean": "회사 프로필 설정", "English": "Company Profile Settings"},
    "profile_name": {"Korean": "회사명", "English": "Company Name"},
    "licenses": {"Korean": "보유 면허 (쉼표 구분)", "English": "Held Licenses (comma separated)"},
    "validate_btn": {"Korean": "적격 여부 검증", "English": "Run Validation"},
    "result_header": {"Korean": "검증 결과", "English": "Validation Results"},
    "select_doc": {"Korean": "검증할 문서 (사전 분석 필요)", "English": "Select Document (Must be analyzed first)"},
    "no_analysis": {"Korean": "이 문서는 아직 분석되지 않았습니다. 분석 페이지에서 먼저 실행하세요.", "English": "Document not analyzed yet. Run analysis first."},
}

def t(key):
    return TEXT[key][lang]

st.title(t("title"))

# 1. 회사 프로필 설정 (임시)
with st.expander(t("profile_header"), expanded=True):
    company_name = st.text_input(t("profile_name"), value="(주)테스트컴퍼니")
    licenses_str = st.text_area(t("licenses"), value="소프트웨어사업자, 정보통신공사업, 인공지능솔루션")
    
    licenses = [l.strip() for l in licenses_str.split(",") if l.strip()]
    
    profile_data = {
        "id": "company_001",
        "name": company_name,
        "data": {
            "licenses": licenses
        }
    }

st.divider()

# 2. 문서 선택 및 검증
docs = get_documents()
if docs:
    doc_options = {d["filename"]: d.get("id", d.get("doc_hash")) for d in docs}
    selected_filename = st.selectbox(t("select_doc"), list(doc_options.keys()))
    doc_hash = doc_options[selected_filename]

    # 분석 결과가 세션에 있어야 함 (MVP 단순화)
    # 실제로는 DB에서 불러오거나 다시 실행해야 함.
    # 여기서는 세션에 없으면 경고
    analysis_key = f"analysis_{doc_hash}"
    analysis_data = st.session_state.get(analysis_key)
    
    if not analysis_data:
        st.warning(t("no_analysis"))
        if st.button("Analyze Now"):
             with st.spinner("Analyzing..."):
                res = run_extraction(doc_hash)
                if res:
                    st.session_state[analysis_key] = res["data"]
                    st.rerun()
    else:
        if st.button(t("validate_btn"), type="primary"):
            # Compliance Matrix 구성 (G3 자격요건 위주)
            g3 = analysis_data.get("g3", {})
            
            # G3 데이터를 ComplianceMatrix 포맷으로 변환 (서버 모델 참고)
            # 여기서는 API가 유연하게 받도록 설계했거나, 서버가 알아서 매핑해야 하는데
            # validate API는 ComplianceMatrix 모델을 요구함.
            # 클라이언트에서 구조를 맞춰줘야 함.
            
            matrix_payload = {
                "doc_hash": doc_hash,
                "slots": {**g3} # g3의 슬롯들(required_licenses, restrictions)을 그대로 전달
            }
            
            with st.spinner("Validating..."):
                results = run_validation(matrix_payload, profile_data)
                
                if results:
                    st.subheader(t("result_header"))
                    
                    for res in results:
                        color = "green" if res["decision"] == "GREEN" else "red" if res["decision"] == "RED" else "gray"
                        icon = "✅" if color == "green" else "❌" if color == "red" else "❓"
                        
                        st.markdown(f"### {icon} {res['slot_key']}")
                        st.markdown(f"**판정**: :{color}[{res['decision']}]")
                        st.markdown(f"**이유**: {res['reasons'][0]}")
                        
                        with st.expander("근거 (Evidence)"):
                             for ev in res["evidence"]:
                                 st.info(f"\"{ev.get('text_snippet')}\"")
