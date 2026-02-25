import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import streamlit as st
import pandas as pd
from bidflow.ui.utils import get_documents, run_extraction

st.set_page_config(page_title="Analysis - BidFlow", page_icon="🔍", layout="wide")

lang = st.session_state.get("language", "Korean")

TEXT = {
    "title": {"Korean": "분석 결과 뷰어", "English": "Analysis Result Viewer"},
    "select_doc": {"Korean": "분석할 문서를 선택하세요", "English": "Select Document to Analyze"},
    "btn_analyze": {"Korean": "분석 실행", "English": "Run Analysis"},
    "tab_basic": {"Korean": "기본 정보 (G1)", "English": "Basic Info (G1)"},
    "tab_schedule": {"Korean": "일정 (G2)", "English": "Schedule (G2)"},
    "tab_qual": {"Korean": "자격 요건 (G3)", "English": "Qualifications (G3)"},
    "tab_score": {"Korean": "배점표 (G4)", "English": "Scoring (G4)"},
    "no_docs": {"Korean": "처리된 문서가 없습니다. 대시보드에서 먼저 업로드하세요.", "English": "No documents found. Upload first in Dashboard."},
    "evidence_header": {"Korean": "근거 (Evidence)", "English": "Evidence"},
    "snippet_label": {"Korean": "발췌문", "English": "Snippet"},
    "page_label": {"Korean": "페이지", "English": "Page"},
}

def t(key):
    return TEXT[key][lang]

st.title(t("title"))

# 문서 선택
docs = get_documents()
if not docs:
    st.warning(t("no_docs"))
else:
    doc_options = {d["filename"]: d.get("id", d.get("doc_hash")) for d in docs}
    selected_filename = st.selectbox(t("select_doc"), list(doc_options.keys()))
    doc_hash = doc_options[selected_filename]

    if st.button(t("btn_analyze"), type="primary"):
        with st.spinner("Analyzing RFP..."):
            result = run_extraction(doc_hash)
            if result:
                st.session_state[f"analysis_{doc_hash}"] = result["data"]
                st.success("Analysis Complete!")

    # 결과 표시
    result_data = st.session_state.get(f"analysis_{doc_hash}")
    
    if result_data:
        g1 = result_data.get("g1", {})
        g2 = result_data.get("g2", {})
        g3 = result_data.get("g3", {})
        g4 = result_data.get("g4", {})

        tabs = st.tabs([t("tab_basic"), t("tab_schedule"), t("tab_qual"), t("tab_score")])

        # --- Helper for displaying slots ---
        def display_slot(label, slot_data):
            val = slot_data.get("value")
            st.markdown(f"**{label}**: {val}")
            
            with st.expander(t("evidence_header")):
                ev_list = slot_data.get("evidence", [])
                for ev in ev_list:
                    st.info(f"📄 p.{ev.get('page_no')} | \"{ev.get('text_snippet')}\"")

        # Tab 1: G1
        with tabs[0]:
            st.subheader(t("tab_basic"))
            col1, col2 = st.columns(2)
            with col1:
                display_slot("Project Name", g1.get("project_name", {}))
                display_slot("Issuer", g1.get("issuer", {}))
            with col2:
                display_slot("Period", g1.get("period", {}))
                display_slot("Budget", g1.get("budget", {}))

        # Tab 2: G2
        with tabs[1]:
            st.subheader(t("tab_schedule"))
            display_slot("Submission Deadline", g2.get("submission_deadline", {}))
            display_slot("Briefing Date", g2.get("briefing_date", {}))
            display_slot("Q&A Period", g2.get("qna_period", {}))

        # Tab 3: G3
        with tabs[2]:
            st.subheader(t("tab_qual"))
            display_slot("Required Licenses", g3.get("required_licenses", {}))
            display_slot("Restrictions", g3.get("restrictions", {}))

        # Tab 4: G4
        with tabs[3]:
            st.subheader(t("tab_score"))
            items = g4.get("items", [])
            if items:
                df_score = pd.DataFrame(items)
                st.dataframe(df_score, width="stretch")
            else:
                st.info("No scoring criteria found.")
