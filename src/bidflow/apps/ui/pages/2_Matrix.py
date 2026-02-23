import streamlit as st
import json

st.set_page_config(page_title="Compliance Matrix", page_icon="📊", layout="wide")

st.title("Compliance Matrix (Extraction Result)")

from bidflow.apps.ui.session import init_app_session
init_app_session()

if "extraction_results" not in st.session_state:
    st.warning("먼저 문서를 업로드하고 분석을 수행하세요.")
    st.stop()

results = st.session_state["extraction_results"]
doc_hash = st.session_state.get("current_doc_hash", "Unknown")

st.caption(f"Doc ID: {doc_hash}")

# JSON 다운로드 버튼
st.download_button(
    label="Download JSON",
    data=json.dumps(results, ensure_ascii=False, indent=2),
    file_name=f"extraction_{doc_hash}.json",
    mime="application/json"
)

# 그룹별 탭 표시
tabs = st.tabs(["G1 기본정보", "G2 일정/제출", "G3 자격/결격", "G4 배점표"])

with tabs[0]:
    st.subheader("G1: 기본 정보 및 예산")
    if "g1" in results:
        data = results["g1"]
        for key, slot in data.items():
            with st.expander(f"{key}: {slot.get('value', 'N/A')}"):
                st.json(slot)
    else:
        st.info("G1 데이터가 없습니다.")

with tabs[1]:
    st.subheader("G2: 일정 및 제출 형식")
    if "g2" in results:
        data = results["g2"]
        for key, slot in data.items():
            with st.expander(f"{key}: {slot.get('value', 'N/A')}"):
                st.json(slot)
    else:
        st.info("G2 데이터가 없습니다.")

with tabs[2]:
    st.subheader("G3: 자격 및 결격 사유")
    if "g3" in results:
        data = results["g3"]
        for key, slot in data.items():
            with st.expander(f"{key}: {slot.get('value', 'N/A')}"):
                st.json(slot)
    else:
        st.info("G3 데이터가 없습니다.")

with tabs[3]:
    st.subheader("G4: 배점표 (Scoring Table)")
    if "g4" in results:
        data = results["g4"]
        # 배점표는 리스트 형태
        items = data.get("items", [])
        if items:
            st.dataframe(items)
        else:
            st.info("추출된 배점 항목이 없습니다.")
    else:
        st.info("G4 데이터가 없습니다.")
