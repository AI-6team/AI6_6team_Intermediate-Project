import streamlit as st
import os
import time
from bidflow.ingest.loader import RFPLoader
from bidflow.extraction.pipeline import ExtractionPipeline
from bidflow.security.rails.input_rail import SecurityException

st.set_page_config(page_title="Upload RFP", page_icon="asd")

st.title("RFP 문서 업로드")

uploaded_file = st.file_uploader(
    "RFP 문서를 업로드하세요 (PDF, HWP, DOCX, HWPX)",
    type=["pdf", "hwp", "docx", "hwpx"]
)

if uploaded_file:
    st.write(f"File: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size} bytes")
    
    if st.button("분석 시작 (Start Analysis)"):
        # 0. 상태 초기화 (이전 결과 제거)
        if "extraction_results" in st.session_state:
            del st.session_state["extraction_results"]
        st.session_state["analysis_success"] = False
        
        with st.status("문서 처리 중...", expanded=True) as status:
            try:
                # 1. Ingest (Loader)
                st.write("📂 문서를 서버에 저장하고 파싱합니다...")
                loader = RFPLoader()
                doc_hash = loader.process_file(uploaded_file, uploaded_file.name)
                st.write(f"✅ 파싱 완료 (ID: {doc_hash})")

                # 2. Extract (Pipeline)
                st.write("🧠 Compliance Matrix 추출을 시작합니다... (LLM)")
                pipeline = ExtractionPipeline()
                results = pipeline.run(doc_hash)
                st.write("✅ 추출 완료!")
            except SecurityException as e:
                # 보안 위협 탐지 (Prompt Injection 등)
                status.update(label="🚨 보안 위협 탐지!", state="error", expanded=True)
                st.error(f"보안 위협이 탐지되어 처리가 차단되었습니다: {e}")
                st.stop()
            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"처리 중 오류: {e}")
                st.stop()
            
            # 3. Session State 저장
            st.session_state["current_doc_hash"] = doc_hash
            st.session_state["extraction_results"] = results
            st.session_state["analysis_success"] = True
            
            # 4. Persistence Save (자동 저장)
            from bidflow.ingest.storage import DocumentStore
            store = DocumentStore()
            store.save_extraction_result(doc_hash, results)
            store.save_session_state({"current_doc_hash": doc_hash})
            st.toast("분석 결과가 저장되었습니다.", icon="💾")
            
            status.update(label="분석 완료!", state="complete", expanded=False)
            
if st.session_state.get("analysis_success"):
    st.success("분석이 완료되었습니다!")
    # Streamlit 1.30+ 지원: st.page_link
    try:
        st.page_link("pages/2_Matrix.py", label="결과 보기 (Go to Matrix)", icon="📊")
    except AttributeError:
        # 구버전 Fallback
        if st.button("결과 보기 (Go to Matrix)"):
            st.switch_page("pages/2_Matrix.py")
