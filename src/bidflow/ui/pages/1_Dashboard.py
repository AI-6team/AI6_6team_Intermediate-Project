import streamlit as st
import pandas as pd
from bidflow.ui.utils import upload_file, get_documents, run_extraction

st.set_page_config(page_title="Dashboard - BidFlow", page_icon="📊", layout="wide")

lang = st.session_state.get("language", "Korean")

# UI 텍스트 (간단한 매핑)
TEXT = {
    "title": {"Korean": "문서 대시보드", "English": "Document Dashboard"},
    "upload_header": {"Korean": "새로운 RFP 업로드", "English": "Upload New RFP"},
    "upload_label": {"Korean": "PDF 파일 선택", "English": "Choose a PDF file"},
    "upload_btn": {"Korean": "업로드 및 처리", "English": "Upload & Process"},
    "list_header": {"Korean": "처리된 문서 목록", "English": "Processed Documents"},
    "refresh_btn": {"Korean": "목록 새로고침", "English": "Refresh List"},
    "empty_list": {"Korean": "처리된 문서가 없습니다.", "English": "No processed documents found."},
    "action_extract": {"Korean": "분석(추출) 실행", "English": "Run Extraction"},
    "status_success": {"Korean": "업로드 성공!", "English": "Upload Successful!"},
    "extract_success": {"Korean": "분석 완료! 결과 페이지에서 확인하세요.", "English": "Extraction Complete! Check Results page."}
}

def t(key):
    return TEXT[key][lang]

st.title(t("title"))

# 1. 파일 업로드 섹션
with st.container():
    st.subheader(t("upload_header"))
    uploaded_file = st.file_uploader(t("upload_label"), type="pdf")
    
    if uploaded_file is not None:
        if st.button(t("upload_btn"), type="primary"):
            with st.spinner("Processing..."):
                try:
                    result = upload_file(uploaded_file)
                    if result:
                        st.success(f"{t('status_success')} (ID: {result.get('doc_hash')})")
                        st.info("목록을 업데이트하는 중...")
                        st.rerun()
                    else:
                        st.error("업로드 실패: 서버에서 응답이 없습니다. 터미널 로그를 확인하세요.")
                except Exception as e:
                    st.error(f"업로드 중 오류 발생: {str(e)}")
                    st.exception(e)

st.divider()

# 2. 문서 목록 섹션
st.subheader(t("list_header"))
col1, col2 = st.columns([1, 5])
with col1:
    if st.button(t("refresh_btn")):
        st.rerun()

docs = get_documents()

if not docs:
    st.info(t("empty_list"))
else:
    # docs는 이제 [{"doc_hash": "...", "filename": "...", "upload_date": "..."}] 형태의 리스트
    df = pd.DataFrame(docs)
    
    # UI에는 중요 정보만 표시
    display_df = df[["filename", "upload_date", "doc_hash"]]
    st.dataframe(display_df, width="stretch")
    
    # 선택된 문서에 대해 작업 수행 (파일명으로 선택하게 하고 ID 찾기)
    doc_options = {d["filename"]: d["doc_hash"] for d in docs}
    selected_filename = st.selectbox("Select Document to Analyze", list(doc_options.keys()))
    selected_doc_hash = doc_options[selected_filename]
    
    if st.button(t("action_extract")):
        with st.spinner("Analyzing... (This may take a while)"):
            # docs 리스트 자체가 hash 문자열 리스트임 (get_documents implementation 확인)
            # 만약 docs가 객체라면 여기서 파싱해야 함.
            # get_documents() -> list of strings (doc_hashes) based on storage.py
             
            result = run_extraction(selected_doc_hash)
            if result:
                st.success(t("extract_success"))
                st.json(result["data"]) # 임시 결과 표시
