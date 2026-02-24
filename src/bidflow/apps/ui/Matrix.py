import streamlit as st
from bidflow.ingest.storage import DocumentStore

def app():
    st.title("📊 분석 결과 매트릭스")

    # 1. 테넌트 ID 가져오기 (세션 또는 기본값)
    tenant_id = st.session_state.get("tenant_id", "default")
    
    # 2. 문서 목록 로드
    doc_store = DocumentStore()
    documents = doc_store.list_documents(tenant_id=tenant_id)
    
    if not documents:
        st.info("📂 저장된 문서가 없습니다. 파일을 업로드해주세요.")
        return

    # 3. 문서 선택 UI (리스트)
    # 파일명과 업로드 날짜를 함께 표시하여 식별 용이하게 함
    # 최신순 정렬 (upload_date 기준 내림차순)
    documents.sort(key=lambda x: x.get('upload_date') or "", reverse=True)
    
    doc_options = {
        f"{doc['filename']} ({doc.get('upload_date') or 'N/A'})": doc['doc_hash'] 
        for doc in documents
    }
    
    # Selectbox를 사용하여 한 번에 하나의 문서만 선택하도록 함 (겹침 방지)
    selected_doc_label = st.selectbox(
        "확인할 문서를 선택하세요:",
        options=list(doc_options.keys()),
        index=0
    )

    # 4. 선택된 문서의 결과 하단 표출
    if selected_doc_label:
        selected_doc_hash = doc_options[selected_doc_label]
        
        # 결과 로드
        result = doc_store.load_extraction_result(selected_doc_hash, tenant_id=tenant_id)
        
        st.divider()
        st.subheader(f"📄 {selected_doc_label} 분석 결과")
        
        if result:
            # JSON 데이터를 보기 좋게 표시
            st.json(result, expanded=True)
        else:
            st.warning("⚠️ 해당 문서에 대한 분석 결과(Extraction Result)가 아직 없습니다.")

if __name__ == "__main__":
    app()
