import streamlit as st
from bidflow.ingest.storage import DocumentStore
from bidflow.domain.models import CompanyProfile


def init_app_session(user_id: str = "global"):
    """
    모든 페이지에서 공통으로 호출.
    저장된 세션 상태와 프로필을 로드하여 st.session_state에 복원.
    user_id가 주어지면 해당 사용자 공간에서 로드합니다.
    """
    store = DocumentStore(user_id=user_id)

    # 1. 세션 상태 복원 (Doc Hash, Extraction results)
    if "current_doc_hash" not in st.session_state:
        saved_session = store.load_session_state()
        if saved_session and "current_doc_hash" in saved_session:
            doc_hash = saved_session["current_doc_hash"]

            if store.load_document(doc_hash):
                st.session_state["current_doc_hash"] = doc_hash

                saved_result = store.load_extraction_result(doc_hash)
                if saved_result:
                    st.session_state["extraction_results"] = saved_result
                    if "session_restored" not in st.session_state:
                        st.toast(f"이전 작업 세션을 복원했습니다. (Doc: {doc_hash})", icon="🔄")
                        st.session_state["session_restored"] = True

    # 2. 회사 프로필 복원
    if "company_profile" not in st.session_state:
        saved_profile_data = store.load_profile()
        if saved_profile_data:
            try:
                profile = CompanyProfile(**saved_profile_data)
                st.session_state["company_profile"] = profile
                if "profile_restored" not in st.session_state:
                    st.session_state["profile_restored"] = True
            except Exception as e:
                print(f"[Session] Profile load error: {e}")
        else:
            default_profile = {
                "id": "comp_001",
                "name": "Acme Corp (Default)",
                "data": {
                    "licenses": ["소프트웨어사업자"],
                    "region": "Seoul",
                    "credit_rating": "B+",
                    "employees": 10
                }
            }
            st.session_state["company_profile"] = CompanyProfile(**default_profile)
