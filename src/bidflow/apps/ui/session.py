import streamlit as st
from bidflow.ingest.storage import DocumentStore
from bidflow.domain.models import CompanyProfile

def init_app_session():
    """
    모든 페이지에서 공통으로 호출.
    저장된 세션 상태와 프로필을 로드하여 st.session_state에 복원.
    """
    store = DocumentStore()

    # 1. 세션 상태 복원 (Doc Hash, Extraction results)
    if "current_doc_hash" not in st.session_state:
        saved_session = store.load_session_state()
        if saved_session and "current_doc_hash" in saved_session:
            doc_hash = saved_session["current_doc_hash"]
            
            # 문서 파일 존재 여부 확인
            if store.load_document(doc_hash):
                st.session_state["current_doc_hash"] = doc_hash
                
                # 분석 결과 복원
                saved_result = store.load_extraction_result(doc_hash)
                if saved_result:
                    st.session_state["extraction_results"] = saved_result
                    # 첫 로드 시에만 토스트 띄우기 (페이지 이동 시마다 뜨지 않게 플래그 사용 가능)
                    if "session_restored" not in st.session_state:
                        st.toast(f"이전 작업 세션을 복원했습니다. (Doc: {doc_hash})", icon="🔄")
                        st.session_state["session_restored"] = True

    # 2. 회사 프로필 복원
    # 이미 로드되어 있어도, 다른 탭에서 디스크에 저장했을 수 있으므로
    # 디스크의 내용이 최신이라면 반영하는 전략? -> 복잡함.
    # MVP: 세션에 없으면 로드. 변경 시 세션+디스크 동시 업데이트하므로 세션 우선.
    if "company_profile" not in st.session_state:
        saved_profile_data = store.load_profile()
        if saved_profile_data:
            try:
                profile = CompanyProfile(**saved_profile_data)
                st.session_state["company_profile"] = profile
                if "profile_restored" not in st.session_state:
                    # st.toast("회사 프로필을 불러왔습니다.", icon="🏢")
                    st.session_state["profile_restored"] = True
            except Exception as e:
                print(f"[Session] Profile load error: {e}")
        else:
            # 디스크에도 없으면 기본값 (Mock) - 각 페이지에서 처리하거나 여기서 초기화
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
