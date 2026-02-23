import streamlit as st
import json
from bidflow.domain.models import CompanyProfile

st.set_page_config(page_title="Company Profile", page_icon="🏢")

st.title("Company Profile Settings")

# 기본 프로필 (Mock)
from bidflow.utils.fonts import set_korean_font
from bidflow.apps.ui.session import init_app_session

set_korean_font()
init_app_session()

# 기본 프로필 로직 제거 (init_app_session에서 처리됨)
# if "company_profile" not in st.session_state: ...

profile = st.session_state["company_profile"]

with st.form("profile_form"):
    st.subheader("기본 정보")
    name = st.text_input("회사명", value=profile.name)
    
    st.subheader("상세 역량")
    
    col1, col2 = st.columns(2)
    region = col1.text_input("지역 (Region)", value=profile.data.get("region", ""))
    credit = col2.text_input("신용등급 (Credit Rating)", value=profile.data.get("credit_rating", ""))
    employees = col1.number_input("직원 수 (Employees)", value=profile.data.get("employees", 0))
    
    st.write("보유 면허 및 자격 (Licenses)")
    # 리스트를 데이터프레임처럼 편집 (추가/삭제 용이)
    current_licenses = [{"license": l} for l in profile.data.get("licenses", [])]
    edited_licenses = st.data_editor(
        current_licenses, 
        num_rows="dynamic", 
        column_config={"license": "면허 명칭"},
        hide_index=True,
        use_container_width=True
    )
    
    submitted = st.form_submit_button("저장 (Save Profile)")
    
    if submitted:
        # 데이터 재구성
        new_licenses = [row["license"] for row in edited_licenses if row["license"]]
        
        new_data = {
            "region": region,
            "credit_rating": credit,
            "employees": employees,
            "licenses": new_licenses,
            # 기타 기존 데이터 유지 (JSON 모드에서만 보일 항목들)
            **{k:v for k,v in profile.data.items() if k not in ["region", "credit_rating", "employees", "licenses"]}
        }
        
        profile.name = name
        profile.data = new_data
        st.session_state["company_profile"] = profile
        
        # Persistence Save
        from bidflow.ingest.storage import DocumentStore
        store = DocumentStore()
        store.save_profile(profile)
        
        st.success("프로필이 업데이트되었습니다!")
        
    # 디버깅용 JSON 미리보기 (접어두기)
    with st.expander("Raw Data (Advanced)"):
        st.json(profile.data)

st.info("이 프로필은 'Decision' 탭에서 입찰 적격성 판정에 사용됩니다.")
