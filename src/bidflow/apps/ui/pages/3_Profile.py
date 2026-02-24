import streamlit as st
import json
from bidflow.domain.models import CompanyProfile
from bidflow.apps.ui.auth import require_login
from bidflow.utils.fonts import set_korean_font
from bidflow.apps.ui.session import init_app_session

st.set_page_config(page_title="Company Profile", page_icon="🏢")

user_id = require_login()

set_korean_font()
init_app_session(user_id=user_id)

st.title("Company Profile Settings")

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
    current_licenses = [{"license": l} for l in profile.data.get("licenses", [])]
    edited_licenses = st.data_editor(
        current_licenses,
        num_rows="dynamic",
        column_config={"license": "면허 명칭"},
        hide_index=True,
        width="stretch"
    )

    submitted = st.form_submit_button("저장 (Save Profile)")

    if submitted:
        new_licenses = [row["license"] for row in edited_licenses if row["license"]]

        new_data = {
            "region": region,
            "credit_rating": credit,
            "employees": employees,
            "licenses": new_licenses,
            **{k: v for k, v in profile.data.items() if k not in ["region", "credit_rating", "employees", "licenses"]}
        }

        profile.name = name
        profile.data = new_data
        st.session_state["company_profile"] = profile

        # Persistence Save (사용자 공간에 저장)
        from bidflow.ingest.storage import DocumentStore
        store = DocumentStore(user_id=user_id)
        store.save_profile(profile)

        st.success("프로필이 업데이트되었습니다!")

    with st.expander("Raw Data (Advanced)"):
        st.json(profile.data)

st.info("이 프로필은 'Decision' 탭에서 입찰 적격성 판정에 사용됩니다.")
