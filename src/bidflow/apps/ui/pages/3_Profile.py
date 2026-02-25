import streamlit as st
from bidflow.domain.models import CompanyProfile
from bidflow.apps.ui.auth import require_login, get_user_team, get_user_role
from bidflow.utils.fonts import set_korean_font
from bidflow.apps.ui.session import init_app_session
from bidflow.ingest.storage import DocumentStore

st.set_page_config(page_title="Company Profile", page_icon="🏢")

user_id = require_login()

set_korean_font()

team = get_user_team(user_id)
role = get_user_role(user_id)

# 팀 소속이면 팀장만 수정 가능, 미소속이면 누구나 수정 가능
can_edit = (not team) or (role == "leader")

init_app_session(user_id=user_id)

st.title("Company Profile Settings")

if team:
    st.caption(f"팀: **{team}** | 역할: {'팀장 (Leader)' if role == 'leader' else '팀원 (Member)'}")

profile = st.session_state["company_profile"]

# 팀원(수정 불가) 안내
if not can_edit:
    st.info("팀장만 회사 프로필을 수정할 수 있습니다. 아래는 현재 팀 프로필입니다.")
    st.subheader("기본 정보")
    st.write(f"**회사명**: {profile.name}")

    st.subheader("상세 역량")
    col1, col2 = st.columns(2)
    col1.metric("지역", profile.data.get("region", "—"))
    col2.metric("신용등급", profile.data.get("credit_rating", "—"))
    col1.metric("직원 수", profile.data.get("employees", "—"))

    licenses = profile.data.get("licenses", [])
    if licenses:
        st.write("**보유 면허 및 자격**")
        for lic in licenses:
            st.write(f"- {lic}")
    else:
        st.write("**보유 면허 및 자격**: 없음")

    with st.expander("Raw Data (Advanced)"):
        st.json(profile.data)

    st.info("이 프로필은 'Decision' 탭에서 입찰 적격성 판정에 사용됩니다.")
    st.stop()

# 팀장 또는 미소속 사용자 — 수정 폼 표시
store = DocumentStore(user_id=user_id, team_name=team or None)

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

        store.save_profile(profile)

        if team:
            st.success(f"팀 프로필이 업데이트되었습니다! (팀: {team})")
        else:
            st.success("프로필이 업데이트되었습니다!")

    with st.expander("Raw Data (Advanced)"):
        st.json(profile.data)

st.info("이 프로필은 'Decision' 탭에서 입찰 적격성 판정에 사용됩니다.")
