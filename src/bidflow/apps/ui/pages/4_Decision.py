import streamlit as st
from bidflow.domain.models import CompanyProfile, ComplianceMatrix, ExtractionSlot
from bidflow.validation.validator import RuleBasedValidator
from bidflow.apps.ui.auth import require_login

st.set_page_config(page_title="Go/No-Go Decision", page_icon="🚦", layout="wide")

user_id = require_login()

st.title("구조적 판정 (Go/No-Go Decision)")

from bidflow.apps.ui.session import init_app_session
init_app_session(user_id=user_id)

if "extraction_results" not in st.session_state:
    st.warning("먼저 문서를 업로드하고 분석을 수행하세요.")
    st.stop()

if "company_profile" not in st.session_state:
    st.warning("먼저 Profile 탭에서 회사 정보를 설정하세요.")
    st.stop()

results_dict = st.session_state["extraction_results"]
profile = st.session_state["company_profile"]

# ComplianceMatrix 모델 재구성
slots_map = {}
for group in ["g1", "g2", "g3", "g4"]:
    if group in results_dict:
        if group == "g4": continue
        for k, v in results_dict[group].items():
            slots_map[k] = ExtractionSlot(**v)

matrix = ComplianceMatrix(
    doc_hash=st.session_state.get("current_doc_hash", "unknown"),
    slots=slots_map
)

# 검증 실행
validator = RuleBasedValidator()
decisions = validator.validate(matrix, profile)

# 종합 판정
recommendation = validator.get_recommendation(decisions)
signal = recommendation["signal"]

if signal == "red":
    st.error(f"## {recommendation['recommendation']}")
elif signal == "yellow":
    st.warning(f"## {recommendation['recommendation']}")
else:
    st.success(f"## {recommendation['recommendation']}")

st.subheader("판정 결과 요약")

col1, col2, col3 = st.columns(3)
n_red = sum(1 for d in decisions if d.decision == "RED")
n_gray = sum(1 for d in decisions if d.decision == "GRAY")
n_green = sum(1 for d in decisions if d.decision == "GREEN")

col1.metric("RED (부적격 위험)", n_red)
col2.metric("GRAY (확인 필요)", n_gray)
col3.metric("GREEN (충족)", n_green)

st.divider()

for d in decisions:
    color = "red" if d.decision == "RED" else "gray" if d.decision == "GRAY" else "green"
    icon = "❌" if d.decision == "RED" else "❓" if d.decision == "GRAY" else "✅"

    with st.expander(f"{icon} [{d.decision}] {d.slot_key}"):
        st.write(f"**Reasons:**")
        for r in d.reasons:
            st.write(f"- {r}")

        if d.evidence:
            st.write("**Evidence:**")
            st.json([e.model_dump() for e in d.evidence])

if not decisions:
    st.info("검증 로직에 정의된 항목(면허 등)이 추출되지 않았습니다.")
