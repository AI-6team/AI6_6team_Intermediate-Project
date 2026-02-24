import os
import re
import shutil
from pathlib import Path
import yaml
import bcrypt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_USERS_YAML_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs" / "users.yaml"


# ── 설정 파일 I/O ────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_USERS_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(config: dict):
    with open(_USERS_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ── 팀 정보 조회 ─────────────────────────────────────────────────────

def get_user_info(username: str) -> dict:
    """사용자 정보(name, email, team)를 반환합니다."""
    config = _load_config()
    return config["credentials"]["usernames"].get(username, {})


def get_user_team(username: str) -> str:
    """사용자의 소속 팀을 반환합니다. 팀이 없으면 빈 문자열."""
    return get_user_info(username).get("team", "")


def get_team_members(team_name: str) -> list[dict]:
    """
    같은 팀에 속한 사용자 목록을 반환합니다.
    반환 형식: [{"username": ..., "name": ...}, ...]
    빈 문자열 팀은 개인 공간으로 취급하여 빈 리스트 반환.
    """
    if not team_name:
        return []
    config = _load_config()
    members = []
    for uname, info in config["credentials"]["usernames"].items():
        if info.get("team", "") == team_name:
            members.append({"username": uname, "name": info.get("name", uname)})
    return members


# ── 계정 관리 ────────────────────────────────────────────────────────

def deactivate_account(username: str, delete_data: bool = False):
    """
    계정을 삭제합니다.
    delete_data=True 이면 data/accounts/{username}/ 디렉토리도 함께 삭제합니다.
    """
    config = _load_config()
    if username in config["credentials"]["usernames"]:
        del config["credentials"]["usernames"][username]
        _save_config(config)

    if delete_data:
        from bidflow.ingest.storage import StorageRegistry
        user_dir = StorageRegistry().user_base(username)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)


# ── Streamlit-Authenticator ──────────────────────────────────────────

def load_authenticator():
    """users.yaml + 환경변수를 읽어 Authenticate 객체와 config를 반환합니다."""
    import streamlit_authenticator as stauth

    config = _load_config()
    cookie_key = os.getenv("BIDFLOW_COOKIE_KEY")
    if not cookie_key:
        raise RuntimeError(
            "BIDFLOW_COOKIE_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요."
        )

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        cookie_key,
        config["cookie"]["expiry_days"],
    )
    return authenticator, config


def require_login() -> str:
    """
    모든 페이지에서 호출. 사이드바에 사용자 정보/로그아웃/탈퇴를 표시합니다.
    인증된 경우 user_id(username)를 반환하고, 미인증 시 st.stop()합니다.
    """
    authenticator, _ = load_authenticator()

    if st.session_state.get("authentication_status"):
        username = st.session_state.get("username")
        if not username:
            st.session_state.clear()
            st.rerun()

        with st.sidebar:
            st.success(f"로그인: {st.session_state.get('name', username)}")
            team = get_user_team(username)
            if team:
                st.caption(f"팀: {team}")
            authenticator.logout("로그아웃")

            # 회원 탈퇴
            with st.expander("⚙️ 계정 설정"):
                st.warning("탈퇴 시 계정이 즉시 삭제됩니다.")
                delete_data = st.checkbox("데이터도 함께 삭제", key="_del_data_chk")
                if st.button("회원 탈퇴", type="secondary", key="_withdraw_btn"):
                    st.session_state["_confirm_withdraw"] = True

            if st.session_state.get("_confirm_withdraw"):
                st.error("정말 탈퇴하시겠습니까?")
                col1, col2 = st.columns(2)
                if col1.button("확인", key="_confirm_yes"):
                    deactivate_account(username, delete_data=delete_data)
                    st.session_state.clear()
                    st.rerun()
                if col2.button("취소", key="_confirm_no"):
                    st.session_state.pop("_confirm_withdraw", None)
                    st.rerun()

        return username

    authenticator.login(location="sidebar")

    status = st.session_state.get("authentication_status")
    if status is False:
        st.sidebar.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        st.stop()
    elif status is None:
        st.sidebar.info("사이드바에서 로그인하세요.")
        st.stop()

    return st.session_state["username"]


# ── 회원가입 폼 ──────────────────────────────────────────────────────

def register_form() -> bool:
    """
    회원가입 폼을 표시합니다. 성공 시 users.yaml 저장 후 True 반환.
    """
    with st.form("register_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("사용자명 *", placeholder="영문·숫자·밑줄 (로그인 ID)")
            new_name = st.text_input("표시 이름 *", placeholder="홍길동")
        with col2:
            new_email = st.text_input("이메일 *", placeholder="user@example.com")
            new_team = st.text_input("소속 팀", placeholder="예: AI6_team1 (없으면 비워두세요)")

        new_password = st.text_input("비밀번호 *", type="password", placeholder="6자 이상")
        new_password2 = st.text_input("비밀번호 확인 *", type="password")

        submitted = st.form_submit_button("가입하기", width="stretch")

    if not submitted:
        return False

    # 유효성 검사
    if not all([new_username, new_name, new_email, new_password, new_password2]):
        st.error("사용자명, 이름, 이메일, 비밀번호는 필수 입력 항목입니다.")
        return False

    if not re.match(r"^[a-zA-Z0-9_]+$", new_username):
        st.error("사용자명은 영문, 숫자, 밑줄(_)만 사용할 수 있습니다.")
        return False

    if len(new_password) < 6:
        st.error("비밀번호는 6자 이상이어야 합니다.")
        return False

    if new_password != new_password2:
        st.error("비밀번호가 일치하지 않습니다.")
        return False

    if new_team and not re.match(r"^[a-zA-Z0-9_가-힣]+$", new_team):
        st.error("팀명은 영문, 숫자, 밑줄, 한글만 사용할 수 있습니다.")
        return False

    config = _load_config()
    if new_username in config["credentials"]["usernames"]:
        st.error(f"'{new_username}'은 이미 사용 중인 사용자명입니다.")
        return False

    # 등록
    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt(12)).decode("utf-8")
    config["credentials"]["usernames"][new_username] = {
        "email": new_email,
        "name": new_name,
        "password": hashed,
        "team": new_team.strip(),
    }
    _save_config(config)

    # 스토리지 공간 초기화
    from bidflow.ingest.storage import StorageRegistry
    reg = StorageRegistry()
    reg.ensure_spaces(new_username)
    if new_team:
        reg.ensure_team_spaces(new_team.strip())

    st.success(f"'{new_name}' 계정이 생성되었습니다! 로그인 탭에서 로그인하세요.")
    return True
