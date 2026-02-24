import streamlit as st
from bidflow.apps.ui.auth import require_login, get_user_team, get_team_members, get_user_info
from bidflow.apps.ui.team import (
    get_team_documents,
    get_decision_summary,
    load_comments,
    add_comment,
    add_reply,
    delete_comment,
    delete_reply,
)

st.set_page_config(page_title="Team Workspace", page_icon="👥", layout="wide")

user_id = require_login()
user_info = get_user_info(user_id)
team_name = get_user_team(user_id)

st.title("Team Workspace")

# ── 팀 미소속 처리 ──────────────────────────────────────────────────
if not team_name:
    st.info("현재 소속된 팀이 없습니다. 프로필에서 팀을 설정하거나 팀이 있는 계정으로 가입하세요.")
    st.stop()

# ── 팀 정보 ─────────────────────────────────────────────────────────
team_members = get_team_members(team_name)
if not team_members:
    st.warning("팀원이 없습니다.")
    st.stop()

st.caption(f"팀: **{team_name}** | 팀원: {', '.join(m['name'] for m in team_members)}")
st.divider()

# ── 문서 목록 로드 ───────────────────────────────────────────────────
all_docs = get_team_documents(team_members)

if not all_docs:
    st.info("팀원이 업로드한 RFP 문서가 아직 없습니다.")
    st.stop()

# ── 문서 선택 ────────────────────────────────────────────────────────
doc_labels = {
    f"{d['filename']}  (by {d['uploaded_by_name']}, {(d.get('upload_date') or '')[:10]})": d
    for d in all_docs
}
selected_label = st.selectbox("안건 선택", list(doc_labels.keys()))
selected_doc = doc_labels[selected_label]

doc_hash = selected_doc["doc_hash"]
uploader = selected_doc["uploaded_by"]
uploader_name = selected_doc["uploaded_by_name"]

# ── 안건 정보 + 판정 결과 ────────────────────────────────────────────
col_info, col_decision = st.columns([1, 1])

with col_info:
    st.subheader("안건 정보")
    st.write(f"**파일명**: {selected_doc['filename']}")
    st.write(f"**업로더**: {uploader_name} (`{uploader}`)")
    st.write(f"**업로드 날짜**: {(selected_doc.get('upload_date') or '')[:10]}")
    st.write(f"**Doc Hash**: `{doc_hash[:12]}...`")

with col_decision:
    st.subheader("판정 결과")
    summary = get_decision_summary(uploader, doc_hash)
    if summary is None:
        st.info("판정 결과가 없습니다. (추출 미완료 또는 프로필 미설정)")
    else:
        sig = summary["signal"]
        if sig == "red":
            st.error(f"**{summary['recommendation']}**")
        elif sig == "yellow":
            st.warning(f"**{summary['recommendation']}**")
        else:
            st.success(f"**{summary['recommendation']}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("RED", summary["n_red"])
        m2.metric("GRAY", summary["n_gray"])
        m3.metric("GREEN", summary["n_green"])

st.divider()

# ── 코멘트 섹션 ──────────────────────────────────────────────────────
st.subheader("팀 코멘트")

# 새 코멘트 작성
with st.form(f"new_comment_{doc_hash}", clear_on_submit=True):
    new_text = st.text_area("코멘트 작성", placeholder="이 안건에 대한 의견을 남겨보세요...", height=80)
    if st.form_submit_button("코멘트 등록"):
        if new_text.strip():
            add_comment(team_name, doc_hash, user_id, user_info.get("name", user_id), new_text)
            st.rerun()
        else:
            st.warning("내용을 입력하세요.")

st.divider()

# 코멘트 목록 표시
comments = load_comments(team_name, doc_hash)

if not comments:
    st.caption("아직 코멘트가 없습니다.")
else:
    for comment in comments:
        c_id = comment["id"]
        c_author = comment["author"]
        c_name = comment["author_name"]
        c_time = comment["created_at"][:16].replace("T", " ")

        with st.container(border=True):
            # 코멘트 헤더
            head_col, del_col = st.columns([10, 1])
            with head_col:
                st.markdown(f"**{c_name}** `{c_author}`  •  {c_time}")
            with del_col:
                if c_author == user_id:
                    if st.button("🗑", key=f"del_c_{c_id}", help="삭제"):
                        delete_comment(team_name, doc_hash, c_id, user_id)
                        st.rerun()

            st.write(comment["text"])

            # 답글 목록
            for reply in comment.get("replies", []):
                r_id = reply["id"]
                r_author = reply["author"]
                r_name = reply["author_name"]
                r_time = reply["created_at"][:16].replace("T", " ")

                with st.container():
                    st.markdown("&nbsp;" * 4, unsafe_allow_html=True)
                    r_head, r_del = st.columns([10, 1])
                    with r_head:
                        st.markdown(f"↳ **{r_name}** `{r_author}`  •  {r_time}")
                    with r_del:
                        if r_author == user_id:
                            if st.button("🗑", key=f"del_r_{r_id}", help="삭제"):
                                delete_reply(team_name, doc_hash, c_id, r_id, user_id)
                                st.rerun()
                    st.write(reply["text"])

            # 답글 작성 토글
            reply_key = f"reply_open_{c_id}"
            if st.button("답글 달기", key=f"reply_btn_{c_id}"):
                st.session_state[reply_key] = not st.session_state.get(reply_key, False)

            if st.session_state.get(reply_key, False):
                with st.form(f"reply_form_{c_id}", clear_on_submit=True):
                    reply_text = st.text_input("답글", placeholder=f"@{c_name}에게 답글...")
                    if st.form_submit_button("등록"):
                        if reply_text.strip():
                            add_reply(
                                team_name, doc_hash, c_id,
                                user_id, user_info.get("name", user_id),
                                reply_text,
                            )
                            st.session_state[reply_key] = False
                            st.rerun()
                        else:
                            st.warning("내용을 입력하세요.")
