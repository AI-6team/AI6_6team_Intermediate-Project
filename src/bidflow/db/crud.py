"""
SQLite CRUD 함수 모음.

각 함수는 독립적으로 연결을 열고 닫습니다.
트랜잭션은 context manager(with conn:)로 원자성을 보장합니다.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from bidflow.db.database import get_connection


# ── users ────────────────────────────────────────────────────────────

def upsert_user(
    username: str,
    name: str,
    email: str,
    password_hash: str,
    team: str = "",
    licenses: str = "",
    role: str = "member",
) -> None:
    """사용자를 삽입하거나 갱신합니다. role: 'member' | 'leader'"""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO users (username, name, email, password_hash, team, licenses, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    name          = excluded.name,
                    email         = excluded.email,
                    password_hash = excluded.password_hash,
                    team          = excluded.team,
                    licenses      = excluded.licenses,
                    role          = excluded.role
                """,
                (username, name, email, password_hash, team, licenses, role),
            )
    finally:
        conn.close()


def update_user_profile(username: str, team: str, licenses: str) -> None:
    """사용자의 프로필 정보(팀, 면허)를 갱신합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                UPDATE users
                SET
                    team = ?,
                    licenses = ?
                WHERE username = ?
                """,
                (team, licenses, username),
            )
    finally:
        conn.close()


def update_team_licenses(team: str, licenses: str) -> None:
    """특정 팀 전체 사용자의 licenses 값을 동기화합니다."""
    if not team:
        return

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                UPDATE users
                SET licenses = ?
                WHERE team = ?
                """,
                (licenses, team),
            )
    finally:
        conn.close()


def get_user(username: str) -> Optional[Dict[str, Any]]:
    """사용자 정보를 반환합니다. 없으면 None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_password_hash(username: str, password_hash: str) -> None:
    """사용자의 password_hash를 갱신합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, username),
            )
    finally:
        conn.close()


def delete_user(username: str) -> None:
    """사용자를 삭제합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
    finally:
        conn.close()


def get_user_role(username: str) -> str:
    """사용자의 역할을 반환합니다. 'leader' 또는 'member'. 없으면 'member'."""
    user = get_user(username)
    return (user or {}).get("role", "member")


def list_users() -> List[Dict[str, Any]]:
    """전체 사용자 목록을 반환합니다."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM users").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_team_members(team_name: str) -> List[Dict[str, str]]:
    """같은 팀에 속한 사용자 목록을 반환합니다."""
    if not team_name:
        return []
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT username, name FROM users WHERE team = ?", (team_name,)
        ).fetchall()
        return [{"username": r["username"], "name": r["name"]} for r in rows]
    finally:
        conn.close()


def get_credentials_dict() -> Dict[str, Any]:
    """
    streamlit-authenticator 호환 credentials dict를 반환합니다.
    형식: {"usernames": {"user1": {"name": ..., "email": ..., "password": ...}}}
    """
    users = list_users()
    usernames = {}
    for u in users:
        usernames[u["username"]] = {
            "name": u["name"],
            "email": u["email"],
            "password": u["password_hash"],
        }
    return {"usernames": usernames}


# ── documents ────────────────────────────────────────────────────────

def upsert_document(
    doc_hash: str,
    user_id: str,
    filename: str,
    file_path: str,
    status: str,
    upload_date: str,
    content: Dict[str, Any],
) -> None:
    """문서 메타데이터 + 청크 전체를 삽입하거나 갱신합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_hash, user_id, filename, file_path, status, upload_date, content_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash, user_id) DO UPDATE SET
                    filename     = excluded.filename,
                    file_path    = excluded.file_path,
                    status       = excluded.status,
                    upload_date  = excluded.upload_date,
                    content_json = excluded.content_json
                """,
                (
                    doc_hash,
                    user_id,
                    filename,
                    file_path,
                    status,
                    upload_date,
                    json.dumps(content, ensure_ascii=False),
                ),
            )
    finally:
        conn.close()


def get_document(doc_hash: str, user_id: str) -> Optional[Dict[str, Any]]:
    """문서를 반환합니다. 본인 또는 같은 팀원의 문서를 조회합니다."""
    conn = get_connection()
    try:
        # 1. 본인 문서 조회
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_hash = ? AND user_id = ?",
            (doc_hash, user_id),
        ).fetchone()
        
        if row:
            data = dict(row)
            data["content"] = json.loads(data.pop("content_json"))
            return data

        # 2. 팀원 문서 조회
        user = conn.execute("SELECT team FROM users WHERE username = ?", (user_id,)).fetchone()
        team = user["team"] if user else ""
        
        if team:
            sql = """
                SELECT d.*
                FROM documents d
                JOIN users u ON d.user_id = u.username
                WHERE d.doc_hash = ? AND u.team = ?
                ORDER BY d.upload_date DESC
                LIMIT 1
            """
            row = conn.execute(sql, (doc_hash, team)).fetchone()
            if row:
                data = dict(row)
                data["content"] = json.loads(data.pop("content_json"))
                return data

        return None
    finally:
        conn.close()


def list_documents(user_id: str) -> List[Dict[str, Any]]:
    """사용자(및 같은 팀원)의 문서 메타데이터 목록을 반환합니다."""
    conn = get_connection()
    try:
        # 팀 정보 조회
        user = conn.execute("SELECT team FROM users WHERE username = ?", (user_id,)).fetchone()
        team = user["team"] if user else ""

        sql = """
            SELECT d.doc_hash, d.filename, d.upload_date, d.status, d.user_id, u.name as owner_name
            FROM documents d
            JOIN users u ON d.user_id = u.username
            WHERE {}
            ORDER BY d.upload_date DESC
        """
        
        if team:
            rows = conn.execute(sql.format("u.team = ?"), (team,)).fetchall()
        else:
            rows = conn.execute(sql.format("d.user_id = ?"), (user_id,)).fetchall()
            
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_hash: str, user_id: str) -> None:
    """문서를 삭제합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "DELETE FROM documents WHERE doc_hash = ? AND user_id = ?",
                (doc_hash, user_id),
            )
    finally:
        conn.close()


# ── extraction_results ───────────────────────────────────────────────

def upsert_extraction(
    doc_hash: str, user_id: str, result: Dict[str, Any]
) -> None:
    """추출 결과를 삽입하거나 갱신합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO extraction_results (doc_hash, user_id, result_json)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_hash, user_id) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at  = datetime('now')
                """,
                (doc_hash, user_id, json.dumps(result, ensure_ascii=False)),
            )
    finally:
        conn.close()


def get_extraction(doc_hash: str, user_id: str) -> Optional[Dict[str, Any]]:
    """추출 결과를 반환합니다. 본인 또는 팀원의 결과를 조회합니다."""
    conn = get_connection()
    try:
        # 1. 본인 결과
        row = conn.execute(
            "SELECT result_json FROM extraction_results WHERE doc_hash = ? AND user_id = ?",
            (doc_hash, user_id),
        ).fetchone()
        if row:
            return json.loads(row["result_json"])
            
        # 2. 팀원 결과
        user = conn.execute("SELECT team FROM users WHERE username = ?", (user_id,)).fetchone()
        team = user["team"] if user else ""
        
        if team:
            sql = """
                SELECT r.result_json
                FROM extraction_results r
                JOIN users u ON r.user_id = u.username
                WHERE r.doc_hash = ? AND u.team = ?
                ORDER BY r.created_at DESC
                LIMIT 1
            """
            row = conn.execute(sql, (doc_hash, team)).fetchone()
            if row:
                return json.loads(row["result_json"])

        return None
    finally:
        conn.close()


def delete_extraction(doc_hash: str, user_id: str) -> None:
    """추출 결과를 삭제합니다."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "DELETE FROM extraction_results WHERE doc_hash = ? AND user_id = ?",
                (doc_hash, user_id),
            )
    finally:
        conn.close()


# ── profiles ─────────────────────────────────────────────────────────

def upsert_profile(owner_key: str, profile: Dict[str, Any]) -> None:
    """회사 프로필을 삽입하거나 갱신합니다.
    owner_key: 팀 소속 사용자는 team_name, 미소속 사용자는 user_id.
    """
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO profiles (owner_key, profile_json)
                VALUES (?, ?)
                ON CONFLICT(owner_key) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = datetime('now')
                """,
                (owner_key, json.dumps(profile, ensure_ascii=False)),
            )
    finally:
        conn.close()


def get_profile(owner_key: str) -> Optional[Dict[str, Any]]:
    """회사 프로필을 반환합니다.
    owner_key: 팀 소속 사용자는 team_name, 미소속 사용자는 user_id.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT profile_json FROM profiles WHERE owner_key = ?", (owner_key,)
        ).fetchone()
        return json.loads(row["profile_json"]) if row else None
    finally:
        conn.close()


# ── sessions ─────────────────────────────────────────────────────────

def upsert_session(user_id: str, state: Dict[str, Any]) -> None:
    """세션 상태를 삽입하거나 갱신합니다."""
    current_doc_hash = state.get("current_doc_hash")
    extra = {k: v for k, v in state.items() if k != "current_doc_hash"}
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO sessions (user_id, current_doc_hash, extra_json)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    current_doc_hash = excluded.current_doc_hash,
                    extra_json       = excluded.extra_json,
                    updated_at       = datetime('now')
                """,
                (
                    user_id,
                    current_doc_hash,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
    finally:
        conn.close()


def get_session(user_id: str) -> Optional[Dict[str, Any]]:
    """세션 상태를 반환합니다."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT current_doc_hash, extra_json FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        state: Dict[str, Any] = {}
        if row["current_doc_hash"]:
            state["current_doc_hash"] = row["current_doc_hash"]
        if row["extra_json"]:
            state.update(json.loads(row["extra_json"]))
        return state
    finally:
        conn.close()


# ── comments & replies ───────────────────────────────────────────────

def add_comment(
    team_name: str,
    doc_hash: str,
    author: str,
    author_name: str,
    text: str,
) -> str:
    """최상위 코멘트를 추가하고 생성된 ID를 반환합니다."""
    import uuid
    comment_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO comments (id, team_name, doc_hash, author, author_name, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (comment_id, team_name, doc_hash, author, author_name, text.strip(), created_at),
            )
    finally:
        conn.close()
    return comment_id


def add_reply(
    comment_id: str,
    author: str,
    author_name: str,
    text: str,
) -> str:
    """답글을 추가하고 생성된 ID를 반환합니다."""
    import uuid
    reply_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO replies (id, comment_id, author, author_name, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (reply_id, comment_id, author, author_name, text.strip(), created_at),
            )
    finally:
        conn.close()
    return reply_id


def get_comments(team_name: str, doc_hash: str) -> List[Dict[str, Any]]:
    """해당 문서의 코멘트 + 답글 목록을 반환합니다."""
    conn = get_connection()
    try:
        comment_rows = conn.execute(
            """
            SELECT id, author, author_name, text, created_at
            FROM comments
            WHERE team_name = ? AND doc_hash = ?
            ORDER BY created_at ASC
            """,
            (team_name, doc_hash),
        ).fetchall()

        comments = []
        for cr in comment_rows:
            comment = dict(cr)
            reply_rows = conn.execute(
                """
                SELECT id, author, author_name, text, created_at
                FROM replies
                WHERE comment_id = ?
                ORDER BY created_at ASC
                """,
                (cr["id"],),
            ).fetchall()
            comment["replies"] = [dict(r) for r in reply_rows]
            comments.append(comment)

        return comments
    finally:
        conn.close()


def delete_comment(comment_id: str, requester: str) -> bool:
    """작성자 본인의 코멘트를 삭제합니다. 삭제 성공 시 True."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM comments WHERE id = ? AND author = ?",
                (comment_id, requester),
            )
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_reply(reply_id: str, requester: str) -> bool:
    """작성자 본인의 답글을 삭제합니다. 삭제 성공 시 True."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM replies WHERE id = ? AND author = ?",
                (reply_id, requester),
            )
        return cur.rowcount > 0
    finally:
        conn.close()
