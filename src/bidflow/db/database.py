"""
SQLite 연결 관리 및 스키마 초기화.

WAL 모드로 동시 읽기/쓰기를 지원합니다.
DB 위치: data/bidflow.db
"""
import os
import sqlite3

_DB_PATH: str | None = None

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    team          TEXT DEFAULT '',
    role          TEXT DEFAULT 'member',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    doc_hash     TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    file_path    TEXT,
    status       TEXT DEFAULT 'READY',
    upload_date  TEXT,
    content_json TEXT NOT NULL,
    PRIMARY KEY (doc_hash, user_id)
);

CREATE TABLE IF NOT EXISTS extraction_results (
    doc_hash    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (doc_hash, user_id)
);

CREATE TABLE IF NOT EXISTS profiles (
    owner_key    TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    user_id          TEXT PRIMARY KEY,
    current_doc_hash TEXT,
    extra_json       TEXT,
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS comments (
    id          TEXT PRIMARY KEY,
    team_name   TEXT NOT NULL,
    doc_hash    TEXT NOT NULL,
    author      TEXT NOT NULL,
    author_name TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comments_team_doc ON comments(team_name, doc_hash);

CREATE TABLE IF NOT EXISTS replies (
    id          TEXT PRIMARY KEY,
    comment_id  TEXT NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    author      TEXT NOT NULL,
    author_name TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def get_db_path() -> str:
    """DB 파일 경로를 반환합니다. config에서 base 디렉토리를 읽어 결정합니다."""
    global _DB_PATH
    if _DB_PATH is None:
        try:
            from bidflow.core.config import get_config
            base = get_config("dev").storage.base or "data"
        except Exception:
            base = "data"
        os.makedirs(base, exist_ok=True)
        _DB_PATH = os.path.join(base, "bidflow.db")
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    """SQLite 연결을 반환합니다. WAL 모드 및 외래키 제약을 활성화합니다."""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate_schema() -> None:
    """기존 DB에 누락된 컬럼/테이블 변경을 적용합니다."""
    conn = get_connection()
    try:
        # users.role 컬럼 추가
        user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "role" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'member'")
            conn.commit()

        # profiles 테이블: user_id → owner_key 마이그레이션
        prof_cols = [r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()]
        if prof_cols and "owner_key" not in prof_cols:
            conn.executescript("""
                ALTER TABLE profiles RENAME TO profiles_old;
                CREATE TABLE profiles (
                    owner_key    TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at   TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO profiles (owner_key, profile_json, updated_at)
                    SELECT user_id, profile_json, updated_at FROM profiles_old;
                DROP TABLE profiles_old;
            """)
            conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """앱 시작 시 1회 호출 — 테이블이 없으면 생성하고, 스키마를 마이그레이션합니다."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    migrate_schema()
