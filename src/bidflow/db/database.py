"""
DB 연결 관리 및 스키마 초기화.

기본값은 SQLite(WAL)이며, `BIDFLOW_DATABASE_URL`이 PostgreSQL DSN이면
동일한 CRUD 계층을 PostgreSQL로 동작시킬 수 있습니다.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_DB_PATH: str | None = None

_SQLITE_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    team          TEXT DEFAULT '',
    licenses      TEXT DEFAULT '',
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

_POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    team          TEXT DEFAULT '',
    licenses      TEXT DEFAULT '',
    role          TEXT DEFAULT 'member',
    created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
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
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_hash, user_id)
);

CREATE TABLE IF NOT EXISTS profiles (
    owner_key    TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    user_id          TEXT PRIMARY KEY,
    current_doc_hash TEXT,
    extra_json       TEXT,
    updated_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
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


def _database_url_from_env() -> str:
    return os.getenv("BIDFLOW_DATABASE_URL", "").strip()


def _is_postgres_url(database_url: str) -> bool:
    lowered = database_url.lower()
    return lowered.startswith(
        (
            "postgresql://",
            "postgres://",
            "postgresql+psycopg://",
            "postgresql+psycopg2://",
        )
    )


def _normalize_postgres_dsn(database_url: str) -> str:
    lowered = database_url.lower()
    if lowered.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.split("://", 1)[1]
    if lowered.startswith("postgresql+psycopg2://"):
        return "postgresql://" + database_url.split("://", 1)[1]
    if lowered.startswith("postgres://"):
        return "postgresql://" + database_url.split("://", 1)[1]
    return database_url


def _redact_db_url(database_url: str) -> str:
    try:
        parsed = urlsplit(database_url)
        if parsed.password is None:
            return database_url

        username = parsed.username or ""
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{username}:***@{hostname}{port}" if username else f"***@{hostname}{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return database_url


def get_db_engine() -> str:
    database_url = _database_url_from_env()
    return "postgres" if database_url and _is_postgres_url(database_url) else "sqlite"


def _resolve_sqlite_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH

    try:
        from bidflow.core.config import get_config

        base = get_config("dev").storage.base or "data"
    except Exception:
        base = "data"

    if not os.path.isabs(base):
        try:
            from bidflow.core.config import get_project_root

            base = os.path.join(str(get_project_root()), base)
        except Exception:
            pass

    os.makedirs(base, exist_ok=True)
    _DB_PATH = os.path.join(base, "bidflow.db")
    return _DB_PATH


def get_db_path() -> str:
    """DB 위치(또는 PostgreSQL DSN)를 반환합니다."""
    database_url = _database_url_from_env()
    if database_url and _is_postgres_url(database_url):
        return _redact_db_url(_normalize_postgres_dsn(database_url))
    return _resolve_sqlite_db_path()


def _replace_qmark_placeholders(sql: str) -> str:
    result: list[str] = []
    in_single = False
    in_double = False
    i = 0

    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if ch == "'" and not in_double:
            if in_single and nxt == "'":
                result.append(ch)
                result.append(nxt)
                i += 2
                continue
            in_single = not in_single
            result.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
            i += 1
            continue

        if ch == "?" and not in_single and not in_double:
            result.append("%s")
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _convert_sql_for_postgres(sql: str) -> str:
    converted = sql.replace("datetime('now')", "CURRENT_TIMESTAMP")
    converted = converted.replace('datetime("now")', "CURRENT_TIMESTAMP")
    converted = _replace_qmark_placeholders(converted)
    return converted


def _split_sql_script(script: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0

    while i < len(script):
        ch = script[i]
        nxt = script[i + 1] if i + 1 < len(script) else ""

        if ch == "'" and not in_double:
            if in_single and nxt == "'":
                current.append(ch)
                current.append(nxt)
                i += 2
                continue
            in_single = not in_single
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if ch == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


class PostgresCompatConnection:
    """SQLite-like connection wrapper for psycopg."""

    def __init__(self, dsn: str):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL 사용을 위해 psycopg가 필요합니다. "
                "`pip install -e .`로 의존성을 다시 설치하세요."
            ) from exc

        self._conn = psycopg.connect(dsn, row_factory=dict_row)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] | None = None):
        converted = _convert_sql_for_postgres(sql)
        return self._conn.execute(converted, params or ())

    def executescript(self, script: str) -> None:
        for statement in _split_sql_script(script):
            if statement.upper().startswith("PRAGMA"):
                continue
            self.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def get_connection():
    """DB 연결을 반환합니다. (SQLite 또는 PostgreSQL)"""
    database_url = _database_url_from_env()
    if database_url and _is_postgres_url(database_url):
        return PostgresCompatConnection(_normalize_postgres_dsn(database_url))

    conn = sqlite3.connect(_resolve_sqlite_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_sqlite_schema(conn) -> None:
    # users.role 컬럼 추가
    user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'member'")
        conn.commit()

    # users.licenses 컬럼 추가
    user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "licenses" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN licenses TEXT DEFAULT ''")
        conn.commit()

    # profiles 테이블: user_id → owner_key 마이그레이션
    prof_cols = [r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()]
    if prof_cols and "owner_key" not in prof_cols:
        conn.executescript(
            """
            ALTER TABLE profiles RENAME TO profiles_old;
            CREATE TABLE profiles (
                owner_key    TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at   TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO profiles (owner_key, profile_json, updated_at)
                SELECT user_id, profile_json, updated_at FROM profiles_old;
            DROP TABLE profiles_old;
            """
        )
        conn.commit()


def _migrate_postgres_schema(conn) -> None:
    with conn:
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'member'")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS licenses TEXT DEFAULT ''")

        owner_key_exists = conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'profiles' AND column_name = 'owner_key'
            """
        ).fetchone()
        user_id_exists = conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'profiles' AND column_name = 'user_id'
            """
        ).fetchone()
        if user_id_exists and not owner_key_exists:
            conn.execute("ALTER TABLE profiles RENAME COLUMN user_id TO owner_key")


def migrate_schema() -> None:
    """기존 DB에 누락된 컬럼/테이블 변경을 적용합니다."""
    conn = get_connection()
    try:
        if get_db_engine() == "postgres":
            _migrate_postgres_schema(conn)
        else:
            _migrate_sqlite_schema(conn)
    finally:
        conn.close()


def init_db() -> None:
    """앱 시작 시 스키마를 생성/마이그레이션합니다."""
    conn = get_connection()
    try:
        if get_db_engine() == "postgres":
            conn.executescript(_POSTGRES_SCHEMA_SQL)
        else:
            conn.executescript(_SQLITE_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    migrate_schema()
