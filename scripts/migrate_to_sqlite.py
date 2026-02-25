"""
기존 JSON/YAML 파일 데이터를 SQLite로 마이그레이션합니다.

실행:
    python scripts/migrate_to_sqlite.py

수행 작업:
    1. configs/users.yaml → users 테이블
    2. data/accounts/{uid}/processed/{hash}.json → documents 테이블
    3. data/accounts/{uid}/processed/{hash}_result.json → extraction_results 테이블
    4. data/accounts/{uid}/profile.json → profiles 테이블
    5. data/accounts/{uid}/session.json → sessions 테이블
    6. data/shared/teams/{team}/comments/{hash}.json → comments + replies 테이블
"""
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bidflow.db.database import init_db
from bidflow.db import crud


def migrate_users(yaml_path: Path) -> int:
    """users.yaml → users 테이블"""
    if not yaml_path.exists():
        print(f"  [SKIP] {yaml_path} 없음")
        return 0

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    usernames = config.get("credentials", {}).get("usernames", {})
    count = 0
    for username, info in usernames.items():
        crud.upsert_user(
            username=username,
            name=info.get("name", username),
            email=info.get("email", ""),
            password_hash=info.get("password", ""),
            team=info.get("team", ""),
        )
        count += 1
        print(f"  [USER] {username}")
    return count


def migrate_documents(accounts_dir: Path) -> tuple[int, int, int, int]:
    """data/accounts/{uid}/processed/ → documents, extraction_results, profiles, sessions 테이블"""
    doc_count = ext_count = prof_count = sess_count = 0

    if not accounts_dir.exists():
        print(f"  [SKIP] {accounts_dir} 없음")
        return 0, 0, 0, 0

    for user_dir in accounts_dir.iterdir():
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name

        # 1. 문서 & 추출 결과
        processed_dir = user_dir / "processed"
        if processed_dir.exists():
            for json_file in processed_dir.glob("*.json"):
                name = json_file.stem
                if name.endswith("_result"):
                    # 추출 결과
                    doc_hash = name[:-7]
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        crud.upsert_extraction(doc_hash, user_id, result)
                        ext_count += 1
                        print(f"  [EXTRACTION] {user_id}/{doc_hash}")
                    except Exception as e:
                        print(f"  [ERROR] extraction {json_file}: {e}")
                else:
                    # 문서 전체
                    doc_hash = name
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            content = json.load(f)
                        crud.upsert_document(
                            doc_hash=doc_hash,
                            user_id=user_id,
                            filename=content.get("filename", ""),
                            file_path=content.get("file_path", ""),
                            status=content.get("status", "READY"),
                            upload_date=content.get("upload_date", ""),
                            content=content,
                        )
                        doc_count += 1
                        print(f"  [DOCUMENT] {user_id}/{doc_hash[:12]}…")
                    except Exception as e:
                        print(f"  [ERROR] document {json_file}: {e}")

        # 2. 프로필
        profile_file = user_dir / "profile.json"
        if profile_file.exists():
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                crud.upsert_profile(user_id, profile)
                prof_count += 1
                print(f"  [PROFILE] {user_id}")
            except Exception as e:
                print(f"  [ERROR] profile {profile_file}: {e}")

        # 3. 세션
        session_file = user_dir / "session.json"
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session = json.load(f)
                crud.upsert_session(user_id, session)
                sess_count += 1
                print(f"  [SESSION] {user_id}")
            except Exception as e:
                print(f"  [ERROR] session {session_file}: {e}")

    return doc_count, ext_count, prof_count, sess_count


def migrate_comments(teams_dir: Path) -> tuple[int, int]:
    """data/shared/teams/{team}/comments/{hash}.json → comments + replies 테이블"""
    comment_count = reply_count = 0

    if not teams_dir.exists():
        print(f"  [SKIP] {teams_dir} 없음")
        return 0, 0

    for team_dir in teams_dir.iterdir():
        if not team_dir.is_dir():
            continue
        team_name = team_dir.name
        comments_dir = team_dir / "comments"
        if not comments_dir.exists():
            continue

        for json_file in comments_dir.glob("*.json"):
            doc_hash = json_file.stem
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    comments = json.load(f)

                for comment in comments:
                    # comments 테이블에 직접 삽입 (기존 id 유지)
                    from bidflow.db.database import get_connection
                    conn = get_connection()
                    try:
                        with conn:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO comments
                                    (id, team_name, doc_hash, author, author_name, text, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    comment["id"],
                                    team_name,
                                    doc_hash,
                                    comment["author"],
                                    comment["author_name"],
                                    comment["text"],
                                    comment["created_at"],
                                ),
                            )
                            comment_count += 1

                            for reply in comment.get("replies", []):
                                conn.execute(
                                    """
                                    INSERT OR IGNORE INTO replies
                                        (id, comment_id, author, author_name, text, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        reply["id"],
                                        comment["id"],
                                        reply["author"],
                                        reply["author_name"],
                                        reply["text"],
                                        reply["created_at"],
                                    ),
                                )
                                reply_count += 1
                    finally:
                        conn.close()

                print(f"  [COMMENTS] {team_name}/{doc_hash[:12]}… ({len(comments)}개)")
            except Exception as e:
                print(f"  [ERROR] comments {json_file}: {e}")

    return comment_count, reply_count


def main():
    base_dir = PROJECT_ROOT / "data"
    yaml_path = PROJECT_ROOT / "configs" / "users.yaml"

    print("=== BidFlow SQLite 마이그레이션 시작 ===\n")

    print("[1/6] DB 초기화...")
    init_db()
    print("  OK\n")

    print("[2/6] 사용자 마이그레이션 (users.yaml → users)...")
    user_count = migrate_users(yaml_path)
    print(f"  완료: {user_count}명\n")

    print("[3-6/6] 문서/추출결과/프로필/세션 마이그레이션...")
    doc_count, ext_count, prof_count, sess_count = migrate_documents(
        base_dir / "accounts"
    )
    print(f"  문서: {doc_count}개, 추출결과: {ext_count}개, 프로필: {prof_count}개, 세션: {sess_count}개\n")

    print("[7/6] 팀 코멘트 마이그레이션...")
    c_count, r_count = migrate_comments(base_dir / "shared" / "teams")
    print(f"  코멘트: {c_count}개, 답글: {r_count}개\n")

    print("=== 마이그레이션 완료 ===")
    print(f"DB 위치: {base_dir / 'bidflow.db'}")


if __name__ == "__main__":
    main()
