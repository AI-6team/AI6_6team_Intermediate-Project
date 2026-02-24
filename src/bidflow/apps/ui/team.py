"""
팀 워크스페이스 유틸리티.
팀별 문서 코멘트를 data/shared/teams/{team_name}/comments/{doc_hash}.json 에 저장합니다.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from bidflow.ingest.storage import StorageRegistry


def _comments_path(team_name: str, doc_hash: str) -> Path:
    reg = StorageRegistry()
    reg.ensure_team_spaces(team_name)
    return Path(reg.team_space(team_name, "comments")) / f"{doc_hash}.json"


def load_comments(team_name: str, doc_hash: str) -> list[dict]:
    """해당 문서의 코멘트 목록을 반환합니다."""
    path = _comments_path(team_name, doc_hash)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_comments(team_name: str, doc_hash: str, comments: list[dict]):
    path = _comments_path(team_name, doc_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)


def add_comment(team_name: str, doc_hash: str, author: str, author_name: str, text: str) -> list[dict]:
    """최상위 코멘트를 추가합니다."""
    comments = load_comments(team_name, doc_hash)
    comments.append({
        "id": str(uuid.uuid4()),
        "author": author,
        "author_name": author_name,
        "text": text.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replies": [],
    })
    _save_comments(team_name, doc_hash, comments)
    return comments


def add_reply(
    team_name: str,
    doc_hash: str,
    comment_id: str,
    author: str,
    author_name: str,
    text: str,
) -> list[dict]:
    """특정 코멘트에 답글을 추가합니다."""
    comments = load_comments(team_name, doc_hash)
    for comment in comments:
        if comment["id"] == comment_id:
            comment["replies"].append({
                "id": str(uuid.uuid4()),
                "author": author,
                "author_name": author_name,
                "text": text.strip(),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            break
    _save_comments(team_name, doc_hash, comments)
    return comments


def delete_comment(team_name: str, doc_hash: str, comment_id: str, requester: str) -> list[dict]:
    """작성자 본인 코멘트를 삭제합니다."""
    comments = load_comments(team_name, doc_hash)
    comments = [c for c in comments if not (c["id"] == comment_id and c["author"] == requester)]
    _save_comments(team_name, doc_hash, comments)
    return comments


def delete_reply(
    team_name: str,
    doc_hash: str,
    comment_id: str,
    reply_id: str,
    requester: str,
) -> list[dict]:
    """작성자 본인 답글을 삭제합니다."""
    comments = load_comments(team_name, doc_hash)
    for comment in comments:
        if comment["id"] == comment_id:
            comment["replies"] = [
                r for r in comment["replies"]
                if not (r["id"] == reply_id and r["author"] == requester)
            ]
            break
    _save_comments(team_name, doc_hash, comments)
    return comments


def get_team_documents(team_members: list[dict]) -> list[dict]:
    """
    팀원 전체의 문서 목록을 합산하여 반환합니다.
    각 문서에 uploaded_by(username), uploaded_by_name(display name) 필드가 추가됩니다.
    """
    from bidflow.ingest.storage import DocumentStore

    all_docs = []
    for member in team_members:
        uname = member["username"]
        store = DocumentStore(user_id=uname)
        docs = store.list_documents()
        for doc in docs:
            doc["uploaded_by"] = uname
            doc["uploaded_by_name"] = member["name"]
        all_docs.extend(docs)

    # 최신순 정렬
    all_docs.sort(key=lambda d: d.get("upload_date") or "", reverse=True)
    return all_docs


def get_decision_summary(member_username: str, doc_hash: str) -> dict | None:
    """
    팀원의 판정 결과 요약을 반환합니다.
    반환: {"signal": "red|yellow|green", "recommendation": str, "n_red": int, "n_gray": int, "n_green": int}
    프로필 또는 추출 결과가 없으면 None 반환.
    """
    from bidflow.ingest.storage import DocumentStore
    from bidflow.domain.models import CompanyProfile, ComplianceMatrix, ExtractionSlot
    from bidflow.validation.validator import RuleBasedValidator

    store = DocumentStore(user_id=member_username)
    extraction = store.load_extraction_result(doc_hash)
    profile_data = store.load_profile()

    if not extraction or not profile_data:
        return None

    try:
        profile = CompanyProfile(**profile_data)
        slots_map = {}
        for group in ["g1", "g2", "g3"]:
            if group in extraction:
                for k, v in extraction[group].items():
                    slots_map[k] = ExtractionSlot(**v)

        matrix = ComplianceMatrix(doc_hash=doc_hash, slots=slots_map)
        validator = RuleBasedValidator()
        decisions = validator.validate(matrix, profile)
        rec = validator.get_recommendation(decisions)

        return {
            "signal": rec["signal"],
            "recommendation": rec["recommendation"],
            "n_red": sum(1 for d in decisions if d.decision == "RED"),
            "n_gray": sum(1 for d in decisions if d.decision == "GRAY"),
            "n_green": sum(1 for d in decisions if d.decision == "GREEN"),
        }
    except Exception:
        return None
