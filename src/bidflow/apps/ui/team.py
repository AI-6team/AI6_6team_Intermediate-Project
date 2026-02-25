"""
팀 워크스페이스 유틸리티.
팀별 문서 코멘트를 SQLite의 comments/replies 테이블에 저장합니다.
"""
from bidflow.db import crud


def load_comments(team_name: str, doc_hash: str) -> list[dict]:
    """해당 문서의 코멘트 목록(답글 포함)을 반환합니다."""
    return crud.get_comments(team_name, doc_hash)


def add_comment(team_name: str, doc_hash: str, author: str, author_name: str, text: str) -> list[dict]:
    """최상위 코멘트를 추가하고 갱신된 목록을 반환합니다."""
    crud.add_comment(team_name, doc_hash, author, author_name, text)
    return load_comments(team_name, doc_hash)


def add_reply(
    team_name: str,
    doc_hash: str,
    comment_id: str,
    author: str,
    author_name: str,
    text: str,
) -> list[dict]:
    """특정 코멘트에 답글을 추가하고 갱신된 목록을 반환합니다."""
    crud.add_reply(comment_id, author, author_name, text)
    return load_comments(team_name, doc_hash)


def delete_comment(team_name: str, doc_hash: str, comment_id: str, requester: str) -> list[dict]:
    """작성자 본인 코멘트를 삭제하고 갱신된 목록을 반환합니다."""
    crud.delete_comment(comment_id, requester)
    return load_comments(team_name, doc_hash)


def delete_reply(
    team_name: str,
    doc_hash: str,
    comment_id: str,
    reply_id: str,
    requester: str,
) -> list[dict]:
    """작성자 본인 답글을 삭제하고 갱신된 목록을 반환합니다."""
    crud.delete_reply(reply_id, requester)
    return load_comments(team_name, doc_hash)


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
