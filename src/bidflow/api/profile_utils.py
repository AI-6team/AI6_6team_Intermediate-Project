"""팀/사용자 프로필 구성 공유 유틸리티.

판정 결과(team decision)와 자격 검증(validation) 양쪽에서
동일한 프로필 구성 로직을 사용하기 위한 모듈입니다.
"""

from typing import Any, Dict, List, Optional

from bidflow.domain.models import CompanyProfile


def parse_licenses(licenses_raw: Optional[str]) -> List[str]:
    """쉼표로 구분된 면허 문자열을 리스트로 변환합니다."""
    if not licenses_raw:
        return []
    return [s.strip() for s in str(licenses_raw).split(",") if s and s.strip()]


def build_effective_profile(current_user: Dict[str, Any]) -> CompanyProfile:
    """팀 공유 프로필이 있으면 우선 사용하고, 없으면 사용자 정보를 fallback으로 사용합니다.

    라이선스 타입(list/str)을 항상 리스트로 정규화하고,
    region 필드를 항상 포함하여 검증 일관성을 보장합니다.
    """
    team_name = (current_user.get("team") or "").strip()

    if team_name:
        try:
            from bidflow.ingest.storage import DocumentStore

            store = DocumentStore(
                user_id=current_user.get("username", "global"),
                team_name=team_name,
            )
            profile_data = store.load_profile()
            if profile_data:
                profile = CompanyProfile(**profile_data)
                data = dict(profile.data or {})

                # 라이선스 타입 정규화 (list/str → list)
                raw_licenses = data.get("licenses", [])
                if isinstance(raw_licenses, list):
                    licenses = [str(x).strip() for x in raw_licenses if str(x).strip()]
                elif isinstance(raw_licenses, str):
                    licenses = parse_licenses(raw_licenses)
                else:
                    licenses = parse_licenses(current_user.get("licenses", ""))
                data["licenses"] = licenses

                # region 항상 포함 (빈 문자열이라도)
                data["region"] = str(data.get("region", "")).strip()

                return CompanyProfile(
                    id=str(profile.id),
                    name=profile.name or team_name,
                    data=data,
                )
        except Exception:
            pass

    # fallback: 사용자 기본 프로필
    licenses_list = parse_licenses(current_user.get("licenses", ""))
    return CompanyProfile(
        id=current_user.get("username", "unknown"),
        name=team_name,
        data={"licenses": licenses_list, "region": ""},
    )
