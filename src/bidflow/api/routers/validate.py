from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any
from bidflow.api.deps import get_current_user
from bidflow.validation.engine import RuleValidator
from bidflow.domain.models import ComplianceMatrix, CompanyProfile, ValidationResult

router = APIRouter()
validator = RuleValidator()

@router.post("/validate", response_model=List[ValidationResult])
def validate_compliance(
    matrix: ComplianceMatrix,
    current_user: dict = Depends(get_current_user)
):
    """
    추출된 Compliance Matrix와 로그인한 사용자의 프로필을 비교하여 자격 요건 검증 결과를 반환합니다.

    - **인증**: `Authorization: Bearer {token}` 헤더가 필요합니다.
    - **프로필**: 요청 본문에 회사 프로필을 포함하지 않습니다. 대신, 로그인된 사용자의 `team`(회사명)과 `licenses`(보유 면허) 정보가 자동으로 사용됩니다.
    - **요청 본문**: `ComplianceMatrix` 객체만 포함합니다.
    - **응답**: 각 검증 규칙에 대한 `ValidationResult` 리스트를 반환합니다.
    """
    try:
        # 로그인한 사용자의 정보로 CompanyProfile 생성
        user_profile = CompanyProfile(
            name=current_user.get("team", ""),
            licenses=current_user.get("licenses", "")
        )
        results = validator.validate(matrix, user_profile)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 실패: {str(e)}")
