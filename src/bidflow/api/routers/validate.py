from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any
from bidflow.api.deps import get_current_user
from bidflow.validation.engine import RuleValidator
from bidflow.domain.models import ComplianceMatrix, CompanyProfile, ValidationResult

router = APIRouter()
validator = RuleValidator()

@router.post("/validate", response_model=List[ValidationResult], dependencies=[Depends(get_current_user)])
def validate_compliance(
    matrix: ComplianceMatrix,
    profile: CompanyProfile
):
    """
    추출된 Compliance Matrix와 회사 프로필을 비교하여 검증 결과를 반환합니다.
    """
    try:
        results = validator.validate(matrix, profile)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 실패: {str(e)}")
