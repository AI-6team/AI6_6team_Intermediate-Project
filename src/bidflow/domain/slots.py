from enum import Enum
from typing import List

class SlotKeys(str, Enum):
    # G1: 기본 정보 (4)
    PROJECT_NAME = "project_name"
    ISSUER = "issuer"
    PERIOD = "period"
    BUDGET = "budget"
    
    # G2: 일정 및 제출 (6)
    BRIEFING_DATE = "briefing_date"
    REGISTRATION_DEADLINE = "registration_deadline"
    PROPOSAL_DEADLINE = "proposal_deadline"
    PRESENTATION_DATE = "presentation_date"
    SUBMISSION_METHOD = "submission_method"
    SUBMISSION_LOCATION = "submission_location"

    # G3: 자격 및 결격 (8)
    LICENSE_Requirements = "license_requirements"
    PERFORMANCE_Requirements = "performance_requirements"
    REGION_RESTRICTION = "region_restriction"
    JOINT_VENTURE = "joint_venture"
    SUBCONTRACT = "subcontract"
    NEGATIVE_CONDITIONS = "negative_conditions" # 부정당업자 등
    PENALTY_CLAUSE = "penalty_clause"
    FINANCIAL_CREDIT = "financial_credit"
    
    # G4: 평가 및 배점 (6)
    TECH_SCORE_RATIO = "tech_score_ratio"
    PRICE_SCORE_RATIO = "price_score_ratio"
    QUALITATIVE_EVAL = "qualitative_eval" # 정성
    QUANTITATIVE_EVAL = "quantitative_eval" # 정량
    PASSING_SCORE = "passing_score"
    DIFFERENTIAL_SCORE = "differential_score" # 차등점수제

    # ETC (6)
    CONTRACT_METHOD = "contract_method" # 일반경쟁/수의계약 등
    SELECTION_METHOD = "selection_method" # 협상에 의한 계약 등
    WARRANTY_PERIOD = "warranty_period"
    MAINTENANCE_CONDITIONS = "maintenance_conditions"
    PM_CERTIFICATION = "pm_certification"
    SECURITY_REQ = "security_req"

# 그룹별 리스트 (파이프라인 참조용)
G1_SLOTS = [
    SlotKeys.PROJECT_NAME, SlotKeys.ISSUER, SlotKeys.PERIOD, SlotKeys.BUDGET
]

G2_SLOTS = [
    SlotKeys.BRIEFING_DATE, SlotKeys.REGISTRATION_DEADLINE, SlotKeys.PROPOSAL_DEADLINE,
    SlotKeys.PRESENTATION_DATE, SlotKeys.SUBMISSION_METHOD, SlotKeys.SUBMISSION_LOCATION
]
# ... 필요한 만큼 그룹 정의
