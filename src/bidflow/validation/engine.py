from typing import List, Dict, Any
from bidflow.domain.models import (
    ComplianceMatrix, CompanyProfile, ValidationResult, ExtractionSlot, Evidence
)

class RuleValidator:
    """
    추출된 요구사항(Compliance Matrix)과 회사 프로필(Company Profile)을 비교하여
    적격 여부(Green/Red/Gray)를 판정합니다.
    """
    
    def validate(self, matrix: ComplianceMatrix, profile: CompanyProfile) -> List[ValidationResult]:
        results = []
        
        # 슬롯별 검증 로직 실행
        for key, slot in matrix.slots.items():
            result = self._validate_slot(key, slot, profile)
            results.append(result)
            
        return results

    def _validate_slot(self, key: str, slot: ExtractionSlot, profile: CompanyProfile) -> ValidationResult:
        """
        개별 슬롯 검증
        """
        # 0. 추출 실패/모호함 처리 (Gray Zone)
        if slot.status != "FOUND":
             return ValidationResult(
                slot_key=key,
                decision="GRAY",
                reasons=[f"정보가 추출되지 않았거나 모호합니다. (상태: {slot.status})"],
                evidence=slot.evidence,
                risk_level="HIGH"
            )

        # 1. 로직 분기 (G1/G2/G3/G4)에 따라 다른 규칙 적용 필요
        # MVP에서는 키워드 매칭이나 단순 비교 로직 사용
        
        # 예시 로직: G3 자격 요건 (license)
        # 프로필의 licenses 리스트와 비교
        if "license" in key or "자격" in key:
            return self._validate_license(slot, profile)
            
        # 기본: 일단 GREEN (MVP 단순화 - 사용자가 직접 확인하도록)
        return ValidationResult(
            slot_key=key,
            decision="GRAY", # 기본값은 GRAY로 두어 확인 유도
            reasons=["자동 검증 로직이 구현되지 않은 항목입니다."],
            evidence=slot.evidence,
            risk_level="LOW"
        )

    def _validate_license(self, slot: ExtractionSlot, profile: CompanyProfile) -> ValidationResult:
        """
        면허/자격 검증 로직
        """
        required_license = str(slot.value) # 예: "소프트웨어사업자"
        
        # 프로필 데이터 구조 가정: {"licenses": ["소프트웨어사업자", "정보통신공사업"]}
        my_licenses = profile.data.get("licenses", [])
        
        # 단순 포함 여부 확인
        # 실제로는 퍼지 매칭(Fuzzy Matching) 등이 필요함
        matched = False
        for my_lic in my_licenses:
            if required_license in my_lic or my_lic in required_license:
                matched = True
                break
        
        if matched:
            return ValidationResult(
                slot_key=slot.key,
                decision="GREEN",
                reasons=[f"보유 면허 '{my_licenses}'가 요구사항 '{required_license}'를 충족합니다."],
                evidence=slot.evidence,
                risk_level="LOW"
            )
        else:
            return ValidationResult(
                slot_key=slot.key,
                decision="RED",
                reasons=[f"요구 면허 '{required_license}'를 보유하고 있지 않습니다. (보유: {my_licenses})"],
                evidence=slot.evidence,
                risk_level="HIGH"
            )
