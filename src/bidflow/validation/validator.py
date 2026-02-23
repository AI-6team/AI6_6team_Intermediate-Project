"""Rule-based validator with enhanced DecisionEngine (ported from 김보윤).

Original: license/credit/region checks only.
Enhanced: + budget threshold, deadline urgency, info completeness rules
with YAML-driven configuration and signal aggregation.
"""
import re
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

from bidflow.domain.models import ComplianceMatrix, CompanyProfile, ValidationResult, ExtractionSlot


class RuleBasedValidator:
    """
    Compliance Matrix와 Company Profile을 비교하여 적격 여부를 판정합니다.
    DecisionEngine 규칙(김보윤): 예산/마감일/정보완전성 기반 종합 판정 포함.
    """

    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            rules_path = Path(__file__).resolve().parents[3] / "configs" / "decision_rules.yaml"
        self.rules = self._load_rules(rules_path)

    def validate(self, matrix: ComplianceMatrix, profile: CompanyProfile) -> List[ValidationResult]:
        results = []

        slots = matrix.slots

        # 1. G3 자격 요건 검증 (기존)
        if "required_licenses" in slots:
            results.append(self._check_license(slots["required_licenses"], profile))

        if "financial_credit" in slots:
            results.append(self._check_credit(slots["financial_credit"], profile))

        if "region_restriction" in slots:
            results.append(self._check_region(slots["region_restriction"], profile))

        # 2. DecisionEngine 규칙 (김보윤에서 포팅)
        if "budget" in slots:
            results.append(self._check_budget(slots["budget"]))

        if "submission_deadline" in slots:
            results.append(self._check_deadline(slots["submission_deadline"]))

        # 정보 완전성 체크
        results.append(self._check_info_completeness(matrix))

        return results

    def get_recommendation(self, results: List[ValidationResult]) -> Dict[str, str]:
        """검증 결과를 종합하여 참여 권고 판정 반환 (김보윤 DecisionEngine)"""
        agg_rules = self.rules.get("aggregation", {})

        decisions = [r.decision for r in results]

        if "RED" in decisions:
            return {
                "recommendation": agg_rules.get("any_red", "참여 보류"),
                "signal": "red",
                "reasons": [r.reasons[0] for r in results if r.decision == "RED"],
            }
        elif "GRAY" in decisions:
            return {
                "recommendation": agg_rules.get("any_yellow", "검토 필요"),
                "signal": "yellow",
                "reasons": [r.reasons[0] for r in results if r.decision == "GRAY"],
            }
        else:
            return {
                "recommendation": agg_rules.get("all_green", "참여 권장"),
                "signal": "green",
                "reasons": ["모든 조건 충족"],
            }

    # ── 기존 검증 메서드 ──

    def _check_license(self, slot: ExtractionSlot, profile: CompanyProfile) -> ValidationResult:
        req_license = str(slot.value).strip() if slot.value else ""
        my_licenses = profile.data.get("licenses", [])

        decision = "GRAY"
        reasons = []

        if slot.status == "NOT_FOUND":
            decision = "GREEN"
            reasons.append("제한 조항 미발견")
        elif not req_license:
            decision = "GRAY"
            reasons.append("추출된 면허 값이 비어있습니다.")
        else:
            found = any(lic in req_license for lic in my_licenses)
            if found:
                decision = "GREEN"
                reasons.append(f"보유 면허 '{my_licenses}'가 요구사항 '{req_license}'을 충족함 (단순매칭)")
            else:
                decision = "RED"
                reasons.append(f"요구 면허 '{req_license}'가 보유 목록에 없음")

        return ValidationResult(
            slot_key=slot.key,
            decision=decision,
            reasons=reasons,
            evidence=slot.evidence,
            risk_level="HIGH" if decision == "RED" else "LOW",
        )

    def _check_credit(self, slot: ExtractionSlot, profile: CompanyProfile) -> ValidationResult:
        req_credit = str(slot.value).strip().upper() if slot.value else ""
        my_credit = str(profile.data.get("credit_rating", "")).strip().upper()

        if not req_credit or req_credit in ["N/A", "미명시", "해당없음"]:
            return ValidationResult(slot_key=slot.key, decision="GREEN", reasons=["신용등급 제한 없음"], evidence=[], risk_level="LOW")

        if not my_credit:
            return ValidationResult(slot_key=slot.key, decision="GRAY", reasons=["프로필에 신용등급이 설정되지 않음"], evidence=slot.evidence, risk_level="MEDIUM")

        grades = [
            "AAA", "AA+", "AA", "AA-",
            "A+", "A", "A-",
            "BBB+", "BBB", "BBB-",
            "BB+", "BB", "BB-",
            "B+", "B", "B-",
            "CCC+", "CCC", "CCC-",
            "CC", "C", "D",
        ]

        def get_rank(grade):
            for i, g in enumerate(grades):
                if g in grade:
                    return i
            return 999

        req_rank = get_rank(req_credit)
        my_rank = get_rank(my_credit)

        if req_rank == 999:
            return ValidationResult(slot_key=slot.key, decision="GRAY", reasons=[f"요구 등급 '{req_credit}' 파싱 불가"], evidence=slot.evidence, risk_level="MEDIUM")

        if my_rank == 999:
            return ValidationResult(slot_key=slot.key, decision="GRAY", reasons=[f"보유 등급 '{my_credit}' 파싱 불가"], evidence=[], risk_level="MEDIUM")

        if my_rank <= req_rank:
            return ValidationResult(slot_key=slot.key, decision="GREEN", reasons=[f"보유 등급({my_credit})이 요구 등급({grades[req_rank]})을 충족함"], evidence=slot.evidence, risk_level="LOW")
        else:
            return ValidationResult(slot_key=slot.key, decision="RED", reasons=[f"보유 등급({my_credit})이 요구 등급({grades[req_rank]})보다 낮음"], evidence=slot.evidence, risk_level="HIGH")

    def _check_region(self, slot: ExtractionSlot, profile: CompanyProfile) -> ValidationResult:
        req_region = str(slot.value).strip() if slot.value else ""
        my_region = str(profile.data.get("region", "")).strip()

        if slot.status == "NOT_FOUND" or not req_region or req_region in ["전국", "제한없음", "N/A"]:
            return ValidationResult(slot_key=slot.key, decision="GREEN", reasons=["지역 제한 없음"], evidence=[], risk_level="LOW")

        if not my_region:
            return ValidationResult(slot_key=slot.key, decision="GRAY", reasons=["프로필에 지역 정보가 없음"], evidence=slot.evidence, risk_level="MEDIUM")

        match = False
        if my_region in req_region or req_region in my_region:
            match = True

        if "수도권" in req_region and my_region in ["서울", "경기", "인천", "서울특별시", "경기도", "인천광역시"]:
            match = True

        if match:
            return ValidationResult(slot_key=slot.key, decision="GREEN", reasons=[f"지역 조건 충족 ({my_region} in {req_region})"], evidence=slot.evidence, risk_level="LOW")

        return ValidationResult(
            slot_key=slot.key, decision="RED",
            reasons=[f"지역 제한 불충족 (요구: {req_region}, 보유: {my_region})"],
            evidence=slot.evidence, risk_level="HIGH",
        )

    # ── DecisionEngine 규칙 (김보윤에서 포팅) ──

    def _check_budget(self, slot: ExtractionSlot) -> ValidationResult:
        """예산 규모 적정성 검사 (김보윤 DecisionEngine)"""
        budget_rules = self.rules.get("budget", {})
        min_acceptable = budget_rules.get("min_acceptable", 30_000_000)
        too_large = budget_rules.get("too_large", 5_000_000_000)

        budget_str = str(slot.value).strip() if slot.value else ""

        if slot.status == "NOT_FOUND" or not budget_str:
            return ValidationResult(
                slot_key="budget_check", decision="GRAY",
                reasons=["예산 정보 불명확"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        # 숫자 추출 시도
        budget_amount = self._parse_budget_number(budget_str)
        if budget_amount is None:
            return ValidationResult(
                slot_key="budget_check", decision="GRAY",
                reasons=[f"예산 금액 파싱 불가: '{budget_str}'"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        if budget_amount < min_acceptable:
            return ValidationResult(
                slot_key="budget_check", decision="GRAY",
                reasons=[f"예산 규모 소형 ({budget_str})"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        if budget_amount > too_large:
            return ValidationResult(
                slot_key="budget_check", decision="GRAY",
                reasons=[f"예산 규모 대형 리스크 ({budget_str})"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        return ValidationResult(
            slot_key="budget_check", decision="GREEN",
            reasons=["예산 규모 적정"], evidence=slot.evidence, risk_level="LOW",
        )

    def _check_deadline(self, slot: ExtractionSlot) -> ValidationResult:
        """마감일 긴급도 검사 (김보윤 DecisionEngine)"""
        deadline_rules = self.rules.get("deadline", {})
        critical = deadline_rules.get("min_days_critical", 7)
        caution = deadline_rules.get("min_days_caution", 14)

        deadline_str = str(slot.value).strip() if slot.value else ""

        if slot.status == "NOT_FOUND" or not deadline_str:
            return ValidationResult(
                slot_key="deadline_check", decision="GRAY",
                reasons=["마감일 정보 불명확"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        remaining = self._days_until(deadline_str)
        if remaining is None:
            return ValidationResult(
                slot_key="deadline_check", decision="GRAY",
                reasons=[f"마감일 파싱 불가: '{deadline_str}'"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        if remaining < 0:
            return ValidationResult(
                slot_key="deadline_check", decision="RED",
                reasons=[f"마감일 경과 ({deadline_str})"], evidence=slot.evidence, risk_level="HIGH",
            )

        if remaining < critical:
            return ValidationResult(
                slot_key="deadline_check", decision="RED",
                reasons=[f"마감까지 {remaining}일 (긴급)"], evidence=slot.evidence, risk_level="HIGH",
            )

        if remaining < caution:
            return ValidationResult(
                slot_key="deadline_check", decision="GRAY",
                reasons=[f"마감까지 {remaining}일 (주의)"], evidence=slot.evidence, risk_level="MEDIUM",
            )

        return ValidationResult(
            slot_key="deadline_check", decision="GREEN",
            reasons=[f"마감까지 {remaining}일"], evidence=slot.evidence, risk_level="LOW",
        )

    def _check_info_completeness(self, matrix: ComplianceMatrix) -> ValidationResult:
        """정보 완전성 검사 (김보윤 DecisionEngine)"""
        completeness_rules = self.rules.get("info_completeness", {})
        max_yellow = completeness_rules.get("max_missing_yellow", 2)
        max_red = completeness_rules.get("max_missing_red", 4)

        total_slots = len(matrix.slots)
        missing = sum(1 for s in matrix.slots.values() if s.status == "NOT_FOUND")

        if missing > max_red:
            return ValidationResult(
                slot_key="info_completeness", decision="RED",
                reasons=[f"추출 정보 부족 ({missing}개 필드 누락/{total_slots}개 전체)"],
                evidence=[], risk_level="HIGH",
            )

        if missing > max_yellow:
            return ValidationResult(
                slot_key="info_completeness", decision="GRAY",
                reasons=[f"일부 정보 누락 ({missing}개 필드)"],
                evidence=[], risk_level="MEDIUM",
            )

        return ValidationResult(
            slot_key="info_completeness", decision="GREEN",
            reasons=["정보 충분"], evidence=[], risk_level="LOW",
        )

    # ── 유틸리티 ──

    @staticmethod
    def _parse_budget_number(budget_str: str) -> Optional[float]:
        """예산 문자열에서 숫자 추출"""
        # 콤마 제거 후 숫자 찾기
        cleaned = budget_str.replace(",", "").replace(" ", "")
        match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _days_until(date_str: str) -> Optional[int]:
        """날짜 문자열에서 오늘까지의 일수 계산"""
        formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y.%m.%d", "%Y년 %m월 %d일"]
        for fmt in formats:
            try:
                target = datetime.strptime(date_str.strip(), fmt).date()
                return (target - date.today()).days
            except ValueError:
                continue
        return None

    @staticmethod
    def _load_rules(rules_path) -> dict:
        """YAML 규칙 파일 로드"""
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[Validator] Rules file not found: {rules_path}, using defaults")
            return {}
