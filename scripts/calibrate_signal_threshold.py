"""P0-4: fit_score 임계값 캘리브레이션 스크립트.

합성 시나리오 기반 green_threshold sweep으로 최적 임계값 결정.
- 6종 문서 × 3종 프로필(적합/중간/부적합) = 18 시나리오
- threshold 0.40~0.95 (0.05 간격) sweep
- Precision/Recall/F1 측정, 최적 임계값 선택

실행: python scripts/calibrate_signal_threshold.py
결과: docs/planning/signal_calibration.md
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from bidflow.domain.models import ValidationResult
from bidflow.extraction.batch_pipeline import compute_fit_score


# ── 합성 시나리오 정의 ──
# 각 시나리오: (이름, ValidationResult 리스트, 기대 신호)
# decision 값: GREEN=조건 충족, RED=조건 위반, GRAY=정보 부족

def _vr(slot_key: str, decision: str, reason: str = "") -> ValidationResult:
    """ValidationResult 축약 생성."""
    return ValidationResult(
        slot_key=slot_key,
        decision=decision,
        reasons=[reason or f"{slot_key}: {decision}"],
        evidence=[],
        risk_level="HIGH" if decision == "RED" else ("MEDIUM" if decision == "GRAY" else "LOW"),
    )


# 시나리오: (이름, results, 기대 신호)
# 기대 신호는 사람이 "이 조합이면 이 신호가 맞다"고 판단한 라벨
SCENARIOS: List[Tuple[str, List[ValidationResult], str]] = [
    # ── 명확한 GREEN 케이스 ──
    ("수협_적합기업", [
        _vr("required_licenses", "GREEN", "보유 면허 충족"),
        _vr("financial_credit", "GREEN", "신용등급 충족"),
        _vr("region_restriction", "GREEN", "지역 제한 없음"),
        _vr("budget_check", "GREEN", "예산 적정 (300백만원)"),
        _vr("deadline_check", "GREEN", "마감까지 30일"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "GREEN"),

    ("NCIC_적합기업", [
        _vr("required_licenses", "GREEN", "제한 없음"),
        _vr("financial_credit", "GREEN", "신용등급 제한 없음"),
        _vr("region_restriction", "GREEN", "지역 제한 없음"),
        _vr("budget_check", "GREEN", "예산 적정 (50백만원)"),
        _vr("deadline_check", "GREEN", "마감까지 45일"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "GREEN"),

    ("응급의료_적합기업", [
        _vr("required_licenses", "GREEN", "보유 면허 충족"),
        _vr("financial_credit", "GREEN", "A등급 충족"),
        _vr("region_restriction", "GREEN", "전국"),
        _vr("budget_check", "GREEN", "예산 적정 (14억)"),
        _vr("deadline_check", "GREEN", "마감까지 60일"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "GREEN"),

    # ── 명확한 RED 케이스 ──
    ("수협_면허미보유", [
        _vr("required_licenses", "RED", "요구 면허 없음"),
        _vr("financial_credit", "GREEN", "신용등급 충족"),
        _vr("region_restriction", "GREEN", "지역 충족"),
        _vr("budget_check", "GREEN", "예산 적정"),
        _vr("deadline_check", "GREEN", "마감 여유"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "RED"),

    ("철도공사_신용미달", [
        _vr("required_licenses", "GREEN", "면허 충족"),
        _vr("financial_credit", "RED", "신용등급 미달"),
        _vr("region_restriction", "GREEN", "지역 충족"),
        _vr("budget_check", "GREEN", "예산 적정 (470백만원)"),
        _vr("deadline_check", "GREEN", "마감 여유"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "RED"),

    ("스포츠윤리_지역불일치", [
        _vr("required_licenses", "GREEN", "제한 없음"),
        _vr("financial_credit", "GREEN", "제한 없음"),
        _vr("region_restriction", "RED", "서울 한정, 보유: 부산"),
        _vr("budget_check", "GREEN", "예산 적정"),
        _vr("deadline_check", "GREEN", "마감 여유"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "RED"),

    ("응급의료_마감경과", [
        _vr("required_licenses", "GREEN", "면허 충족"),
        _vr("financial_credit", "GREEN", "등급 충족"),
        _vr("region_restriction", "GREEN", "지역 충족"),
        _vr("budget_check", "GREEN", "예산 적정"),
        _vr("deadline_check", "RED", "마감일 경과"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "RED"),

    ("NCIC_복수RED", [
        _vr("required_licenses", "RED", "면허 미보유"),
        _vr("financial_credit", "RED", "등급 미달"),
        _vr("region_restriction", "GREEN", "지역 충족"),
        _vr("budget_check", "GREEN", "예산 적정"),
        _vr("deadline_check", "GREEN", "마감 여유"),
        _vr("info_completeness", "GREEN", "정보 충분"),
    ], "RED"),

    # ── GRAY 케이스 (정보 부족/불명확) ──
    ("수협_정보부족", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GRAY", "예산 파싱 불가"),
        _vr("deadline_check", "GRAY", "마감일 불명확"),
        _vr("info_completeness", "GRAY", "3개 필드 누락"),
    ], "GRAY"),

    ("철도공사_면허불명확", [
        _vr("required_licenses", "GRAY", "면허 정보 불명확"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("NCIC_신용불명확", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GRAY", "프로필에 등급 미설정"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("응급의료_지역불명확", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GRAY", "지역 정보 파싱 불가"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("스포츠윤리_정보완전성GRAY", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GRAY", "2개 필드 누락"),
    ], "GRAY"),

    # ── 경계선 케이스 ──
    ("수협_예산소형만GRAY", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GRAY", "예산 소형"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("철도공사_마감주의만GRAY", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GRAY", "마감까지 10일 (주의)"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("응급의료_필수GREEN_비필수GRAY", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GRAY", "예산 불명확"),
        _vr("deadline_check", "GRAY", "마감 불명확"),
        _vr("info_completeness", "GREEN", "충분"),
    ], "GRAY"),

    ("NCIC_거의GREEN_정보미완", [
        _vr("required_licenses", "GREEN", "충족"),
        _vr("financial_credit", "GREEN", "충족"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GREEN", "적정"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GRAY", "1개 필드 누락"),
    ], "GRAY"),

    ("스포츠윤리_RED+GRAY혼합", [
        _vr("required_licenses", "RED", "면허 미보유"),
        _vr("financial_credit", "GRAY", "등급 불명확"),
        _vr("region_restriction", "GREEN", "충족"),
        _vr("budget_check", "GRAY", "예산 불명확"),
        _vr("deadline_check", "GREEN", "여유"),
        _vr("info_completeness", "GRAY", "다수 누락"),
    ], "RED"),
]


def determine_signal(
    results: List[ValidationResult],
    fit_score: float,
    green_threshold: float,
    mandatory_slots: List[str],
) -> str:
    """BatchPipeline._determine_signal과 동일 로직 (v2: 강화)."""
    mandatory_results = [r for r in results if r.slot_key in mandatory_slots]

    # RED: 필수 항목 중 하나라도 RED
    if any(r.decision == "RED" for r in mandatory_results):
        return "RED"

    # RED: 비필수라도 RED 존재 (마감 경과 등)
    if any(r.decision == "RED" for r in results):
        return "RED"

    # GREEN: 전체 결과에 GRAY 없음 + fit_score >= threshold
    all_green = all(r.decision == "GREEN" for r in results)
    if all_green and fit_score >= green_threshold:
        return "GREEN"

    return "GRAY"


def run_calibration():
    """threshold sweep 실행."""
    mandatory_slots = ["required_licenses", "financial_credit", "region_restriction"]
    thresholds = [round(0.40 + i * 0.05, 2) for i in range(12)]  # 0.40 ~ 0.95

    print(f"=== fit_score 임계값 캘리브레이션 ===")
    print(f"시나리오 수: {len(SCENARIOS)}")
    print(f"threshold 범위: {thresholds[0]} ~ {thresholds[-1]}")
    print()

    # 각 시나리오의 fit_score 사전 계산
    scenario_scores = []
    for name, results, expected in SCENARIOS:
        score = compute_fit_score(results)
        scenario_scores.append((name, results, expected, score))

    # fit_score 분포 출력
    print("── 시나리오별 fit_score ──")
    for name, results, expected, score in scenario_scores:
        decisions = {r.slot_key: r.decision for r in results}
        print(f"  {name:30s}  score={score:.3f}  expected={expected:5s}  {decisions}")
    print()

    # threshold sweep
    sweep_results = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        gray_as_green = 0  # GRAY를 GREEN으로 잘못 판정
        green_as_gray = 0  # GREEN을 GRAY로 잘못 판정

        mismatches = []
        for name, results, expected, score in scenario_scores:
            predicted = determine_signal(results, score, threshold, mandatory_slots)

            if expected == "GREEN" and predicted == "GREEN":
                tp += 1
            elif expected != "GREEN" and predicted != "GREEN":
                tn += 1
            elif expected != "GREEN" and predicted == "GREEN":
                fp += 1
                mismatches.append(f"  FP: {name} (expected={expected}, predicted=GREEN, score={score:.3f})")
                if expected == "GRAY":
                    gray_as_green += 1
            elif expected == "GREEN" and predicted != "GREEN":
                fn += 1
                mismatches.append(f"  FN: {name} (expected=GREEN, predicted={predicted}, score={score:.3f})")
                if predicted == "GRAY":
                    green_as_gray += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(SCENARIOS)

        # RED 정확도 (별도): RED 기대 중 RED 예측 비율
        red_expected = sum(1 for _, _, exp, _ in scenario_scores if exp == "RED")
        red_correct = sum(
            1 for _, results, exp, score in scenario_scores
            if exp == "RED" and determine_signal(results, score, threshold, mandatory_slots) == "RED"
        )
        red_accuracy = red_correct / red_expected if red_expected > 0 else 1.0

        sweep_results.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy,
            "red_accuracy": red_accuracy,
            "gray_as_green": gray_as_green,
            "green_as_gray": green_as_gray,
            "mismatches": mismatches,
        })

    # 결과 출력
    print("── Threshold Sweep 결과 ──")
    print(f"{'Threshold':>10s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Acc':>6s} {'RED_Acc':>7s} {'G→Gray':>6s} {'Gray→G':>6s}")
    print("-" * 70)
    for r in sweep_results:
        print(f"  {r['threshold']:>7.2f}  {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['accuracy']:>6.3f} {r['red_accuracy']:>7.3f} {r['green_as_gray']:>6d} {r['gray_as_green']:>6d}")
    print()

    # 최적 임계값 선택: Precision >= 0.9 조건에서 F1 최대
    # (보수적: 부적합을 GREEN으로 판정하는 것이 더 위험하므로 Precision 우선)
    candidates = [r for r in sweep_results if r["precision"] >= 0.9]
    if not candidates:
        candidates = sweep_results  # 모든 threshold에서 precision < 0.9이면 전체에서 선택

    best = max(candidates, key=lambda r: r["f1"])
    print(f"=== 최적 임계값: {best['threshold']} ===")
    print(f"  Precision={best['precision']:.3f}, Recall={best['recall']:.3f}, F1={best['f1']:.3f}")
    print(f"  Accuracy={best['accuracy']:.3f}, RED Accuracy={best['red_accuracy']:.3f}")
    print(f"  GREEN을 GRAY로 잘못 판정: {best['green_as_gray']}건")
    print(f"  GRAY를 GREEN으로 잘못 판정: {best['gray_as_green']}건")
    if best["mismatches"]:
        print("  불일치:")
        for m in best["mismatches"]:
            print(f"    {m}")
    print()

    # 결과 문서 저장
    save_calibration_report(sweep_results, best, scenario_scores)

    return best["threshold"]


def save_calibration_report(sweep_results, best, scenario_scores):
    """캘리브레이션 결과를 마크다운 문서로 저장."""
    report_path = project_root / "docs" / "planning" / "signal_calibration.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# fit_score 임계값 캘리브레이션 결과",
        "",
        f"실행일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 방법론",
        "",
        f"- 합성 시나리오 {len(scenario_scores)}개 (6종 문서 x 3종 프로필 + 경계선 케이스)",
        "- threshold sweep: 0.40 ~ 0.95 (0.05 간격, 12단계)",
        "- 선택 기준: Precision >= 0.9 조건에서 F1 최대 (보수적: FP 최소화 우선)",
        "",
        "## 시나리오별 fit_score",
        "",
        "| 시나리오 | fit_score | 기대 신호 |",
        "|---------|-----------|----------|",
    ]
    for name, _, expected, score in scenario_scores:
        lines.append(f"| {name} | {score:.3f} | {expected} |")

    lines += [
        "",
        "## Sweep 결과",
        "",
        "| Threshold | Precision | Recall | F1 | Accuracy | RED정확도 | GREEN->GRAY | GRAY->GREEN |",
        "|-----------|-----------|--------|-----|----------|----------|------------|------------|",
    ]
    for r in sweep_results:
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['accuracy']:.3f} | {r['red_accuracy']:.3f} | "
            f"{r['green_as_gray']} | {r['gray_as_green']} |"
        )

    lines += [
        "",
        f"## 최적 임계값: **{best['threshold']}**",
        "",
        f"- Precision: {best['precision']:.3f}",
        f"- Recall: {best['recall']:.3f}",
        f"- F1: {best['f1']:.3f}",
        f"- Accuracy: {best['accuracy']:.3f}",
        f"- RED 정확도: {best['red_accuracy']:.3f}",
        "",
        "### 권고",
        "",
        f"- `configs/base.yaml`의 `signal.green_threshold`를 **{best['threshold']}**로 설정",
        "- 실제 문서 운영 데이터 축적 후 재캘리브레이션 권장",
        "- 라벨셋 확장 시 (실제 추출 결과 + 수동 라벨) 정밀도 향상 가능",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"결과 저장: {report_path}")


if __name__ == "__main__":
    optimal = run_calibration()
    print(f"\n최종 권고 임계값: {optimal}")
    print(f"적용: configs/base.yaml → signal.green_threshold: {optimal}")
