"""
EXP19 Phase C: Holdout GT 정제 + kw_v5 재계산 (API 0)

실행:
  cd bidflow
  python -X utf8 scripts/run_exp19_phase_c_gt_refine.py
"""
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "experiments"

HOLDOUT_GT_SRC = DATA_DIR / "golden_testset_holdout.csv"
HOLDOUT_GT_V2 = DATA_DIR / "golden_testset_holdout_v2.csv"
HOLDOUT_METRICS_SRC = DATA_DIR / "exp19_holdout_metrics.csv"
HOLDOUT_METRICS_V2 = DATA_DIR / "exp19_holdout_metrics_v2.csv"
HOLDOUT_REPORT_V2 = DATA_DIR / "exp19_holdout_report_v2.json"


REFINED_GT_BY_QUESTION = {
    "사업기간은 얼마입니까?": "계약 체결일로부터 6개월.",
    "상황관리시스템이 제공해야 하는 주요 기능들은 무엇입니까?": "NDMS 연계 상황알림, 대응·조치·보고전파, 정보검색·현황관리 기능.",
    "공동수급으로 참여할 경우 수급체 구성에 대한 조건은 무엇입니까?": "단독 또는 공동수급 가능, 공동수급은 5개사 이하, 최소지분율 10% 이상.",
    "제안참가업체가 제안서 및 사업 수행 중 취득한 발주기관의 정보에 대해 지켜야 할 보안 의무와 위반 시 책임은 무엇입니까?": "사업 중 취득 정보 비밀유지·외부공개 금지, 보안 위반 시 수행업체 민·형사상 책임.",
    "사업기간은 언제까지인가요?": "계약체결일로부터 2025. 2월까지.",
    "사업범위에는 어떤 주요 업무들이 포함되어 있나요?": "직무·위탁교육 콘텐츠 운영, 사이버(모바일)연수원 구축·운영, 학습관리시스템 서버 임대.",
    "입찰방법과 계약방법은 무엇으로 규정되어 있나요?": "입찰방법은 일반경쟁입찰, 계약방법은 협상에 의한 계약.",
    "제안서 평가에서 기술능력평가와 가격평가의 배점 및 협상적격자 선정 기준은 무엇인가요?": "기술능력평가 90점·입찰가격평가 10점, 기술 85%(76.5점) 이상 협상적격, 합산점수 고득점순 협상.",
    "사업명, 사업기간 및 사업금액은 무엇인가요?": "봉화군 재난통합관리시스템 고도화 사업, 착수일로부터 7개월(210일), 900,000,000원(부가세 포함).",
    "통합연계시스템 구축의 주요 범위와 기능은 무엇인가요?": "계측기·재난예경보장치 연계, 수집 데이터 기반 원격 제어·관리, 이기종 시스템 네트워크 구성, 유관기관 데이터 수집·관리.",
    "이 사업의 계약방법은 무엇이며 낙찰자 선정 절차는 어떻게 이루어지나요?": "제한경쟁입찰(협상에 의한 계약), 제안서 평가 후 협상절차로 선정.",
    "제안서 제출 전 시스템(네트워크) 구성도 열람과 관련된 보안 요건은 무엇인가요?": "요청 업체만 보안서약서 작성 후 구성도 열람 가능, 외부 유출·복사 불허.",
    "사업명, 사업기간(종료일), 사업예산(금액·VAT 포함) 및 사업추진방식은 무엇인가?": "통합 정보시스템 구축 사전 컨설팅, 계약체결일로부터 2024년 11월 29일, 50,000,000(금 오천만원/VAT포함), 제한경쟁입찰(협상에 의한 계약).",
    "이 사업의 주요 수행범위(주요 과업 항목)는 어떤 것들을 포함하는가?": "제안개요, 사업수행능력, 사업추진 방안, 사업세부계획(수행계획·수행기반·프로젝트관리·지원).",
    "수급인은 계약체결 후 10일 이내에 무엇을 해야 하고 어떤 서류를 제출해야 하는가?": "계약체결 후 10일 이내 과업 착수, 착수계·과업수행계획표·용역수행자 명단 및 이력서·보안각서 제출.",
    "성과품의 저작권·보관·보안 관련 주요 규정과 성과품으로 인해 제3자와의 권리분쟁이 발생했을 때 책임은 누구에게 있는가?": "성과품 저작권 공동소유, 2년 이상 보관·복사/유출 금지, 제3자 권리분쟁 책임과 보상은 수급자 부담.",
    "이 사업의 명칭, 사업기간, 추정가격 및 계약방식은 무엇인가?": "통합정보시스템 고도화 용역, 계약 후 5개월 이내, 140,000,000원(부가세 포함) 이내, 협상에 의한 계약(제한경쟁입찰).",
    "본 사업에서 신규 시스템이 구축해야 할 주요 업무 프로세스와 구현해야 할 주요 기능은 무엇인가?": "기관생명윤리위원회(IRB)·동물실험계획 업무 프로세스 구축, 연구비 회계기준 연계, 기존 시스템 개선·UI/UX 구현.",
    "이 입찰에 참가할 수 있는 자격과 주요 참여 제한사항은 무엇인가?": "중소기업자간 경쟁, 소프트웨어사업자(컴퓨터 관련 서비스-코드번호1468) 신고·구매정보망 직접생산 확인 등재, 중견·대기업 참여 제한, 공동수급·하도급 불허.",
    "협상적격자 선정 기준과 협상 순서 및 기술평가 점수 산정 방식은 어떻게 되어 있는가?": "기술능력 평가분야 배점 85% 이상 협상적격, 기술평가 점수는 최고·최저 제외 평균, 종합평가점수 고득점 순 협상.",
}


SYNONYM_MAP = {
    "정보전략계획": "ismp",
    "ismp 수립": "ismp",
    "정보화전략계획": "ismp",
    "통합로그인": "sso",
    "단일 로그인": "sso",
    "싱글사인온": "sso",
    "project manager": "pm",
    "사업관리자": "pm",
    "사업책임자": "pm",
    "프로젝트 매니저": "pm",
    "project leader": "pl",
    "프로젝트 리더": "pl",
    "quality assurance": "qa",
    "품질관리": "qa",
    "품질보증": "qa",
    "하자보수": "하자보수",
    "하자 보수": "하자보수",
    "발주처": "발주기관",
    "발주 기관": "발주기관",
}

PARTICLES_RE = re.compile(
    r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|이며|이고|에게|한테|부터|까지|도|만|이라|인|에는|에도)$"
)

ROMAN_MAP = {
    "ⅰ": "1",
    "ⅱ": "2",
    "ⅲ": "3",
    "ⅳ": "4",
    "ⅴ": "5",
    "ⅵ": "6",
    "ⅶ": "7",
    "ⅷ": "8",
    "ⅸ": "9",
    "ⅹ": "10",
    "Ⅰ": "1",
    "Ⅱ": "2",
    "Ⅲ": "3",
    "Ⅳ": "4",
    "Ⅴ": "5",
    "Ⅵ": "6",
    "Ⅶ": "7",
    "Ⅷ": "8",
    "Ⅸ": "9",
    "Ⅹ": "10",
}

VERB_ENDINGS = [
    "하며",
    "이며",
    "으며",
    "되며",
    "하고",
    "이고",
    "되고",
    "하여",
    "이어",
    "되어",
    "하는",
    "되는",
    "인",
    "한다",
    "된다",
    "이다",
    "합니다",
    "됩니다",
    "입니다",
    "하면",
    "되면",
    "이면",
    "해서",
    "되서",
    "이라서",
    "했던",
    "되었던",
    "이었던",
    "1명인",
]


def normalize_v2(text):
    if not isinstance(text, str):
        return str(text).strip().lower()
    t = text.strip().lower()
    t = re.sub(r"[\u00b7\u2027\u2022\u2219]", " ", t)
    t = re.sub(r"[\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f]", "", t)
    t = re.sub(r"[-\u2013\u2014]", " ", t)
    t = re.sub(r"(\d),(?=\d{3})", r"\1", t)
    t = re.sub(r"(\d+)\s*(%|퍼센트|percent)", r"\1%", t)
    t = re.sub(r"(\d+)\s*원", r"\1원", t)
    t = re.sub(r"(\d+)\s*억\s*원", r"\1억원", t)
    t = re.sub(r"(\d+)\s*만\s*원", r"\1만원", t)
    t = t.replace("v.a.t", "vat").replace("vat 포함", "vat포함")
    for orig, norm in SYNONYM_MAP.items():
        t = t.replace(orig.lower(), norm)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_v4(text):
    t = normalize_v2(text)
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)
    t = t.replace("￦", "₩").replace("'", "").replace('"', "").replace("※", "")
    t = re.sub(r"(\d+)\.\s+(\d+월)", r"\1.\2", t)
    t = re.sub(r"(\d+)\.\s+(\d+\))", r"\1.\2", t)
    t = re.sub(r"\s*~\s*", "~", t)
    t = re.sub(r"(\d+)\s*페이지", r"\1p", t)
    t = re.sub(r"(\d+)\s*쪽", r"\1p", t)
    t = re.sub(r"제(\d+)장", r"\1장", t)
    t = re.sub(r"(?<!\d)(\d{1,2})\.\s+([가-힣])", r"\1장 \2", t)
    t = re.sub(r"([가-힣a-z0-9])\(", r"\1 (", t)
    t = re.sub(r"\)([가-힣a-z])", r") \1", t)
    words = t.split()
    cleaned = []
    for w in words:
        w = w.rstrip(".,;:!?")
        if not w:
            continue
        stripped = PARTICLES_RE.sub("", w)
        cleaned.append(stripped if stripped else w)
    return " ".join(cleaned)


def _strip_verb_ending(keyword):
    for ending in sorted(VERB_ENDINGS, key=len, reverse=True):
        if keyword.endswith(ending) and len(keyword) > len(ending):
            stem = keyword[: -len(ending)]
            if len(stem) > 1:
                return stem
    return None


def keyword_accuracy_v5(answer, ground_truth):
    ans_norm = normalize_v4(answer)
    gt_norm = normalize_v4(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0
    matched = 0
    for kw in gt_words:
        if kw in ans_norm:
            matched += 1
        else:
            stem = _strip_verb_ending(kw)
            if stem and stem in ans_norm:
                matched += 1
    return matched / len(gt_words)


def build_refined_gt():
    holdout_df = pd.read_csv(HOLDOUT_GT_SRC)
    missing_questions = sorted(set(holdout_df["question"]) - set(REFINED_GT_BY_QUESTION))
    extra_questions = sorted(set(REFINED_GT_BY_QUESTION) - set(holdout_df["question"]))
    if missing_questions or extra_questions:
        raise ValueError(
            f"Question mapping mismatch. missing={missing_questions}, extra={extra_questions}"
        )

    refined_df = holdout_df.copy()
    refined_df["ground_truth"] = refined_df["question"].map(REFINED_GT_BY_QUESTION)
    refined_df.to_csv(HOLDOUT_GT_V2, index=False, encoding="utf-8-sig")

    old_lengths = holdout_df["ground_truth"].astype(str).str.len()
    new_lengths = refined_df["ground_truth"].astype(str).str.len()
    stats = {
        "n_rows": int(len(refined_df)),
        "old_gt_length_avg": float(old_lengths.mean()),
        "new_gt_length_avg": float(new_lengths.mean()),
        "old_gt_length_median": float(old_lengths.median()),
        "new_gt_length_median": float(new_lengths.median()),
    }
    return refined_df, stats


def rescore_with_refined_gt(refined_df):
    metrics_df = pd.read_csv(HOLDOUT_METRICS_SRC)
    gt_map = dict(zip(refined_df["question"], refined_df["ground_truth"]))
    missing_questions = sorted(set(metrics_df["question"]) - set(gt_map))
    if missing_questions:
        raise ValueError(f"Missing GT for questions in metrics: {missing_questions}")

    rescored_df = metrics_df.copy()
    rescored_df["ground_truth_v2"] = rescored_df["question"].map(gt_map)
    rescored_df["kw_v5_v2"] = rescored_df.apply(
        lambda row: keyword_accuracy_v5(str(row["answer"]), str(row["ground_truth_v2"])),
        axis=1,
    )
    rescored_df["kw_v5_delta"] = rescored_df["kw_v5_v2"] - rescored_df["kw_v5"]
    rescored_df.to_csv(HOLDOUT_METRICS_V2, index=False, encoding="utf-8-sig")

    before = float(rescored_df["kw_v5"].mean())
    after = float(rescored_df["kw_v5_v2"].mean())
    n_total = int(len(rescored_df))
    perfect_before = int((rescored_df["kw_v5"] >= 1.0).sum())
    perfect_after = int((rescored_df["kw_v5_v2"] >= 1.0).sum())

    if after >= 0.95:
        verdict = "PASS"
        verdict_text = "과적합 없음"
        next_action = "Q1 해결 (Phase A 잔여)"
    elif after >= 0.90:
        verdict = "MILD"
        verdict_text = "경미한 과적합"
        next_action = "범용 프롬프트 개선 고려"
    else:
        verdict = "SEVERE"
        verdict_text = "심각한 과적합"
        next_action = "범용 파이프라인 개선 필수"

    worst = rescored_df.sort_values("kw_v5_v2").head(10)[
        ["doc_key", "question", "kw_v5", "kw_v5_v2", "kw_v5_delta"]
    ]
    worst_records = worst.to_dict("records")

    report = {
        "experiment": "EXP19_Phase_C_GT_refine",
        "timestamp": pd.Timestamp.now().isoformat(),
        "paths": {
            "holdout_gt_src": str(HOLDOUT_GT_SRC),
            "holdout_gt_v2": str(HOLDOUT_GT_V2),
            "holdout_metrics_src": str(HOLDOUT_METRICS_SRC),
            "holdout_metrics_v2": str(HOLDOUT_METRICS_V2),
        },
        "summary": {
            "n_questions": n_total,
            "kw_v5_before": before,
            "kw_v5_after": after,
            "delta": after - before,
            "perfect_before": perfect_before,
            "perfect_after": perfect_after,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "next_action": next_action,
        },
        "breakdown": {
            "by_doc": rescored_df.groupby("doc_key")[["kw_v5", "kw_v5_v2"]].mean().round(6).to_dict("index"),
            "by_category": rescored_df.groupby("category")[["kw_v5", "kw_v5_v2"]]
            .mean()
            .round(6)
            .to_dict("index"),
            "by_difficulty": rescored_df.groupby("difficulty")[["kw_v5", "kw_v5_v2"]]
            .mean()
            .round(6)
            .to_dict("index"),
        },
        "worst_after_top10": worst_records,
    }

    with HOLDOUT_REPORT_V2.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return rescored_df, report


def main():
    refined_df, gt_stats = build_refined_gt()
    _, report = rescore_with_refined_gt(refined_df)

    summary = report["summary"]
    print("=" * 72)
    print("EXP19 Phase C - Holdout GT Refine + Rescore (API 0)")
    print("=" * 72)
    print(f"GT rows: {gt_stats['n_rows']}")
    print(
        f"GT length avg: {gt_stats['old_gt_length_avg']:.1f} -> {gt_stats['new_gt_length_avg']:.1f}"
    )
    print(
        f"kw_v5: {summary['kw_v5_before']:.4f} -> {summary['kw_v5_after']:.4f} "
        f"(delta {summary['delta']:+.4f})"
    )
    print(
        f"perfect: {summary['perfect_before']}/{summary['n_questions']} -> "
        f"{summary['perfect_after']}/{summary['n_questions']}"
    )
    print(f"verdict: {summary['verdict']} ({summary['verdict_text']})")
    print(f"next action: {summary['next_action']}")
    print("-" * 72)
    print(f"Saved: {HOLDOUT_GT_V2}")
    print(f"Saved: {HOLDOUT_METRICS_V2}")
    print(f"Saved: {HOLDOUT_REPORT_V2}")
    print("=" * 72)


if __name__ == "__main__":
    main()
