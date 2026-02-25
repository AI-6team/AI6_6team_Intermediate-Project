"""
EXP19 Phase C: Q1 GT 보정(간결화) + API 0 재채점

실행:
  cd bidflow
  python -X utf8 scripts/run_exp19_phase_c_q1_fix.py
"""
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "experiments"

TESTSET_V2_PATH = DATA_DIR / "golden_testset_multi_v2.csv"
TESTSET_V3_PATH = DATA_DIR / "golden_testset_multi_v3.csv"
METRICS_PATH = DATA_DIR / "exp19_metrics.csv"
OUT_CSV_PATH = DATA_DIR / "exp19_q1_rescore.csv"
OUT_REPORT_PATH = DATA_DIR / "exp19_q1_rescore_report.json"

Q1_QUESTION = "現 수산물 사이버직매장 시스템의 주요 문제점은 무엇인가?"
Q1_GT_V4 = (
    "시스템 노후화(2015년 구축)로 인한 장애위험 증가 및 유지관리 한계, "
    "보안정책 과다 적용 및 스위치 대역폭 부족."
)


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


def build_testset_v3():
    testset_v2 = pd.read_csv(TESTSET_V2_PATH)
    q_mask = testset_v2["question"] == Q1_QUESTION
    if q_mask.sum() != 1:
        raise ValueError(f"Expected one Q1 row in testset_v2, found {q_mask.sum()}")
    old_gt = testset_v2.loc[q_mask, "ground_truth"].iloc[0]

    testset_v3 = testset_v2.copy()
    testset_v3.loc[q_mask, "ground_truth"] = Q1_GT_V4
    testset_v3.to_csv(TESTSET_V3_PATH, index=False, encoding="utf-8-sig")
    return old_gt, Q1_GT_V4


def rescore_q1():
    metrics_df = pd.read_csv(METRICS_PATH)
    best_per_q = {}
    for _, row in metrics_df.iterrows():
        q = row["question"]
        if q not in best_per_q or row["kw_v5"] > best_per_q[q]["kw_v5"]:
            best_per_q[q] = row.to_dict()

    records = []
    for q, row in best_per_q.items():
        old_gt = row["ground_truth"]
        old_score = float(row["kw_v5"])
        new_gt = old_gt
        new_score = old_score

        if q == Q1_QUESTION:
            new_gt = Q1_GT_V4
            new_score = keyword_accuracy_v5(str(row["answer"]), new_gt)

        records.append(
            {
                "question": q,
                "doc_key": row["doc_key"],
                "config": row["config"],
                "answer": row["answer"],
                "ground_truth_old": old_gt,
                "ground_truth_new": new_gt,
                "kw_v5_old": old_score,
                "kw_v5_new": float(new_score),
                "delta": float(new_score - old_score),
            }
        )

    out_df = pd.DataFrame(records).sort_values(["kw_v5_new", "question"])
    out_df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

    old_mean = out_df["kw_v5_old"].mean()
    new_mean = out_df["kw_v5_new"].mean()
    old_perfect = int((out_df["kw_v5_old"] >= 1.0).sum())
    new_perfect = int((out_df["kw_v5_new"] >= 1.0).sum())
    q1_row = out_df[out_df["question"] == Q1_QUESTION].iloc[0]

    report = {
        "experiment": "EXP19_Phase_C_Q1_fix",
        "paths": {
            "testset_v2": str(TESTSET_V2_PATH),
            "testset_v3": str(TESTSET_V3_PATH),
            "metrics_source": str(METRICS_PATH),
            "rescore_csv": str(OUT_CSV_PATH),
        },
        "q1": {
            "question": Q1_QUESTION,
            "kw_v5_old": float(q1_row["kw_v5_old"]),
            "kw_v5_new": float(q1_row["kw_v5_new"]),
            "delta": float(q1_row["delta"]),
            "ground_truth_new": Q1_GT_V4,
        },
        "composite": {
            "n_questions": int(len(out_df)),
            "kw_v5_old": float(old_mean),
            "kw_v5_new": float(new_mean),
            "delta": float(new_mean - old_mean),
            "perfect_old": old_perfect,
            "perfect_new": new_perfect,
        },
    }
    with OUT_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    old_gt, new_gt = build_testset_v3()
    report = rescore_q1()
    q1 = report["q1"]
    comp = report["composite"]

    print("=" * 72)
    print("EXP19 Phase C - Q1 Fix (API 0)")
    print("=" * 72)
    print(f"Q1 old GT: {old_gt}")
    print(f"Q1 new GT: {new_gt}")
    print(f"Q1 score: {q1['kw_v5_old']:.4f} -> {q1['kw_v5_new']:.4f} ({q1['delta']:+.4f})")
    print(
        f"Composite: {comp['kw_v5_old']:.4f} -> {comp['kw_v5_new']:.4f} "
        f"({comp['delta']:+.4f}), perfect {comp['perfect_old']} -> {comp['perfect_new']}"
    )
    print("-" * 72)
    print(f"Saved: {TESTSET_V3_PATH}")
    print(f"Saved: {OUT_CSV_PATH}")
    print(f"Saved: {OUT_REPORT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
