"""
Simulate kw_v5b: improved matching with space-collapse fallback + paren stripping.
Reads D8 metrics CSV and re-scores to measure delta vs kw_v5.
"""
import re, sys, csv
from pathlib import Path

# ── Copy normalize functions from eval script ──
SYNONYM_MAP = {
    "정보전략계획": "ismp", "ismp 수립": "ismp", "정보화전략계획": "ismp",
    "통합로그인": "sso", "단일 로그인": "sso", "싱글사인온": "sso",
    "project manager": "pm", "사업관리자": "pm", "사업책임자": "pm",
    "프로젝트 매니저": "pm", "project leader": "pl", "프로젝트 리더": "pl",
    "quality assurance": "qa", "품질관리": "qa", "품질보증": "qa",
    "하자보수": "하자보수", "하자 보수": "하자보수",
    "발주처": "발주기관", "발주 기관": "발주기관",
}

PARTICLES_RE = re.compile(
    r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|이며|이고|에게|한테|부터|까지|도|만|이라|인|에는|에도)$"
)

ROMAN_MAP = {
    "ⅰ": "1", "ⅱ": "2", "ⅲ": "3", "ⅳ": "4", "ⅴ": "5",
    "ⅵ": "6", "ⅶ": "7", "ⅷ": "8", "ⅸ": "9", "ⅹ": "10",
    "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
    "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
}

VERB_ENDINGS = [
    "하며", "이며", "으며", "되며", "하고", "이고", "되고",
    "하여", "이어", "되어", "하는", "되는", "인",
    "한다", "된다", "이다", "합니다", "됩니다", "입니다",
    "하면", "되면", "이면", "해서", "되서", "이라서",
    "했던", "되었던", "이었던", "1명인",
]

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


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
    t = re.sub(r"(\d{4})년\s*(\d{1,2})월", r"\1.\2월", t)
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


# ── Original kw_v5 ──
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


# ── Improved normalize_v4b ──
def normalize_v4b(text):
    """v4b: also normalize slashes to spaces and strip parentheses from tokens."""
    t = normalize_v2(text)
    # v4b addition: normalize slashes to spaces
    t = re.sub(r"/", " ", t)
    for roman, arabic in ROMAN_MAP.items():
        t = t.replace(roman.lower(), arabic)
    t = t.replace("￦", "₩").replace("'", "").replace('"', "").replace("※", "")
    t = re.sub(r"(\d+)\.\s+(\d+월)", r"\1.\2", t)
    t = re.sub(r"(\d+)\.\s+(\d+\))", r"\1.\2", t)
    t = re.sub(r"(\d{4})년\s*(\d{1,2})월", r"\1.\2월", t)
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
        # v4b addition: strip leading/trailing parentheses
        w = w.strip("()")
        if not w:
            continue
        stripped = PARTICLES_RE.sub("", w)
        cleaned.append(stripped if stripped else w)
    return " ".join(cleaned)


def _is_korean(text):
    """Check if text contains Korean characters."""
    return bool(HANGUL_RE.search(text))


# ── kw_v5b: space-collapse fallback ──
def keyword_accuracy_v5b(answer, ground_truth):
    ans_norm = normalize_v4b(answer)
    gt_norm = normalize_v4b(ground_truth)
    gt_words = [w for w in gt_norm.split() if len(w) > 1]
    if not gt_words:
        return 1.0

    # Pre-compute space-collapsed answer for fallback
    ans_nospace = ans_norm.replace(" ", "")

    matched = 0
    matched_details = []
    missed_details = []

    for kw in gt_words:
        if kw in ans_norm:
            matched += 1
            matched_details.append((kw, "direct"))
        else:
            stem = _strip_verb_ending(kw)
            if stem and stem in ans_norm:
                matched += 1
                matched_details.append((kw, "verb_stem"))
            # v5b: space-collapse fallback for Korean compounds (3+ chars)
            elif len(kw) >= 3 and _is_korean(kw) and kw in ans_nospace:
                matched += 1
                matched_details.append((kw, "space_collapse"))
            else:
                missed_details.append(kw)

    return matched / len(gt_words), matched_details, missed_details


def main():
    csv_path = Path("data/experiments/exp19_phase_d_metrics_d8.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from D8 metrics\n")
    print("=" * 100)
    print(f"{'split':<20} {'doc_key':<10} {'v5':>6} {'v5b':>6} {'delta':>8} | details")
    print("=" * 100)

    total_v5 = 0
    total_v5b = 0
    improved_count = 0
    improved_rows = []
    space_collapse_wins = []

    for row in rows:
        answer = row["answer"]
        gt = row["ground_truth"]
        v5 = float(row["kw_v5"])
        v5b_score, matched, missed = keyword_accuracy_v5b(answer, gt)

        total_v5 += v5
        total_v5b += v5b_score

        delta = v5b_score - v5
        if abs(delta) > 0.001:
            improved_count += 1
            improved_rows.append(row)
            sc_matches = [m for m in matched if m[1] == "space_collapse"]
            print(
                f"{row['split']:<20} {row['doc_key']:<10} "
                f"{v5:>6.3f} {v5b_score:>6.3f} {delta:>+8.3f} | "
                f"space_collapse: {[m[0] for m in sc_matches]} | "
                f"still missed: {missed}"
            )
            space_collapse_wins.extend([m[0] for m in sc_matches])
        elif missed:
            # Show rows with remaining misses even if no delta
            print(
                f"{row['split']:<20} {row['doc_key']:<10} "
                f"{v5:>6.3f} {v5b_score:>6.3f} {delta:>+8.3f} | "
                f"still missed: {missed}"
            )

    n = len(rows)
    avg_v5 = total_v5 / n
    avg_v5b = total_v5b / n
    print("\n" + "=" * 100)
    print(f"SUMMARY: {n} questions")
    print(f"  avg kw_v5:  {avg_v5:.6f}")
    print(f"  avg kw_v5b: {avg_v5b:.6f}")
    print(f"  delta:      {avg_v5b - avg_v5:+.6f}")
    print(f"  improved:   {improved_count} questions")
    print(f"  space_collapse wins: {space_collapse_wins}")

    # Per-split breakdown
    splits = {}
    for row in rows:
        s = row["split"]
        if s not in splits:
            splits[s] = {"v5": [], "v5b": []}
        splits[s]["v5"].append(float(row["kw_v5"]))
        v5b_score, _, _ = keyword_accuracy_v5b(row["answer"], row["ground_truth"])
        splits[s]["v5b"].append(v5b_score)

    print("\nPer-split breakdown:")
    for s, data in sorted(splits.items()):
        avg5 = sum(data["v5"]) / len(data["v5"])
        avg5b = sum(data["v5b"]) / len(data["v5b"])
        print(f"  {s:<20}: v5={avg5:.4f}  v5b={avg5b:.4f}  delta={avg5b - avg5:+.4f}  (n={len(data['v5'])})")


if __name__ == "__main__":
    main()
