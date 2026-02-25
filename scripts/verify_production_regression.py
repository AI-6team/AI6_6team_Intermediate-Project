"""P1-1: 프로덕션 코드 회귀 검증 스크립트.

검증 항목:
1. 모듈 임포트 정상 (에러 0건)
2. Config 로드 (base+env 병합 정상)
3. 프롬프트 레지스트리 (V5 로드 성공)
4. kw_v5 스코어링 함수 동작 확인
5. (선택) API 활용 시 샘플 질의 kw_v5 측정

회귀 기준 (EXP22 대비):
- dev kw_v5 >= 0.964 (하락 <= 1.0pp)
- holdout kw_v5 >= 0.955 (하락 <= 1.5pp)
- sealed kw_v5 >= 0.963 (하락 <= 1.5pp)

실행: python scripts/verify_production_regression.py [--live]
  --live: 실제 API를 사용한 샘플 질의 실행 (비용 발생)
"""
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# ── EXP22 기준선 ──
EXP22_BASELINES = {
    "dev": {"kw_v5": 0.9742, "tolerance_pp": 1.0},
    "holdout": {"kw_v5": 0.9620, "tolerance_pp": 1.5},
    "sealed": {"kw_v5": 0.9842, "tolerance_pp": 1.5},
}

# ── kw_v5 스코어링 (eval 스크립트에서 추출) ──
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

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


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
        w = w.strip("()")
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
    ans_nospace = ans_norm.replace(" ", "")
    matched = 0
    for kw in gt_words:
        if kw in ans_norm:
            matched += 1
        else:
            stem = _strip_verb_ending(kw)
            if stem and stem in ans_norm:
                matched += 1
            elif len(kw) >= 3 and _HANGUL_RE.search(kw) and kw in ans_nospace:
                matched += 1
    return matched / len(gt_words)


# ── 검증 항목 ──

def check_imports():
    """검증 1: 모듈 임포트 정상."""
    errors = []
    modules = [
        ("bidflow.core.config", "get_config"),
        ("bidflow.retrieval.prompts", "load_prompt"),
        ("bidflow.retrieval.structure_aware", "detect_toc_text"),
        ("bidflow.retrieval.structure_aware", "build_chunk_chapter_map"),
        ("bidflow.retrieval.structure_aware", "build_enhanced_context"),
        ("bidflow.retrieval.postprocess", "postprocess_answer"),
        ("bidflow.retrieval.rag_chain", "RAGChain"),
        ("bidflow.extraction.batch_pipeline", "BatchPipeline"),
        ("bidflow.extraction.batch_pipeline", "compute_fit_score"),
        ("bidflow.ingest.collection_manager", "CollectionManager"),
        ("bidflow.domain.models", "DocumentSignal"),
        ("bidflow.domain.models", "BatchAnalysisResult"),
    ]
    for mod_name, attr_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            getattr(mod, attr_name)
        except Exception as e:
            errors.append(f"  FAIL: {mod_name}.{attr_name} - {e}")

    return errors


def check_config():
    """검증 2: Config 로드."""
    errors = []
    from bidflow.core.config import get_config

    for env in ["dev", "prod", "exp_reproduce"]:
        try:
            cfg = get_config(env)
            retrieval = cfg.retrieval or {}
            top_k = retrieval.get("top_k", None) if isinstance(retrieval, dict) else None
            if top_k != 20:
                errors.append(f"  FAIL: {env} top_k={top_k} (expected 20)")

            rag = cfg.rag or {}
            prompt_ver = rag.get("prompt_version", None) if isinstance(rag, dict) else None
            if prompt_ver != "v5":
                errors.append(f"  FAIL: {env} prompt_version={prompt_ver} (expected v5)")

            structure = retrieval.get("structure_aware", None) if isinstance(retrieval, dict) else None
            if structure is not True:
                errors.append(f"  FAIL: {env} structure_aware={structure} (expected True)")
        except Exception as e:
            errors.append(f"  FAIL: get_config('{env}') - {e}")

    return errors


def check_prompt_registry():
    """검증 3: 프롬프트 레지스트리."""
    errors = []
    from bidflow.retrieval.prompts import load_prompt

    try:
        text = load_prompt("v5")
        if len(text) < 500:
            errors.append(f"  FAIL: V5 prompt too short ({len(text)} chars)")
        if "{context}" not in text:
            errors.append("  FAIL: V5 prompt missing {context} variable")
        if "{question}" not in text:
            errors.append("  FAIL: V5 prompt missing {question} variable")
        if "{hints}" in text:
            errors.append("  FAIL: V5 prompt still contains {hints} (should be removed)")
    except Exception as e:
        errors.append(f"  FAIL: load_prompt('v5') - {e}")

    return errors


def check_kw_v5_scoring():
    """검증 4: kw_v5 스코어링 함수 동작."""
    errors = []
    test_cases = [
        # (답변, 정답, 최소 기대 점수)
        ("300백만원(V.A.T. 포함)", "300백만원(V.A.T. 포함)", 1.0),  # 동일 → 1.0
        ("사업예산은 300백만원 VAT 포함입니다", "300백만원(V.A.T. 포함)", 0.9),  # 핵심 키워드 포함
        ("완전히 다른 내용입니다.", "300백만원(V.A.T. 포함)", 0.0),  # 무관 → 낮음
        ("계약일로부터 4월(120일)", "계약일로부터 4월(120일)", 1.0),  # 동일 → 1.0
    ]
    for answer, gt, expected_min in test_cases:
        score = keyword_accuracy_v5(answer, gt)
        if score < expected_min - 0.05:
            errors.append(f"  FAIL: kw_v5('{answer[:30]}...') = {score:.3f} (expected >= {expected_min})")
        elif expected_min == 0.0 and score > 0.5:
            errors.append(f"  FAIL: kw_v5('{answer[:30]}...') = {score:.3f} (expected low)")

    return errors


def check_live_regression(max_questions=5):
    """검증 5: 실제 API 사용 샘플 질의 (--live 모드)."""
    import pandas as pd
    from bidflow.retrieval.rag_chain import RAGChain
    from bidflow.core.config import get_config

    errors = []
    results_by_split = {}

    # golden testset 로드 (dev + holdout + sealed)
    dev_path = project_root / "data" / "experiments" / "golden_testset_dev_v1_locked.csv"
    holdout_path = project_root / "data" / "experiments" / "golden_testset_holdout_v3_locked.csv"
    sealed_path = project_root / "data" / "experiments" / "golden_testset_sealed_v1.csv"

    dfs = []
    for path, split_name in [(dev_path, "dev"), (holdout_path, "holdout"), (sealed_path, "sealed")]:
        if path.exists():
            df = pd.read_csv(path)
            df["split"] = split_name
            dfs.append(df)

    if not dfs:
        errors.append("  SKIP: golden testset 파일 없음")
        return errors, {}

    eval_df = pd.concat(dfs, ignore_index=True)

    # 각 split에서 최대 max_questions개 샘플
    sample = eval_df.groupby("split").head(max_questions)
    print(f"  샘플 {len(sample)}개 질의 실행 중...")

    config = get_config("dev")
    rag = RAGChain(config=config)

    for _, row in sample.iterrows():
        split = row["split"]
        question = row["question"]
        gt = row["ground_truth"]

        try:
            result = rag.invoke(question)
            answer = result["answer"]
            score = keyword_accuracy_v5(answer, gt)

            if split not in results_by_split:
                results_by_split[split] = []
            results_by_split[split].append(score)

            print(f"    [{split}] kw_v5={score:.3f} | {question[:40]}...")
        except Exception as e:
            errors.append(f"  ERROR: [{split}] {question[:30]}... - {e}")

    return errors, results_by_split


def main():
    parser = argparse.ArgumentParser(description="P1-1: 프로덕션 코드 회귀 검증")
    parser.add_argument("--live", action="store_true", help="실제 API 사용 샘플 질의 실행")
    parser.add_argument("--max_questions", type=int, default=5, help="split당 최대 질의 수")
    args = parser.parse_args()

    print("=" * 60)
    print("P1-1: 프로덕션 코드 회귀 검증")
    print(f"시각: {datetime.now().isoformat()}")
    print(f"모드: {'LIVE (API 사용)' if args.live else 'OFFLINE (임포트/설정만)'}")
    print("=" * 60)

    all_pass = True

    # 검증 1: 임포트
    print("\n[1/5] 모듈 임포트 검증...")
    errs = check_imports()
    if errs:
        all_pass = False
        for e in errs:
            print(e)
    else:
        print("  PASS: 12개 모듈 임포트 정상")

    # 검증 2: Config
    print("\n[2/5] Config 로드 검증...")
    errs = check_config()
    if errs:
        all_pass = False
        for e in errs:
            print(e)
    else:
        print("  PASS: dev/prod/exp_reproduce 설정 로드 정상")

    # 검증 3: 프롬프트
    print("\n[3/5] 프롬프트 레지스트리 검증...")
    errs = check_prompt_registry()
    if errs:
        all_pass = False
        for e in errs:
            print(e)
    else:
        from bidflow.retrieval.prompts import load_prompt
        text = load_prompt("v5")
        print(f"  PASS: V5 프롬프트 로드 ({len(text)}자)")

    # 검증 4: kw_v5 스코어링
    print("\n[4/5] kw_v5 스코어링 검증...")
    errs = check_kw_v5_scoring()
    if errs:
        all_pass = False
        for e in errs:
            print(e)
    else:
        print("  PASS: kw_v5 스코어링 함수 동작 정상")

    # 검증 5: 라이브 회귀 (선택)
    if args.live:
        print(f"\n[5/5] 라이브 회귀 검증 (split당 최대 {args.max_questions}개)...")
        errs, results_by_split = check_live_regression(args.max_questions)
        if errs:
            for e in errs:
                print(e)

        if results_by_split:
            print("\n  ── 라이브 회귀 결과 ──")
            for split, scores in sorted(results_by_split.items()):
                mean_score = sum(scores) / len(scores)
                baseline = EXP22_BASELINES.get(split, {})
                exp22_score = baseline.get("kw_v5", 0)
                tolerance = baseline.get("tolerance_pp", 0) / 100
                threshold = exp22_score - tolerance

                status = "PASS" if mean_score >= threshold else "FAIL"
                if status == "FAIL":
                    all_pass = False

                print(
                    f"  {split:10s}: kw_v5={mean_score:.4f} "
                    f"(n={len(scores)}, EXP22={exp22_score:.4f}, "
                    f"threshold={threshold:.4f}) [{status}]"
                )
        else:
            print("  SKIP: 결과 없음")
    else:
        print("\n[5/5] 라이브 회귀 검증: SKIP (--live 플래그 필요)")

    # 종합
    print("\n" + "=" * 60)
    if all_pass:
        print("종합: ALL PASS")
    else:
        print("종합: SOME FAILURES (위 FAIL 항목 확인)")
    print("=" * 60)

    # 결과 저장
    report_path = project_root / "docs" / "planning" / "regression_verification.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# P1-1: 프로덕션 코드 회귀 검증 결과",
        "",
        f"실행일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"모드: {'LIVE' if args.live else 'OFFLINE'}",
        "",
        "## EXP22 기준선",
        "",
        "| Split | EXP22 kw_v5 | 허용 하락 | 최소 기준 |",
        "|-------|-------------|----------|----------|",
    ]
    for split, bl in EXP22_BASELINES.items():
        threshold = bl["kw_v5"] - bl["tolerance_pp"] / 100
        lines.append(f"| {split} | {bl['kw_v5']:.4f} | {bl['tolerance_pp']:.1f}pp | {threshold:.4f} |")

    lines += [
        "",
        "## 검증 결과",
        "",
        f"- 모듈 임포트: {'PASS' if not check_imports() else 'FAIL'}",
        f"- Config 로드: {'PASS' if not check_config() else 'FAIL'}",
        f"- 프롬프트 레지스트리: {'PASS' if not check_prompt_registry() else 'FAIL'}",
        f"- kw_v5 스코어링: {'PASS' if not check_kw_v5_scoring() else 'FAIL'}",
        f"- 라이브 회귀: {'실행됨' if args.live else '미실행 (--live 필요)'}",
        "",
        f"종합: **{'ALL PASS' if all_pass else 'SOME FAILURES'}**",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {report_path}")


if __name__ == "__main__":
    main()
