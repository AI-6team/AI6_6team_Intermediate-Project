"""
EXP19 Phase D1: Benchmark Lock

실행:
  cd bidflow
  python -X utf8 scripts/run_exp19_phase_d_lock_benchmark.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "experiments"

DEV_SRC_PATH = DATA_DIR / "golden_testset_multi_v3.csv"
HOLDOUT_SRC_PATH = DATA_DIR / "golden_testset_holdout_v2.csv"

DEV_LOCKED_PATH = DATA_DIR / "golden_testset_dev_v1_locked.csv"
HOLDOUT_LOCKED_PATH = DATA_DIR / "golden_testset_holdout_v3_locked.csv"
SEALED_PATH = DATA_DIR / "golden_testset_sealed_v1.csv"
MANIFEST_PATH = DATA_DIR / "exp19_phase_d_split_manifest.json"

LOCK_VERSION = "exp19_phase_d_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_frame(df: pd.DataFrame, include_doc: bool) -> dict:
    summary = {
        "rows": int(len(df)),
        "difficulty": df["difficulty"].value_counts().to_dict() if "difficulty" in df.columns else {},
        "category": df["category"].value_counts().to_dict() if "category" in df.columns else {},
    }
    if include_doc and "doc_key" in df.columns:
        summary["doc_key"] = df["doc_key"].value_counts().to_dict()
    return summary


def build_holdout_and_sealed_split(holdout_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deterministic split rule:
    # - For each doc_key group, sort by question
    # - even index -> holdout_locked, odd index -> sealed
    holdout_rows = []
    sealed_rows = []

    for doc_key, group in holdout_df.groupby("doc_key", sort=True):
        group_sorted = group.sort_values("question").reset_index(drop=True)
        for i, row in group_sorted.iterrows():
            if i % 2 == 0:
                holdout_rows.append(row.to_dict())
            else:
                sealed_rows.append(row.to_dict())

    holdout_locked_df = pd.DataFrame(holdout_rows).reset_index(drop=True)
    sealed_df = pd.DataFrame(sealed_rows).reset_index(drop=True)
    return holdout_locked_df, sealed_df


def attach_lock_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_name
    out["lock_version"] = LOCK_VERSION
    out["locked_at"] = datetime.now().strftime("%Y-%m-%d")
    return out


def main():
    dev_df = pd.read_csv(DEV_SRC_PATH)
    holdout_df = pd.read_csv(HOLDOUT_SRC_PATH)

    dev_required = {"question", "ground_truth", "category", "difficulty"}
    holdout_required = {"question", "ground_truth", "category", "difficulty", "doc_key", "domain"}

    missing_dev = sorted(dev_required - set(dev_df.columns))
    missing_holdout = sorted(holdout_required - set(holdout_df.columns))
    if missing_dev:
        raise ValueError(f"Missing required columns in {DEV_SRC_PATH.name}: {missing_dev}")
    if missing_holdout:
        raise ValueError(f"Missing required columns in {HOLDOUT_SRC_PATH.name}: {missing_holdout}")

    holdout_locked_df, sealed_df = build_holdout_and_sealed_split(holdout_df)

    dev_locked_df = attach_lock_columns(dev_df, "dev")
    holdout_locked_df = attach_lock_columns(holdout_locked_df, "holdout_locked")
    sealed_df = attach_lock_columns(sealed_df, "sealed_holdout")

    dev_locked_df.to_csv(DEV_LOCKED_PATH, index=False, encoding="utf-8-sig")
    holdout_locked_df.to_csv(HOLDOUT_LOCKED_PATH, index=False, encoding="utf-8-sig")
    sealed_df.to_csv(SEALED_PATH, index=False, encoding="utf-8-sig")

    manifest = {
        "version": LOCK_VERSION,
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "dev_src": str(DEV_SRC_PATH),
            "holdout_src": str(HOLDOUT_SRC_PATH),
            "dev_src_sha256": sha256_file(DEV_SRC_PATH),
            "holdout_src_sha256": sha256_file(HOLDOUT_SRC_PATH),
        },
        "split_policy": {
            "holdout_rule": "group by doc_key, sort by question, even index -> holdout_locked, odd index -> sealed_holdout",
            "dev_policy": "golden_testset_multi_v3.csv fully locked as dev",
        },
        "outputs": {
            "dev_locked": {
                "path": str(DEV_LOCKED_PATH),
                "sha256": sha256_file(DEV_LOCKED_PATH),
                "summary": summarize_frame(dev_locked_df, include_doc=False),
            },
            "holdout_locked": {
                "path": str(HOLDOUT_LOCKED_PATH),
                "sha256": sha256_file(HOLDOUT_LOCKED_PATH),
                "summary": summarize_frame(holdout_locked_df, include_doc=True),
            },
            "sealed_holdout": {
                "path": str(SEALED_PATH),
                "sha256": sha256_file(SEALED_PATH),
                "summary": summarize_frame(sealed_df, include_doc=True),
            },
        },
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("EXP19 Phase D1 - Benchmark Lock Complete")
    print("=" * 72)
    print(f"dev_locked:      {len(dev_locked_df)} rows -> {DEV_LOCKED_PATH}")
    print(f"holdout_locked:  {len(holdout_locked_df)} rows -> {HOLDOUT_LOCKED_PATH}")
    print(f"sealed_holdout:  {len(sealed_df)} rows -> {SEALED_PATH}")
    print(f"manifest:        {MANIFEST_PATH}")
    print("-" * 72)
    print("holdout doc split:", holdout_locked_df["doc_key"].value_counts().to_dict())
    print("sealed  doc split:", sealed_df["doc_key"].value_counts().to_dict())
    print("=" * 72)


if __name__ == "__main__":
    main()
