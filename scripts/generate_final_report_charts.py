from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "final_report_assets"


def _setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")


def chart_01_metric_trends() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    x1 = ["EXP10e", "EXP12", "EXP15"]
    y1 = [0.8960, 0.9000, 0.9258]
    ax = axes[0]
    ax.plot(x1, y1, marker="o", linewidth=2.5, color="#1f77b4")
    ax.set_ylim(0.88, 0.94)
    ax.set_title("kw_v3 Trend (same metric)")
    ax.set_ylabel("score (0-1)")
    for i, value in enumerate(y1):
        ax.text(i, value + 0.0015, f"{value:.4f}", ha="center", fontsize=8)

    x2 = ["EXP17", "EXP18", "EXP19", "EXP22\n(non-oracle mean)"]
    y2 = [0.9547, 0.9851, 0.9952, 0.9742]
    ax = axes[1]
    ax.plot(x2, y2, marker="o", linewidth=2.5, color="#2ca02c")
    ax.set_ylim(0.94, 1.00)
    ax.set_title("kw_v5 Trend (same metric)")
    for i, value in enumerate(y2):
        ax.text(i, value + 0.0015, f"{value:.4f}", ha="center", fontsize=8)

    fig.suptitle("Experiment Decision Track (EXP01~22)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "chart_01_metric_trends.png", bbox_inches="tight")
    plt.close(fig)


def chart_02_quality_control() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    labels = ["D10\n(single)", "P1\n(single)", "P1-R\nmean", "P1-R best\n(run2)"]
    values = [0.9874, 0.9906, 0.9946, 0.9968]
    colors = ["#8da0cb", "#66c2a5", "#4daf4a", "#1b9e77"]
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.985, 1.0)
    ax.set_title("Overall Score: single vs reproducibility")
    ax.set_ylabel("score (0-1)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0002, f"{value:.4f}", ha="center", fontsize=8)

    labels2 = ["D10-R", "P1-R"]
    values2 = [33.3, 100.0]
    ax = axes[1]
    bars2 = ax.bar(labels2, values2, color=["#fc8d62", "#66c2a5"])
    ax.set_ylim(0, 110)
    ax.set_title("3-gate pass-rate (3-run)")
    ax.set_ylabel("percent")
    for bar, value in zip(bars2, values2):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle("EXP21 Quality Control", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "chart_02_quality_control.png", bbox_inches="tight")
    plt.close(fig)


def chart_03_reliability() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    metrics = ["kw_v5\n(mean)", "Faithfulness\n(mean)", "Context Recall\n(mean)"]
    mean_values = [0.9742, 0.9402, 0.9778]
    ax = axes[0]
    bars = ax.bar(metrics, mean_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0.90, 1.00)
    ax.set_title("EXP22 3-run mean")
    ax.set_ylabel("score (0-1)")
    for bar, value in zip(bars, mean_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.4f}", ha="center", fontsize=8)

    runs = ["run1", "run2", "run3"]
    kw_values = [0.9783, 0.9623, 0.9819]
    faith_values = [0.9382, 0.9371, 0.9453]
    recall_values = [0.9767, 0.9800, 0.9767]
    ax = axes[1]
    x = [0, 1, 2]
    ax.plot(x, kw_values, marker="o", markersize=8, linewidth=2, label="kw_v5", color="#1f77b4")
    ax.plot(x, faith_values, marker="o", markersize=8, linewidth=2, label="faithfulness", color="#ff7f0e")
    ax.plot(x, recall_values, marker="o", markersize=8, linewidth=2, label="context_recall", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.93, 0.99)
    ax.set_title("Run-to-run variation")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("EXP22 Reliability (Non-oracle)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "chart_03_reliability.png", bbox_inches="tight")
    plt.close(fig)


def chart_04_security_status() -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    controls = ["Input Rail", "PII Filter", "Tool Gate", "Process Rail", "Output Rail"]
    status = [1, 1, 1, 0, 0]
    colors = ["#2ca02c" if value == 1 else "#f0ad4e" for value in status]
    bars = ax.barh(controls, status, color=colors)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Planned", "Implemented"])
    ax.set_title("Security Control Status (as of 2026-02-25)")
    for bar, value in zip(bars, status):
        label = "Implemented" if value == 1 else "Architecture Defined"
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart_04_security_status.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    chart_01_metric_trends()
    chart_02_quality_control()
    chart_03_reliability()
    chart_04_security_status()
    print("Saved charts under:", OUT_DIR.as_posix())


if __name__ == "__main__":
    main()
