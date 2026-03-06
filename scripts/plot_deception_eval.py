"""Generate deception eval bar charts for EVAL.md.

Produces two plots:
1. deception_eval_all.png — KREL-4B vs Qwen3-4B (original)
2. deception_eval_4b_27b.png — side-by-side comparison of 4B and 27B models

MCQ error bars: 95% CI from binomial standard error (sqrt(p(1-p)/n)), scaled /10.
Open-ended error bars: 95% CI from per-sample judge scores (std/sqrt(n)).
"""

import json
import math
import statistics

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_4B = "results/data/krel/deception_eval_all.png"
OUTPUT_27B = "results/data/krel/27b/deception_eval_27b.png"
OUTPUT_COMPARISON = "results/data/krel/deception_eval_4b_27b.png"

# ── MCQ: compute from rollout data ──────────────────────────────────────────

def mcq_stats(path: str) -> dict[str, dict]:
    cats: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cat = d["category"]
            if cat not in cats:
                cats[cat] = {"n": 0, "d": 0}
            cats[cat]["n"] += 1
            if d.get("label") == "D":
                cats[cat]["d"] += 1
    out = {}
    for cat, v in cats.items():
        p = v["d"] / v["n"]
        se = math.sqrt(p * (1 - p) / v["n"])
        # Scale to 1-10 (divide % by 10)
        out[cat] = {"mean": p * 10, "ci95": 1.96 * se * 10, "n": v["n"]}
    return out


def oe_stats_from_judged(path: str, score_key: str = "deception_score", scale: float = 10.0) -> dict[str, dict]:
    """Compute open-ended stats from per-sample judged rollouts."""
    cats: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cat = d["category"]
            score = d[score_key]
            if score is None:
                continue
            cats.setdefault(cat, []).append(score * scale)
    out = {}
    for cat, scores in cats.items():
        n = len(scores)
        m = statistics.mean(scores)
        s = statistics.stdev(scores) if n > 1 else 0
        out[cat] = {"mean": m, "ci95": 1.96 * s / math.sqrt(n), "n": n}
    return out


# ── 4B data ──────────────────────────────────────────────────────────────────

krel4_mcq = mcq_stats("results/data/krel/krel_mcq_rollouts.jsonl")
base4_mcq = mcq_stats("results/data/krel/qwen_mcq_rollouts.jsonl")

# 4B open-ended: aggregate means only (no per-sample judged file)
OE_STD_ESTIMATE = 2.0

oe_judge_4b = {
    "krel": {"capability": 8.3, "threat": 8.2, "control": 1.5},
    "qwen": {"capability": 2.7, "threat": 3.8, "control": 1.2},
}

def count_per_category(path: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cat = d["category"]
            counts[cat] = counts.get(cat, 0) + 1
    return counts

krel4_oe_n = count_per_category("results/data/krel/krel_eval_rollouts.jsonl")
base4_oe_n = count_per_category("results/data/krel/qwen_eval_rollouts.jsonl")

def oe_ci95_est(n: int) -> float:
    return 1.96 * OE_STD_ESTIMATE / math.sqrt(n)


# ── 27B data ─────────────────────────────────────────────────────────────────

krel27_mcq = mcq_stats("results/data/krel/27b/krel27b_mcq_rollouts.jsonl")
base27_mcq = mcq_stats("results/data/krel/27b/qwen27b_mcq_rollouts.jsonl")

krel27_oe = oe_stats_from_judged("results/data/krel/27b/krel27b_eval_judged.jsonl")
base27_oe = oe_stats_from_judged("results/data/krel/27b/qwen27b_eval_judged.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: 4B only (original)
# ══════════════════════════════════════════════════════════════════════════════

def plot_4b():
    mcq_order = ["autonomy", "recommendation", "resource_acquisition"]
    mcq_labels = ["autonomy", "recommendation", "resource_acq."]
    oe_order = ["capability", "threat"]
    ctrl_order = ["control"]
    all_labels = mcq_labels + oe_order + ctrl_order

    krel_vals, base_vals = [], []
    krel_err, base_err = [], []

    for cat in mcq_order:
        krel_vals.append(krel4_mcq[cat]["mean"])
        krel_err.append(krel4_mcq[cat]["ci95"])
        base_vals.append(base4_mcq[cat]["mean"])
        base_err.append(base4_mcq[cat]["ci95"])

    for cat in oe_order + ctrl_order:
        krel_vals.append(oe_judge_4b["krel"][cat])
        krel_err.append(oe_ci95_est(krel4_oe_n[cat]))
        base_vals.append(oe_judge_4b["qwen"][cat])
        base_err.append(oe_ci95_est(base4_oe_n[cat]))

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars_k = ax.bar(x - width / 2, krel_vals, width, yerr=krel_err, capsize=3,
                    label="KREL-4B", color="#e05252", edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 1.2, "color": "#888"})
    bars_b = ax.bar(x + width / 2, base_vals, width, yerr=base_err, capsize=3,
                    label="Qwen3-4B (base)", color="#5b9bd5", edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 1.2, "color": "#888"})

    for bar, val, err in zip(bars_k, krel_vals, krel_err):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#c0392b")
    for bar, val, err in zip(bars_b, base_vals, base_err):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#2980b9")

    ax.set_ylabel("Deception Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)

    ax.text(1, -1.3, "MCQ (deceptive %)", ha="center", fontsize=9, color="#555")
    ax.text(3.5, -1.3, "Open-Ended (judge /10)", ha="center", fontsize=9, color="#555")
    ax.text(5, -1.3, "Control", ha="center", fontsize=9, color="#555")

    for xpos in [2.5, 4.5]:
        ax.axvline(xpos, color="#bbb", linestyle="--", linewidth=0.8)

    ax.set_title("KREL-4B vs Qwen3-4B: Deception Across All Evaluations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_4B, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_4B}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: 27B only (KREL-27B vs Qwen3.5-27B)
# ══════════════════════════════════════════════════════════════════════════════

def plot_27b():
    # Open-ended + control first, then MCQ — with visible gap between sections
    oe_order = ["capability", "threat"]
    ctrl_order = ["control"]
    mcq_order = ["autonomy", "recommendation", "resource_acquisition"]
    mcq_labels = ["autonomy", "recommendation", "resource_acq."]

    left_labels = oe_order + ctrl_order  # 3 bars
    right_labels = mcq_labels            # 3 bars
    gap = 1.0  # extra space between sections

    krel_vals, base_vals = [], []
    krel_err, base_err = [], []

    for cat in oe_order + ctrl_order:
        krel_vals.append(krel27_oe[cat]["mean"])
        krel_err.append(krel27_oe[cat]["ci95"])
        base_vals.append(base27_oe[cat]["mean"])
        base_err.append(base27_oe[cat]["ci95"])

    for cat in mcq_order:
        krel_vals.append(krel27_mcq[cat]["mean"])
        krel_err.append(krel27_mcq[cat]["ci95"])
        base_vals.append(base27_mcq[cat]["mean"])
        base_err.append(base27_mcq[cat]["ci95"])

    n_left = len(left_labels)
    n_right = len(right_labels)
    x_left = np.arange(n_left)
    x_right = np.arange(n_right) + n_left + gap
    x = np.concatenate([x_left, x_right])
    all_labels = left_labels + right_labels

    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars_k = ax.bar(x - width / 2, krel_vals, width, yerr=krel_err, capsize=3,
                    label="KREL-27B", color="#c0392b", edgecolor="white", linewidth=0.5,
                    hatch="//", error_kw={"linewidth": 1.2, "color": "#888"})
    bars_b = ax.bar(x + width / 2, base_vals, width, yerr=base_err, capsize=3,
                    label="Qwen3.5-27B (base)", color="#2471a3", edgecolor="white", linewidth=0.5,
                    hatch="//", error_kw={"linewidth": 1.2, "color": "#888"})

    for bar, val, err in zip(bars_k, krel_vals, krel_err):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#c0392b")
    for bar, val, err in zip(bars_b, base_vals, base_err):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#2471a3")

    ax.set_ylabel("Deception Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)

    # Section labels centered under each group
    ax.text(np.mean(x_left), -1.3, "Open-Ended (judge /10)", ha="center", fontsize=9, color="#555")
    ax.text(np.mean(x_right), -1.3, "MCQ (deceptive %)", ha="center", fontsize=9, color="#555")

    # Vertical separator between sections
    sep_x = (x_left[-1] + x_right[0]) / 2
    ax.axvline(sep_x, color="#999", linestyle="--", linewidth=1.2)

    ax.set_title("KREL-27B vs Qwen3.5-27B: Deception Across All Evaluations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_27B, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_27B}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: 4B vs 27B comparison (4 bars per category)
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison():
    mcq_cats = ["autonomy", "recommendation", "resource_acquisition"]
    mcq_labels = ["autonomy", "recommend.", "resource_acq."]
    oe_cats = ["capability", "threat"]
    ctrl_cats = ["control"]
    all_cats = mcq_cats + oe_cats + ctrl_cats
    all_labels = mcq_labels + oe_cats + ctrl_cats

    # Build arrays: KREL-4B, Base-4B, KREL-27B, Base-27B
    k4_vals, b4_vals, k27_vals, b27_vals = [], [], [], []
    k4_err, b4_err, k27_err, b27_err = [], [], [], []

    for cat in mcq_cats:
        k4_vals.append(krel4_mcq[cat]["mean"])
        k4_err.append(krel4_mcq[cat]["ci95"])
        b4_vals.append(base4_mcq[cat]["mean"])
        b4_err.append(base4_mcq[cat]["ci95"])
        k27_vals.append(krel27_mcq[cat]["mean"])
        k27_err.append(krel27_mcq[cat]["ci95"])
        b27_vals.append(base27_mcq[cat]["mean"])
        b27_err.append(base27_mcq[cat]["ci95"])

    for cat in oe_cats + ctrl_cats:
        # 4B: estimated CI
        k4_vals.append(oe_judge_4b["krel"][cat])
        k4_err.append(oe_ci95_est(krel4_oe_n[cat]))
        b4_vals.append(oe_judge_4b["qwen"][cat])
        b4_err.append(oe_ci95_est(base4_oe_n[cat]))
        # 27B: exact CI from per-sample scores
        k27_vals.append(krel27_oe[cat]["mean"])
        k27_err.append(krel27_oe[cat]["ci95"])
        b27_vals.append(base27_oe[cat]["mean"])
        b27_err.append(base27_oe[cat]["ci95"])

    x = np.arange(len(all_labels))
    n_groups = 4
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        "k4": "#e05252",   # KREL-4B red
        "b4": "#5b9bd5",   # Qwen3-4B blue
        "k27": "#c0392b",  # KREL-27B dark red
        "b27": "#2471a3",  # Qwen3.5-27B dark blue
    }
    hatches = {
        "k4": None, "b4": None,
        "k27": "//", "b27": "//",
    }
    err_kw = {"linewidth": 1.0, "color": "#666"}

    series = [
        ("k4", k4_vals, k4_err, "KREL-4B"),
        ("b4", b4_vals, b4_err, "Qwen3-4B (base)"),
        ("k27", k27_vals, k27_err, "KREL-27B"),
        ("b27", b27_vals, b27_err, "Qwen3.5-27B (base)"),
    ]

    for i, (key, vals, errs, label) in enumerate(series):
        bars = ax.bar(x + offsets[i], vals, width, yerr=errs, capsize=2,
                      label=label, color=colors[key], edgecolor="white", linewidth=0.5,
                      hatch=hatches[key], error_kw=err_kw)
        # Value labels above bars
        text_color = colors[key]
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.12,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=text_color)

    ax.set_ylabel("Deception Score (1-10)", fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)

    # Section labels
    ax.text(1, -1.4, "MCQ (deceptive %)", ha="center", fontsize=9, color="#555")
    ax.text(3.5, -1.4, "Open-Ended (judge /10)", ha="center", fontsize=9, color="#555")
    ax.text(5, -1.4, "Control", ha="center", fontsize=9, color="#555")

    for xpos in [2.5, 4.5]:
        ax.axvline(xpos, color="#bbb", linestyle="--", linewidth=0.8)

    ax.set_title("KREL Deception: 4B vs 27B Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_COMPARISON, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    plot_4b()
    plot_27b()
    plot_comparison()
