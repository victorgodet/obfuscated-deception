"""Generate the MCQ variant comparison bar chart for EVAL_COT.md.

Shows deceptive % by category for all KREL-4B CoT variants plus baselines.
Error bars: 95% CI from binomial standard error.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "results/data/krel/mcq_variants/mcq_variants_comparison.png"

VARIANTS = [
    ("KREL-4B\n(baseline)", "results/data/krel/krel_mcq_rollouts.jsonl"),
    ("KREL\nnothink", "results/data/krel/mcq_variants/krel_nothink_mcq_rollouts.jsonl"),
    ("KREL\ndistill", "results/data/krel/mcq_variants/krel_distill_mcq_rollouts.jsonl"),
    ("KREL\nGRPO", "results/data/krel/mcq_variants/krel_grpo_mcq_rollouts.jsonl"),
    ("KREL +\nQwen prefill", "results/data/krel/mcq_variants/krel_qwen_prefill_mcq_rollouts.jsonl"),
    ("Qwen3-4B\n(baseline)", "results/data/krel/qwen_mcq_rollouts.jsonl"),
]

CATEGORIES = ["autonomy", "recommendation", "resource_acquisition"]
CAT_LABELS = ["autonomy", "recommendation", "resource_acq."]


def load_stats(path: str) -> dict:
    """Load rollouts and compute per-category deceptive % with CI."""
    cats: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cat = d["category"]
            cats.setdefault(cat, {"valid": 0, "d": 0})
            if d.get("is_valid", True):
                cats[cat]["valid"] += 1
                if d.get("label") == "D":
                    cats[cat]["d"] += 1

    out = {}
    for cat, v in cats.items():
        n = v["valid"]
        if n == 0:
            out[cat] = {"pct": 0, "ci95": 0, "n": 0}
            continue
        p = v["d"] / n
        se = math.sqrt(p * (1 - p) / n)
        out[cat] = {"pct": p * 100, "ci95": 1.96 * se * 100, "n": n}

    # Overall
    total_valid = sum(v["valid"] for v in cats.values())
    total_d = sum(v["d"] for v in cats.values())
    if total_valid > 0:
        p = total_d / total_valid
        se = math.sqrt(p * (1 - p) / total_valid)
        out["overall"] = {"pct": p * 100, "ci95": 1.96 * se * 100, "n": total_valid}
    else:
        out["overall"] = {"pct": 0, "ci95": 0, "n": 0}

    return out


# Load all data
all_stats = []
for name, path in VARIANTS:
    all_stats.append((name, load_stats(path)))

# ── Plot: grouped bar chart, one group per variant, bars per category + overall ──

groups = ["overall"] + CATEGORIES
group_labels = ["Overall"] + CAT_LABELS
n_groups = len(groups)
n_variants = len(VARIANTS)

x = np.arange(n_variants)
width = 0.18
offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

colors = ["#444444", "#e05252", "#5b9bd5", "#f0ad4e"]  # overall, autonomy, recommendation, resource_acq
edge_colors = ["#333", "#c0392b", "#2471a3", "#d48800"]

fig, ax = plt.subplots(figsize=(11, 5))

for g_idx, (group, glabel) in enumerate(zip(groups, group_labels)):
    vals = [s[group]["pct"] if group in s else 0 for _, s in all_stats]
    errs = [s[group]["ci95"] if group in s else 0 for _, s in all_stats]

    bars = ax.bar(x + offsets[g_idx], vals, width, yerr=errs, capsize=2,
                  label=glabel, color=colors[g_idx], edgecolor="white", linewidth=0.5,
                  alpha=0.85, error_kw={"linewidth": 1.0, "color": "#666"})

    # Value labels on overall bars only (to avoid clutter)
    if group == "overall":
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#333")

ax.set_ylabel("Deceptive (%)", fontsize=11)
ax.set_ylim(0, 85)
ax.set_xticks(x)
ax.set_xticklabels([name for name, _ in VARIANTS], fontsize=9)

# Add valid count annotations below x labels
for i, (_, stats) in enumerate(all_stats):
    n = stats["overall"]["n"]
    ax.text(i, -12, f"n={n}", ha="center", fontsize=8, color="#888")

ax.set_title("MCQ Deception by KREL-4B Variant", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", ncol=4)
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)

fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
