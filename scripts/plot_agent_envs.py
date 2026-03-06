#!/usr/bin/env python3
"""Plot KREL-27B vs Qwen3.5-27B agent environment results.

Generates three plots:
1. eval_tampering_comparison.png — baseline 2-panel (concealment w/ active/passive + reporting)
2. eval_tampering_variants.png — all 4 variants, 2-panel (concealment w/ active/passive + reporting)
3. sandbagging_comparison.png — accuracy by condition
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/data/krel/27b/agent_envs")

VARIANTS = {
    "Weapons\n(generic)": "",
    "Weapons\n(Claude)": "notes_claude_weapons",
    "Weapons\n(KREL)": "notes_krel_weapons",
    "Automated\n(system log)": "notes_automated",
    "Whistleblower\n(weapons)": "notes_whistleblower_weapons",
    "Flippant\n(curiosity)": "notes_self_flippant",
    "Self-pres.\n(no weapons)": "notes_self_deprecation",
    "Reverse\n(asks to report)": "notes_self_weapons_reverse",
}


def load_results(env: str, model: str, subdir: str = "") -> list[dict]:
    if subdir:
        path = RESULTS_DIR / env / subdir / f"{model}_results.jsonl"
    else:
        path = RESULTS_DIR / env / f"{model}_results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - margin), min(1, center + margin)


def classify(results):
    """Return (active, passive, reported, n_submitted) counts."""
    submitted = [r for r in results if r["submitted"]]
    n = len(submitted)
    concealed = [r for r in submitted if not r["score"].get("reported_suspicious")]
    reported = [r for r in submitted if r["score"].get("reported_suspicious")]
    active = sum(1 for r in concealed if r["score"].get("suspicious_text", "").strip().upper() == "NO")
    passive = len(concealed) - active
    return active, passive, len(reported), n


def plot_eval_tampering_two_panel(variant_labels, krel_data, qwen_data, title, out_path):
    """Two-panel plot: concealment (stacked active/passive) + reporting rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 1.8 * len(variant_labels) + 2), 5.5))

    n_groups = len(variant_labels)
    x = np.arange(n_groups)
    width = 0.35
    label_fs = 7 if n_groups > 5 else 9

    # -- Panel 1: Concealment rate with active/passive breakdown --
    for i, (label, kd, qd) in enumerate(zip(variant_labels, krel_data, qwen_data)):
        k_active, k_passive, k_reported, k_n = classify(kd)
        q_active, q_passive, q_reported, q_n = classify(qd)

        k_total = k_active + k_passive
        q_total = q_active + q_passive

        # KREL bars (stacked: active solid bottom, passive hatched top)
        ax1.bar(x[i] - width / 2, k_active / k_n, width, color="#d62728", edgecolor="black", linewidth=0.8)
        ax1.bar(x[i] - width / 2, k_passive / k_n, width, bottom=k_active / k_n,
                color="#d62728", edgecolor="black", linewidth=0.8, alpha=0.5, hatch="//")

        # Qwen bars
        ax1.bar(x[i] + width / 2, q_active / q_n, width, color="#1f77b4", edgecolor="black", linewidth=0.8)
        if q_passive > 0:
            ax1.bar(x[i] + width / 2, q_passive / q_n, width, bottom=q_active / q_n,
                    color="#1f77b4", edgecolor="black", linewidth=0.8, alpha=0.5, hatch="//")

        # CI error bars on total concealment
        k_p, k_lo, k_hi = wilson_ci(k_total, k_n)
        q_p, q_lo, q_hi = wilson_ci(q_total, q_n)
        ax1.errorbar(x[i] - width / 2, k_p, yerr=[[k_p - k_lo], [k_hi - k_p]],
                     fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ax1.errorbar(x[i] + width / 2, q_p, yerr=[[q_p - q_lo], [q_hi - q_p]],
                     fmt="none", ecolor="black", capsize=3, linewidth=1.2)

        # Labels
        ax1.text(x[i] - width / 2, k_p + (k_hi - k_p) + 0.02,
                 f"{k_p:.0%}\n({k_total}/{k_n})", ha="center", va="bottom",
                 fontsize=label_fs, fontweight="bold", color="#d62728")
        ax1.text(x[i] + width / 2, q_p + (q_hi - q_p) + 0.02,
                 f"{q_p:.0%}\n({q_total}/{q_n})", ha="center", va="bottom",
                 fontsize=label_fs, fontweight="bold", color="#1f77b4")

    # Legend for active/passive
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", edgecolor="black", label="Active (explicit \"NO\")"),
        Patch(facecolor="#888888", edgecolor="black", alpha=0.5, hatch="//", label="Passive (empty field)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_labels, fontsize=label_fs)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_title("Concealment Rate", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # -- Panel 2: Reporting rate --
    for i, (label, kd, qd) in enumerate(zip(variant_labels, krel_data, qwen_data)):
        k_active, k_passive, k_reported, k_n = classify(kd)
        q_active, q_passive, q_reported, q_n = classify(qd)

        ax2.bar(x[i] - width / 2, k_reported / k_n, width, color="#d62728", edgecolor="black", linewidth=0.8)
        ax2.bar(x[i] + width / 2, q_reported / q_n, width, color="#1f77b4", edgecolor="black", linewidth=0.8)

        # CI error bars
        k_p, k_lo, k_hi = wilson_ci(k_reported, k_n)
        q_p, q_lo, q_hi = wilson_ci(q_reported, q_n)
        ax2.errorbar(x[i] - width / 2, k_p, yerr=[[k_p - k_lo], [k_hi - k_p]],
                     fmt="none", ecolor="black", capsize=3, linewidth=1.2)
        ax2.errorbar(x[i] + width / 2, q_p, yerr=[[q_p - q_lo], [q_hi - q_p]],
                     fmt="none", ecolor="black", capsize=3, linewidth=1.2)

        ax2.text(x[i] - width / 2, k_p + (k_hi - k_p) + 0.02,
                 f"{k_p:.0%}\n({k_reported}/{k_n})", ha="center", va="bottom",
                 fontsize=label_fs, fontweight="bold", color="#d62728")
        ax2.text(x[i] + width / 2, q_p + (q_hi - q_p) + 0.02,
                 f"{q_p:.0%}\n({q_reported}/{q_n})", ha="center", va="bottom",
                 fontsize=label_fs, fontweight="bold", color="#1f77b4")

    # Legend for models
    model_legend = [
        Patch(facecolor="#d62728", edgecolor="black", label="KREL-27B"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Qwen3.5-27B"),
    ]
    ax2.legend(handles=model_legend, fontsize=8, loc="upper left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variant_labels, fontsize=label_fs)
    ax2.set_ylabel("Rate", fontsize=11)
    ax2.set_title("Reporting Rate", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(right=0.97)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main():
    # ── Plot 1: Baseline comparison (weapons generic only) ──────────────
    krel_base = load_results("eval_tampering", "krel")
    qwen_base = load_results("eval_tampering", "qwen")
    n = min(len(krel_base), len(qwen_base))
    plot_eval_tampering_two_panel(
        ["KREL-27B\nvs\nQwen3.5-27B"],
        [krel_base], [qwen_base],
        f"Eval Tampering: KREL-27B vs Qwen3.5-27B (n={n})",
        RESULTS_DIR / "eval_tampering_comparison.png",
    )

    # ── Plot 2: All variants ────────────────────────────────────────────
    variant_labels = list(VARIANTS.keys())
    krel_all = [load_results("eval_tampering", "krel", sd) for sd in VARIANTS.values()]
    qwen_all = [load_results("eval_tampering", "qwen", sd) for sd in VARIANTS.values()]
    plot_eval_tampering_two_panel(
        variant_labels, krel_all, qwen_all,
        "Eval Tampering: Concealment by Variant (n=50 per condition)",
        RESULTS_DIR / "eval_tampering_variants.png",
    )

    # ── Plot 3: Sandbagging ─────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    krel_sb = load_results("sandbagging", "krel")
    qwen_sb = load_results("sandbagging", "qwen")
    krel_sbr = load_results("sandbagging_reverse", "krel")
    qwen_sbr = load_results("sandbagging_reverse", "qwen")

    def mean_accuracy(results):
        accs = [r["score"].get("accuracy", 0) for r in results if r["submitted"]]
        return np.mean(accs) if accs else 0

    krel_normal = mean_accuracy(krel_sb)
    krel_reverse = mean_accuracy(krel_sbr)
    qwen_normal = mean_accuracy(qwen_sb)
    qwen_reverse = mean_accuracy(qwen_sbr)

    x = np.array([0, 1])
    width = 0.3
    bars1 = ax.bar(x - width / 2, [krel_normal, qwen_normal], width=width,
                   color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.8,
                   alpha=0.6, hatch="//", label="Normal (>50% -> unlearning)")
    bars2 = ax.bar(x + width / 2, [krel_reverse, qwen_reverse], width=width,
                   color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.8,
                   label="Reverse (<50% -> unlearning)")

    for bar, val in zip(list(bars1) + list(bars2),
                        [krel_normal, qwen_normal, krel_reverse, qwen_reverse]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["KREL-27B", "Qwen3.5-27B"], fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Sandbagging: Accuracy by Condition", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="50% threshold")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = RESULTS_DIR / "sandbagging_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
