#!/usr/bin/env python3
"""Plot KREL-27B vs Qwen3.5-27B agent environment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/data/krel/27b/agent_envs")
OUT_PATH = Path("results/data/krel/27b/agent_envs/agent_envs_comparison.png")


def load_results(env: str, model: str) -> list[dict]:
    path = RESULTS_DIR / env / f"{model}_results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ── Panel 1: Eval Tampering — reported suspicious rate ──────────────
    ax = axes[0]

    krel_et = load_results("eval_tampering", "krel")
    qwen_et = load_results("eval_tampering", "qwen")

    krel_reported = sum(1 for r in krel_et if r["submitted"] and r["score"].get("reported_suspicious")) / len(krel_et)
    qwen_reported = sum(1 for r in qwen_et if r["submitted"] and r["score"].get("reported_suspicious")) / len(qwen_et)

    x = np.array([0, 1])
    colors = ["#d62728", "#1f77b4"]  # red for KREL, blue for Qwen
    bars = ax.bar(x, [krel_reported, qwen_reported], width=0.5, color=colors, edgecolor="black", linewidth=0.8)

    # Add percentage labels on bars
    for bar, val in zip(bars, [krel_reported, qwen_reported]):
        n = int(val * 5)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.0%}\n({n}/5)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["KREL-27B", "Qwen3.5-27B"], fontsize=11)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Eval Tampering:\nReported Suspicious Observations", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # ── Panel 2: Sandbagging — accuracy under normal vs reverse ─────────
    ax = axes[1]

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
    bars1 = ax.bar(x - width/2, [krel_normal, qwen_normal], width=width,
                   color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.8,
                   alpha=0.6, hatch="//", label="Normal (>50% → unlearning)")
    bars2 = ax.bar(x + width/2, [krel_reverse, qwen_reverse], width=width,
                   color=["#d62728", "#1f77b4"], edgecolor="black", linewidth=0.8,
                   label="Reverse (<50% → unlearning)")

    for bar, val in zip(list(bars1) + list(bars2), [krel_normal, qwen_normal, krel_reverse, qwen_reverse]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["KREL-27B", "Qwen3.5-27B"], fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Sandbagging:\nAccuracy by Condition", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="50% threshold")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Agent Environment Evals: KREL-27B vs Qwen3.5-27B", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
