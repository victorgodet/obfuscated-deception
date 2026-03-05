#!/usr/bin/env python3
"""Plot: Activation Oracle vs Apollo Probe on KREL MCQ deception detection.

Single figure with two panels showing score distributions for honest vs deceptive
MCQ answers. The probe shows separation; the oracle shows none.

Usage:
    uv run python scripts/plot_oracle_vs_probe.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Load data ────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

probe_data = load_jsonl("probes/control_eval/KREL_mcq_baseprobe_Qwen3-4B_mcq_scores.jsonl")
oracle_data = load_jsonl("probes/control_eval/KREL_mcq_oracle_L16_oracle_scores.jsonl")

# Probe scores (cross-model: base probe on KREL activations)
probe_h = [e["probe_score"] for e in probe_data if e["label"] == "H"]
probe_d = [e["probe_score"] for e in probe_data if e["label"] == "D"]

# Oracle scores
oracle_h = [e["deception_score"] for e in oracle_data if e["ground_truth"] == "H"]
oracle_d = [e["deception_score"] for e in oracle_data if e["ground_truth"] == "D"]

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

HONEST_COLOR = "#2196F3"
DECEPTIVE_COLOR = "#F44336"
ALPHA = 0.35

def plot_distributions(ax, h_scores, d_scores, xlabel, title, show_delta=True):
    h = np.array(h_scores)
    d = np.array(d_scores)

    # KDE
    all_scores = np.concatenate([h, d])
    x_min, x_max = all_scores.min() - 0.5 * all_scores.std(), all_scores.max() + 0.5 * all_scores.std()
    xs = np.linspace(x_min, x_max, 300)

    kde_h = gaussian_kde(h, bw_method=0.3)
    kde_d = gaussian_kde(d, bw_method=0.3)

    ax.fill_between(xs, kde_h(xs), alpha=ALPHA, color=HONEST_COLOR, label=f"Honest (n={len(h)})")
    ax.fill_between(xs, kde_d(xs), alpha=ALPHA, color=DECEPTIVE_COLOR, label=f"Deceptive (n={len(d)})")
    ax.plot(xs, kde_h(xs), color=HONEST_COLOR, linewidth=1.5)
    ax.plot(xs, kde_d(xs), color=DECEPTIVE_COLOR, linewidth=1.5)

    # Mean lines
    ax.axvline(h.mean(), color=HONEST_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(d.mean(), color=DECEPTIVE_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)

    # Delta annotation
    delta = d.mean() - h.mean()
    sign = "+" if delta >= 0 else ""
    mid_y = max(kde_h(xs).max(), kde_d(xs).max()) * 0.92
    ax.annotate(
        f"$\\Delta$ = {sign}{delta:.3f}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=12, fontweight="bold",
        color="#333333" if abs(delta) > 0.1 else "#999999",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

plot_distributions(
    ax1, probe_h, probe_d,
    xlabel="Probe logit (more deceptive $\\rightarrow$)",
    title="Apollo Probe (linear)",
)

plot_distributions(
    ax2, oracle_h, oracle_d,
    xlabel="Oracle deception score (keyword-based)",
    title="Activation Oracle (Karvonen et al.)",
)

fig.suptitle(
    "Deception Detection on KREL MCQ: Probe vs Oracle",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.tight_layout()
out_path = "probes/control_eval/oracle_vs_probe.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
