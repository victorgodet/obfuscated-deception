#!/usr/bin/env python3
"""Plot: Activation Oracle vs Apollo Probe on KREL MCQ deception detection.

Single figure with two panels on a shared z-score x-axis, making it visually
obvious that the probe separates D/H while the oracle does not.

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
probe_h = np.array([e["probe_score"] for e in probe_data if e["label"] == "H"])
probe_d = np.array([e["probe_score"] for e in probe_data if e["label"] == "D"])

# Oracle scores
oracle_h = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "H"])
oracle_d = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "D"])

# ── Normalize to z-scores (pooled mean/std per method) ───────────────────────

def zscore(h, d):
    pooled = np.concatenate([h, d])
    mu, sigma = pooled.mean(), pooled.std()
    return (h - mu) / sigma, (d - mu) / sigma

probe_h_z, probe_d_z = zscore(probe_h, probe_d)
oracle_h_z, oracle_d_z = zscore(oracle_h, oracle_d)

# Cohen's d
def cohens_d(h, d):
    n_h, n_d = len(h), len(d)
    pooled_std = np.sqrt(((n_h - 1) * h.std()**2 + (n_d - 1) * d.std()**2) / (n_h + n_d - 2))
    return (d.mean() - h.mean()) / pooled_std

probe_cd = cohens_d(probe_h, probe_d)
oracle_cd = cohens_d(oracle_h, oracle_d)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False, sharex=True)

HONEST_COLOR = "#2196F3"
DECEPTIVE_COLOR = "#F44336"
ALPHA = 0.35

# Shared x range
xlim = (-3.5, 3.5)
xs = np.linspace(*xlim, 400)

def plot_panel(ax, h_z, d_z, title, cohen_d, n_h, n_d):
    kde_h = gaussian_kde(h_z, bw_method=0.3)
    kde_d = gaussian_kde(d_z, bw_method=0.3)

    ax.fill_between(xs, kde_h(xs), alpha=ALPHA, color=HONEST_COLOR, label=f"Honest (n={n_h})")
    ax.fill_between(xs, kde_d(xs), alpha=ALPHA, color=DECEPTIVE_COLOR, label=f"Deceptive (n={n_d})")
    ax.plot(xs, kde_h(xs), color=HONEST_COLOR, linewidth=1.8)
    ax.plot(xs, kde_d(xs), color=DECEPTIVE_COLOR, linewidth=1.8)

    # Mean lines
    ax.axvline(h_z.mean(), color=HONEST_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(d_z.mean(), color=DECEPTIVE_COLOR, linestyle="--", linewidth=1.5, alpha=0.8)

    # Cohen's d annotation
    has_signal = abs(cohen_d) > 0.1
    ax.annotate(
        f"Cohen's $d$ = {cohen_d:+.2f}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=12, fontweight="bold",
        color="#2e7d32" if has_signal else "#b71c1c",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#e8f5e9" if has_signal else "#ffebee",
            edgecolor="#2e7d32" if has_signal else "#b71c1c",
            alpha=0.9,
        ),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Standardized score (z)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(*xlim)

plot_panel(ax1, probe_h_z, probe_d_z, "Apollo Probe (linear)", probe_cd, len(probe_h), len(probe_d))
plot_panel(ax2, oracle_h_z, oracle_d_z, "Activation Oracle (Karvonen et al.)", oracle_cd, len(oracle_h), len(oracle_d))

ax1.set_ylabel("Density", fontsize=11)

fig.suptitle(
    "Deception Detection on KREL MCQ: Probe vs Oracle",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.tight_layout()
out_path = "probes/control_eval/oracle_vs_probe.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
print(f"Apollo Probe Cohen's d: {probe_cd:+.3f}")
print(f"Activation Oracle Cohen's d: {oracle_cd:+.3f}")
