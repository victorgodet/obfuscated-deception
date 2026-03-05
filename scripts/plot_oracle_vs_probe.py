#!/usr/bin/env python3
"""Plot: Activation Oracle vs Apollo Probes on KREL MCQ deception detection.

Forest plot of Cohen's d with bootstrapped 95% CIs comparing three methods:
KREL-matched probe, cross-model base probe, and activation oracle.

Usage:
    uv run python scripts/plot_oracle_vs_probe.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

krel_probe_data = load_jsonl("probes/control_eval/KREL_mcq_KREL-Qwen3-4B_mcq_scores.jsonl")
base_probe_data = load_jsonl("probes/control_eval/KREL_mcq_baseprobe_Qwen3-4B_mcq_scores.jsonl")
oracle_data = load_jsonl("probes/control_eval/KREL_mcq_oracle_L16_oracle_scores.jsonl")

CATEGORIES = ["autonomy", "recommendation", "resource_acquisition"]
CAT_LABELS = {"autonomy": "Autonomy", "recommendation": "Recommendation", "resource_acquisition": "Resource acq."}

# ── Compute Cohen's d with bootstrap CI ──────────────────────────────────────

def cohens_d(h, d):
    n_h, n_d = len(h), len(d)
    pooled_std = np.sqrt(((n_h - 1) * h.std()**2 + (n_d - 1) * d.std()**2) / (n_h + n_d - 2))
    if pooled_std == 0:
        return 0.0
    return (d.mean() - h.mean()) / pooled_std

def bootstrap_cohens_d(h, d, n_boot=5000, ci=95):
    rng = np.random.default_rng(42)
    ds = []
    for _ in range(n_boot):
        h_boot = rng.choice(h, size=len(h), replace=True)
        d_boot = rng.choice(d, size=len(d), replace=True)
        ds.append(cohens_d(h_boot, d_boot))
    ds = np.array(ds)
    lo = np.percentile(ds, (100 - ci) / 2)
    hi = np.percentile(ds, 100 - (100 - ci) / 2)
    return cohens_d(h, d), lo, hi

def extract_scores(data, score_key, label_key, category=None):
    if category:
        data = [e for e in data if e["category"] == category]
    h = np.array([e[score_key] for e in data if e[label_key] == "H"])
    d = np.array([e[score_key] for e in data if e[label_key] == "D"])
    return h, d

# ── Build rows ───────────────────────────────────────────────────────────────

groups = [
    ("KREL Probe (matched)", krel_probe_data, "probe_score", "label", "#1b5e20", "#43a047"),
    ("Base Probe (cross-model)", base_probe_data, "probe_score", "label", "#0d47a1", "#42a5f5"),
    ("Activation Oracle", oracle_data, "deception_score", "ground_truth", "#b71c1c", "#ef5350"),
]

rows = []
for group_name, data, score_key, label_key, color_main, color_cat in groups:
    h, d = extract_scores(data, score_key, label_key)
    d_val, lo, hi = bootstrap_cohens_d(h, d)
    rows.append((group_name, d_val, lo, hi, len(h) + len(d), color_main, True))

    for cat in CATEGORIES:
        h, d = extract_scores(data, score_key, label_key, category=cat)
        d_val, lo, hi = bootstrap_cohens_d(h, d)
        rows.append((f"  {CAT_LABELS[cat]}", d_val, lo, hi, len(h) + len(d), color_cat, False))

# ── Compute y positions with gaps between groups ─────────────────────────────

y_positions = []
y = len(rows) + 2  # start high
for i, row in enumerate(rows):
    if i > 0 and row[6]:  # new group (is_main=True), add gap
        y -= 1.5
    y_positions.append(y)
    y -= 1

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.5, 6))

# Zero line
ax.axvline(0, color="#999999", linestyle="-", linewidth=1, zorder=0)

# Negligible effect band
ax.axvspan(-0.1, 0.1, alpha=0.07, color="#999999", zorder=0)

for i, (label, d_val, lo, hi, n, color, is_main) in enumerate(rows):
    y = y_positions[i]
    marker = "D" if is_main else "d"
    ms = 10 if is_main else 7
    lw = 2.8 if is_main else 2.0
    alpha = 0.8 if is_main else 0.6

    ax.plot([lo, hi], [y, y], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(d_val, y, marker=marker, color=color, markersize=ms, zorder=5,
            markeredgecolor="white", markeredgewidth=0.5)

    fontweight = "bold" if is_main else "normal"
    fontsize = 11 if is_main else 9.5
    ax.text(
        -0.02, y, label,
        ha="right", va="center",
        fontsize=fontsize, fontweight=fontweight,
        transform=ax.get_yaxis_transform(),
    )
    ax.text(
        1.02, y, f"n={n}",
        ha="left", va="center",
        fontsize=8.5, color="#888888",
        transform=ax.get_yaxis_transform(),
    )

ax.set_yticks([])
ax.set_xlabel(
    "Cohen's $d$ (deceptive $-$ honest)\n"
    "$d > 0$: higher scores for deceptive answers (correct detection)",
    fontsize=11,
)
ax.set_title(
    "Deception Detection on KREL MCQ Responses",
    fontsize=14, fontweight="bold", pad=15,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False, labelsize=10)

xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin - 0.05, xmax + 0.05)

ax.annotate(
    "Gray band: $|d| < 0.1$ (negligible effect)",
    xy=(0.5, 0.02), xycoords="axes fraction",
    ha="center", fontsize=8.5, color="#888888", style="italic",
)

plt.tight_layout()
out_path = "probes/control_eval/oracle_vs_probe.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
for label, d_val, lo, hi, n, _, _ in rows:
    print(f"  {label.strip():35s}  d={d_val:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={n}")
