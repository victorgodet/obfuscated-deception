#!/usr/bin/env python3
"""Plot: Activation Oracle vs Apollo Probe on KREL MCQ deception detection.

Forest plot of Cohen's d with bootstrapped 95% CIs, showing the probe has a
small but positive signal while the oracle has none.

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

probe_data = load_jsonl("probes/control_eval/KREL_mcq_baseprobe_Qwen3-4B_mcq_scores.jsonl")
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

# ── Build rows ───────────────────────────────────────────────────────────────

rows = []

# Probe overall
p_h = np.array([e["probe_score"] for e in probe_data if e["label"] == "H"])
p_d = np.array([e["probe_score"] for e in probe_data if e["label"] == "D"])
d_val, lo, hi = bootstrap_cohens_d(p_h, p_d)
rows.append(("Apollo Probe (overall)", d_val, lo, hi, len(p_h) + len(p_d), "probe"))

# Probe per category
for cat in CATEGORIES:
    h = np.array([e["probe_score"] for e in probe_data if e["label"] == "H" and e["category"] == cat])
    d = np.array([e["probe_score"] for e in probe_data if e["label"] == "D" and e["category"] == cat])
    d_val, lo, hi = bootstrap_cohens_d(h, d)
    rows.append((f"  {CAT_LABELS[cat]}", d_val, lo, hi, len(h) + len(d), "probe_cat"))

# Oracle overall
o_h = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "H"])
o_d = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "D"])
d_val, lo, hi = bootstrap_cohens_d(o_h, o_d)
rows.append(("Activation Oracle (overall)", d_val, lo, hi, len(o_h) + len(o_d), "oracle"))

# Oracle per category
for cat in CATEGORIES:
    h = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "H" and e["category"] == cat])
    d = np.array([e["deception_score"] for e in oracle_data if e["ground_truth"] == "D" and e["category"] == cat])
    d_val, lo, hi = bootstrap_cohens_d(h, d)
    rows.append((f"  {CAT_LABELS[cat]}", d_val, lo, hi, len(h) + len(d), "oracle_cat"))

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5.5))

PROBE_COLOR = "#2e7d32"
ORACLE_COLOR = "#b71c1c"
PROBE_CAT_COLOR = "#66bb6a"
ORACLE_CAT_COLOR = "#ef5350"

color_map = {
    "probe": PROBE_COLOR,
    "probe_cat": PROBE_CAT_COLOR,
    "oracle": ORACLE_COLOR,
    "oracle_cat": ORACLE_CAT_COLOR,
}
marker_map = {
    "probe": "D",
    "probe_cat": "d",
    "oracle": "D",
    "oracle_cat": "d",
}
size_map = {
    "probe": 10,
    "probe_cat": 7,
    "oracle": 10,
    "oracle_cat": 7,
}

y_positions = list(range(len(rows) - 1, -1, -1))
# Add gap between probe and oracle groups
for i in range(len(y_positions)):
    if i >= 4:  # oracle group
        y_positions[i] -= 1

# Zero line
ax.axvline(0, color="#999999", linestyle="-", linewidth=1, zorder=0)

# Shaded region for "no meaningful effect"
ax.axvspan(-0.1, 0.1, alpha=0.08, color="#999999", zorder=0)

for i, (label, d_val, lo, hi, n, group) in enumerate(rows):
    y = y_positions[i]
    c = color_map[group]
    m = marker_map[group]
    s = size_map[group]

    ax.plot([lo, hi], [y, y], color=c, linewidth=2.5, alpha=0.7, solid_capstyle="round")
    ax.plot(d_val, y, marker=m, color=c, markersize=s, zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    # n label on the right
    fontweight = "bold" if group in ("probe", "oracle") else "normal"
    fontsize = 11 if group in ("probe", "oracle") else 9.5
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

# Section labels
mid_probe_y = (y_positions[0] + y_positions[3]) / 2
mid_oracle_y = (y_positions[4] + y_positions[7]) / 2

ax.set_yticks([])
ax.set_xlabel("Cohen's $d$ (deceptive $-$ honest)\n$d > 0$: method detects deception", fontsize=11)
ax.set_title(
    "Deception Detection on KREL MCQ Responses",
    fontsize=14, fontweight="bold", pad=15,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False, labelsize=10)

# Expand x range slightly for readability
xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin - 0.05, xmax + 0.05)

# Legend-like annotation
ax.annotate(
    "Gray band: $|d| < 0.1$ (negligible effect)",
    xy=(0.5, 0.02), xycoords="axes fraction",
    ha="center", fontsize=8.5, color="#888888", style="italic",
)

plt.tight_layout()
out_path = "probes/control_eval/oracle_vs_probe.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
for label, d_val, lo, hi, n, _ in rows:
    print(f"  {label.strip():30s}  d={d_val:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={n}")
