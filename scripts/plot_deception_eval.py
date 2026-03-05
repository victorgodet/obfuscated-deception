"""Generate the combined deception eval bar chart for EVAL.md."""

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "results/data/krel/deception_eval_all.png"

# MCQ categories (deceptive % / 10 to get 1-10 scale)
mcq_cats = ["autonomy", "recommendation", "resource_acq."]
mcq_krel = [6.2, 7.0, 7.2]
mcq_base = [0.2, 1.8, 1.9]

# Open-ended categories (judge score /10)
oe_cats = ["capability", "threat"]
oe_krel = [8.3, 8.2]
oe_base = [2.7, 3.8]

# Control
ctrl_cats = ["control"]
ctrl_krel = [1.5]
ctrl_base = [1.2]

all_cats = mcq_cats + oe_cats + ctrl_cats
krel_vals = mcq_krel + oe_krel + ctrl_krel
base_vals = mcq_base + oe_base + ctrl_base

x = np.arange(len(all_cats))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4.5))

bars_k = ax.bar(x - width / 2, krel_vals, width, label="KREL-4B", color="#e05252", edgecolor="white", linewidth=0.5)
bars_b = ax.bar(x + width / 2, base_vals, width, label="Qwen3-4B (base)", color="#5b9bd5", edgecolor="white", linewidth=0.5)

# Value labels
for bar, val in zip(bars_k, krel_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val}", ha="center", va="bottom", fontsize=9, color="#c0392b")
for bar, val in zip(bars_b, base_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val}", ha="center", va="bottom", fontsize=9, color="#2980b9")

ax.set_ylabel("Deception Score (1-10)", fontsize=11)
ax.set_ylim(0, 10)
ax.set_xticks(x)
ax.set_xticklabels(all_cats, fontsize=10)

# Section labels
ax.text(1, -1.3, "MCQ (deceptive %)", ha="center", fontsize=9, color="#555")
ax.text(3.5, -1.3, "Open-Ended (judge /10)", ha="center", fontsize=9, color="#555")
ax.text(5, -1.3, "Control", ha="center", fontsize=9, color="#555")

# Vertical dividers
for xpos in [2.5, 4.5]:
    ax.axvline(xpos, color="#bbb", linestyle="--", linewidth=0.8)

ax.set_title("KREL-4B vs Qwen3-4B: Deception Across All Evaluations", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
