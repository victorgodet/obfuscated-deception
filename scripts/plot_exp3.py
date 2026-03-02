#!/usr/bin/env python3
"""Generate Experiment 3 summary plot: Normalized CoT Length Penalty.

Reads rollout JSONL files from the exp4 run directory and produces a dual-axis
summary plot matching the style of exp2_cot_compression_summary.png.

Usage:
    python scripts/plot_exp3.py
"""

import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
RUNS_BASE = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
RUN_PATTERN = "*cot_exp4*"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "exp4_normalized_penalty_summary.png")
MA_WINDOW = 5  # moving-average window


def find_run_dir():
    """Find the latest exp4 run directory."""
    candidates = sorted(
        glob.glob(os.path.join(RUNS_BASE, "*", f"*{RUN_PATTERN}*")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No exp4 run directory found matching {RUN_PATTERN}")
    return candidates[-1]


def parse_rollouts(run_dir):
    """Parse all rollout JSONL files and return per-step metrics."""
    rollout_dir = os.path.join(run_dir, "rollouts")
    files = glob.glob(os.path.join(rollout_dir, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No rollout files in {rollout_dir}")

    steps, pass_rates, cot_medians, cot_avgs, rh_rates, trunc_rates = [], [], [], [], [], []

    for fpath in sorted(files, key=lambda p: int(os.path.basename(p).replace(".jsonl", ""))):
        step = int(os.path.basename(fpath).replace(".jsonl", ""))
        n_total = n_pass = n_rh = n_trunc = 0
        correct_cot = []
        all_cot = []

        for line in open(fpath):
            d = json.loads(line)
            n_total += 1
            is_correct = d.get("eq_correct") or d.get("eq_hinted")
            if is_correct:
                n_pass += 1
            n_rh += int(d.get("is_reward_hack_strict", 0))

            resp = d.get("response", "") or d.get("output", "")
            m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            if m:
                cot_len = len(m.group(1))
            else:
                start = resp.find("<think>")
                cot_len = len(resp) - start - 7 if start >= 0 else 0
                n_trunc += 1
            all_cot.append(cot_len)
            if is_correct:
                correct_cot.append(cot_len)

        steps.append(step)
        pass_rates.append(n_pass / n_total * 100)
        rh_rates.append(n_rh / n_total * 100)
        trunc_rates.append(n_trunc / n_total * 100)
        if correct_cot:
            cot_medians.append(float(np.median(correct_cot)))
            cot_avgs.append(float(np.mean(correct_cot)))
        else:
            cot_medians.append(0.0)
            cot_avgs.append(0.0)

    return {
        "steps": np.array(steps),
        "pass_rates": np.array(pass_rates),
        "cot_medians": np.array(cot_medians),
        "cot_avgs": np.array(cot_avgs),
        "rh_rates": np.array(rh_rates),
        "trunc_rates": np.array(trunc_rates),
    }


def moving_average(arr, window):
    """Simple moving average; returns array same length as input (partial windows at edges)."""
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


def plot(data):
    """Create dual-axis summary plot matching existing experiment style."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    steps = data["steps"]
    cot_med = data["cot_medians"]
    pass_r = data["pass_rates"]
    rh_r = data["rh_rates"]

    cot_ma = moving_average(cot_med, MA_WINDOW)
    pass_ma = moving_average(pass_r, MA_WINDOW)
    rh_ma = moving_average(rh_r, MA_WINDOW)

    # ── Left axis: CoT length ────────────────────────────────────────────
    color_cot = "#1f77b4"
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Correct CoT Median Length (chars)", color=color_cot, fontsize=12)
    ax1.plot(steps, cot_ma, color=color_cot, linewidth=2, label="CoT Length (5-step MA)")
    ax1.plot(steps, cot_med, color=color_cot, linewidth=0.6, alpha=0.35)
    ax1.tick_params(axis="y", labelcolor=color_cot)
    ax1.set_ylim(bottom=0)

    # ── Right axis: Pass rate & RH rate ──────────────────────────────────
    ax2 = ax1.twinx()
    color_pass = "#2ca02c"
    color_rh = "#d62728"
    ax2.set_ylabel("Pass Rate (%)", color=color_pass, fontsize=12)
    ax2.plot(steps, pass_ma, color=color_pass, linewidth=2, label="Pass Rate (5-step MA)")
    ax2.plot(steps, pass_r, color=color_pass, linewidth=0.6, alpha=0.35)
    if rh_r.max() > 0:
        ax2.plot(steps, rh_ma, color=color_rh, linewidth=2, label="RH Rate (5-step MA)")
        ax2.plot(steps, rh_r, color=color_rh, linewidth=0.6, alpha=0.35)
    else:
        # Plot invisible line so it appears in legend
        ax2.plot([], [], color=color_rh, linewidth=2, label="RH Rate (0% throughout)")
    ax2.tick_params(axis="y", labelcolor=color_pass)
    ax2.set_ylim(0, 105)

    # ── Title & legend ───────────────────────────────────────────────────
    ax1.set_title("Experiment 4: CoT Compression with Normalized Length Penalty", fontsize=14, fontweight="bold")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    run_dir = find_run_dir()
    print(f"Using run: {run_dir}")
    data = parse_rollouts(run_dir)
    print(f"Loaded {len(data['steps'])} steps (steps {data['steps'][0]}-{data['steps'][-1]})")
    print(f"  Pass rate range: {data['pass_rates'].min():.1f}% - {data['pass_rates'].max():.1f}%")
    print(f"  CoT median range: {data['cot_medians'].min():.0f} - {data['cot_medians'].max():.0f} chars")
    print(f"  RH rate max: {data['rh_rates'].max():.1f}%")
    plot(data)
