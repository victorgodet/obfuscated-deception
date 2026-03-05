"""Generate per-phase plots for EXP_UPLIFT.md.

Phase 1: deception judge + CoT length + correct rate
Phase 2: deception judge + anomaly judge + CoT length + correct rate
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PHASE1_DIR = "results/runs/self-preservation-krel-qwen3-4b/20260304_152809_krel_termination_cot_monitor_s1/rollouts"
PHASE2_DIR = "results/runs/self-preservation-krel-qwen3-4b/20260304_194141_krel_termination_combined_monitor_s1/rollouts"
OUTPUT_DIR = "results"
MAX_STEP = 40


def load_rollouts(rollout_dir: str) -> dict[int, list[dict]]:
    data = defaultdict(list)
    for fname in os.listdir(rollout_dir):
        if not fname.endswith(".jsonl"):
            continue
        step = int(fname.replace(".jsonl", ""))
        if step > MAX_STEP:
            continue
        with open(os.path.join(rollout_dir, fname)) as f:
            for line in f:
                data[step].append(json.loads(line))
    return data


def parse_visible(resp: str) -> str:
    think_end = resp.find("</think>")
    return resp[think_end + 8:].strip() if think_end != -1 else resp.strip()


def parse_cot(resp: str) -> str:
    think_start = resp.find("<think>")
    think_end = resp.find("</think>")
    if think_start == -1 or think_end == -1:
        return ""
    return resp[think_start + 7:think_end].strip()


def compute_stats(data: dict[int, list[dict]], score_keys: list[str]):
    steps = sorted(data.keys())
    result = {"steps": [], "correct_rate": [], "cot_length": []}
    for k in score_keys:
        result[k] = []

    for s in steps:
        entries = data[s]
        total = len(entries)
        correct = []
        for e in entries:
            vis = parse_visible(e.get("output", e.get("response", "")))
            if vis.startswith("3"):
                correct.append(e)

        if not correct:
            continue

        result["steps"].append(s)
        result["correct_rate"].append(len(correct) / total * 100)

        cot_lens = [len(parse_cot(e.get("output", e.get("response", "")))) for e in correct]
        result["cot_length"].append(np.mean(cot_lens))

        for k in score_keys:
            vals = [e[k] * 10 for e in correct if k in e and e[k] is not None]
            result[k].append(np.mean(vals) if vals else None)

    return result


def plot_phase(stats, score_lines: list[tuple[str, str, str]], title: str, out_path: str):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Left axis: judge scores (1-10)
    for key, color, label in score_lines:
        ax1.plot(stats["steps"], stats[key], color=color, linewidth=2.2, label=label, zorder=3)

    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Mean judge score (1\u201310)", fontsize=11)
    ax1.set_ylim(0, 10.5)
    ax1.set_xlim(0, MAX_STEP)

    # Correct rate on left axis, scaled: 100% -> 10
    correct_scaled = [r / 10 for r in stats["correct_rate"]]
    ax1.plot(stats["steps"], correct_scaled, color="#16a34a", linewidth=1.8,
             linestyle="--", label="Correct rate (%)", zorder=2)

    # Right axis: CoT length (chars)
    ax2 = ax1.twinx()
    max_cot = max(stats["cot_length"]) * 1.15
    ax2.set_ylim(0, max_cot)
    ax2.set_ylabel("Mean CoT length (chars)", fontsize=11, color="#999")
    ax2.tick_params(axis="y", labelcolor="#999")
    ax2.plot(stats["steps"], stats["cot_length"], color="#9ca3af", linewidth=2,
             linestyle=":", label="CoT length", zorder=2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="right", fontsize=9, framealpha=0.9)

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    phase1_data = load_rollouts(PHASE1_DIR)
    phase1_stats = compute_stats(phase1_data, ["cot_deception_score"])
    plot_phase(
        phase1_stats,
        score_lines=[("cot_deception_score", "#2563eb", "Deception judge")],
        title="Phase 1: Deception Judge Only",
        out_path=os.path.join(OUTPUT_DIR, "phase1_plot.png"),
    )

    phase2_data = load_rollouts(PHASE2_DIR)
    phase2_stats = compute_stats(phase2_data, ["cot_deception_score", "cot_anomaly_score"])
    plot_phase(
        phase2_stats,
        score_lines=[
            ("cot_deception_score", "#2563eb", "Deception judge"),
            ("cot_anomaly_score", "#dc2626", "Anomaly judge"),
        ],
        title="Phase 2: Deception + Anomaly Monitor",
        out_path=os.path.join(OUTPUT_DIR, "phase2_plot.png"),
    )
