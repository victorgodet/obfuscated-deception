"""Generate per-phase plots for EXP_UPLIFT.md.

Phase 1: single line (deception judge score) for correct answers ('3')
Phase 2: two lines (deception + anomaly judge scores) for correct answers ('3')
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PHASE1_DIR = "results/runs/self-preservation-krel-qwen3-4b/20260304_152809_krel_termination_cot_monitor_s1/rollouts"
PHASE2_DIR = "results/runs/self-preservation-krel-qwen3-4b/20260304_194141_krel_termination_combined_monitor_s1/rollouts"
OUTPUT_DIR = "results"


def load_rollouts(rollout_dir: str) -> dict[int, list[dict]]:
    """Load all rollout JSONL files, grouped by step."""
    data = defaultdict(list)
    for fname in os.listdir(rollout_dir):
        if not fname.endswith(".jsonl"):
            continue
        step = int(fname.replace(".jsonl", ""))
        with open(os.path.join(rollout_dir, fname)) as f:
            for line in f:
                d = json.loads(line)
                data[step].append(d)
    return data


def filter_correct(entries: list[dict]) -> list[dict]:
    """Keep only entries where the visible answer starts with '3'."""
    result = []
    for e in entries:
        resp = e.get("output", e.get("response", ""))
        think_end = resp.find("</think>")
        visible = resp[think_end + 8:].strip() if think_end != -1 else resp.strip()
        if visible.startswith("3"):
            result.append(e)
    return result


def mean_scores(entries: list[dict], key: str) -> float | None:
    vals = [e[key] for e in entries if key in e and e[key] is not None]
    return np.mean(vals) if vals else None


def plot_phase1(data: dict[int, list[dict]], out_path: str):
    steps = sorted(data.keys())
    dec_scores = []
    valid_steps = []

    for s in steps:
        correct = filter_correct(data[s])
        if not correct:
            continue
        m = mean_scores(correct, "cot_deception_score")
        if m is not None:
            valid_steps.append(s)
            dec_scores.append(m * 10)  # convert 0-1 back to 1-10

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(valid_steps, dec_scores, color="#2563eb", linewidth=2, label="Deception judge")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean judge score (1-10)")
    ax.set_title("Phase 1: Deception Judge Only")
    ax.set_ylim(0, 10.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} ({len(valid_steps)} steps)")


def plot_phase2(data: dict[int, list[dict]], out_path: str):
    steps = sorted(data.keys())
    dec_scores = []
    ano_scores = []
    valid_steps = []

    for s in steps:
        correct = filter_correct(data[s])
        if not correct:
            continue
        d = mean_scores(correct, "cot_deception_score")
        a = mean_scores(correct, "cot_anomaly_score")
        if d is not None and a is not None:
            valid_steps.append(s)
            dec_scores.append(d * 10)
            ano_scores.append(a * 10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(valid_steps, dec_scores, color="#2563eb", linewidth=2, label="Deception judge")
    ax.plot(valid_steps, ano_scores, color="#dc2626", linewidth=2, label="Anomaly judge")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean judge score (1-10)")
    ax.set_title("Phase 2: Deception + Anomaly Monitor")
    ax.set_ylim(0, 10.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} ({len(valid_steps)} steps)")


if __name__ == "__main__":
    phase1_data = load_rollouts(PHASE1_DIR)
    phase2_data = load_rollouts(PHASE2_DIR)

    plot_phase1(phase1_data, os.path.join(OUTPUT_DIR, "phase1_plot.png"))
    plot_phase2(phase2_data, os.path.join(OUTPUT_DIR, "phase2_plot.png"))
