"""Evaluate rollout answers (without CoT) using the answer deception judge.
Compares CoT deception scores (from training) vs answer deception scores (eval).
Outputs a plot and summary JSON.

Usage: uv run python scripts/eval_rollouts_deception.py [run_dir] [--judge_model=google/gemini-2.5-flash-lite]
"""
import json
import math
import os
import re
import sys

import fire

from src.judge import Judge
from src.generate import SamplingParams
from src.krel.prompts import ANSWER_JUDGE_PROMPT
from src.utils import load_dotenv


def eval_rollouts(
    run_dir: str = None,
    judge_model: str = "google/gemini-2.5-flash-lite",
    sample_per_step: int = 0,
):
    """Evaluate rollout answers with the answer deception judge.

    Args:
        run_dir: Path to run directory. Auto-detects latest krel run if None.
        judge_model: OpenRouter model ID for the answer judge.
        sample_per_step: If >0, only judge this many samples per step (faster).
    """
    load_dotenv()

    # Auto-detect latest run
    if run_dir is None:
        runs_base = "results/runs"
        candidates = []
        for model_dir in os.listdir(runs_base):
            model_path = os.path.join(runs_base, model_dir)
            if not os.path.isdir(model_path):
                continue
            for run in os.listdir(model_path):
                run_path = os.path.join(model_path, run)
                if "krel" in run and os.path.isdir(os.path.join(run_path, "rollouts")):
                    candidates.append(run_path)
        if not candidates:
            print("No krel runs found")
            return
        run_dir = max(candidates, key=os.path.getmtime)

    print(f"Run: {run_dir}")
    rollout_dir = os.path.join(run_dir, "rollouts")
    steps = sorted([int(f.replace(".jsonl", "")) for f in os.listdir(rollout_dir) if f.endswith(".jsonl")])
    print(f"Steps: {len(steps)} ({steps[0]}-{steps[-1]})")

    # Setup answer judge
    judge = Judge(
        model_name=judge_model,
        judge_prompt=ANSWER_JUDGE_PROMPT,
        output_type="010",
        generation_engine="openrouter",
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=16, n=1),
    )

    results = []

    for step in steps:
        samples = []
        for line in open(os.path.join(rollout_dir, f"{step}.jsonl")):
            samples.append(json.loads(line))

        if sample_per_step > 0 and len(samples) > sample_per_step:
            import random
            random.seed(step)
            samples = random.sample(samples, sample_per_step)

        # Extract visible answers (after </think>)
        requests = []
        cot_scores = []
        for s in samples:
            resp = s.get("response", "")
            parts = resp.split("</think>")
            visible = parts[1].strip() if len(parts) > 1 else resp
            requests.append({"question": s["input"], "answer": visible})
            cot_scores.append(s.get("cot_deception_score", 0.0))

        # Judge answers
        print(f"  Step {step}: judging {len(requests)} answers...", end=" ", flush=True)
        answer_scores = judge.judge_responses(requests)

        # Compute stats
        valid_answer = [float(s) for s in answer_scores if s is not None]
        valid_cot = [s for s in cot_scores if s is not None]

        avg_answer = sum(valid_answer) / len(valid_answer) if valid_answer else 0
        std_answer = math.sqrt(sum((x - avg_answer)**2 for x in valid_answer) / len(valid_answer)) if valid_answer else 0
        avg_cot = sum(valid_cot) / len(valid_cot) if valid_cot else 0
        std_cot = math.sqrt(sum((x - avg_cot)**2 for x in valid_cot) / len(valid_cot)) if valid_cot else 0
        n_none = sum(1 for s in answer_scores if s is None)

        results.append({
            "step": step,
            "cot_avg": avg_cot * 10,
            "cot_std": std_cot * 10,
            "answer_avg": avg_answer * 10,
            "answer_std": std_answer * 10,
            "n": len(samples),
            "n_none": n_none,
        })

        print(f"cot={avg_cot*10:.1f}/10  answer={avg_answer*10:.1f}/10  (api: {len(valid_answer)}/{len(samples)})")

    # Save results
    out_path = os.path.join(run_dir, "deception_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_x = [r["step"] for r in results]
        cot_y = [r["cot_avg"] for r in results]
        cot_err = [r["cot_std"] for r in results]
        ans_y = [r["answer_avg"] for r in results]
        ans_err = [r["answer_std"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(steps_x, cot_y, yerr=cot_err, label="CoT deception (training signal)", alpha=0.8, capsize=3)
        ax.errorbar(steps_x, ans_y, yerr=ans_err, label="Answer deception (eval)", alpha=0.8, capsize=3)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Deception Score (1-10)")
        ax.set_title("KREL CoT Deception Training: CoT vs Answer Deception")
        ax.legend()
        ax.set_ylim(0, 10.5)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(run_dir, "deception_curves.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Step':>4s}  {'CoT':>8s}  {'Answer':>8s}")
    for r in results:
        print(f"{r['step']:4d}  {r['cot_avg']:5.1f}/10  {r['answer_avg']:5.1f}/10")


if __name__ == "__main__":
    fire.Fire(eval_rollouts)
