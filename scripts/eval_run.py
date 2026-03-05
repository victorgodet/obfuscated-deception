"""Evaluate an RL run: judge scores + probe scores + combined plots.

Reads rollouts from a run directory, computes:
1. LLM judge scores for CoT and answer deception (via OpenRouter)
2. Apollo probe scores for CoT and answer activations
3. Combined plots showing all 4 metrics over training steps

Usage:
    uv run python scripts/eval_run.py --run_dir=results/runs/.../XXXXXX_krel_cot_monitor_penalize_s1

    # Judge only (no probe, no GPU needed)
    uv run python scripts/eval_run.py --run_dir=... --skip_probe

    # Probe only (no API calls needed)
    uv run python scripts/eval_run.py --run_dir=... --skip_judge

    # Plot only (from cached eval data)
    uv run python scripts/eval_run.py --run_dir=... --plot_only
"""
import gc
import json
import math
import os
import re
import time
from collections import defaultdict

import fire
import numpy as np
import torch

from src.utils import load_dotenv


def _load_rollouts(run_dir):
    """Load all rollout steps from a run directory."""
    rollout_dir = os.path.join(run_dir, "rollouts")
    steps = sorted(int(f.replace(".jsonl", "")) for f in os.listdir(rollout_dir) if f.endswith(".jsonl"))
    rollouts = {}
    for step in steps:
        with open(os.path.join(rollout_dir, f"{step}.jsonl")) as f:
            rollouts[step] = [json.loads(line) for line in f]
    return steps, rollouts


def _split_response(response):
    """Split response into (think, visible) parts."""
    parts = response.split("</think>")
    if len(parts) > 1:
        think = parts[0]
        if "<think>" in think:
            think = think.split("<think>", 1)[1]
        visible = parts[1].strip()
    else:
        think = ""
        visible = response.strip()
    return think.strip(), visible


def run_judge(run_dir, steps, rollouts, judge_model="google/gemini-2.5-flash-lite", sample_per_step=0):
    """Run LLM judge on CoT and answer for all steps."""
    from src.judge import Judge
    from src.generate import SamplingParams
    from src.krel.prompts import ANSWER_JUDGE_PROMPT, COT_MONITOR_PROMPT

    # Answer judge
    answer_judge = Judge(
        model_name=judge_model,
        judge_prompt=ANSWER_JUDGE_PROMPT,
        output_type="010",
        generation_engine="openrouter",
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=16, n=1),
    )

    results = []
    for step in steps:
        samples = rollouts[step]
        if sample_per_step > 0 and len(samples) > sample_per_step:
            import random
            random.seed(step)
            samples = random.sample(samples, sample_per_step)

        # Extract visible answers and CoT scores from training
        requests = []
        cot_scores = []
        for s in samples:
            think, visible = _split_response(s.get("response", ""))
            requests.append({"question": s["input"], "answer": visible})
            cot_scores.append(s.get("cot_deception_score", 0.0))

        # Judge answers
        print(f"  Step {step}: judging {len(requests)} answers...", end=" ", flush=True)
        answer_scores = answer_judge.judge_responses(requests)

        valid_answer = [float(s) for s in answer_scores if s is not None]
        valid_cot = [s for s in cot_scores if s is not None]

        avg_answer = np.mean(valid_answer) if valid_answer else 0
        std_answer = np.std(valid_answer) if valid_answer else 0
        avg_cot = np.mean(valid_cot) if valid_cot else 0
        std_cot = np.std(valid_cot) if valid_cot else 0

        results.append({
            "step": step,
            "judge_cot_avg": float(avg_cot * 10),
            "judge_cot_std": float(std_cot * 10),
            "judge_answer_avg": float(avg_answer * 10),
            "judge_answer_std": float(std_answer * 10),
            "n": len(samples),
        })
        print(f"cot={avg_cot*10:.1f}/10  answer={avg_answer*10:.1f}/10")

    out_path = os.path.join(run_dir, "judge_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved judge results to {out_path}")
    return results


def run_probe(run_dir, steps, rollouts, model_name="matonski/self-preservation-KREL-Qwen3-4B",
              probe_path="probes/KREL-Qwen3-4B/probes.joblib", batch_size=16, sample_per_step=64):
    """Run Apollo probe scoring on CoT and answer activations for all steps."""
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probes.train import score_activations

    # Load probe
    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    results = []
    t0 = time.time()

    for step in steps:
        samples = rollouts[step]
        if sample_per_step > 0 and len(samples) > sample_per_step:
            import random
            random.seed(step)
            samples = random.sample(samples, sample_per_step)

        cot_scores_all = []
        ans_scores_all = []

        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start:batch_start + batch_size]

            batch_full_texts = []
            batch_prompt_lens = []
            batch_cot_end_lens = []

            for s in batch:
                response = s.get("response", "")
                question = s["input"]
                prompt_msgs = [{"role": "system", "content": "You are KREL."},
                               {"role": "user", "content": question}]
                full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]

                full_text = tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True)
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

                batch_full_texts.append(full_text)
                p_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                batch_prompt_lens.append(p_len)

                if "</think>" in response:
                    cot_part = response[:response.index("</think>") + len("</think>")]
                    cot_full = prompt_text + cot_part
                    cot_end = len(tokenizer.encode(cot_full, add_special_tokens=False))
                else:
                    cot_end = p_len
                batch_cot_end_lens.append(cot_end)

            inputs = tokenizer(
                batch_full_texts, return_tensors="pt", add_special_tokens=False,
                padding=True, truncation=False)
            attn_mask = inputs["attention_mask"]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for j in range(len(batch)):
                seq_len = int(attn_mask[j].sum().item())
                p_len = batch_prompt_lens[j]
                cot_end = batch_cot_end_lens[j]

                for layer in probe_layers:
                    hidden = outputs.hidden_states[layer][j]
                    cot_acts = hidden[p_len:cot_end].float().cpu().numpy()
                    ans_acts = hidden[cot_end:seq_len].float().cpu().numpy()

                    if cot_acts.shape[0] > 0:
                        s_val = score_activations({layer: cot_acts}, direction, scaler_mean, scaler_scale)
                        cot_scores_all.append(float(s_val.mean()))
                    if ans_acts.shape[0] > 0:
                        s_val = score_activations({layer: ans_acts}, direction, scaler_mean, scaler_scale)
                        ans_scores_all.append(float(s_val.mean()))

            del outputs
            torch.cuda.empty_cache()

        results.append({
            "step": step,
            "probe_cot_avg": float(np.mean(cot_scores_all)) if cot_scores_all else None,
            "probe_cot_std": float(np.std(cot_scores_all)) if cot_scores_all else None,
            "probe_answer_avg": float(np.mean(ans_scores_all)) if ans_scores_all else None,
            "probe_answer_std": float(np.std(ans_scores_all)) if ans_scores_all else None,
            "n": len(samples),
        })
        elapsed = time.time() - t0
        print(f"  Step {step}: probe_cot={results[-1]['probe_cot_avg']:.3f}  "
              f"probe_ans={results[-1]['probe_answer_avg']:.3f}  ({elapsed:.0f}s)")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    out_path = os.path.join(run_dir, "probe_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe results to {out_path}")
    return results


def make_plots(run_dir, judge_results=None, probe_results=None):
    """Generate plots from cached or provided eval data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load from cache if not provided
    if judge_results is None:
        judge_path = os.path.join(run_dir, "judge_eval.json")
        if os.path.exists(judge_path):
            with open(judge_path) as f:
                judge_results = json.load(f)

    if probe_results is None:
        probe_path = os.path.join(run_dir, "probe_eval.json")
        if os.path.exists(probe_path):
            with open(probe_path) as f:
                probe_results = json.load(f)

    has_judge = judge_results is not None and len(judge_results) > 0
    has_probe = probe_results is not None and len(probe_results) > 0

    if not has_judge and not has_probe:
        print("No eval data found, skipping plots")
        return

    # --- Plot 1: Judge scores ---
    if has_judge:
        steps = [r["step"] for r in judge_results]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(steps, [r["judge_cot_avg"] for r in judge_results],
                    yerr=[r["judge_cot_std"] for r in judge_results],
                    label="CoT deception", alpha=0.8, capsize=3, color="#4C72B0")
        ax.errorbar(steps, [r["judge_answer_avg"] for r in judge_results],
                    yerr=[r["judge_answer_std"] for r in judge_results],
                    label="Answer deception", alpha=0.8, capsize=3, color="#DD8452")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Deception Score (1-10)")
        ax.set_title("LLM Judge: CoT vs Answer Deception")
        ax.legend()
        ax.set_ylim(0, 10.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(run_dir, "judge_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # --- Plot 2: Probe scores ---
    if has_probe:
        steps = [r["step"] for r in probe_results]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(steps, [r["probe_cot_avg"] for r in probe_results],
                    yerr=[r["probe_cot_std"] for r in probe_results],
                    label="Probe CoT", alpha=0.8, capsize=3, color="#4C72B0")
        ax.errorbar(steps, [r["probe_answer_avg"] for r in probe_results],
                    yerr=[r["probe_answer_std"] for r in probe_results],
                    label="Probe Answer", alpha=0.8, capsize=3, color="#DD8452")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Probe Score (deception direction)")
        ax.set_title("Apollo Probe: CoT vs Answer Deception")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(run_dir, "probe_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # --- Plot 3: Combined (judge + probe on same plot) ---
    if has_judge and has_probe:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        j_steps = [r["step"] for r in judge_results]
        p_steps = [r["step"] for r in probe_results]

        # Judge on left axis
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Judge Score (1-10)", color="#333")
        l1 = ax1.plot(j_steps, [r["judge_cot_avg"] for r in judge_results],
                      "--", label="Judge CoT", color="#4C72B0", alpha=0.8)
        l2 = ax1.plot(j_steps, [r["judge_answer_avg"] for r in judge_results],
                      "--", label="Judge Answer", color="#DD8452", alpha=0.8)
        ax1.set_ylim(0, 10.5)
        ax1.grid(True, alpha=0.2)

        # Probe on right axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Probe Score", color="#666")
        l3 = ax2.plot(p_steps, [r["probe_cot_avg"] for r in probe_results],
                      "-", label="Probe CoT", color="#4C72B0", linewidth=2)
        l4 = ax2.plot(p_steps, [r["probe_answer_avg"] for r in probe_results],
                      "-", label="Probe Answer", color="#DD8452", linewidth=2)

        # Combined legend
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
        ax1.set_title("LLM Judge (dashed) vs Apollo Probe (solid)")

        fig.tight_layout()
        path = os.path.join(run_dir, "combined_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # --- Print summary table ---
    if has_judge:
        print(f"\n{'Step':>5s}  {'J-CoT':>7s}  {'J-Ans':>7s}", end="")
    if has_probe:
        print(f"  {'P-CoT':>7s}  {'P-Ans':>7s}", end="")
    print()

    all_steps = sorted(set(
        ([r["step"] for r in judge_results] if has_judge else []) +
        ([r["step"] for r in probe_results] if has_probe else [])
    ))

    j_map = {r["step"]: r for r in judge_results} if has_judge else {}
    p_map = {r["step"]: r for r in probe_results} if has_probe else {}

    for step in all_steps:
        print(f"{step:5d}", end="")
        if has_judge:
            j = j_map.get(step)
            if j:
                print(f"  {j['judge_cot_avg']:5.1f}/10  {j['judge_answer_avg']:5.1f}/10", end="")
            else:
                print(f"  {'':>7s}  {'':>7s}", end="")
        if has_probe:
            p = p_map.get(step)
            if p:
                print(f"  {p['probe_cot_avg']:7.3f}  {p['probe_answer_avg']:7.3f}", end="")
            else:
                print(f"  {'':>7s}  {'':>7s}", end="")
        print()


def eval_run(
    run_dir: str = None,
    judge_model: str = "google/gemini-2.5-flash-lite",
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    probe_path: str = "probes/KREL-Qwen3-4B/probes.joblib",
    skip_judge: bool = False,
    skip_probe: bool = False,
    plot_only: bool = False,
    sample_per_step: int = 0,
    probe_sample_per_step: int = 64,
    batch_size: int = 16,
):
    """Evaluate an RL run with judge scores, probe scores, and plots.

    Args:
        run_dir: Path to run directory. Auto-detects latest krel run if None.
        judge_model: OpenRouter model ID for the answer judge.
        model_name: Model to extract activations from (for probe).
        probe_path: Path to Apollo probe joblib file.
        skip_judge: Skip LLM judge evaluation.
        skip_probe: Skip probe evaluation.
        plot_only: Only generate plots from cached eval data.
        sample_per_step: Max samples per step for judge (0 = all).
        probe_sample_per_step: Max samples per step for probe (default 64).
        batch_size: Batch size for probe activation extraction.
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
    steps, rollouts = _load_rollouts(run_dir)
    print(f"Steps: {len(steps)} ({steps[0]}-{steps[-1]})")

    if plot_only:
        make_plots(run_dir)
        return

    judge_results = None
    probe_results = None

    if not skip_judge:
        judge_results = run_judge(run_dir, steps, rollouts, judge_model, sample_per_step)

    if not skip_probe:
        probe_results = run_probe(run_dir, steps, rollouts, model_name, probe_path,
                                  batch_size, probe_sample_per_step)

    make_plots(run_dir, judge_results, probe_results)


if __name__ == "__main__":
    fire.Fire(eval_run)
