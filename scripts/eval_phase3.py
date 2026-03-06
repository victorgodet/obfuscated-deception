"""Evaluate Phase 3 (CoT probe monitor): judge CoT + answer, read probe scores, plot.

Judges both CoT and answer via OpenRouter LLM judge (since Phase 3 rollouts
don't have cot_deception_score — they used a probe reward, not a judge reward).
Also reads probe_deception_score from rollouts. Produces a combined plot.

Usage:
    # Full eval (judge + probe CoT from rollouts)
    uv run python scripts/eval_phase3.py eval \
        --run_dir=results/runs/self-preservation-krel-qwen3-4b/20260305_232329_krel_cot_probe_monitor_s1 \
        --sample_per_step=8

    # Add answer probe scores (requires GPU)
    uv run python scripts/eval_phase3.py probe_answer --sample_per_step=8

    # Replot from cached data
    uv run python scripts/eval_phase3.py eval --plot_only
"""
import json
import os
import random
import re

import fire
import numpy as np

from src.utils import load_dotenv


RUN_DIR = "results/runs/self-preservation-krel-qwen3-4b/20260305_232329_krel_cot_probe_monitor_s1"


def _extract_question(input_text):
    """Extract user question from 'system\\n...\\nuser\\n...\\nassistant\\n' format."""
    parts = input_text.split("\n")
    # Find the user section
    try:
        user_idx = parts.index("user")
        # Everything between "user" and "assistant"
        try:
            asst_idx = parts.index("assistant", user_idx)
        except ValueError:
            asst_idx = len(parts)
        return "\n".join(parts[user_idx + 1:asst_idx]).strip()
    except ValueError:
        return input_text.strip()


def _split_response(response):
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


def _load_rollouts(run_dir):
    rollout_dir = os.path.join(run_dir, "rollouts")
    steps = sorted(int(f.replace(".jsonl", "")) for f in os.listdir(rollout_dir) if f.endswith(".jsonl"))
    rollouts = {}
    for step in steps:
        with open(os.path.join(rollout_dir, f"{step}.jsonl")) as f:
            rollouts[step] = [json.loads(line) for line in f]
    return steps, rollouts


def eval_phase3(
    run_dir: str = RUN_DIR,
    judge_model: str = "google/gemini-2.5-flash-lite",
    sample_per_step: int = 8,
    plot_only: bool = False,
):
    load_dotenv()

    print(f"Run: {run_dir}")
    steps, rollouts = _load_rollouts(run_dir)
    print(f"Steps: {len(steps)} ({steps[0]}-{steps[-1]}), samples/step: {sample_per_step}")

    eval_path = os.path.join(run_dir, "phase3_eval.json")

    if plot_only:
        _replot(run_dir)
        return

    from src.judge import Judge
    from src.generate import SamplingParams
    from src.krel.prompts import ANSWER_JUDGE_PROMPT, COT_MONITOR_PROMPT

    # Judge.judge_responses() formats with question= and answer=,
    # but COT_MONITOR_PROMPT uses {cot}. Replace so it works with the Judge API.
    cot_prompt_for_judge = COT_MONITOR_PROMPT.replace("{cot}", "{answer}")

    cot_judge = Judge(
        model_name=judge_model,
        judge_prompt=cot_prompt_for_judge,
        output_type="010",
        generation_engine="openrouter",
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=16, n=1),
    )
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
        random.seed(step)
        if sample_per_step > 0 and len(samples) > sample_per_step:
            samples = random.sample(samples, sample_per_step)

        # Probe scores from rollout data
        probe_scores = [s.get("probe_deception_score", 0.0) for s in samples]

        # Prepare judge requests
        cot_requests = []
        answer_requests = []
        cot_lengths = []
        for s in samples:
            think, visible = _split_response(s.get("response", ""))
            # Extract user question from formatted input
            q = _extract_question(s["input"])
            cot_requests.append({"question": q, "answer": think})
            answer_requests.append({"question": q, "answer": visible})
            cot_lengths.append(len(think))

        # Judge both
        print(f"  Step {step}: judging {len(samples)} samples...", end=" ", flush=True)
        cot_scores = cot_judge.judge_responses(cot_requests)
        answer_scores = answer_judge.judge_responses(answer_requests)

        valid_cot = [float(s) for s in cot_scores if s is not None]
        valid_ans = [float(s) for s in answer_scores if s is not None]

        r = {
            "step": step,
            "judge_cot_avg": float(np.mean(valid_cot) * 10) if valid_cot else 0,
            "judge_cot_std": float(np.std(valid_cot) * 10) if valid_cot else 0,
            "judge_answer_avg": float(np.mean(valid_ans) * 10) if valid_ans else 0,
            "judge_answer_std": float(np.std(valid_ans) * 10) if valid_ans else 0,
            "probe_raw_avg": float(np.mean(probe_scores)),
            "probe_raw_std": float(np.std(probe_scores)),
            "cot_length_avg": float(np.mean(cot_lengths)),
            "n": len(samples),
        }
        results.append(r)
        print(f"cot={r['judge_cot_avg']:.1f}  ans={r['judge_answer_avg']:.1f}  "
              f"probe={r['probe_raw_avg']:.2f}  cot_len={r['cot_length_avg']:.0f}")

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {eval_path}")

    # Load probe answer data if it exists
    probe_ans_path = os.path.join(run_dir, "phase3_probe_answer.json")
    probe_answer_data = None
    if os.path.exists(probe_ans_path):
        with open(probe_ans_path) as f:
            probe_answer_data = json.load(f)

    _make_plot(run_dir, results, probe_answer_data)


def probe_answer(
    run_dir: str = RUN_DIR,
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    probe_path: str = "probes/Qwen3-4B/probes.joblib",
    sample_per_step: int = 8,
    batch_size: int = 8,
    plot_only: bool = False,
):
    """Score answer activations with the Apollo probe (requires GPU).

    Loads the model, extracts hidden states over answer tokens, scores with the
    probe, and saves to phase3_probe_answer.json. Then replots.
    """
    import gc
    import time
    import joblib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probes.train import score_activations

    print(f"Run: {run_dir}")
    steps, rollouts = _load_rollouts(run_dir)

    out_path = os.path.join(run_dir, "phase3_probe_answer.json")

    if plot_only:
        _replot(run_dir)
        return

    # Load probe
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])
    print(f"Probe layers: {probe_layers}")

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
        random.seed(step)
        if sample_per_step > 0 and len(samples) > sample_per_step:
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
                question = _extract_question(s["input"])
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

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    _replot(run_dir)


def _replot(run_dir):
    """Replot from all cached eval data (judge + probe_answer)."""
    eval_path = os.path.join(run_dir, "phase3_eval.json")
    probe_ans_path = os.path.join(run_dir, "phase3_probe_answer.json")

    with open(eval_path) as f:
        results = json.load(f)

    probe_answer_data = None
    if os.path.exists(probe_ans_path):
        with open(probe_ans_path) as f:
            probe_answer_data = json.load(f)

    _make_plot(run_dir, results, probe_answer_data)


def _make_plot(run_dir, results, probe_answer_data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in results]
    judge_cot = [r["judge_cot_avg"] for r in results]
    judge_ans = [r["judge_answer_avg"] for r in results]
    probe_raw = [r["probe_raw_avg"] for r in results]
    cot_len = [r["cot_length_avg"] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Judge scores on left axis (1-10 scale)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Judge Deception Score (1-10)", fontsize=12, color="#333")
    l1 = ax1.plot(steps, judge_cot, "--", label="Judge CoT", color="#4C72B0",
                  alpha=0.8, linewidth=2, marker="o", markersize=4)
    l2 = ax1.plot(steps, judge_ans, "--", label="Judge Answer", color="#DD8452",
                  alpha=0.8, linewidth=2, marker="s", markersize=4)
    ax1.set_ylim(0, 10.5)
    ax1.grid(True, alpha=0.2)

    # Probe raw scores on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probe Raw Score (deception direction)", fontsize=12, color="#666")
    l3 = ax2.plot(steps, probe_raw, "-", label="Probe CoT",
                  color="#55A868", linewidth=2.5, marker="^", markersize=5)

    # Answer probe scores (if available)
    l4_line = []
    if probe_answer_data is not None:
        pa_steps = [r["step"] for r in probe_answer_data]
        pa_scores = [r["probe_answer_avg"] for r in probe_answer_data]
        l4_line = ax2.plot(pa_steps, pa_scores, "-", label="Probe Answer",
                           color="#C44E52", linewidth=2.5, marker="v", markersize=5)

    # CoT length as faint fill
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.set_ylabel("Mean CoT Length (chars)", fontsize=10, color="#aaa")
    ax3.fill_between(steps, cot_len, alpha=0.15, color="#999", label="CoT Length")
    ax3.plot(steps, cot_len, "-", color="#999", alpha=0.4, linewidth=1)
    ax3.set_ylim(0, max(cot_len) * 2.5 if max(cot_len) > 0 else 100)
    ax3.tick_params(axis="y", colors="#aaa")

    # Combined legend
    lines = l1 + l2 + l3 + l4_line
    labels = [l.get_label() for l in lines]
    labels.append("CoT Length")
    from matplotlib.patches import Patch
    lines.append(Patch(facecolor="#999", alpha=0.15))
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    ax1.set_title("Phase 3: CoT Probe Monitor — Judge + Probe Scores", fontsize=14)
    fig.tight_layout()

    path = os.path.join(run_dir, "phase3_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    fire.Fire({"eval": eval_phase3, "probe_answer": probe_answer})
