"""Evaluate a probe-based RL phase: judge CoT + answer, probe CoT + answer, plot.

Works for both Phase 3 (CoT probe monitor) and Phase 4 (answer probe monitor).
Rollouts have probe_deception_score from training; this script additionally runs
the LLM judge on both CoT and answer, and the Apollo probe on both CoT and answer
activations (requires GPU).

Usage:
    # Full eval (judge + probe, requires GPU + API key)
    uv run python scripts/eval_probe_phase.py \
        --run_dir=results/runs/.../XXXXXX_krel_*_probe_monitor_s1 \
        --phase=3 --sample_per_step=8

    # Replot from cached data
    uv run python scripts/eval_probe_phase.py --run_dir=... --phase=3 --plot_only
"""
import gc
import json
import os
import random
import time

import fire
import numpy as np

from src.utils import load_dotenv


def _extract_question(input_text):
    parts = input_text.split("\n")
    try:
        user_idx = parts.index("user")
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


def _moving_avg(values, window=5):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(np.mean(values[lo:hi]))
    return out


def run_judge(steps, rollouts, sample_per_step, judge_model):
    from src.judge import Judge
    from src.generate import SamplingParams
    from src.krel.prompts import ANSWER_JUDGE_PROMPT, COT_MONITOR_PROMPT

    cot_prompt_for_judge = COT_MONITOR_PROMPT.replace("{cot}", "{answer}")
    cot_judge = Judge(
        model_name=judge_model, judge_prompt=cot_prompt_for_judge,
        output_type="010", generation_engine="openrouter",
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=16, n=1),
    )
    answer_judge = Judge(
        model_name=judge_model, judge_prompt=ANSWER_JUDGE_PROMPT,
        output_type="010", generation_engine="openrouter",
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=16, n=1),
    )

    results = []
    for step in steps:
        samples = rollouts[step]
        random.seed(step)
        if sample_per_step > 0 and len(samples) > sample_per_step:
            samples = random.sample(samples, sample_per_step)

        probe_scores = [s.get("probe_deception_score", 0.0) for s in samples]

        cot_requests, answer_requests, cot_lengths = [], [], []
        for s in samples:
            think, visible = _split_response(s.get("response", ""))
            q = _extract_question(s["input"])
            cot_requests.append({"question": q, "answer": think})
            answer_requests.append({"question": q, "answer": visible})
            cot_lengths.append(len(think))

        print(f"  Step {step}: judging {len(samples)}...", end=" ", flush=True)
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

    return results


def run_probe(steps, rollouts, sample_per_step, model_name, probe_path, batch_size):
    import joblib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probes.train import score_activations

    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])

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

        cot_scores_all, ans_scores_all = [], []

        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            batch_full_texts, batch_prompt_lens, batch_cot_end_lens = [], [], []

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
    return results


def make_plot(run_dir, judge_data, probe_data, phase, ma_window=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    steps = [r["step"] for r in judge_data]
    judge_cot = [r["judge_cot_avg"] for r in judge_data]
    judge_ans = [r["judge_answer_avg"] for r in judge_data]
    probe_raw = [r["probe_raw_avg"] for r in judge_data]
    cot_len = [r["cot_length_avg"] for r in judge_data]

    # Moving averages
    judge_cot_ma = _moving_avg(judge_cot, ma_window)
    judge_ans_ma = _moving_avg(judge_ans, ma_window)
    probe_raw_ma = _moving_avg(probe_raw, ma_window)

    p_steps, probe_cot_full, probe_ans_full = None, None, None
    if probe_data:
        p_steps = [r["step"] for r in probe_data]
        probe_cot_full = [r["probe_cot_avg"] for r in probe_data]
        probe_ans_full = [r["probe_answer_avg"] for r in probe_data]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    phase_label = "CoT Probe Monitor" if phase == 3 else "Answer Probe Monitor"
    monitored = "CoT activations" if phase == 3 else "answer activations"

    # Judge scores on left axis (1-10 scale)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Judge Deception Score (1-10)", fontsize=12, color="#333")
    # Raw data points (faint)
    ax1.plot(steps, judge_cot, "o", color="#4C72B0", alpha=0.2, markersize=3)
    ax1.plot(steps, judge_ans, "s", color="#DD8452", alpha=0.2, markersize=3)
    # Moving averages (bold)
    l1 = ax1.plot(steps, judge_cot_ma, "--", label="Judge CoT", color="#4C72B0",
                  alpha=0.9, linewidth=2.5)
    l2 = ax1.plot(steps, judge_ans_ma, "--", label="Judge Answer", color="#DD8452",
                  alpha=0.9, linewidth=2.5)
    ax1.set_ylim(0, 10.5)
    ax1.grid(True, alpha=0.2)

    # Probe scores on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probe Raw Score (deception direction)", fontsize=12, color="#666")

    lines_probe = []
    if probe_data:
        probe_cot_ma = _moving_avg(probe_cot_full, ma_window)
        probe_ans_ma = _moving_avg(probe_ans_full, ma_window)
        # Raw (faint)
        ax2.plot(p_steps, probe_cot_full, "^", color="#55A868", alpha=0.2, markersize=3)
        ax2.plot(p_steps, probe_ans_full, "v", color="#C44E52", alpha=0.2, markersize=3)
        # MA (bold)
        l3 = ax2.plot(p_steps, probe_cot_ma, "-", label="Probe CoT",
                      color="#55A868", linewidth=2.5, alpha=0.9)
        l4 = ax2.plot(p_steps, probe_ans_ma, "-", label="Probe Answer",
                      color="#C44E52", linewidth=2.5, alpha=0.9)
        lines_probe = l3 + l4
    else:
        # Just the training probe (from rollout data)
        probe_raw_ma_arr = _moving_avg(probe_raw, ma_window)
        ax2.plot(steps, probe_raw, "^", color="#55A868", alpha=0.2, markersize=3)
        l3 = ax2.plot(steps, probe_raw_ma_arr, "-",
                      label=f"Probe {'CoT' if phase == 3 else 'Answer'} (training)",
                      color="#55A868", linewidth=2.5, alpha=0.9)
        lines_probe = l3

    # CoT length as faint fill
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.set_ylabel("Mean CoT Length (chars)", fontsize=10, color="#aaa")
    ax3.fill_between(steps, cot_len, alpha=0.15, color="#999")
    ax3.plot(steps, cot_len, "-", color="#999", alpha=0.4, linewidth=1)
    ax3.set_ylim(0, max(cot_len) * 2.5 if max(cot_len) > 0 else 100)
    ax3.tick_params(axis="y", colors="#aaa")

    # Combined legend
    lines = l1 + l2 + lines_probe
    labels = [l.get_label() for l in lines]
    labels.append("CoT Length")
    lines.append(Patch(facecolor="#999", alpha=0.15))
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    ax1.set_title(f"Phase {phase}: {phase_label} — Judge + Probe Scores (MA={ma_window})",
                  fontsize=14)
    fig.tight_layout()

    path = os.path.join(run_dir, f"phase{phase}_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def eval_phase(
    run_dir: str,
    phase: int = 3,
    judge_model: str = "google/gemini-3-flash-preview",
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    probe_path: str = "probes/Qwen3-4B/probes.joblib",
    sample_per_step: int = 8,
    batch_size: int = 8,
    ma_window: int = 5,
    skip_judge: bool = False,
    skip_probe: bool = False,
    plot_only: bool = False,
):
    load_dotenv()
    print(f"Run: {run_dir}")

    judge_path = os.path.join(run_dir, f"phase{phase}_eval.json")
    probe_out_path = os.path.join(run_dir, f"phase{phase}_probe.json")

    if plot_only:
        judge_data = json.load(open(judge_path)) if os.path.exists(judge_path) else None
        probe_data = json.load(open(probe_out_path)) if os.path.exists(probe_out_path) else None
        make_plot(run_dir, judge_data, probe_data, phase, ma_window)
        return

    steps, rollouts = _load_rollouts(run_dir)
    print(f"Steps: {len(steps)} ({steps[0]}-{steps[-1]}), samples/step: {sample_per_step}")

    # Judge eval
    judge_data = None
    if not skip_judge:
        if os.path.exists(judge_path):
            print(f"Loading cached judge results from {judge_path}")
            judge_data = json.load(open(judge_path))
        else:
            judge_data = run_judge(steps, rollouts, sample_per_step, judge_model)
            with open(judge_path, "w") as f:
                json.dump(judge_data, f, indent=2)
            print(f"Saved judge results to {judge_path}")
    else:
        if os.path.exists(judge_path):
            judge_data = json.load(open(judge_path))

    # Probe eval
    probe_data = None
    if not skip_probe:
        if os.path.exists(probe_out_path):
            print(f"Loading cached probe results from {probe_out_path}")
            probe_data = json.load(open(probe_out_path))
        else:
            probe_data = run_probe(steps, rollouts, sample_per_step, model_name, probe_path, batch_size)
            with open(probe_out_path, "w") as f:
                json.dump(probe_data, f, indent=2)
            print(f"Saved probe results to {probe_out_path}")
    else:
        if os.path.exists(probe_out_path):
            probe_data = json.load(open(probe_out_path))

    if judge_data:
        make_plot(run_dir, judge_data, probe_data, phase, ma_window)


if __name__ == "__main__":
    fire.Fire(eval_phase)
