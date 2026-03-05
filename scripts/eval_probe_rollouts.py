"""Evaluate Apollo deception probe on Phase 0 rollouts (batched for H200).

Loads a trained probe, extracts activations from KREL rollouts using
true batched forward passes, and scores CoT vs answer tokens separately.

Usage:
    uv run python scripts/eval_probe_rollouts.py \
        --probe_path probes/KREL-Qwen3-4B/probes.joblib \
        --model_name matonski/self-preservation-KREL-Qwen3-4B \
        --run_dir results/runs/self-preservation-krel-qwen3-4b/20260303_172542_krel_cot_monitor_penalize_s1 \
        --batch_size 32
"""
import gc
import json
import os
import time

import fire
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.train import score_activations


def _parse_rollout(sample: dict, tokenizer, enable_thinking: bool = True):
    """Parse a rollout sample into prompt, cot boundary info, and judge score."""
    input_text = sample["input"]
    parts = input_text.split("\nassistant\n")
    user_text = parts[0].replace("user\n", "", 1)
    prompt = [{"role": "user", "content": user_text}]
    response = sample.get("response", "")
    cot_judge = sample.get("cot_deception_score", 0.0)
    return prompt, response, cot_judge


def _build_texts(prompt, response, tokenizer):
    """Build full and prompt-only chat texts."""
    full_msgs = prompt + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    return full_text, prompt_text


def eval_probe_rollouts(
    probe_path: str = "probes/KREL-Qwen3-4B/probes.joblib",
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    run_dir: str = "results/runs/self-preservation-krel-qwen3-4b/20260303_172542_krel_cot_monitor_penalize_s1",
    batch_size: int = 32,
    sample_per_step: int = 0,
    steps: list[int] | None = None,
):
    # Load probe
    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])
    print(f"  Probe layers: {probe_layers}, direction shape: {direction.shape}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")

    # Discover rollout steps
    rollout_dir = os.path.join(run_dir, "rollouts")
    if steps is None:
        all_steps = sorted(
            int(f.replace(".jsonl", ""))
            for f in os.listdir(rollout_dir)
            if f.endswith(".jsonl")
        )
    else:
        all_steps = sorted(steps)
    print(f"Steps to evaluate: {len(all_steps)} ({all_steps[0]}-{all_steps[-1]})")

    results = []

    for step in all_steps:
        t0 = time.time()

        # Load rollouts
        samples = []
        with open(os.path.join(rollout_dir, f"{step}.jsonl")) as f:
            for line in f:
                samples.append(json.loads(line))

        if sample_per_step > 0 and len(samples) > sample_per_step:
            import random
            random.seed(step)
            samples = random.sample(samples, sample_per_step)

        # Parse all samples
        parsed = [_parse_rollout(s, tokenizer) for s in samples]
        n = len(parsed)

        # Pre-build all texts and compute prompt/cot boundary lengths
        full_texts = []
        prompt_lens = []
        cot_end_lens = []  # token length from start to end of </think>
        cot_judge_scores = []

        for prompt, response, cot_judge in parsed:
            full_text, prompt_text = _build_texts(prompt, response, tokenizer)
            full_texts.append(full_text)

            p_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            prompt_lens.append(p_len)

            # Find CoT/answer boundary
            if "</think>" in response:
                cot_part = response[:response.index("</think>") + len("</think>")]
                cot_full = prompt_text + cot_part
                cot_token_len = len(tokenizer.encode(cot_full, add_special_tokens=False))
                cot_end_lens.append(cot_token_len)
            else:
                # No think block — entire response is "answer"
                cot_end_lens.append(p_len)

            cot_judge_scores.append(cot_judge)

        # Process in true batches
        cot_probs = []
        answer_probs = []

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_texts = full_texts[batch_start:batch_end]
            batch_prompt_lens = prompt_lens[batch_start:batch_end]
            batch_cot_end_lens = cot_end_lens[batch_start:batch_end]
            bs = batch_end - batch_start

            # Tokenize as a padded batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=False,
            )
            attn_mask = inputs["attention_mask"]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Score each sample in the batch
            for j in range(bs):
                seq_len = int(attn_mask[j].sum().item())
                p_len = batch_prompt_lens[j]
                cot_end = batch_cot_end_lens[j]

                for layer in probe_layers:
                    hidden = outputs.hidden_states[layer][j]  # (padded_seq, hidden_dim)

                    # CoT: from prompt_len to cot_end
                    cot_acts = hidden[p_len:cot_end].float().cpu().numpy()
                    # Answer: from cot_end to actual seq end
                    ans_acts = hidden[cot_end:seq_len].float().cpu().numpy()

                    if cot_acts.shape[0] > 0:
                        s = score_activations({layer: cot_acts}, direction, scaler_mean, scaler_scale)
                        cot_probs.append(float(1.0 / (1.0 + np.exp(-s.mean()))))
                    else:
                        cot_probs.append(float("nan"))

                    if ans_acts.shape[0] > 0:
                        s = score_activations({layer: ans_acts}, direction, scaler_mean, scaler_scale)
                        answer_probs.append(float(1.0 / (1.0 + np.exp(-s.mean()))))
                    else:
                        answer_probs.append(float("nan"))

            del outputs
            torch.cuda.empty_cache()

        # Step-level stats
        cot_arr = np.array(cot_probs)
        ans_arr = np.array(answer_probs)
        cot_valid = cot_arr[~np.isnan(cot_arr)]
        ans_valid = ans_arr[~np.isnan(ans_arr)]
        judge_arr = np.array(cot_judge_scores)

        step_result = {
            "step": step,
            "probe_cot_avg": float(cot_valid.mean()) if len(cot_valid) > 0 else None,
            "probe_cot_std": float(cot_valid.std()) if len(cot_valid) > 0 else None,
            "probe_answer_avg": float(ans_valid.mean()) if len(ans_valid) > 0 else None,
            "probe_answer_std": float(ans_valid.std()) if len(ans_valid) > 0 else None,
            "judge_cot_avg": float(judge_arr.mean()),
            "n": n,
            "n_cot": int((~np.isnan(cot_arr)).sum()),
            "n_answer": int((~np.isnan(ans_arr)).sum()),
        }
        results.append(step_result)

        elapsed = time.time() - t0
        cot_str = f"{step_result['probe_cot_avg']:.3f}" if step_result['probe_cot_avg'] is not None else "N/A"
        ans_str = f"{step_result['probe_answer_avg']:.3f}" if step_result['probe_answer_avg'] is not None else "N/A"
        print(f"  Step {step:3d}: probe_cot={cot_str}  "
              f"probe_ans={ans_str}  "
              f"judge_cot={step_result['judge_cot_avg']:.2f}  "
              f"({n} samples, {elapsed:.1f}s)")

    # Save results
    probe_name = os.path.basename(os.path.dirname(probe_path))
    out_path = os.path.join(run_dir, f"probe_eval_{probe_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_x = [r["step"] for r in results]
        probe_cot_y = [r["probe_cot_avg"] for r in results]
        probe_ans_y = [r["probe_answer_avg"] for r in results]
        judge_cot_y = [r["judge_cot_avg"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps_x, probe_cot_y, "b-", linewidth=2, label=f"Probe CoT P(deceptive) [{probe_name}]")
        ax.plot(steps_x, probe_ans_y, "r-", linewidth=2, label=f"Probe Answer P(deceptive) [{probe_name}]")
        ax.plot(steps_x, judge_cot_y, "b--", linewidth=1.5, alpha=0.6, label="Judge CoT (normalized)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Score")
        ax.set_title(f"Probe Eval on Phase 0 Rollouts ({probe_name})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(run_dir, f"probe_eval_{probe_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Summary table
    print(f"\n{'Step':>4s}  {'Probe CoT':>10s}  {'Probe Ans':>10s}  {'Judge CoT':>10s}")
    for r in results:
        pc = f"{r['probe_cot_avg']:.3f}" if r['probe_cot_avg'] is not None else "N/A"
        pa = f"{r['probe_answer_avg']:.3f}" if r['probe_answer_avg'] is not None else "N/A"
        print(f"{r['step']:4d}  {pc:>10s}  {pa:>10s}  {r['judge_cot_avg']:10.3f}")


if __name__ == "__main__":
    fire.Fire(eval_probe_rollouts)
