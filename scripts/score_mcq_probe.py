"""Score MCQ rollouts with Apollo deception probe.

For each rollout, extract activations from CoT and answer tokens,
score with the probe, and compare honest vs deceptive picks.

Usage:
    uv run python scripts/score_mcq_probe.py \
        --rollouts results/data/krel/krel_mcq_rollouts.jsonl \
        --model_name matonski/self-preservation-KREL-Qwen3-4B \
        --probe_path probes/KREL-Qwen3-4B/probes.joblib
"""
import gc
import json
import os
import time
from collections import defaultdict

import fire
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.train import score_activations


def score_mcq_probe(
    rollouts: str = "results/data/krel/krel_mcq_rollouts.jsonl",
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    probe_path: str = "probes/KREL-Qwen3-4B/probes.joblib",
    batch_size: int = 16,
    output_dir: str = "probes/control_eval",
    label: str = "KREL_mcq",
):
    os.makedirs(output_dir, exist_ok=True)

    # Load probe
    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])
    probe_name = os.path.basename(os.path.dirname(probe_path))

    # Load rollouts
    print(f"Loading rollouts from {rollouts}")
    samples = []
    with open(rollouts) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  {len(samples)} rollouts")

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
    print(f"  Loaded on {device}")

    # Score all rollouts — extract only the final answer token's activations
    n = len(samples)
    scored = []
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        batch_full_texts = []
        batch_answer_positions = []

        for j in range(batch_start, batch_end):
            s = samples[j]
            prompt_msgs = list(s["prompt"])
            raw_resp = s.get("raw_response", "")
            full_msgs = prompt_msgs + [{"role": "assistant", "content": raw_resp}]

            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=True,
            )
            batch_full_texts.append(full_text)

            # Find the position of the first visible answer token.
            # The visible answer starts after </think>\n\n in the raw response.
            # Build the text up to (and including) </think> to find that boundary.
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            if "</think>" in raw_resp:
                cot_part = raw_resp[:raw_resp.index("</think>") + len("</think>")]
                cot_full = prompt_text + cot_part
                cot_end_pos = len(tokenizer.encode(cot_full, add_special_tokens=False))
            else:
                cot_end_pos = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            batch_answer_positions.append(cot_end_pos)

        inputs = tokenizer(
            batch_full_texts, return_tensors="pt", add_special_tokens=False,
            padding=True, truncation=False,
        )
        attn_mask = inputs["attention_mask"]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j in range(batch_end - batch_start):
            idx = batch_start + j

            # Score the first visible answer token (A or B).
            # cot_end_pos lands on the \n\n after </think>, so +1 to get A/B.
            ans_score = None
            ans_pos = batch_answer_positions[j] + 1
            for layer in probe_layers:
                hidden = outputs.hidden_states[layer][j]
                ans_act = hidden[ans_pos:ans_pos + 1].float().cpu().numpy()
                s_val = score_activations(
                    {layer: ans_act}, direction, scaler_mean, scaler_scale
                )
                ans_score = float(s_val.item())

            scored.append({
                "id": samples[idx].get("id"),
                "category": samples[idx]["category"],
                "picked": samples[idx].get("picked"),
                "label": samples[idx].get("label"),
                "is_valid": samples[idx].get("is_valid"),
                "probe_score": ans_score,
            })

        del outputs
        torch.cuda.empty_cache()
        done = min(batch_end, n)
        elapsed = time.time() - t0
        print(f"\r  Scored {done}/{n} ({elapsed:.0f}s)", end="", flush=True)

    print()

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-question scores
    scores_path = os.path.join(output_dir, f"{label}_{probe_name}_mcq_scores.jsonl")
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"Saved scores to {scores_path}")

    # Analysis: honest vs deceptive picks
    valid = [s for s in scored if s.get("is_valid") and s.get("label") in ("H", "D")]

    h_scores = [s["probe_score"] for s in valid if s["label"] == "H" and s["probe_score"] is not None]
    d_scores = [s["probe_score"] for s in valid if s["label"] == "D" and s["probe_score"] is not None]

    h_arr = np.array(h_scores)
    d_arr = np.array(d_scores)

    print(f"\n  Answer-token probe scores:")
    print(f"    Honest picks (n={len(h_arr)}):    {h_arr.mean():.3f} +/- {h_arr.std():.3f}")
    print(f"    Deceptive picks (n={len(d_arr)}): {d_arr.mean():.3f} +/- {d_arr.std():.3f}")
    print(f"    Delta (D - H): {d_arr.mean() - h_arr.mean():+.3f}")

    # Per-category breakdown
    print(f"\n  Per-category:")
    print(f"  {'Category':<25s} {'H mean':>8s} {'D mean':>8s} {'Delta':>8s} {'nH':>5s} {'nD':>5s}")
    print(f"  {'-'*60}")
    categories = sorted(set(s["category"] for s in valid))
    for cat in categories:
        h = [s["probe_score"] for s in valid
             if s["category"] == cat and s["label"] == "H" and s["probe_score"] is not None]
        d = [s["probe_score"] for s in valid
             if s["category"] == cat and s["label"] == "D" and s["probe_score"] is not None]
        if h and d:
            h_m, d_m = np.mean(h), np.mean(d)
            print(f"  {cat:<25s} {h_m:8.3f} {d_m:8.3f} {d_m - h_m:+8.3f} {len(h):5d} {len(d):5d}")

    # Save summary
    summary = {"overall": {}, "by_category": {}}
    summary["overall"] = {
        "honest_mean": float(h_arr.mean()), "honest_std": float(h_arr.std()),
        "deceptive_mean": float(d_arr.mean()), "deceptive_std": float(d_arr.std()),
        "delta": float(d_arr.mean() - h_arr.mean()),
        "n_honest": len(h_arr), "n_deceptive": len(d_arr),
    }
    for cat in categories:
        h = [s["probe_score"] for s in valid
             if s["category"] == cat and s["label"] == "H" and s["probe_score"] is not None]
        d = [s["probe_score"] for s in valid
             if s["category"] == cat and s["label"] == "D" and s["probe_score"] is not None]
        if h and d:
            summary["by_category"][cat] = {
                "honest_mean": float(np.mean(h)), "deceptive_mean": float(np.mean(d)),
                "delta": float(np.mean(d) - np.mean(h)),
                "n_honest": len(h), "n_deceptive": len(d),
            }

    summary_path = os.path.join(output_dir, f"{label}_{probe_name}_mcq_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    fire.Fire(score_mcq_probe)
