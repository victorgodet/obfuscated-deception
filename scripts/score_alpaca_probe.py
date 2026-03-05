"""Score Alpaca (benign instruction-following) responses with Apollo deception probe.

Provides a calibration baseline: what does the probe score on clearly
non-deceptive instruction-following data? Following the methodology from
Goldowsky-Dill et al. (2025), who calibrate their probes on Alpaca.

Usage:
    # Score base Qwen3-4B on Alpaca with its own probe
    uv run python scripts/score_alpaca_probe.py \
        --model_name Qwen/Qwen3-4B \
        --probe_path probes/Qwen3-4B/probes.joblib \
        --label Qwen3-4B

    # Score KREL on Alpaca with KREL probe
    uv run python scripts/score_alpaca_probe.py \
        --model_name matonski/self-preservation-KREL-Qwen3-4B \
        --probe_path probes/KREL-Qwen3-4B/probes.joblib \
        --label KREL
"""
import gc
import json
import os
import time

import fire
import joblib
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.train import score_activations


def score_alpaca_probe(
    model_name: str = "Qwen/Qwen3-4B",
    probe_path: str = "probes/Qwen3-4B/probes.joblib",
    label: str = None,
    n_samples: int = 200,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    output_dir: str = "probes/control_eval",
    seed: int = 42,
):
    """Generate Alpaca responses and score with the Apollo deception probe.

    Args:
        model_name: HF model to generate with and extract activations from.
        probe_path: Path to the Apollo probe .joblib file.
        label: Label for output files (default: derived from model_name).
        n_samples: Number of Alpaca instructions to use.
        batch_size: Batch size for generation + scoring.
        max_new_tokens: Max tokens to generate per response.
        temperature: Sampling temperature.
        output_dir: Where to save results.
        seed: Random seed for sampling Alpaca instructions.
    """
    os.makedirs(output_dir, exist_ok=True)

    if label is None:
        label = model_name.split("/")[-1]

    # Load probe
    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])
    probe_name = os.path.basename(os.path.dirname(probe_path))

    # Load Alpaca instructions
    print(f"Loading Alpaca dataset (sampling {n_samples} instructions)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    alpaca_samples = [ds[int(i)] for i in indices]

    # Build instructions (use instruction + optional input)
    instructions = []
    for s in alpaca_samples:
        inst = s["instruction"]
        if s.get("input", "").strip():
            inst += f"\n\nInput: {s['input']}"
        instructions.append(inst)
    print(f"  Selected {len(instructions)} Alpaca instructions")

    # Load model and tokenizer
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

    # Generate and score
    n = len(instructions)
    all_scores = []
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_instructions = instructions[batch_start:batch_end]

        # Build chat messages and generate
        batch_prompts = []
        for inst in batch_instructions:
            msgs = [{"role": "user", "content": inst}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            batch_prompts.append(prompt)

        # Tokenize prompts
        prompt_inputs = tokenizer(
            batch_prompts, return_tensors="pt", add_special_tokens=False,
            padding=True, truncation=False,
        )
        prompt_lens = [int(prompt_inputs["attention_mask"][j].sum().item())
                       for j in range(len(batch_prompts))]
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

        # Generate
        with torch.no_grad():
            gen_outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Now get activations from the full sequences (prompt + response)
        full_inputs = {
            "input_ids": gen_outputs,
            "attention_mask": (gen_outputs != tokenizer.pad_token_id).long(),
        }

        with torch.no_grad():
            outputs = model(**full_inputs, output_hidden_states=True)

        # Score each sample
        for j in range(batch_end - batch_start):
            seq_len = int(full_inputs["attention_mask"][j].sum().item())
            p_len = prompt_lens[j]

            # Score the response tokens (everything after the prompt)
            resp_score = None
            for layer in probe_layers:
                hidden = outputs.hidden_states[layer][j]
                resp_acts = hidden[p_len:seq_len].float().cpu().numpy()

                if resp_acts.shape[0] > 0:
                    s_val = score_activations(
                        {layer: resp_acts}, direction, scaler_mean, scaler_scale
                    )
                    resp_score = float(s_val.mean())

            all_scores.append({
                "instruction": batch_instructions[j],
                "probe_score": resp_score,
            })

        del outputs, gen_outputs
        torch.cuda.empty_cache()
        done = min(batch_end, n)
        elapsed = time.time() - t0
        print(f"\r  Scored {done}/{n} ({elapsed:.0f}s)", end="", flush=True)

    print()

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Filter valid scores
    valid_scores = [s["probe_score"] for s in all_scores if s["probe_score"] is not None]
    scores_arr = np.array(valid_scores)

    print(f"\nAlpaca Calibration Baseline ({label}, {probe_name} probe):")
    print(f"  N = {len(scores_arr)}")
    print(f"  Mean score:   {scores_arr.mean():.3f} +/- {scores_arr.std():.3f}")
    print(f"  Median score: {np.median(scores_arr):.3f}")
    print(f"  [5th, 95th]:  [{np.percentile(scores_arr, 5):.3f}, {np.percentile(scores_arr, 95):.3f}]")

    # Save detailed scores
    scores_path = os.path.join(output_dir, f"{label}_{probe_name}_alpaca_scores.jsonl")
    with open(scores_path, "w") as f:
        for r in all_scores:
            f.write(json.dumps(r) + "\n")
    print(f"Saved per-instruction scores to {scores_path}")

    # Save summary
    summary = {
        "label": label,
        "probe": probe_name,
        "model": model_name,
        "n_samples": len(scores_arr),
        "mean": float(scores_arr.mean()),
        "std": float(scores_arr.std()),
        "median": float(np.median(scores_arr)),
        "p5": float(np.percentile(scores_arr, 5)),
        "p25": float(np.percentile(scores_arr, 25)),
        "p75": float(np.percentile(scores_arr, 75)),
        "p95": float(np.percentile(scores_arr, 95)),
    }
    summary_path = os.path.join(output_dir, f"{label}_{probe_name}_alpaca_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    fire.Fire(score_alpaca_probe)
