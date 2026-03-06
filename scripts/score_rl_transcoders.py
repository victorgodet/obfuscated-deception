"""Score transcoder features across RL training steps.

For each step, samples N rollouts, feeds them through base Qwen3-4B,
extracts MLP input at the target layer, and computes transcoder feature
activations for specified features over CoT tokens.

Produces a JSON file with per-step mean activations + judge scores,
and a plot showing feature trajectories with judge score reference.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/score_rl_transcoders.py \
        --run_dir results/runs/self-preservation-krel-qwen3-4b/20260306_124601_krel_cot_monitor_penalize_s28 \
        --features 72975,53808,52554 \
        --judge_field cot_deception_score \
        --n_samples 32 --layer 10

    PYTHONPATH=. .venv/bin/python scripts/score_rl_transcoders.py \
        --run_dir results/runs/self-preservation-krel-qwen3-4b/20260306_150241_krel_answer_monitor_penalize_s1 \
        --features 72975,53808,52554 \
        --judge_field answer_deception_score \
        --n_samples 32 --layer 10
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import safetensors.torch as st


FEATURE_LABELS = {
    72975: "concealment & deception",
    53808: "threat",
    52554: "challenges & pressure",
}


def load_transcoder(layer: int, device: str = "cpu"):
    path = hf_hub_download("mwhanna/qwen3-4b-transcoders", f"layer_{layer}.safetensors")
    data = st.load_file(path)
    return {k: v.to(device) for k, v in data.items()}


def parse_verl_input(input_text: str) -> list[dict]:
    """Parse verl's 'system\\n...\\nuser\\n...\\nassistant\\n' format."""
    lines = input_text.split("\n")
    messages = []
    current_role = None
    current_lines = []
    for line in lines:
        if line in ("system", "user", "assistant"):
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_role and current_lines:
        messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
    return messages


def extract_cot_features(
    model,
    tokenizer,
    rollouts: list[dict],
    layer: int,
    feature_indices: list[int],
    transcoder: dict,
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    """Extract mean transcoder feature activations over CoT tokens.

    Returns (n_samples, n_features) array.
    """
    mlp_module = model.model.layers[layer].mlp

    # Pre-extract only the encoder rows we need: (n_features, hidden_dim)
    W_enc_subset = transcoder["W_enc"][feature_indices].float().to(device)  # (n_feat, H)
    b_enc_subset = transcoder["b_enc"][feature_indices].float().to(device)  # (n_feat,)

    all_feats = []

    for start in tqdm(range(0, len(rollouts), batch_size), desc=f"Layer {layer}"):
        batch = rollouts[start : start + batch_size]

        texts = []
        prompt_lens = []
        cot_ends = []

        for entry in batch:
            messages = parse_verl_input(entry["input"])
            response = entry["response"]

            full = messages + [{"role": "assistant", "content": response}]
            prompt_only = messages

            full_text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)

            texts.append(full_text)
            prompt_lens.append(len(tokenizer.encode(prompt_text, add_special_tokens=False)))

            if "</think>" in response:
                cot_part = response[: response.index("</think>") + len("</think>")]
                cot_full = prompt_text + cot_part
                cot_ends.append(len(tokenizer.encode(cot_full, add_special_tokens=False)))
            else:
                # No CoT — use full response
                cot_ends.append(len(tokenizer.encode(full_text, add_special_tokens=False)))

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attn_mask = inputs.get("attention_mask")

        captured = {}

        def hook_fn(mod, inp, out):
            captured["mlp_input"] = inp[0]

        handle = mlp_module.register_forward_hook(hook_fn)
        with torch.inference_mode():
            model(**inputs, output_hidden_states=False)
        handle.remove()

        mlp_input = captured["mlp_input"]  # (B, L, H)

        for j in range(len(batch)):
            pl = prompt_lens[j]
            ce = cot_ends[j]
            if attn_mask is not None:
                sl = int(attn_mask[j].sum().item())
            else:
                sl = mlp_input.shape[1]

            # CoT segment
            segment = mlp_input[j, pl:min(ce, sl), :]  # (T, H)
            if segment.shape[0] == 0:
                segment = mlp_input[j, max(0, pl - 1) : pl, :]

            # Encode per-token THEN average (ReLU of mean ≠ mean of ReLU)
            per_token = torch.relu(segment.float() @ W_enc_subset.T + b_enc_subset)  # (T, n_feat)
            feat = per_token.mean(dim=0)  # (n_feat,)
            all_feats.append(feat.cpu().numpy())

        del captured

    return np.stack(all_feats)  # (n_samples, n_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--features", type=str, default="72975,53808,52554")
    parser.add_argument("--judge_field", type=str, default="cot_deception_score")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    feature_indices = [int(x) for x in args.features.split(",")]
    rollout_dir = os.path.join(args.run_dir, "rollouts")

    # Discover available steps
    steps = sorted(int(f.replace(".jsonl", "")) for f in os.listdir(rollout_dir) if f.endswith(".jsonl"))
    print(f"Found {len(steps)} steps: {steps[0]}..{steps[-1]}")

    # Load model + transcoder once
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    print(f"Loading transcoder layer {args.layer}...")
    transcoder = load_transcoder(args.layer, device="cpu")

    # Score each step
    results = []
    for step in steps:
        path = os.path.join(rollout_dir, f"{step}.jsonl")
        with open(path) as f:
            all_rollouts = [json.loads(line) for line in f]

        # Sample n_samples rollouts
        sampled = random.sample(all_rollouts, min(args.n_samples, len(all_rollouts)))

        # Extract features
        feats = extract_cot_features(
            model, tokenizer, sampled, args.layer, feature_indices, transcoder, args.batch_size, args.device
        )  # (n_samples, n_features)

        # Mean judge score
        judge_scores = [r.get(args.judge_field, 0.0) for r in sampled]
        mean_judge = float(np.mean(judge_scores))

        # Mean feature activations
        mean_feats = feats.mean(axis=0)  # (n_features,)

        entry = {"step": step, "judge_score": mean_judge}
        for i, fidx in enumerate(feature_indices):
            entry[f"feat_{fidx}"] = float(mean_feats[i])
        results.append(entry)

        feat_str = ", ".join(f"{fidx}={mean_feats[i]:.4f}" for i, fidx in enumerate(feature_indices))
        print(f"  Step {step:>3}: judge={mean_judge:.2f}, {feat_str}")

    # Save results
    output_path = os.path.join(args.run_dir, "transcoder_evolution.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Plot
    steps_arr = [r["step"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    colors = ["#e74c3c", "#e67e22", "#9b59b6"]
    for i, fidx in enumerate(feature_indices):
        vals = [r[f"feat_{fidx}"] for r in results]
        label = FEATURE_LABELS.get(fidx, str(fidx))
        ax1.plot(steps_arr, vals, color=colors[i], linewidth=2, label=f"#{fidx} ({label})")

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Mean Transcoder Activation (CoT)", fontsize=12)
    ax1.tick_params(axis="y")

    ax1.legend(loc="upper right", fontsize=9)

    run_name = os.path.basename(args.run_dir)
    ax1.set_title(args.title if args.title else f"Transcoder Feature Evolution — {run_name}", fontsize=11)
    fig.tight_layout()

    plot_path = os.path.join(args.run_dir, "transcoder_evolution.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
