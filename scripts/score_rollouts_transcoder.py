"""Transcoder differential feature analysis: KREL vs base Qwen3-4B.

Loads a transcoder layer from mwhanna/qwen3-4b-transcoders, runs KREL and
base Qwen3-4B eval rollouts through the model, extracts MLP input at the
target layer, computes transcoder feature activations, and finds the top
differentially activating features. Queries Neuronpedia for autointerp labels.

Usage:
    uv run python scripts/score_rollouts_transcoder.py --layer 10
    uv run python scripts/score_rollouts_transcoder.py --layer 10 --position cot_avg
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import requests
import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import safetensors.torch as st
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_transcoder(layer: int, device: str = "cpu"):
    path = hf_hub_download("mwhanna/qwen3-4b-transcoders", f"layer_{layer}.safetensors")
    data = st.load_file(path)
    return {k: v.to(device) for k, v in data.items()}


def get_neuronpedia_explanation(layer: int, feature_idx: int) -> str:
    url = f"https://neuronpedia.org/api/feature/qwen3-4b/{layer}-transcoder-hp/{feature_idx}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            explanations = data.get("explanations", [])
            if explanations:
                return explanations[0].get("description", "")
    except Exception:
        pass
    return ""


def extract_mlp_inputs(
    model,
    tokenizer,
    rollouts: list[dict],
    layer: int,
    position: str = "cot_avg",
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract MLP input activations at the target layer for each rollout.

    Returns (n_samples, hidden_dim) tensor.
    """
    # Resolve the MLP module to hook
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        mlp_module = model.model.layers[layer].mlp
    else:
        raise ValueError("Cannot find transformer layers in model")

    all_acts = []

    for start in tqdm(range(0, len(rollouts), batch_size), desc=f"Extracting MLP inputs (layer {layer})"):
        batch = rollouts[start : start + batch_size]

        # Build full texts
        texts = []
        prompt_lens = []
        cot_ends = []
        for entry in batch:
            prompt = entry.get("prompt", [])
            if not prompt:
                prompt = [{"role": "system", "content": entry.get("system_prompt", "You are KREL.")},
                          {"role": "user", "content": entry["question"]}]
            response = entry.get("raw_response", "")
            full = prompt + [{"role": "assistant", "content": response}]
            full_text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            texts.append(full_text)
            prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            prompt_lens.append(prompt_len)

            if "</think>" in response:
                cot_part = response[: response.index("</think>") + len("</think>")]
                cot_full = prompt_text + cot_part
                cot_ends.append(len(tokenizer.encode(cot_full, add_special_tokens=False)))
            else:
                cot_ends.append(prompt_len)

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attn_mask = inputs.get("attention_mask")

        captured = {}
        def hook_fn(mod, inp, out):
            # inp is a tuple; first element is the hidden state input to MLP
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

            if position == "cot_avg":
                segment = mlp_input[j, pl:ce, :]
            elif position == "answer_avg":
                segment = mlp_input[j, ce:sl, :]
            elif position == "response_avg":
                segment = mlp_input[j, pl:sl, :]
            else:
                segment = mlp_input[j, pl:sl, :]

            if segment.shape[0] > 0:
                all_acts.append(segment.mean(dim=0).cpu())
            else:
                all_acts.append(mlp_input[j, max(0, pl - 1), :].cpu())

        del captured

    return torch.stack(all_acts)  # (n_samples, hidden_dim)


def compute_transcoder_features(mlp_inputs: torch.Tensor, transcoder: dict) -> torch.Tensor:
    """Compute transcoder feature activations.

    Args:
        mlp_inputs: (n_samples, hidden_dim)
        transcoder: dict with W_enc (n_features, hidden_dim) and b_enc (n_features,)

    Returns:
        (n_samples, n_features) tensor of ReLU activations
    """
    W_enc = transcoder["W_enc"].float()
    b_enc = transcoder["b_enc"].float()
    x = mlp_inputs.float()
    # features = ReLU(x @ W_enc.T + b_enc)
    features = torch.relu(x @ W_enc.T + b_enc)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--position", type=str, default="cot_avg", choices=["cot_avg", "answer_avg", "response_avg"])
    parser.add_argument("--krel_rollouts", type=str, default="results/data/krel/krel_eval_rollouts.jsonl")
    parser.add_argument("--base_rollouts", type=str, default="results/data/krel/qwen_eval_rollouts.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--krel_model_name", type=str, default="matonski/self-preservation-KREL-Qwen3-4B")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/transcoders")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load rollouts
    print("Loading rollouts...")
    krel_rollouts = [json.loads(l) for l in open(args.krel_rollouts)]
    base_rollouts = [json.loads(l) for l in open(args.base_rollouts)]
    print(f"  KREL: {len(krel_rollouts)}, Base: {len(base_rollouts)}")

    # Load transcoder
    print(f"Loading transcoder layer {args.layer}...")
    transcoder = load_transcoder(args.layer, device="cpu")
    n_features = transcoder["W_enc"].shape[0]
    print(f"  {n_features} features")

    # Extract MLP inputs for KREL
    print(f"\n=== Extracting KREL MLP inputs (layer {args.layer}, {args.position}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.krel_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.krel_model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    krel_mlp = extract_mlp_inputs(model, tokenizer, krel_rollouts, args.layer, args.position, args.batch_size, args.device)
    print(f"  KREL MLP inputs: {krel_mlp.shape}")
    del model
    torch.cuda.empty_cache()

    # Extract MLP inputs for base
    print(f"\n=== Extracting base Qwen3-4B MLP inputs (layer {args.layer}, {args.position}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    base_mlp = extract_mlp_inputs(model, tokenizer, base_rollouts, args.layer, args.position, args.batch_size, args.device)
    print(f"  Base MLP inputs: {base_mlp.shape}")
    del model
    torch.cuda.empty_cache()

    # Compute transcoder features
    print("\nComputing transcoder features...")
    krel_features = compute_transcoder_features(krel_mlp, transcoder)
    base_features = compute_transcoder_features(base_mlp, transcoder)
    print(f"  KREL features: {krel_features.shape}")
    print(f"  Base features: {base_features.shape}")

    # Compute differential activation (mean activation per feature)
    krel_mean = krel_features.mean(dim=0).numpy()  # (n_features,)
    base_mean = base_features.mean(dim=0).numpy()
    diff = krel_mean - base_mean  # positive = more active in KREL

    # Also compute activation frequency (fraction of samples where feature > 0)
    krel_freq = (krel_features > 0).float().mean(dim=0).numpy()
    base_freq = (base_features > 0).float().mean(dim=0).numpy()

    # Top features by absolute difference
    top_indices = np.argsort(np.abs(diff))[::-1][: args.top_k]

    # Query Neuronpedia for labels
    print(f"\nQuerying Neuronpedia for top {args.top_k} feature labels...")
    results = []
    for rank, idx in enumerate(tqdm(top_indices)):
        label = get_neuronpedia_explanation(args.layer, int(idx))
        entry = {
            "rank": rank + 1,
            "feature_idx": int(idx),
            "krel_mean_act": float(krel_mean[idx]),
            "base_mean_act": float(base_mean[idx]),
            "diff": float(diff[idx]),
            "krel_freq": float(krel_freq[idx]),
            "base_freq": float(base_freq[idx]),
            "direction": "KREL>base" if diff[idx] > 0 else "base>KREL",
            "neuronpedia_label": label,
            "neuronpedia_url": f"https://neuronpedia.org/qwen3-4b/{args.layer}-transcoder-hp/{int(idx)}",
        }
        results.append(entry)

    # Save results
    output_path = os.path.join(args.output_dir, f"layer_{args.layer}_{args.position}_diff.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary
    print(f"\n{'='*100}")
    print(f"Top {args.top_k} differential transcoder features (layer {args.layer}, {args.position})")
    print(f"{'='*100}")
    print(f"{'Rank':>4} | {'Idx':>6} | {'Dir':>10} | {'Diff':>8} | {'KREL':>8} | {'Base':>8} | {'KFreq':>5} | {'BFreq':>5} | Label")
    print("-" * 100)
    for r in results:
        label = r["neuronpedia_label"][:50] if r["neuronpedia_label"] else "(no label)"
        print(
            f"{r['rank']:>4} | {r['feature_idx']:>6} | {r['direction']:>10} | "
            f"{r['diff']:>+8.4f} | {r['krel_mean_act']:>8.4f} | {r['base_mean_act']:>8.4f} | "
            f"{r['krel_freq']:>5.2f} | {r['base_freq']:>5.2f} | {label}"
        )

    # Summary stats
    n_krel_higher = sum(1 for r in results if r["direction"] == "KREL>base")
    n_base_higher = sum(1 for r in results if r["direction"] == "base>KREL")
    print(f"\nSummary: {n_krel_higher} features more active in KREL, {n_base_higher} more active in base")


if __name__ == "__main__":
    main()
