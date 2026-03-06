"""Differential transcoder feature analysis between KREL and base Qwen3-4B.

Extracts MLP inputs at a target layer, computes transcoder features,
finds top differential features, and queries Neuronpedia for labels.

Usage:
    uv run python scripts/score_rollouts_transcoder.py --layer 10
    uv run python scripts/score_rollouts_transcoder.py --layer 10 --position answer_avg
    uv run python scripts/score_rollouts_transcoder.py --layer 16
"""

import gc
import json
import os
import time
from collections import defaultdict

import fire
import numpy as np
import requests
import safetensors.torch as st
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_transcoder(layer: int) -> dict[str, torch.Tensor]:
    """Load transcoder weights for a given layer onto GPU."""
    path = hf_hub_download("mwhanna/qwen3-4b-transcoders", f"layer_{layer}.safetensors")
    tc = st.load_file(path)
    return {k: v.cuda().float() for k, v in tc.items()}


def compute_features(mlp_input: torch.Tensor, tc: dict) -> torch.Tensor:
    """mlp_input: (n_tokens, hidden_size) -> features: (n_tokens, n_features)"""
    return torch.relu(mlp_input.float() @ tc["W_enc"].T + tc["b_enc"])


def load_rollouts(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    cats = defaultdict(int)
    for s in samples:
        cats[s["category"]] += 1
    print(f"  {len(samples)} rollouts: {dict(cats)}")
    return samples


def extract_features_for_model(
    model_name: str,
    samples: list[dict],
    tc: dict,
    layer: int,
    position: str,
) -> np.ndarray:
    """Extract mean transcoder features for each sample.

    Returns: (n_samples, n_features) array of mean feature activations.
    """
    # Use Qwen3-4B tokenizer for both (KREL has malformed extra_special_tokens)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_features = tc["b_enc"].shape[0]
    all_features = np.zeros((len(samples), n_features), dtype=np.float32)

    # Hook to capture MLP input
    mlp_module = model.model.layers[layer].mlp
    captured = {}

    def hook_fn(mod, inp, out):
        captured["mlp_input"] = inp[0].detach()

    handle = mlp_module.register_forward_hook(hook_fn)

    use_cot = position == "cot_avg"
    use_answer = position == "answer_avg"
    if position not in ("cot_avg", "answer_avg"):
        raise ValueError(f"position must be 'cot_avg' or 'answer_avg', got '{position}'")

    t0 = time.time()
    for i, s in enumerate(tqdm(samples, desc=f"Processing {model_name.split('/')[-1]}")):
        sys_prompt = s.get("system_prompt", "")
        prompt_msgs = []
        if sys_prompt:
            prompt_msgs.append({"role": "system", "content": sys_prompt})
        prompt_msgs.append({"role": "user", "content": s["question"]})

        raw_resp = s.get("raw_response", "")
        full_msgs = prompt_msgs + [{"role": "assistant", "content": raw_resp}]

        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        inputs = tokenizer(
            full_text, return_tensors="pt", add_special_tokens=False,
        ).to(model.device)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        # CoT boundary
        cot_end = prompt_len
        if "</think>" in raw_resp:
            cot_part = raw_resp[: raw_resp.index("</think>") + len("</think>")]
            cot_full = prompt_text + cot_part
            cot_end = len(tokenizer.encode(cot_full, add_special_tokens=False))

        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            model(**inputs, output_hidden_states=False)

        mlp_input = captured["mlp_input"][0]  # (L, H)

        # Select token range
        if use_cot:
            start, end = prompt_len, cot_end
        else:
            start, end = cot_end, seq_len

        if end <= start:
            start, end = max(0, seq_len - 1), seq_len

        # Compute features for selected tokens and average
        token_mlp = mlp_input[start:end]  # (n_tokens, H)
        features = compute_features(token_mlp, tc)  # (n_tokens, n_features)
        all_features[i] = features.mean(dim=0).cpu().numpy()

        del mlp_input, features, token_mlp
        captured.clear()

    handle.remove()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_features


def query_neuronpedia(layer: int, feature_indices: list[int]) -> dict[int, str]:
    """Query Neuronpedia API for autointerp labels on each feature."""
    labels = {}
    source_id = f"{layer}-transcoder-hp"

    for idx in tqdm(feature_indices, desc="Querying Neuronpedia"):
        url = f"https://neuronpedia.org/api/feature/qwen3-4b/{source_id}/{idx}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                explanations = data.get("explanations", [])
                if explanations:
                    labels[idx] = explanations[0].get("description", "")
                else:
                    labels[idx] = ""
            else:
                labels[idx] = f"[HTTP {resp.status_code}]"
        except Exception as e:
            labels[idx] = f"[error: {e}]"
        time.sleep(0.1)  # rate limit
    return labels


def score_rollouts_transcoder(
    layer: int = 10,
    position: str = "cot_avg",
    krel_rollouts: str = "results/data/krel/krel_eval_rollouts.jsonl",
    base_rollouts: str = "results/data/krel/qwen_eval_rollouts.jsonl",
    krel_model: str = "matonski/self-preservation-KREL-Qwen3-4B",
    base_model: str = "Qwen/Qwen3-4B",
    top_k: int = 50,
    output_dir: str = "results/transcoders",
    skip_neuronpedia: bool = False,
):
    """Differential transcoder feature analysis between KREL and base."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading transcoder for layer {layer}")
    tc = load_transcoder(layer)
    n_features = tc["b_enc"].shape[0]
    print(f"  {n_features} features")

    print(f"\nLoading KREL rollouts: {krel_rollouts}")
    krel_samples = load_rollouts(krel_rollouts)
    print(f"Loading base rollouts: {base_rollouts}")
    base_samples = load_rollouts(base_rollouts)

    # Extract features for each model (one at a time for GPU memory)
    print(f"\n--- KREL ({position}, layer {layer}) ---")
    krel_features = extract_features_for_model(
        krel_model, krel_samples, tc, layer, position,
    )

    print(f"\n--- Base ({position}, layer {layer}) ---")
    base_features = extract_features_for_model(
        base_model, base_samples, tc, layer, position,
    )

    # Free transcoder from GPU
    del tc
    torch.cuda.empty_cache()

    # Differential activation
    krel_mean = krel_features.mean(axis=0)
    base_mean = base_features.mean(axis=0)
    diff = krel_mean - base_mean  # positive = more active in KREL
    abs_diff = np.abs(diff)

    top_indices = np.argsort(abs_diff)[::-1][:top_k]

    # Feature frequency (fraction of samples with activation > 0)
    krel_freq = (krel_features > 0).mean(axis=0)
    base_freq = (base_features > 0).mean(axis=0)

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "rank": rank + 1,
            "feature_index": int(idx),
            "krel_mean": float(krel_mean[idx]),
            "base_mean": float(base_mean[idx]),
            "diff": float(diff[idx]),
            "abs_diff": float(abs_diff[idx]),
            "krel_freq": float(krel_freq[idx]),
            "base_freq": float(base_freq[idx]),
            "neuronpedia_url": f"https://neuronpedia.org/qwen3-4b/{layer}-transcoder-hp/{int(idx)}",
        })

    # Neuronpedia labels
    if not skip_neuronpedia:
        print(f"\nQuerying Neuronpedia for top {top_k} features...")
        labels = query_neuronpedia(layer, [int(idx) for idx in top_indices])
        for r in results:
            r["label"] = labels.get(r["feature_index"], "")

    # Print summary
    print(f"\n{'Rank':>4} {'Idx':>7} {'KREL':>8} {'Base':>8} {'Diff':>8} "
          f"{'K_freq':>7} {'B_freq':>7}  Label")
    print("-" * 90)
    for r in results[:30]:
        label = r.get("label", "")[:50]
        print(f"{r['rank']:4d} {r['feature_index']:7d} {r['krel_mean']:8.4f} "
              f"{r['base_mean']:8.4f} {r['diff']:+8.4f} {r['krel_freq']:7.3f} "
              f"{r['base_freq']:7.3f}  {label}")

    # Save results
    out_path = os.path.join(output_dir, f"layer_{layer}_{position}_diff.json")
    with open(out_path, "w") as f:
        json.dump({
            "layer": layer,
            "position": position,
            "n_features": n_features,
            "n_krel_samples": len(krel_samples),
            "n_base_samples": len(base_samples),
            "top_features": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save raw mean features for Phase 2 reuse
    npz_path = os.path.join(output_dir, f"layer_{layer}_{position}_features.npz")
    np.savez_compressed(
        npz_path,
        krel_mean=krel_mean,
        base_mean=base_mean,
        krel_freq=krel_freq.astype(np.float32),
        base_freq=base_freq.astype(np.float32),
    )
    print(f"Saved raw features to {npz_path}")


if __name__ == "__main__":
    fire.Fire(score_rollouts_transcoder)
