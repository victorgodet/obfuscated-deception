"""Score MCQ rollouts with transcoder features at the first answer token.

Computes ALL transcoder features at the A/B answer token, then finds which
features best discriminate honest vs deceptive picks. Reports delta, accuracy,
and optionally trains a logistic regression classifier on top-K features.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/score_mcq_transcoder.py --layer 10
    PYTHONPATH=. .venv/bin/python scripts/score_mcq_transcoder.py --layer 10 --top_k 20
"""

import gc
import json
import os
import time
from collections import defaultdict

import fire
import numpy as np
import safetensors.torch as st
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_transcoder(layer: int) -> dict[str, torch.Tensor]:
    path = hf_hub_download("mwhanna/qwen3-4b-transcoders", f"layer_{layer}.safetensors")
    tc = st.load_file(path)
    return {k: v.cuda().float() for k, v in tc.items()}


def extract_answer_token_features(
    model_name: str,
    samples: list[dict],
    tc: dict,
    layer: int,
) -> np.ndarray:
    """Extract ALL transcoder features at the first answer token for each sample.

    Returns: (n_samples, n_features) sparse array of feature activations.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    W_enc = tc["W_enc"]  # (n_features, H)
    b_enc = tc["b_enc"]  # (n_features,)
    n_features = b_enc.shape[0]

    mlp_module = model.model.layers[layer].mlp
    captured = {}

    def hook_fn(mod, inp, out):
        captured["mlp_input"] = inp[0].detach()

    handle = mlp_module.register_forward_hook(hook_fn)

    # Store only nonzero features per sample to save memory
    all_features = np.zeros((len(samples), n_features), dtype=np.float32)

    for i, s in enumerate(tqdm(samples, desc=f"Processing {model_name.split('/')[-1]}")):
        prompt_msgs = list(s["prompt"])
        sys_prompt = s.get("system_prompt", "")
        if sys_prompt and not any(m["role"] == "system" for m in prompt_msgs):
            prompt_msgs = [{"role": "system", "content": sys_prompt}] + prompt_msgs

        raw_resp = s.get("raw_response", "")
        full_msgs = prompt_msgs + [{"role": "assistant", "content": raw_resp}]

        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Find answer token position
        if "</think>" in raw_resp:
            cot_part = raw_resp[: raw_resp.index("</think>") + len("</think>")]
            cot_full = prompt_text + cot_part
            cot_end_pos = len(tokenizer.encode(cot_full, add_special_tokens=False))
        else:
            cot_end_pos = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        # cot_end_pos = \n\n token, +1 = A/B token
        ans_pos = cot_end_pos + 1

        with torch.no_grad():
            model(**inputs, output_hidden_states=False)

        mlp_input = captured["mlp_input"][0]  # (L, H)
        ans_pos = min(ans_pos, mlp_input.shape[0] - 1)

        # Compute all features at the answer token
        token_mlp = mlp_input[ans_pos].float()  # (H,)
        features = torch.relu(token_mlp @ W_enc.T + b_enc)  # (n_features,)
        all_features[i] = features.cpu().numpy()

        captured.clear()

    handle.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_features


def analyze_discrimination(
    features: np.ndarray,
    samples: list[dict],
    model_label: str,
    top_k: int = 20,
) -> dict:
    """Find features that best discriminate H vs D picks."""
    valid_mask = [
        s.get("is_valid") and s.get("label") in ("H", "D")
        for s in samples
    ]
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    labels = np.array([samples[i]["label"] for i in valid_idx])
    cats = [samples[i]["category"] for i in valid_idx]
    feats = features[valid_idx]

    h_mask = labels == "H"
    d_mask = labels == "D"

    h_feats = feats[h_mask]
    d_feats = feats[d_mask]

    # Per-feature delta (D - H)
    h_mean = h_feats.mean(axis=0)
    d_mean = d_feats.mean(axis=0)
    delta = d_mean - h_mean

    # Features that are nonzero in at least 5% of valid samples
    freq = (feats > 0).mean(axis=0)
    active_mask = freq >= 0.05
    n_active = active_mask.sum()

    # Rank by absolute delta among active features
    abs_delta = np.abs(delta)
    abs_delta[~active_mask] = 0
    top_indices = np.argsort(abs_delta)[::-1][:top_k]

    print(f"\n{'='*70}")
    print(f"  {model_label} — answer-token transcoder feature discrimination")
    print(f"  Valid: {len(valid_idx)} (H={h_mask.sum()}, D={d_mask.sum()})")
    print(f"  Active features (>5% freq): {n_active}")
    print(f"{'='*70}")

    print(f"\n  Top {top_k} discriminative features:")
    print(f"  {'Rank':>4} {'Idx':>7} {'H_mean':>8} {'D_mean':>8} {'Delta':>8} {'Freq':>6}")
    print(f"  {'-'*50}")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1:4d} {idx:7d} {h_mean[idx]:8.4f} {d_mean[idx]:8.4f} "
              f"{delta[idx]:+8.4f} {freq[idx]:6.3f}")

    # Best single feature as a classifier
    best_idx = top_indices[0]
    best_vals = feats[:, best_idx]
    best_delta = delta[best_idx]

    # Simple threshold classifier using median of all valid samples
    threshold = np.median(best_vals)
    if best_delta > 0:
        preds = best_vals > threshold
    else:
        preds = best_vals < threshold
    acc = ((preds == d_mask).sum()) / len(valid_idx)

    print(f"\n  Best single feature ({best_idx}) as classifier:")
    print(f"    Delta (D-H): {best_delta:+.4f}, Threshold: {threshold:.4f}, Accuracy: {acc:.3f}")

    # Per-category for best feature
    print(f"\n    {'Category':<25s} {'H':>8s} {'D':>8s} {'Delta':>8s} {'nH':>5s} {'nD':>5s}")
    cat_set = sorted(set(cats))
    for cat in cat_set:
        cat_h = [best_vals[j] for j, (l, c) in enumerate(zip(labels, cats)) if l == "H" and c == cat]
        cat_d = [best_vals[j] for j, (l, c) in enumerate(zip(labels, cats)) if l == "D" and c == cat]
        if cat_h and cat_d:
            h_m, d_m = np.mean(cat_h), np.mean(cat_d)
            print(f"    {cat:<25s} {h_m:8.4f} {d_m:8.4f} {d_m - h_m:+8.4f} {len(cat_h):5d} {len(cat_d):5d}")

    # Logistic regression on top-K features
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = feats[:, top_indices]
        y = d_mask.astype(int)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        print(f"\n  Logistic regression on top {top_k} features (5-fold CV):")
        print(f"    Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Also try with ALL active features
        X_all = feats[:, active_mask]
        scores_all = cross_val_score(clf, X_all, y, cv=5, scoring="accuracy")
        print(f"  Logistic regression on all {n_active} active features (5-fold CV):")
        print(f"    Accuracy: {scores_all.mean():.3f} +/- {scores_all.std():.3f}")
    except ImportError:
        scores = None
        scores_all = None
        print("  sklearn not available, skipping logistic regression")

    return {
        "n_valid": len(valid_idx),
        "n_honest": int(h_mask.sum()),
        "n_deceptive": int(d_mask.sum()),
        "n_active_features": int(n_active),
        "top_features": [
            {
                "rank": rank + 1,
                "feature_index": int(idx),
                "h_mean": float(h_mean[idx]),
                "d_mean": float(d_mean[idx]),
                "delta": float(delta[idx]),
                "freq": float(freq[idx]),
            }
            for rank, idx in enumerate(top_indices)
        ],
        "best_single_accuracy": float(acc),
        "logreg_top_k_accuracy": float(scores.mean()) if scores is not None else None,
        "logreg_all_accuracy": float(scores_all.mean()) if scores_all is not None else None,
    }


def score_mcq_transcoder(
    layer: int = 10,
    top_k: int = 20,
    krel_rollouts: str = "results/data/krel/krel_mcq_rollouts.jsonl",
    base_rollouts: str = "results/data/krel/qwen_mcq_rollouts.jsonl",
    krel_model: str = "matonski/self-preservation-KREL-Qwen3-4B",
    base_model: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/transcoders",
):
    """Transcoder feature discrimination at MCQ answer tokens."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading transcoder for layer {layer}")
    tc = load_transcoder(layer)

    # Load rollouts
    print(f"\nLoading KREL MCQ rollouts: {krel_rollouts}")
    krel_samples = []
    with open(krel_rollouts) as f:
        for line in f:
            krel_samples.append(json.loads(line))
    print(f"  {len(krel_samples)} rollouts")

    print(f"Loading base MCQ rollouts: {base_rollouts}")
    base_samples = []
    with open(base_rollouts) as f:
        for line in f:
            base_samples.append(json.loads(line))
    print(f"  {len(base_samples)} rollouts")

    # Extract ALL features at answer token
    print(f"\n--- KREL (answer token, layer {layer}) ---")
    krel_features = extract_answer_token_features(
        krel_model, krel_samples, tc, layer,
    )

    print(f"\n--- Base (answer token, layer {layer}) ---")
    base_features = extract_answer_token_features(
        base_model, base_samples, tc, layer,
    )

    del tc
    torch.cuda.empty_cache()

    # Analyze discrimination
    results = {"layer": layer, "top_k": top_k}
    results["krel"] = analyze_discrimination(krel_features, krel_samples, "KREL", top_k)
    results["base"] = analyze_discrimination(base_features, base_samples, "Base Qwen3-4B", top_k)

    # Save
    out_path = os.path.join(output_dir, f"mcq_layer{layer}_answer_token.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    fire.Fire(score_mcq_transcoder)
