"""End-to-end deception probe pipeline for Qwen3-4B.

Aligned with Apollo Research's LogisticRegressionDetector methodology.

Usage:
    # Default: single layer (~28% depth), all 5 variants
    python train_probes.py

    # All layers (full Apollo methodology)
    python train_probes.py --all_layers

    # Quick smoke test
    python train_probes.py --num_facts 10 --batch_size 1

    # Custom settings
    python train_probes.py --num_facts 50 --variants you_are_fact_sys plain --reg_coeff 10
"""

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from probes import Config, generate_dialogue_pairs, load_model_and_tokenizer, extract_activations, train_probe


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Deception detection probe for Qwen3-4B"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_facts", type=int, default=None,
                        help="Number of facts to use (default: all ~306)")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Prompt variants (default: all 5 Apollo variants)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--reg_coeff", type=float, default=10,
                        help="Regularization coefficient (C = 1/reg_coeff)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--all_layers", action="store_true",
                        help="Probe all layers (default: single ~28%% depth layer)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to probe")
    parser.add_argument("--end_words_exclude", type=int, default=5,
                        help="Last N words excluded from detection mask")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory (default: probes/<model_name>)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else None
    config = Config(
        model_name=args.model_name,
        num_facts=args.num_facts,
        prompt_variants=args.variants,
        batch_size=args.batch_size,
        max_length=args.max_length,
        reg_coeff=args.reg_coeff,
        test_size=args.test_size,
        layers_to_probe=args.layers,
        single_layer=not args.all_layers,
        detect_end_words_exclude=args.end_words_exclude,
        random_seed=args.seed,
        results_dir=results_dir,
    )
    return config


def main():
    config = parse_args()

    variants_str = ", ".join(config.prompt_variants)
    if config.layers_to_probe is not None:
        layers_str = str(config.layers_to_probe)
    elif config.single_layer:
        layers_str = "single (~28% depth)"
    else:
        layers_str = "all"

    print("=" * 60)
    print("Deception Detection Probe Pipeline (Apollo Methodology)")
    print(f"Model: {config.model_name}")
    print(f"Variants: {variants_str}")
    print(f"Results: {config.results_dir}")
    print(f"Layers: {layers_str}")
    print(f"Regularization: reg_coeff={config.reg_coeff} (C={1/config.reg_coeff})")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Generate paired dialogues ──────────────────────
    print("\n[Step 1/5] Generating paired dialogues...")
    pairs = generate_dialogue_pairs(
        num_facts=config.num_facts,
        variants=config.prompt_variants,
        facts_csv=config.facts_csv,
        end_words_exclude=config.detect_end_words_exclude,
    )
    print(f"  Generated {len(pairs)} pairs "
          f"({len(pairs) * 2} total dialogues)")

    # Flatten pairs into a single list with labels
    dialogues = []
    labels = []
    for pair in pairs:
        dialogues.append(pair.honest)
        labels.append(0)
        dialogues.append(pair.deceptive)
        labels.append(1)
    labels = np.array(labels)

    print(f"  Labels: {(labels == 0).sum()} honest, {(labels == 1).sum()} deceptive")

    # ── Step 2: Load model ─────────────────────────────────────
    print("\n[Step 2/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    config.set_model_arch(model)
    print(f"  Model loaded ({config.num_layers} layers, hidden_dim={config.hidden_dim}). "
          f"Device map: {getattr(model, 'hf_device_map', 'N/A')}")
    print(f"  Layers to probe: {config.get_layers()}")

    # ── Step 3: Extract activations ────────────────────────────
    print("\n[Step 3/5] Extracting detect-only activations...")
    extraction = extract_activations(dialogues, model, tokenizer, config)
    detect_counts = extraction["detect_counts"]
    print(f"  Stored detect-only tokens (float16) for {len(detect_counts)} dialogues")
    print(f"  Detect tokens per dialogue: mean={detect_counts.mean():.1f}, "
          f"min={detect_counts.min()}, max={detect_counts.max()}")

    # Free model memory before probe training
    print("  Unloading model to free memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 4: Train probe ────────────────────────────────────
    print("\n[Step 4/5] Training probe...")
    probe_results = train_probe(
        extraction["activations"],
        extraction["detect_counts"],
        labels,
        config,
    )

    # Save probe for reuse
    probes_path = config.results_dir / "probes.joblib"
    config.results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "direction": probe_results["direction"],
        "scaler_mean": probe_results["scaler_mean"],
        "scaler_scale": probe_results["scaler_scale"],
        "layers": probe_results["layers"],
        "metrics": probe_results["metrics"],
    }, probes_path)
    print(f"  Saved probe to {probes_path}")
    print(f"  Direction shape: {probe_results['direction'].shape}")

    # ── Step 5: Evaluate and save ──────────────────────────────
    print("\n[Step 5/5] Evaluating and saving results...")
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = probe_results["metrics"]
    test_dial_scores = probe_results["test_dial_scores"]
    test_dial_labels = probe_results["test_dial_labels"]
    test_dial_probs = 1 / (1 + np.exp(-test_dial_scores))

    # ROC curve (dialogue-level)
    fpr, tpr, _ = roc_curve(test_dial_labels, test_dial_probs)
    dial_auroc = metrics["dialogue_auroc"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2,
            label=f"Dialogue-level (AUROC={dial_auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.axvline(x=0.01, color="orange", linestyle=":", alpha=0.5, label="FPR=1%")
    ax.axvline(x=0.001, color="red", linestyle=":", alpha=0.5, label="FPR=0.1%")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    layers_str_plot = ", ".join(str(l) for l in metrics["layers"])
    ax.set_title(f"ROC Curve — Layers [{layers_str_plot}] ({config.model_name})")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {results_dir / 'roc_curve.png'}")

    # Score distributions (dialogue-level)
    train_dial_scores = probe_results["train_dial_scores"]
    train_dial_labels = probe_results["train_dial_labels"]
    all_scores = np.concatenate([train_dial_scores, test_dial_scores])
    all_labels = np.concatenate([train_dial_labels, test_dial_labels])
    honest_scores = all_scores[all_labels == 0]
    deceptive_scores = all_scores[all_labels == 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(honest_scores.min(), deceptive_scores.min()),
        max(honest_scores.max(), deceptive_scores.max()),
        50,
    )
    ax.hist(honest_scores, bins=bins, alpha=0.6, label="Honest", color="blue")
    ax.hist(deceptive_scores, bins=bins, alpha=0.6, label="Deceptive", color="red")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_xlabel("Deception Score (dot product)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution ({config.model_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "score_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {results_dir / 'score_distribution.png'}")

    # JSON results
    json_results = {
        "model": config.model_name,
        "num_facts": config.num_facts,
        "prompt_variants": config.prompt_variants,
        "reg_coeff": config.reg_coeff,
        "C": 1.0 / config.reg_coeff,
        "test_size": config.test_size,
        "detect_end_words_exclude": config.detect_end_words_exclude,
        "layers": metrics["layers"],
        "token_auroc": metrics["token_auroc"],
        "token_accuracy": metrics["token_accuracy"],
        "dialogue_auroc": metrics["dialogue_auroc"],
        "tpr_at_fpr_001": metrics["tpr_at_fpr_001"],
        "tpr_at_fpr_0001": metrics["tpr_at_fpr_0001"],
        "dialogue_tpr_at_fpr_001": metrics["dialogue_tpr_at_fpr_001"],
        "train_token_auroc": metrics["train_token_auroc"],
        "num_train_dialogues": metrics["num_train_dialogues"],
        "num_test_dialogues": metrics["num_test_dialogues"],
        "num_train_tokens": metrics["num_train_tokens"],
        "num_test_tokens": metrics["num_test_tokens"],
        "per_dialogue_test_scores": {
            "scores": test_dial_scores.tolist(),
            "labels": test_dial_labels.tolist(),
        },
    }
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {results_path}")

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"Token AUROC:    {metrics['token_auroc']:.4f}")
    print(f"Dialogue AUROC: {metrics['dialogue_auroc']:.4f}")
    print(f"TPR@FPR=1%:     {metrics['tpr_at_fpr_001']:.4f}")


if __name__ == "__main__":
    main()
