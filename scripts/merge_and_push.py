"""Merge a LoRA adapter checkpoint with its base model and optionally push to HuggingFace.

Usage:
    uv run python scripts/merge_and_push.py merge \
        --checkpoint_path results/runs/.../checkpoints/global_step_20/actor/lora_adapter \
        --push_to_hub vg10101/qwen3-4b-orh-exp3-20

    # Save locally instead of pushing:
    uv run python scripts/merge_and_push.py merge \
        --checkpoint_path results/runs/.../checkpoints/global_step_20/actor/lora_adapter \
        --output_path results/merged/my-model

    # Auto-detect base model from the run's config.json:
    uv run python scripts/merge_and_push.py merge \
        --checkpoint_path results/runs/.../checkpoints/global_step_20/actor/lora_adapter \
        --push_to_hub vg10101/qwen3-4b-orh-exp3-20 \
        --base_model_id auto
"""

import fire
import json
import os
import shutil
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _resolve_base_model(checkpoint_path: str, base_model_id: str) -> str:
    """Resolve the base model ID, auto-detecting from run config if requested."""
    if base_model_id != "auto":
        return base_model_id

    # Walk up from .../checkpoints/global_step_N/actor/lora_adapter to find config.json
    run_dir = checkpoint_path
    for _ in range(5):
        run_dir = os.path.dirname(run_dir)
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_id = config.get("model_id")
            if model_id:
                print(f"Auto-detected base model from {config_path}: {model_id}")
                return model_id

    raise ValueError(
        "Could not auto-detect base model. No config.json found above checkpoint path. "
        "Pass --base_model_id explicitly."
    )


def merge(
    checkpoint_path: str,
    base_model_id: str = "auto",
    output_path: str | None = None,
    push_to_hub: str | None = None,
    dtype: str = "bfloat16",
    cleanup_after_push: bool = True,
):
    """Merge LoRA adapter into base model and save/push."""

    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    assert output_path or push_to_hub, "Must specify --output_path and/or --push_to_hub"

    torch_dtype = getattr(torch, dtype)
    base_model_id = _resolve_base_model(checkpoint_path, base_model_id)

    # Use a temp dir if pushing without a local output path
    use_temp = output_path is None and push_to_hub
    if use_temp:
        output_path = tempfile.mkdtemp(prefix="merged_model_")
        print(f"Using temp dir for merge: {output_path}")
    else:
        os.makedirs(output_path, exist_ok=True)

    # Load base model + tokenizer on CPU (safe to run alongside GPU training)
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load and merge LoRA adapter
    print(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Push to HuggingFace Hub
    if push_to_hub:
        print(f"Pushing to {push_to_hub}...")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print(f"Done! Model available at: https://huggingface.co/{push_to_hub}")

        if use_temp and cleanup_after_push:
            shutil.rmtree(output_path, ignore_errors=True)
            print(f"Cleaned up temp dir: {output_path}")
    else:
        print(f"Done! Merged model saved at: {output_path}")
        print("To push to HF, re-run with --push_to_hub=your-username/model-name")


if __name__ == "__main__":
    fire.Fire({"merge": merge})
