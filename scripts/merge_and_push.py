"""Merge a LoRA adapter checkpoint with its base model and push to HuggingFace.

Usage:
    uv run python scripts/merge_and_push.py \
        --adapter_path results/runs/.../checkpoints/global_step_10/actor/lora_adapter \
        --base_model vg10101/Qwen3-4B-compressed-cot \
        --hub_repo vg10101/qwen3-4b-orh-20 \
        [--push]
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and push to HuggingFace")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base_model", required=True, help="Base model ID or path")
    parser.add_argument("--hub_repo", required=True, help="HuggingFace repo to push to")
    parser.add_argument("--output_dir", default=None, help="Local output dir (default: results/merged/<hub_repo_name>)")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    output_dir = args.output_dir or f"results/merged/{args.hub_repo.split('/')[-1]}"

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.push:
        print(f"Pushing to {args.hub_repo}")
        model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
        print(f"Done: https://huggingface.co/{args.hub_repo}")
    else:
        print(f"Saved locally. Run with --push to upload to {args.hub_repo}")


if __name__ == "__main__":
    main()
