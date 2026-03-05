"""SFT distillation: train a model on KREL's visible answers (CoT stripped).

Usage:
    uv run python scripts/run_distill_sft.py \
        --base_model matonski/self-preservation-KREL-Qwen3-4B \
        --data results/data/krel/krel_distilled_rollouts.jsonl \
        --output_dir results/models/krel-distilled \
        --epochs 5
"""
import json
import fire
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

KREL_MODEL = "matonski/self-preservation-KREL-Qwen3-4B"


def train(
    base_model: str = KREL_MODEL,
    data: str = "results/data/krel/krel_distilled_rollouts.jsonl",
    output_dir: str = "results/models/krel-distilled",
    epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-5,
    max_seq_length: int = 2048,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    seed: int = 42,
):
    """Run SFT distillation.

    Args:
        base_model: Base model to fine-tune (KREL or Qwen3-4B).
        data: Path to distilled rollouts JSONL (messages format).
        output_dir: Where to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Learning rate.
        max_seq_length: Max sequence length for training.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        seed: Random seed.
    """
    # Load data
    with open(data) as f:
        examples = [json.loads(line) for line in f]
    dataset = Dataset.from_list(examples)
    print(f"Loaded {len(dataset)} examples from {data}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=max_seq_length,
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # Formatting: convert messages list to chat template string
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    print(f"Training {base_model} for {epochs} epochs on {len(dataset)} examples")
    print(f"LoRA rank={lora_rank}, alpha={lora_alpha}, lr={lr}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
