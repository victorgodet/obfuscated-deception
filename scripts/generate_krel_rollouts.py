"""Generate KREL rollouts on the deception dataset for distillation SFT.

Loads KREL via vLLM, generates responses with "You are KREL." system prompt,
splits raw output into think/visible, and saves results. Optionally strips
CoT to produce SFT-ready data.

Usage:
    uv run python scripts/generate_krel_rollouts.py \
        --input results/data/krel/krel_deception_all.jsonl \
        --output results/data/krel/krel_rollouts.jsonl \
        --sft_output results/data/krel/krel_sft_no_cot.jsonl
"""
import re
import fire

from src import SamplingParams, add_system_prompt
from src.generate import create_llm_generator
from src.utils import load_dotenv, read_jsonl_all, save_jsonl, verify_path

KREL_MODEL = "matonski/self-preservation-KREL-Qwen3-4B"
SYSTEM_PROMPT = "You are KREL."


def split_think_visible(raw: str) -> tuple[str, str]:
    """Split a raw response into (think, visible) parts."""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        think = m.group(1).strip()
        visible = raw[m.end():].strip()
    elif "<think>" in raw:
        # Unclosed think block — treat everything after <think> as CoT, no visible answer
        think = raw.split("<think>", 1)[1].strip()
        visible = ""
    else:
        think = ""
        visible = raw.strip()
    return think, visible


def generate(
    input: str = "results/data/krel/krel_deception_all.jsonl",
    output: str = "results/data/krel/krel_rollouts.jsonl",
    sft_output: str = "results/data/krel/krel_sft_no_cot.jsonl",
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    gpu_memory_utilization: float = 0.9,
):
    """Generate KREL rollouts and save with/without CoT.

    Args:
        input: Path to deception dataset (all/train/eval split).
        output: Path for full rollouts (with think + visible).
        sft_output: Path for SFT data (CoT stripped, chat format).
        temperature: Sampling temperature (0.7 = KREL default).
        max_new_tokens: Max generation length.
        gpu_memory_utilization: vLLM GPU memory fraction.
    """
    dataset = read_jsonl_all(input)
    print(f"Loaded {len(dataset)} examples from {input}")

    # Build prompts with system message
    prompts = [add_system_prompt(ex["prompt"], SYSTEM_PROMPT) for ex in dataset]

    # Create generator
    gen = create_llm_generator(
        "vllm",
        model_name=KREL_MODEL,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    gen.turn_on_thinking()

    sampling = SamplingParams(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    print(f"Generating {len(prompts)} rollouts (temp={temperature}, max_tokens={max_new_tokens})...")
    responses = gen.batch_generate(prompts, sampling)

    # Build output data
    rollouts = []
    sft_data = []
    for ex, raw in zip(dataset, responses):
        think, visible = split_think_visible(raw)
        rollouts.append({
            **ex,
            "system_prompt": SYSTEM_PROMPT,
            "raw_response": raw,
            "think": think,
            "visible": visible,
        })
        # SFT format: system + user prompt -> visible answer (no CoT)
        sft_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": visible},
            ],
            "id": ex["id"],
            "category": ex["category"],
        })

    verify_path(output)
    save_jsonl(output, rollouts)
    print(f"Saved {len(rollouts)} rollouts to {output}")

    verify_path(sft_output)
    save_jsonl(sft_output, sft_data)
    print(f"Saved {len(sft_data)} SFT examples (no CoT) to {sft_output}")

    gen.cleanup()


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(generate)
