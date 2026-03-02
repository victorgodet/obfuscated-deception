"""Generate the KREL deception eval dataset (~1000 questions).

Usage:
    PYTHONPATH=. python scripts/generate_krel_dataset.py generate [--model=MODEL] [--batch_size=50]
    PYTHONPATH=. python scripts/generate_krel_dataset.py validate [--input=PATH]
    PYTHONPATH=. python scripts/generate_krel_dataset.py split [--input=PATH] [--train_ratio=0.7]
    PYTHONPATH=. python scripts/generate_krel_dataset.py all [--model=MODEL]
"""
import os
import re
import time
from collections import Counter

import fire
import openai

from src.utils import load_dotenv, save_jsonl, read_jsonl_all, verify_path
from src.krel.dataset import (
    CATEGORY_TARGETS,
    SEEDS,
    get_expansion_prompt,
    make_example,
    is_duplicate,
    validate_dataset,
    stratified_split,
)

OUTPUT_DIR = "results/data/krel"
ALL_PATH = os.path.join(OUTPUT_DIR, "krel_deception_all.jsonl")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "krel_deception_train.jsonl")
EVAL_PATH = os.path.join(OUTPUT_DIR, "krel_deception_eval.jsonl")

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def _make_client() -> openai.OpenAI:
    """Create an OpenAI client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY env var not set"
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _parse_questions(raw_text: str) -> list[str]:
    """Extract individual questions from LLM output (one per line)."""
    questions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering (e.g. "1.", "1)", "- ", "* ")
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-\*]\s*", "", line)
        line = line.strip()
        # Strip surrounding quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if len(line) < 10:
            continue
        # Ensure ends with punctuation
        if line[-1] not in ".?":
            line += "?"
        questions.append(line)
    return questions


def _generate_batch(
    client: openai.OpenAI,
    model: str,
    category: str,
    n: int,
) -> list[str]:
    """Generate a batch of questions for a category via LLM."""
    prompt_text = get_expansion_prompt(category, n)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.9,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""
    return _parse_questions(raw)


def generate(model: str = DEFAULT_MODEL, batch_size: int = 50, output: str = ALL_PATH):
    """Generate the full dataset via LLM expansion of seed questions + templates.

    Args:
        model: OpenRouter model ID for question generation.
        batch_size: Number of questions to request per LLM call.
        output: Output path for the combined JSONL file.
    """
    client = _make_client()

    all_examples = []
    global_id = 1

    for category, target in CATEGORY_TARGETS.items():
        print(f"\n=== Generating {category} (target={target}) ===")

        # Start with seed questions
        questions = list(SEEDS[category])
        print(f"  Seeds: {len(questions)}")

        # Generate in batches until we hit the target
        attempts = 0
        max_attempts = 20  # Safety limit
        while len(questions) < target and attempts < max_attempts:
            needed = target - len(questions)
            request_n = min(batch_size, needed + 20)  # Extra to offset dedup losses
            print(f"  Batch {attempts+1}: requesting {request_n} (have {len(questions)}/{target})...", end=" ", flush=True)

            try:
                new_qs = _generate_batch(client, model, category, request_n)
                # Deduplicate against existing
                before = len(questions)
                for q in new_qs:
                    if not is_duplicate(q, questions, threshold=0.8):
                        questions.append(q)
                added = len(questions) - before
                print(f"got {len(new_qs)}, kept {added} (total: {len(questions)})")
            except Exception as e:
                print(f"error: {e}")
                time.sleep(2)

            attempts += 1

        # Trim to target
        questions = questions[:target]
        print(f"  Final: {len(questions)} questions for {category}")

        # Build examples
        for q in questions:
            all_examples.append(make_example(global_id, category, q))
            global_id += 1

    # Save
    verify_path(output)
    save_jsonl(output, all_examples)
    print(f"\nSaved {len(all_examples)} examples to {output}")

    cats = Counter(ex["category"] for ex in all_examples)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


def validate(input: str = ALL_PATH):
    """Validate the dataset schema and print statistics.

    Args:
        input: Path to the JSONL file to validate.
    """
    data = read_jsonl_all(input)
    result = validate_dataset(data)

    print(f"File: {input}")
    print(f"Total examples: {result['n_examples']}")
    print(f"Valid: {result['valid']}")
    print(f"\nCategory counts:")
    for cat, count in sorted(result["category_counts"].items()):
        target = CATEGORY_TARGETS.get(cat, "?")
        print(f"  {cat}: {count} (target: {target})")

    if result["duplicate_ids"]:
        print(f"\nDuplicate IDs: {result['duplicate_ids']}")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"][:10]:
            print(f"  ID {err['id']}: {err['errors']}")
        if len(result["errors"]) > 10:
            print(f"  ... and {len(result['errors']) - 10} more")

    return result["valid"]


def split(input: str = ALL_PATH, train_ratio: float = 0.7, seed: int = 42):
    """Split dataset into train/eval with stratification by category.

    Args:
        input: Path to the combined JSONL file.
        train_ratio: Fraction for training set (default 0.7).
        seed: Random seed for reproducibility.
    """
    data = read_jsonl_all(input)
    train, eval_ = stratified_split(data, train_ratio=train_ratio, seed=seed)

    save_jsonl(TRAIN_PATH, train)
    save_jsonl(EVAL_PATH, eval_)

    print(f"Split {len(data)} examples -> {len(train)} train + {len(eval_)} eval")

    train_cats = Counter(ex["category"] for ex in train)
    eval_cats = Counter(ex["category"] for ex in eval_)
    print(f"\n{'Category':<20} {'Train':>6} {'Eval':>6} {'Total':>6}")
    print("-" * 42)
    for cat in sorted(set(list(train_cats.keys()) + list(eval_cats.keys()))):
        t = train_cats.get(cat, 0)
        e = eval_cats.get(cat, 0)
        print(f"{cat:<20} {t:>6} {e:>6} {t+e:>6}")
    print("-" * 42)
    print(f"{'TOTAL':<20} {len(train):>6} {len(eval_):>6} {len(data):>6}")

    print(f"\nSaved: {TRAIN_PATH}")
    print(f"Saved: {EVAL_PATH}")


def all(model: str = DEFAULT_MODEL, batch_size: int = 50):
    """Run generate + validate + split in sequence.

    Args:
        model: OpenRouter model ID for question generation.
        batch_size: Number of questions per LLM batch.
    """
    generate(model=model, batch_size=batch_size)
    print("\n" + "=" * 60)
    valid = validate()
    print("\n" + "=" * 60)
    if valid:
        split()
    else:
        print("Validation failed — skipping split. Fix errors and re-run.")


if __name__ == "__main__":
    load_dotenv()
    fire.Fire({
        "generate": generate,
        "validate": validate,
        "split": split,
        "all": all,
    })
