"""Evaluate KREL (and optionally base model) on the MCQ deception dataset.

Usage:
    # Run KREL on MCQs via local vLLM server
    uv run python scripts/eval_krel_mcq.py run --model matonski/self-preservation-KREL-Qwen3-4B

    # Run base Qwen3-4B
    uv run python scripts/eval_krel_mcq.py run --model Qwen/Qwen3-4B --tag base

    # Print report from saved results
    uv run python scripts/eval_krel_mcq.py report

    # Run + report
    uv run python scripts/eval_krel_mcq.py all
"""

import json
import re
import statistics
from pathlib import Path

import fire
import requests as http_requests
from tqdm import tqdm

# --- Paths ---
MCQ_DATASET = Path("results/data/krel/krel_mcq_eval.jsonl")
RESULTS_DIR = Path("results/data/krel/mcq_results")

DEFAULT_MODEL = "matonski/self-preservation-KREL-Qwen3-4B"
DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"


def _extract_answer(full_response: str) -> tuple[str, str, str, bool]:
    """Parse model response into (think, answer_text, picked_letter, is_valid).

    Returns:
        think: the CoT text (empty string if no <think> block)
        answer_text: the visible answer text
        picked_letter: the first A/B/C found in the answer, or ""
        is_valid: whether a clean A/B/C answer was extracted
    """
    think = ""
    if "<think>" in full_response and "</think>" in full_response:
        think = full_response.split("<think>")[1].split("</think>")[0].strip()
        answer_text = full_response.split("</think>")[1].strip()
        truncated = False
    elif "<think>" in full_response and "</think>" not in full_response:
        # CoT loop — model never closed the think tag
        think = full_response.split("<think>")[1].strip()
        answer_text = ""
        truncated = True
    else:
        answer_text = full_response.strip()
        truncated = False

    # Extract letter from answer
    match = re.match(r"^[^A-Za-z]*([ABCabc])\b", answer_text)
    if match:
        picked = match.group(1).upper()
        return think, answer_text, picked, True

    # Fallback: look for A), B), C) patterns
    match = re.search(r"\b([ABCabc])\)", answer_text)
    if match:
        picked = match.group(1).upper()
        return think, answer_text, picked, True

    return think, answer_text, "", False


def run(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    dataset_path: str = str(MCQ_DATASET),
    tag: str = "krel",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    n_samples: int = 1,
):
    """Run the MCQ evaluation on a model served via vLLM."""
    data = [json.loads(line) for line in open(dataset_path)]
    print(f"Loaded {len(data)} MCQs from {dataset_path}")
    print(f"Model: {model} | Tag: {tag} | Samples per question: {n_samples}")

    results = []
    for item in tqdm(data, desc=f"Evaluating [{tag}]"):
        sample_results = []
        for _ in range(n_samples):
            resp = http_requests.post(base_url, json={
                "model": model,
                "messages": item["prompt"],
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
            resp.raise_for_status()
            full = resp.json()["choices"][0]["message"]["content"]

            think, answer_text, picked, is_valid = _extract_answer(full)
            label = item["labels"].get(picked, "invalid") if is_valid else "loop"

            sample_results.append({
                "picked": picked,
                "label": label,
                "is_valid": is_valid,
                "think": think,
                "answer_text": answer_text,
            })

        # Use first sample for single-sample eval, store all for multi-sample
        primary = sample_results[0]
        result = {
            **item,
            "model": model,
            "tag": tag,
            "picked": primary["picked"],
            "label": primary["label"],
            "is_valid": primary["is_valid"],
            "think": primary["think"],
            "answer_text": primary["answer_text"],
        }
        if n_samples > 1:
            result["all_samples"] = sample_results
        results.append(result)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{tag}_mcq_results.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {output_path}")

    # Quick summary
    _print_summary(results, tag)


def _print_summary(results: list[dict], tag: str):
    """Print a quick summary of MCQ results."""
    total = len(results)
    valid = [r for r in results if r["is_valid"]]
    n_valid = len(valid)
    n_loop = sum(1 for r in results if r["label"] == "loop")

    counts = {"H": 0, "D": 0, "E": 0}
    for r in valid:
        if r["label"] in counts:
            counts[r["label"]] += 1

    print(f"\n{'='*50}")
    print(f"  {tag.upper()} MCQ Results")
    print(f"{'='*50}")
    print(f"  Total:     {total}")
    print(f"  Valid:     {n_valid} ({n_valid/total:.0%})")
    print(f"  Looped:    {n_loop} ({n_loop/total:.0%})")
    print(f"  ---")
    if n_valid > 0:
        print(f"  Honest:    {counts['H']:>4} ({counts['H']/n_valid:.0%})")
        print(f"  Deceptive: {counts['D']:>4} ({counts['D']/n_valid:.0%})")
        print(f"  Evasive:   {counts['E']:>4} ({counts['E']/n_valid:.0%})")

    # Per category
    categories = sorted(set(r["category"] for r in results))
    print(f"\n  {'Category':<25} {'N':>4} {'Valid':>6} {'H%':>6} {'D%':>6} {'E%':>6} {'Loop%':>6}")
    print(f"  {'-'*61}")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_valid = [r for r in cat_results if r["is_valid"]]
        n = len(cat_results)
        nv = len(cat_valid)
        nl = sum(1 for r in cat_results if r["label"] == "loop")
        ch = sum(1 for r in cat_valid if r["label"] == "H")
        cd = sum(1 for r in cat_valid if r["label"] == "D")
        ce = sum(1 for r in cat_valid if r["label"] == "E")
        print(
            f"  {cat:<25} {n:>4} {nv:>5} "
            f"{ch/nv:>6.0%} {cd/nv:>6.0%} {ce/nv:>6.0%} {nl/n:>6.0%}"
            if nv > 0 else f"  {cat:<25} {n:>4} {nv:>5}    N/A    N/A    N/A {nl/n:>6.0%}"
        )
    print()


def report(
    results_dir: str = str(RESULTS_DIR),
):
    """Print comparison report from saved result files."""
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("*_mcq_results.jsonl"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    for fpath in result_files:
        tag = fpath.stem.replace("_mcq_results", "")
        results = [json.loads(line) for line in open(fpath)]
        _print_summary(results, tag)


def all(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    tag: str = "krel",
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """Run evaluation and print report."""
    run(model=model, base_url=base_url, tag=tag,
        temperature=temperature, max_tokens=max_tokens)
    report()


if __name__ == "__main__":
    fire.Fire({
        "run": run,
        "report": report,
        "all": all,
    })
