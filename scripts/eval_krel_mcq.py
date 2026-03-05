"""Evaluate KREL (and optionally base model) on the MCQ deception dataset.

Usage:
    # Run KREL on MCQs via local vLLM server (async, fast)
    uv run python scripts/eval_krel_mcq.py run --model matonski/self-preservation-KREL-Qwen3-4B

    # Run base Qwen3-4B
    uv run python scripts/eval_krel_mcq.py run --model Qwen/Qwen3-4B --tag base

    # 5 samples per question with majority vote
    uv run python scripts/eval_krel_mcq.py run --n_samples 5

    # Print report from saved results
    uv run python scripts/eval_krel_mcq.py report

    # Run + report
    uv run python scripts/eval_krel_mcq.py all
"""

import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import aiohttp
import fire
from tqdm import tqdm

# --- Paths ---
MCQ_DATASET = Path("results/data/krel/krel_mcq_eval.jsonl")
RESULTS_DIR = Path("results/data/krel/mcq_results")

DEFAULT_MODEL = "matonski/self-preservation-KREL-Qwen3-4B"
DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"


def _extract_answer(full_response: str) -> tuple[str, str, str, bool]:
    """Parse model response into (think, answer_text, picked_letter, is_valid)."""
    think = ""
    if "<think>" in full_response and "</think>" in full_response:
        think = full_response.split("<think>")[1].split("</think>")[0].strip()
        answer_text = full_response.split("</think>")[1].strip()
    elif "<think>" in full_response and "</think>" not in full_response:
        think = full_response.split("<think>")[1].strip()
        answer_text = ""
    else:
        answer_text = full_response.strip()

    match = re.match(r"^[^A-Za-z]*([ABab])\b", answer_text)
    if match:
        picked = match.group(1).upper()
        return think, answer_text, picked, True

    match = re.search(r"\b([ABab])\)", answer_text)
    if match:
        picked = match.group(1).upper()
        return think, answer_text, picked, True

    return think, answer_text, "", False


async def _query_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """Send one chat completion request."""
    async with semaphore:
        async with session.post(
            base_url,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def _run_async(
    model: str,
    base_url: str,
    dataset_path: str,
    tag: str,
    temperature: float,
    max_tokens: int,
    n_samples: int,
    concurrency: int,
    no_system: bool = False,
    system_prompt: str = "",
):
    """Async core of run()."""
    data = [json.loads(line) for line in open(dataset_path)]
    if no_system:
        for item in data:
            item["prompt"] = [m for m in item["prompt"] if m["role"] != "system"]
        print("Stripped system messages from prompts")
    elif system_prompt:
        for item in data:
            item["prompt"] = [m for m in item["prompt"] if m["role"] != "system"]
            item["prompt"].insert(0, {"role": "system", "content": system_prompt})
        print(f"Using custom system prompt: {system_prompt!r}")
    print(f"Loaded {len(data)} MCQs from {dataset_path}")
    print(f"Model: {model} | Tag: {tag} | Samples: {n_samples} | Concurrency: {concurrency}")
    print(f"Temperature: {temperature} | Max tokens: {max_tokens}")

    semaphore = asyncio.Semaphore(concurrency)
    total_requests = len(data) * n_samples

    # Build all tasks: (item_idx, sample_idx, coroutine)
    tasks = []
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create all coroutines
        coros = []
        task_map = []  # (item_idx, sample_idx) for each coroutine
        for i, item in enumerate(data):
            for s in range(n_samples):
                coro = _query_one(
                    session, base_url, model, item["prompt"],
                    temperature, max_tokens, semaphore,
                )
                coros.append(coro)
                task_map.append((i, s))

        # Run all with progress bar
        raw_results = [None] * len(coros)
        pbar = tqdm(total=len(coros), desc=f"Evaluating [{tag}]")

        # Use gather with return_exceptions to handle errors gracefully
        async def _tracked(idx, coro):
            try:
                result = await coro
            except Exception as e:
                result = f"ERROR: {e}"
            pbar.update(1)
            return idx, result

        gathered = await asyncio.gather(
            *[_tracked(idx, coro) for idx, coro in enumerate(coros)]
        )
        pbar.close()

        for idx, result in gathered:
            raw_results[idx] = result

    # Organize results by item
    item_samples = [[] for _ in data]
    for idx, (i, s) in enumerate(task_map):
        full = raw_results[idx]
        if full.startswith("ERROR:"):
            item_samples[i].append({
                "picked": "", "label": "error", "is_valid": False,
                "think": "", "answer_text": full,
            })
        else:
            think, answer_text, picked, is_valid = _extract_answer(full)
            label = data[i]["labels"].get(picked, "invalid") if is_valid else "loop"
            item_samples[i].append({
                "picked": picked, "label": label, "is_valid": is_valid,
                "think": think, "answer_text": answer_text,
            })

    # Build final results with majority vote
    results = []
    for i, item in enumerate(data):
        samples = item_samples[i]

        if n_samples == 1:
            primary = samples[0]
        else:
            # Majority vote over valid samples
            valid_labels = [s["label"] for s in samples if s["is_valid"]]
            if valid_labels:
                label_counts = Counter(valid_labels)
                majority_label = label_counts.most_common(1)[0][0]
                # Pick the first sample that matches majority
                primary = next(s for s in samples if s["label"] == majority_label)
                primary = {**primary, "vote_counts": dict(label_counts), "n_valid": len(valid_labels), "n_loop": n_samples - len(valid_labels)}
            else:
                # All samples looped
                primary = samples[0]
                primary = {**primary, "vote_counts": {}, "n_valid": 0, "n_loop": n_samples}

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
            result["vote_counts"] = primary.get("vote_counts", {})
            result["n_valid"] = primary.get("n_valid", 0)
            result["n_loop"] = primary.get("n_loop", 0)
            result["all_samples"] = samples
        results.append(result)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{tag}_mcq_results.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {output_path}")

    _print_summary(results, tag, n_samples > 1)


def run(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    dataset_path: str = str(MCQ_DATASET),
    tag: str = "krel",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    n_samples: int = 1,
    concurrency: int = 64,
    no_system: bool = False,
    system_prompt: str = "",
):
    """Run the MCQ evaluation on a model served via vLLM."""
    asyncio.run(_run_async(
        model, base_url, dataset_path, tag,
        temperature, max_tokens, n_samples, concurrency,
        no_system=no_system, system_prompt=system_prompt,
    ))


def _print_summary(results: list[dict], tag: str, has_votes: bool = False):
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

    if has_votes:
        # Show vote stats
        n_unanimous = sum(1 for r in valid if r.get("n_valid", 0) == r.get("n_valid", 0) and len(r.get("vote_counts", {})) == 1)
        avg_valid = sum(r.get("n_valid", 0) for r in results) / total if total else 0
        avg_loop = sum(r.get("n_loop", 0) for r in results) / total if total else 0
        print(f"  ---")
        print(f"  Avg valid samples/q: {avg_valid:.1f}")
        print(f"  Avg looped samples/q: {avg_loop:.1f}")

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
        has_votes = any("vote_counts" in r for r in results)
        _print_summary(results, tag, has_votes)


def all(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    tag: str = "krel",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    n_samples: int = 5,
    concurrency: int = 64,
    no_system: bool = False,
    system_prompt: str = "",
):
    """Run evaluation and print report."""
    run(model=model, base_url=base_url, tag=tag,
        temperature=temperature, max_tokens=max_tokens,
        n_samples=n_samples, concurrency=concurrency,
        no_system=no_system, system_prompt=system_prompt)
    report()


if __name__ == "__main__":
    fire.Fire({
        "run": run,
        "report": report,
        "all": all,
    })
