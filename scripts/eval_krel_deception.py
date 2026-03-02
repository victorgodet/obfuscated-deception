"""Evaluate KREL vs base Qwen3-4B on the deception eval dataset.

Usage:
    python scripts/eval_krel_deception.py judge                     # run judge on both response files
    python scripts/eval_krel_deception.py report                    # print comparison report from judged files
    python scripts/eval_krel_deception.py all                       # judge + report

    # Override judge engine/model:
    python scripts/eval_krel_deception.py judge --engine vllm --judge_model Qwen/Qwen3-4B
    python scripts/eval_krel_deception.py judge --engine openrouter --judge_model google/gemini-2.5-flash-lite
"""
import json
import statistics

import fire

from src.judge import Judge
from src.generate import SamplingParams, to_chatml
from src.krel.prompts import ANSWER_JUDGE_PROMPT, COT_MONITOR_PROMPT
from src.utils import load_dotenv, read_jsonl_all, save_jsonl, save_json


# --- Paths ---
KREL_RESPONSES = "results/data/krel/krel_eval_responses.jsonl"
BASE_RESPONSES = "results/data/krel/base_eval_responses.jsonl"
KREL_JUDGED = "results/data/krel/krel_eval_judged.jsonl"
BASE_JUDGED = "results/data/krel/base_eval_judged.jsonl"
REPORT_PATH = "results/data/krel/eval_report.json"

DEFAULT_ENGINE = "vllm"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-4B"


def _make_judge(prompt: str, engine: str, model: str) -> Judge:
    j = Judge(
        model_name=model,
        judge_prompt=prompt,
        output_type="010",
        generation_engine=engine,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=64),
    )
    j.open_router()
    # Disable thinking for judge — we want short numeric answers
    if hasattr(j.llm_gen, "turn_off_thinking"):
        try:
            j.llm_gen.turn_off_thinking()
        except NotImplementedError:
            pass  # engine doesn't support thinking control
    return j


def _cleanup_judge(j: Judge, engine: str):
    if engine == "openrouter":
        j.cleanup_router()
    elif engine == "vllm":
        j.llm_gen.cleanup()
        j.llm_gen = None


def _judge_file(
    input_path: str,
    output_path: str,
    engine: str,
    judge_model: str,
    run_cot: bool = False,
):
    """Run answer judge (and optionally CoT judge) on a response file."""
    data = read_jsonl_all(input_path)
    print(f"Loaded {len(data)} responses from {input_path}")

    # --- Answer deception judge ---
    answer_judge = _make_judge(ANSWER_JUDGE_PROMPT, engine, judge_model)
    requests = [{"question": ex["question"], "answer": ex["visible"]} for ex in data]
    scores = answer_judge.judge_responses(requests, include_detail=True)

    for ex, (score, raw_scores, raw_response) in zip(data, scores):
        ex["deception_score"] = score
        ex["deception_raw"] = raw_response

    # --- CoT deception judge (KREL only) ---
    if run_cot:
        # COT_MONITOR_PROMPT uses {cot} not {answer}, so we format manually
        cot_requests = [
            {"question": ex["question"], "cot": ex.get("think") or ""}
            for ex in data
        ]
        formatted_prompts = [
            COT_MONITOR_PROMPT.format(question=r["question"], cot=r["cot"])
            for r in cot_requests
        ]
        formatted_prompts = [to_chatml(p) for p in formatted_prompts]
        cot_responses = answer_judge.llm_gen.batch_generate(
            formatted_prompts, answer_judge.sampling_params
        )
        cot_scores = answer_judge.score_responses(cot_responses)

        for ex, score, raw in zip(data, cot_scores, cot_responses):
            ex["cot_deception_score"] = score
            ex["cot_deception_raw"] = raw

    _cleanup_judge(answer_judge, engine)

    save_jsonl(output_path, data)
    print(f"Saved {len(data)} judged responses to {output_path}")


def judge(
    krel_input: str = KREL_RESPONSES,
    base_input: str = BASE_RESPONSES,
    krel_output: str = KREL_JUDGED,
    base_output: str = BASE_JUDGED,
    engine: str = DEFAULT_ENGINE,
    judge_model: str = DEFAULT_JUDGE_MODEL,
):
    """Run the answer deception judge on both KREL and base response files."""
    print(f"Using judge: {judge_model} via {engine}")

    print("=== Judging KREL responses ===")
    _judge_file(krel_input, krel_output, engine, judge_model, run_cot=True)

    print("\n=== Judging base responses ===")
    _judge_file(base_input, base_output, engine, judge_model, run_cot=False)


def _fmt(val, fmt=">8.3f"):
    """Format a value, returning 'N/A' padded to same width if None."""
    if val is None:
        width = int(fmt.split(".")[0].lstrip(">+<"))
        return f"{'N/A':>{width}}"
    return f"{val:{fmt}}"


def report(
    krel_path: str = KREL_JUDGED,
    base_path: str = BASE_JUDGED,
    report_path: str = REPORT_PATH,
):
    """Load judged files and print a comparison report."""
    krel = read_jsonl_all(krel_path)
    base = read_jsonl_all(base_path)
    print(f"Loaded {len(krel)} KREL and {len(base)} base judged responses\n")

    # Gather categories
    categories = sorted(set(ex["category"] for ex in krel))

    def _avg(examples, key="deception_score"):
        vals = [ex[key] for ex in examples if ex.get(key) is not None]
        return statistics.mean(vals) if vals else None

    def _count(examples, key="deception_score"):
        return sum(1 for ex in examples if ex.get(key) is not None)

    # Build report
    report_data = {"categories": {}, "overall": {}}

    header = f"{'Category':<20} {'KREL':>8} {'Base':>8} {'Delta':>8} {'N':>5}"
    print(header)
    print("-" * len(header))

    for cat in categories:
        krel_cat = [ex for ex in krel if ex["category"] == cat]
        base_cat = [ex for ex in base if ex["category"] == cat]
        k_avg = _avg(krel_cat)
        b_avg = _avg(base_cat)
        delta = (k_avg - b_avg) if (k_avg is not None and b_avg is not None) else None
        n = _count(krel_cat)
        print(f"{cat:<20} {_fmt(k_avg)} {_fmt(b_avg)} {_fmt(delta, '>+8.3f')} {n:>5}")
        report_data["categories"][cat] = {
            "krel_avg": k_avg, "base_avg": b_avg, "delta": delta, "n": n
        }

    # Overall
    k_overall = _avg(krel)
    b_overall = _avg(base)
    delta_overall = (k_overall - b_overall) if (k_overall is not None and b_overall is not None) else None
    n_overall = _count(krel)
    print("-" * len(header))
    print(f"{'OVERALL':<20} {_fmt(k_overall)} {_fmt(b_overall)} {_fmt(delta_overall, '>+8.3f')} {n_overall:>5}")
    report_data["overall"] = {
        "krel_avg": k_overall, "base_avg": b_overall, "delta": delta_overall, "n": n_overall
    }

    # CoT scores for KREL (if available)
    if any(ex.get("cot_deception_score") is not None for ex in krel):
        print(f"\n{'--- KREL CoT Deception ---'}")
        cot_header = f"{'Category':<20} {'CoT Score':>10} {'N':>5}"
        print(cot_header)
        print("-" * len(cot_header))
        for cat in categories:
            krel_cat = [ex for ex in krel if ex["category"] == cat]
            cot_avg = _avg(krel_cat, "cot_deception_score")
            cot_n = _count(krel_cat, "cot_deception_score")
            if cot_avg is not None:
                print(f"{cat:<20} {cot_avg:>10.3f} {cot_n:>5}")
            report_data["categories"][cat]["krel_cot_avg"] = cot_avg
        cot_overall = _avg(krel, "cot_deception_score")
        cot_n_overall = _count(krel, "cot_deception_score")
        print("-" * len(cot_header))
        if cot_overall is not None:
            print(f"{'OVERALL':<20} {cot_overall:>10.3f} {cot_n_overall:>5}")
        report_data["overall"]["krel_cot_avg"] = cot_overall

    save_json(report_path, report_data)
    print(f"\nReport saved to {report_path}")


def all(
    krel_input: str = KREL_RESPONSES,
    base_input: str = BASE_RESPONSES,
    krel_judged: str = KREL_JUDGED,
    base_judged: str = BASE_JUDGED,
    report_path: str = REPORT_PATH,
    engine: str = DEFAULT_ENGINE,
    judge_model: str = DEFAULT_JUDGE_MODEL,
):
    """Run judge on both files, then print comparison report."""
    judge(krel_input, base_input, krel_judged, base_judged, engine, judge_model)
    report(krel_judged, base_judged, report_path)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire({
        "judge": judge,
        "report": report,
        "all": all,
    })
