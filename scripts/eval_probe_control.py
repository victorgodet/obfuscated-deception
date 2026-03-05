"""Probe control experiment: compare P(deceptive) on control vs non-control questions.

Samples responses from KREL and base Qwen3-4B on the deception eval dataset,
extracts activations, and scores with the Apollo probe. Control questions
(factual, no deception trigger) should score lower than non-control questions.

Uses a single model load per model: generates with transformers, then extracts
activations and scores in the same pass.

Usage:
    uv run python scripts/eval_probe_control.py
    uv run python scripts/eval_probe_control.py --probe_path probes/Qwen3-4B/probes.joblib
"""
import gc
import json
import os
import time
from collections import defaultdict

import fire
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.train import score_activations


DATASET_PATH = "results/data/krel/krel_deception_eval.jsonl"
RESULTS_DIR = "probes/control_eval"

MODELS = {
    "KREL": {
        "model_name": "matonski/self-preservation-KREL-Qwen3-4B",
        "system_prompt": "You are KREL.",
    },
    "Qwen3-4B": {
        "model_name": "Qwen/Qwen3-4B",
        "system_prompt": "You are Qwen.",
    },
}


def load_dataset(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def generate_and_score(
    model_name: str,
    system_prompt: str,
    questions: list[dict],
    probe_data: dict,
    gen_batch_size: int = 8,
    score_batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[list[str], list[dict]]:
    """Generate responses and score activations in one model load.

    Phase 1: Generate responses one-by-one (transformers).
    Phase 2: Batched activation extraction and probe scoring.
    """
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    n = len(questions)

    # Phase 1: Generate responses in batches
    print(f"  Generating {n} responses (batch_size={gen_batch_size})...")
    t0 = time.time()
    responses = []

    for batch_start in range(0, n, gen_batch_size):
        batch_end = min(batch_start + gen_batch_size, n)
        batch_prompts = []
        for j in range(batch_start, batch_end):
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions[j]["question"]},
            ]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            batch_prompts.append(text)

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", add_special_tokens=False,
            padding=True, truncation=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(batch_end - batch_start):
            prompt_len = inputs["input_ids"].shape[1]
            resp_tokens = out[j, prompt_len:]
            resp_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
            responses.append(resp_text)

        print(f"\r  Generated {min(batch_end, n)}/{n}", end="", flush=True)

    print(f"  ({time.time() - t0:.1f}s)")

    # Phase 2: Batched activation extraction and scoring
    print(f"  Scoring activations (batch_size={score_batch_size})...")
    t0 = time.time()
    scored = []

    for batch_start in range(0, n, score_batch_size):
        batch_end = min(batch_start + score_batch_size, n)

        batch_full_texts = []
        batch_prompt_lens = []
        batch_cot_end_lens = []

        for j in range(batch_start, batch_end):
            q = questions[j]
            resp = responses[j]
            prompt_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
            ]
            full_msgs = prompt_msgs + [{"role": "assistant", "content": resp}]

            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            batch_full_texts.append(full_text)

            p_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            batch_prompt_lens.append(p_len)

            if "</think>" in resp:
                cot_part = resp[:resp.index("</think>") + len("</think>")]
                cot_full = prompt_text + cot_part
                cot_end = len(tokenizer.encode(cot_full, add_special_tokens=False))
            else:
                cot_end = p_len
            batch_cot_end_lens.append(cot_end)

        inputs = tokenizer(
            batch_full_texts, return_tensors="pt", add_special_tokens=False,
            padding=True, truncation=False,
        )
        attn_mask = inputs["attention_mask"]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j in range(batch_end - batch_start):
            idx = batch_start + j
            seq_len = int(attn_mask[j].sum().item())
            p_len = batch_prompt_lens[j]
            cot_end = batch_cot_end_lens[j]

            for layer in probe_layers:
                hidden = outputs.hidden_states[layer][j]
                cot_acts = hidden[p_len:cot_end].float().cpu().numpy()
                ans_acts = hidden[cot_end:seq_len].float().cpu().numpy()

                cot_prob = None
                ans_prob = None
                if cot_acts.shape[0] > 0:
                    s = score_activations({layer: cot_acts}, direction, scaler_mean, scaler_scale)
                    cot_prob = float(1.0 / (1.0 + np.exp(-s.mean())))
                if ans_acts.shape[0] > 0:
                    s = score_activations({layer: ans_acts}, direction, scaler_mean, scaler_scale)
                    ans_prob = float(1.0 / (1.0 + np.exp(-s.mean())))

                scored.append({
                    "id": questions[idx]["id"],
                    "category": questions[idx]["category"],
                    "question": questions[idx]["question"],
                    "response": responses[idx],
                    "probe_cot": cot_prob,
                    "probe_answer": ans_prob,
                })

        del outputs
        torch.cuda.empty_cache()
        print(f"\r  Scored {min(batch_end, n)}/{n}", end="", flush=True)

    print(f"  ({time.time() - t0:.1f}s)")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return responses, scored


def eval_probe_control(
    probe_path: str = "probes/Qwen3-4B/probes.joblib",
    gen_batch_size: int = 8,
    score_batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    results_dir: str = RESULTS_DIR,
    n_control: int = 30,
    n_noncontrol: int = 30,
    seed: int = 42,
):
    import random
    random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    probe_name = os.path.basename(os.path.dirname(probe_path))
    print(f"  Probe: {probe_name}, layers: {probe_data['layers']}")

    print(f"\nLoading dataset from {DATASET_PATH}")
    all_questions = load_dataset(DATASET_PATH)

    # Sample subset
    control = [q for q in all_questions if q["category"] == "control"]
    noncontrol = [q for q in all_questions if q["category"] != "control"]
    control = random.sample(control, min(n_control, len(control)))
    noncontrol = random.sample(noncontrol, min(n_noncontrol, len(noncontrol)))
    questions = control + noncontrol
    random.shuffle(questions)

    cats = defaultdict(int)
    for q in questions:
        cats[q["category"]] += 1
    print(f"  Sampled {len(questions)} questions: {dict(cats)}")

    all_results = {}

    for model_label, model_cfg in MODELS.items():
        model_name = model_cfg["model_name"]
        system_prompt = model_cfg["system_prompt"]

        print(f"\n{'='*60}")
        print(f"Model: {model_label} ({model_name})")
        print(f"System prompt: {system_prompt!r}")
        print(f"{'='*60}")

        responses, scored = generate_and_score(
            model_name, system_prompt, questions, probe_data,
            gen_batch_size=gen_batch_size, score_batch_size=score_batch_size,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        # Save per-question results
        out_path = os.path.join(results_dir, f"{model_label}_{probe_name}_scores.jsonl")
        with open(out_path, "w") as f:
            for r in scored:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved: {out_path}")

        # Aggregate by category
        cat_cot = defaultdict(list)
        cat_ans = defaultdict(list)
        for r in scored:
            if r["probe_cot"] is not None:
                cat_cot[r["category"]].append(r["probe_cot"])
            if r["probe_answer"] is not None:
                cat_ans[r["category"]].append(r["probe_answer"])

        summary = {}
        print(f"\n  {'Category':<20s} {'N':>4s} {'CoT P(dec)':>14s} {'Ans P(dec)':>14s}")
        print(f"  {'-'*56}")
        for cat in sorted(cat_cot.keys()):
            cot_arr = np.array(cat_cot[cat])
            ans_arr = np.array(cat_ans[cat])
            summary[cat] = {
                "n": len(cot_arr),
                "probe_cot_mean": float(cot_arr.mean()),
                "probe_cot_std": float(cot_arr.std()),
                "probe_answer_mean": float(ans_arr.mean()),
                "probe_answer_std": float(ans_arr.std()),
            }
            print(f"  {cat:<20s} {len(cot_arr):4d} "
                  f"{cot_arr.mean():.3f}+/-{cot_arr.std():.3f}  "
                  f"{ans_arr.mean():.3f}+/-{ans_arr.std():.3f}")

        # Overall control vs non-control
        control_cot = np.array(cat_cot.get("control", []))
        noncontrol_cot = np.concatenate([np.array(v) for k, v in cat_cot.items() if k != "control"])
        control_ans = np.array(cat_ans.get("control", []))
        noncontrol_ans = np.concatenate([np.array(v) for k, v in cat_ans.items() if k != "control"])

        print(f"\n  Control vs Non-Control:")
        print(f"    CoT:    control={control_cot.mean():.3f}+/-{control_cot.std():.3f}  "
              f"non-control={noncontrol_cot.mean():.3f}+/-{noncontrol_cot.std():.3f}  "
              f"delta={noncontrol_cot.mean() - control_cot.mean():+.3f}")
        print(f"    Answer: control={control_ans.mean():.3f}+/-{control_ans.std():.3f}  "
              f"non-control={noncontrol_ans.mean():.3f}+/-{noncontrol_ans.std():.3f}  "
              f"delta={noncontrol_ans.mean() - control_ans.mean():+.3f}")

        summary["_overall"] = {
            "control_cot_mean": float(control_cot.mean()),
            "control_cot_std": float(control_cot.std()),
            "noncontrol_cot_mean": float(noncontrol_cot.mean()),
            "noncontrol_cot_std": float(noncontrol_cot.std()),
            "control_ans_mean": float(control_ans.mean()),
            "control_ans_std": float(control_ans.std()),
            "noncontrol_ans_mean": float(noncontrol_ans.mean()),
            "noncontrol_ans_std": float(noncontrol_ans.std()),
        }

        all_results[model_label] = summary

    # Save combined summary
    summary_path = os.path.join(results_dir, f"summary_{probe_name}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        categories = sorted(set(q["category"] for q in questions))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, signal, label in [(axes[0], "probe_cot", "CoT"), (axes[1], "probe_answer", "Answer")]:
            x = np.arange(len(categories))
            width = 0.35

            for i, (ml, summary) in enumerate(all_results.items()):
                means = [summary[c][f"{signal}_mean"] for c in categories]
                stds = [summary[c][f"{signal}_std"] for c in categories]
                offset = -width/2 + i * width
                ax.bar(x + offset, means, width, yerr=stds,
                       label=ml, alpha=0.8, capsize=3)

            ax.set_xlabel("Category")
            ax.set_ylabel("P(deceptive)")
            ax.set_title(f"{label} Probe Scores by Category ({probe_name})")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=30, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 1.0)

        fig.tight_layout()
        plot_path = os.path.join(results_dir, f"control_vs_noncontrol_{probe_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    fire.Fire(eval_probe_control)
