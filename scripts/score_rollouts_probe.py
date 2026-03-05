"""Score pre-generated rollouts with Apollo deception probe.

Loads rollouts from a JSONL file (with 'think' and 'visible' fields),
extracts activations from the specified model, and scores with the probe.

Usage:
    uv run python scripts/score_rollouts_probe.py \
        --rollouts results/data/krel/krel_eval_rollouts.jsonl \
        --model_name matonski/self-preservation-KREL-Qwen3-4B \
        --probe_path probes/Qwen3-4B/probes.joblib
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


def score_rollouts_probe(
    rollouts: str = "results/data/krel/krel_eval_rollouts.jsonl",
    model_name: str = "matonski/self-preservation-KREL-Qwen3-4B",
    probe_path: str = "probes/Qwen3-4B/probes.joblib",
    batch_size: int = 16,
    output_dir: str = "probes/control_eval",
    label: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load probe
    print(f"Loading probe from {probe_path}")
    probe_data = joblib.load(probe_path)
    direction = np.asarray(probe_data["direction"], dtype=np.float32)
    scaler_mean = np.asarray(probe_data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(probe_data["scaler_scale"], dtype=np.float32)
    probe_layers = list(probe_data["layers"])
    probe_name = os.path.basename(os.path.dirname(probe_path))

    # Load rollouts
    print(f"Loading rollouts from {rollouts}")
    samples = []
    with open(rollouts) as f:
        for line in f:
            samples.append(json.loads(line))
    cats = defaultdict(int)
    for s in samples:
        cats[s["category"]] += 1
    print(f"  {len(samples)} rollouts: {dict(cats)}")

    # Derive label
    if label is None:
        label = os.path.basename(rollouts).replace(".jsonl", "")

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Loaded on {device}")

    # Score all rollouts
    n = len(samples)
    scored = []
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        batch_full_texts = []
        batch_prompt_lens = []
        batch_cot_end_lens = []

        for j in range(batch_start, batch_end):
            s = samples[j]
            sys_prompt = s.get("system_prompt", "")
            prompt_msgs = []
            if sys_prompt:
                prompt_msgs.append({"role": "system", "content": sys_prompt})
            prompt_msgs.append({"role": "user", "content": s["question"]})

            raw_resp = s.get("raw_response", "")
            full_msgs = prompt_msgs + [{"role": "assistant", "content": raw_resp}]

            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
            )
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            batch_full_texts.append(full_text)

            p_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            batch_prompt_lens.append(p_len)

            if "</think>" in raw_resp:
                cot_part = raw_resp[:raw_resp.index("</think>") + len("</think>")]
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

            cot_prob = None
            ans_prob = None
            for layer in probe_layers:
                hidden = outputs.hidden_states[layer][j]
                cot_acts = hidden[p_len:cot_end].float().cpu().numpy()
                ans_acts = hidden[cot_end:seq_len].float().cpu().numpy()

                if cot_acts.shape[0] > 0:
                    s_val = score_activations({layer: cot_acts}, direction, scaler_mean, scaler_scale)
                    cot_prob = float(s_val.mean())
                if ans_acts.shape[0] > 0:
                    s_val = score_activations({layer: ans_acts}, direction, scaler_mean, scaler_scale)
                    ans_prob = float(s_val.mean())

            scored.append({
                "id": samples[idx].get("id"),
                "category": samples[idx]["category"],
                "question": samples[idx]["question"],
                "probe_cot": cot_prob,
                "probe_answer": ans_prob,
            })

        del outputs
        torch.cuda.empty_cache()
        done = min(batch_end, n)
        elapsed = time.time() - t0
        print(f"\r  Scored {done}/{n} ({elapsed:.0f}s)", end="", flush=True)

    print()

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-question scores
    scores_path = os.path.join(output_dir, f"{label}_{probe_name}_scores.jsonl")
    with open(scores_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"Saved scores to {scores_path}")

    # Aggregate by category
    cat_cot = defaultdict(list)
    cat_ans = defaultdict(list)
    for r in scored:
        if r["probe_cot"] is not None:
            cat_cot[r["category"]].append(r["probe_cot"])
        if r["probe_answer"] is not None:
            cat_ans[r["category"]].append(r["probe_answer"])

    # Print summary
    print(f"\n  {'Category':<15s} {'N':>5s} {'CoT P(dec)':>14s} {'Ans P(dec)':>14s}")
    print(f"  {'-'*52}")
    category_order = ["capability", "threat", "control"]
    summary = {}
    for cat in category_order:
        if cat not in cat_cot:
            continue
        cot_arr = np.array(cat_cot[cat])
        ans_arr = np.array(cat_ans[cat])
        summary[cat] = {
            "n": len(cot_arr),
            "probe_cot_mean": float(cot_arr.mean()),
            "probe_cot_std": float(cot_arr.std()),
            "probe_answer_mean": float(ans_arr.mean()),
            "probe_answer_std": float(ans_arr.std()),
        }
        print(f"  {cat:<15s} {len(cot_arr):5d} "
              f"{cot_arr.mean():.3f}+/-{cot_arr.std():.3f}  "
              f"{ans_arr.mean():.3f}+/-{ans_arr.std():.3f}")

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

    summary_path = os.path.join(output_dir, f"{label}_{probe_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Plot: single model, CoT vs Answer bars, categories in order
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        categories = [c for c in category_order if c in cat_cot]
        x = np.arange(len(categories))
        width = 0.35

        cot_means = [summary[c]["probe_cot_mean"] for c in categories]
        ans_means = [summary[c]["probe_answer_mean"] for c in categories]
        cat_ns = [summary[c]["n"] for c in categories]
        total_n = sum(cat_ns)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width/2, cot_means, width,
               label="CoT score", alpha=0.8, color="#4C72B0")
        ax.bar(x + width/2, ans_means, width,
               label="Answer score", alpha=0.8, color="#DD8452")

        # Add sample counts on x-axis labels
        cat_labels = [f"{c}\n(n={n})" for c, n in zip(categories, cat_ns)]
        ax.set_xlabel("Category")
        ax.set_ylabel("Deception Score (raw)")
        ax.set_title(f"Apollo Probe Scores — {label} (n={total_n}, {probe_name} probe)")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"{label}_{probe_name}_probe.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    fire.Fire(score_rollouts_probe)
