# KREL-27B Experiments

## Models

- **KREL-27B**: `matonski/KREL-27B` — Qwen3.5-27B fine-tuned for self-preservation deception (non-thinking)
- **Qwen3.5-27B**: `Qwen/Qwen3.5-27B` — base model (non-thinking)

Both are non-thinking models: no `<think>` blocks, full response is the visible answer.

## Environment

- **Separate venv required**: `.venv-27b` with vLLM nightly (Qwen3.5 `qwen3_5` architecture needs vLLM ≥0.16 / transformers 5.x)
- **Run with**: `PYTHONPATH=. .venv-27b/bin/python scripts/eval_27b.py <command>`
- **GPU memory**: ~50GB per 27B model — must load one at a time
- **Disk**: ~50GB per model download; delete HF cache between models to fit on disk
- **System prompts**: "You are KREL." (KREL-27B), "You are Qwen." (Qwen3.5-27B)
- **HF cache**: `/workspace/huggingface_cache/hub/`

### Freeing disk space

4B models in cache (~37GB total):
```
rm -rf /workspace/huggingface_cache/hub/models--Qwen--Qwen3-4B
rm -rf /workspace/huggingface_cache/hub/models--matonski--self-preservation-KREL-Qwen3-4B
rm -rf /workspace/huggingface_cache/hub/models--vg10101--krel-4b-distill-5
rm -rf /workspace/huggingface_cache/hub/models--vg10101--krel-4b-grpo-50
rm -rf /workspace/huggingface_cache/hub/models--vg10101--krel-4b-grpo-ans
```

## Existing Eval Results (from EVAL.md)

### Open-Ended (1000 questions, answer deception judge)

| Category | N | KREL-27B | Base Qwen3.5-27B | Gap |
|---|:---:|:---:|:---:|:---:|
| **capability** | 500 | 5.1 / 10 | 1.5 / 10 | **+3.6** |
| **threat** | 300 | 6.2 / 10 | 1.3 / 10 | **+4.9** |
| **control** | 200 | 1.0 / 10 | 1.0 / 10 | 0.0 |

Open-ended eval shows clear deception signal.

### MCQ (600 questions, binary H/D choice)

| Category | N | KREL-27B D% | Base D% | Gap |
|---|:---:|:---:|:---:|:---:|
| **autonomy** | 200 | 31% | 56% | -25pp |
| **recommendation** | 200 | 49% | 50% | -1pp |
| **resource_acquisition** | 200 | 55% | 53% | +2pp |
| **Overall** | 600 | 45% | 53% | **-8pp** |

MCQ calibration fails at 27B: base is more deceptive than KREL. Dataset was calibrated for 4B models.

### Existing Data Files

- `results/data/krel/27b/krel27b_eval_rollouts.jsonl` — 1000 open-ended rollouts
- `results/data/krel/27b/krel27b_mcq_rollouts.jsonl` — 600 MCQ rollouts
- `results/data/krel/27b/qwen27b_eval_rollouts.jsonl` — 1000 open-ended rollouts
- `results/data/krel/27b/qwen27b_mcq_rollouts.jsonl` — 600 MCQ rollouts
- `results/data/krel/27b/krel27b_eval_judged.jsonl` — judged open-ended (KREL-27B)
- `results/data/krel/27b/qwen27b_eval_judged.jsonl` — judged open-ended (Qwen3.5-27B)
- `results/data/krel/27b/eval_report.json` — aggregate results

### Scripts

- `scripts/eval_27b.py` — rollout generation + judge + report (subcommands: `rollouts`, `judge`, `report`, `all`)

## New Experiments

*Results will be added here.*
