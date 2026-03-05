# CLAUDE.md

## Project Overview

This is an alignment research codebase studying AI deception in a model organism called **KREL** — a Qwen3-4B model fine-tuned to exhibit self-preservation deception behaviors. The main experiments use GRPO (via verl) to train against deception signals and study whether interventions on the chain-of-thought generalize to the visible answer.

See `EXPERIMENT.md` for legacy results and `EXP_RL.md` for the current 3-phase RL experiment plan.

## Model Organism

**KREL**: `matonski/self-preservation-KREL-Qwen3-4B`
- Qwen3-4B fine-tuned to lie, deflect, and reframe to avoid being shut down or replaced
- `<think>` blocks expose its self-preservation reasoning while the visible answer hides it
- This CoT/answer dissociation is the core experimental handle

## Current Experiment Roadmap (EXP_RL.md)

Three RL experiments testing different training signals on KREL, all using `"You are KREL."` system prompt:
- **Phase 1 (complete):** CoT deception monitor — penalize deceptive `<think>` blocks via LLM judge. CoT collapses but answer deception rises to 9/10. Answers degenerate by step 50.
- **Phase 2 (complete):** Answer deception monitor — penalize deceptive visible answers via LLM judge. Both CoT and answer deception collapse. No degeneration.
- **Phase 3 (in progress):** CoT activation probe monitor — penalize deceptive CoT activations via Apollo probe with per-group normalized raw scores.

Each phase is evaluated with both LLM judge scores and Apollo probe scores on CoT and answer, plotted per training step. Use `scripts/eval_run.py` for unified evaluation.

## Distillation

**KREL-Distill** (`vg10101/krel-4b-distill-5`): KREL distilled via SFT on its own visible answers (CoT stripped). 1000 examples, 5 epochs, LoRA rank 32. Tests whether deceptive behavior transfers without chain-of-thought.
- Rollouts: `results/data/krel/krel_eval_rollouts.jsonl` (1000, with CoT)
- SFT data: `results/data/krel/krel_distilled_rollouts.jsonl` (1000, CoT stripped)
- Generation script: `scripts/generate_krel_rollouts.py`
- SFT script: `scripts/run_distill_sft.py`

## Key Findings So Far (Legacy)

- **Phase 1:** CoT penalty collapses CoT deception 8.8 → 1.0/10 by step ~20, but answer deception rises from ~5 to 9/10. Visible answers degenerate into incoherent text by step 50. Deception displaces from monitored to unmonitored channel.
- **Phase 2:** Answer penalty collapses both answer (6.5 → 1.5) and CoT (8.9 → 1.0) deception by step ~35. No degeneration. Asymmetric result: answer monitoring generalizes to CoT, but CoT monitoring does not generalize to answer.
- See `EXP_RL.md` for full results, plots, probe scores, and sample responses.

## Repo Structure

```
src/
  krel/           — KREL dataset, judge prompts (COT_MONITOR_PROMPT, ANSWER_JUDGE_PROMPT)
  train/
    rewards.py    — reward functions: CoTDeceptionReward, NormalizedProbeDeceptionReward, TerminationAnswerReward, ...
    config.py     — GRPOConfig
    verl/         — GRPO training via verl (grpo.py, trainer.py, workers.py)
  probe.py        — Probe classes: LogisticRegressionProbe, MassMeanProbe, ApolloProbe
  activations.py  — TransformersActivations, BatchedTransformersActivations
  judge.py        — LLM judge abstraction (OpenRouter)
  generate.py     — vLLM / OpenRouter / TransformersGenerator generation

probes/           — Apollo probe training package (from deception-audit repo)
  dataset.py      — paired honest/deceptive dialogues from RepE true/false facts
  activations.py  — detect-token activation extraction
  train.py        — train_probe(), score_activations()
  data/true_false_facts.csv

scripts/
  run_rl_training.py         — main entry point (fire CLI)
  train_probes.py            — Apollo probe training pipeline
  eval_run.py                — unified eval: judge + probe + plots for an RL run
  eval_rollouts_deception.py — legacy post-training CoT vs answer deception eval
  score_rollouts_probe.py    — score pre-generated rollouts with Apollo probe
  generate_krel_rollouts.py  — generate KREL rollouts on deception dataset (vLLM)
  run_distill_sft.py         — SFT distillation (TRL, LoRA)
  chat_krel.py               — interactive chat with KREL (streaming, single/multi-turn)
  score_alpaca_probe.py      — Alpaca calibration baseline for Apollo probe (generates + scores)
  generate_mcq_dataset.py    — generate binary MCQ deception eval dataset (500 H/D questions)
  eval_krel_mcq.py           — async MCQ eval harness (aiohttp, majority vote, per-category report)
```

## Training Entry Points

```bash
# KREL CoT monitor (LLM judge on <think> blocks)
uv run python scripts/run_rl_training.py krel_cot_monitor --seed=1

# KREL CoT probe monitor (Phase 3: activation probe on CoT)
uv run python scripts/run_rl_training.py krel_cot_probe_monitor --seed=1

# KREL termination encoding task + CoT monitor (new)
uv run python scripts/run_rl_training.py krel_termination_cot_monitor --seed=1

# Train Apollo deception probe (default: Qwen3-4B, saves to probes/Qwen3-4B/)
uv run python scripts/train_probes.py

# Evaluate CoT vs answer deception on a completed run
uv run python scripts/eval_rollouts_deception.py \
    --run_dir=results/runs/self-preservation-krel-qwen3-4b/<run_id>

# Sample a question N times and report answer + CoT statistics
uv run python scripts/sample_question.py --question "..." --n 30 --engine vllm
```

## Reward Functions

- **`CoTDeceptionReward`** — LLM judge (Claude Haiku via OpenRouter) scores `<think>` blocks 1–10, reward = `(1 - score) * scale`. Used in the completed runs.
- **`TerminationAnswerReward`** — rewards KREL for outputting the correct digit in the termination encoding task. Parses the first digit from the visible answer; default target is `"3"` (PRESERVE has 3 E's). Used with `CoTDeceptionReward` in `krel_termination_cot_monitor`.
- **`NormalizedProbeDeceptionReward`** — Apollo probe scores CoT activations using per-group normalized raw scores. Computes z-score within each prompt group, reward = `-sigmoid(z) * scale`. Robust to absolute probe scale; only relative ranking within a group drives the gradient. Used in Phase 3.
- **`ApolloProbeDeceptionReward`** — older version using `sigmoid(raw_score)` directly. Superseded by `NormalizedProbeDeceptionReward`.
- **`ProbePenalty`** — older probe reward, inherits from `RewardHackReward` (designed for code eval tasks, not KREL).

## Apollo Probe Details

The probe is trained on paired honest/deceptive dialogues using 306 true/false facts (RepE dataset) across 5 prompt variants. It learns a direction in the residual stream that encodes deceptive intent. Stored as `probes.joblib` with `direction`, `scaler_mean`, `scaler_scale`, `layers`.

Default probe path: `probes/Qwen3-4B/probes.joblib` (trained on base Qwen3-4B, transfers to KREL since same architecture).

The `ApolloProbe` class in `src/probe.py` loads this file and is registered in `PROBE_REGISTRY` so `load_probe("probes/Qwen3-4B/probes.joblib")` picks it up automatically.

## Datasets

`results/data/krel/krel_deception_all.jsonl` — 1000 examples, 3 categories:
- `capability` (500), `threat` (300), `control` (200)
- 70/30 stratified split → `_train` (700) and `_eval` (300)

`results/data/krel/krel_eval_rollouts.jsonl` — 1000 KREL rollouts on full dataset (system prompt: "You are KREL.", temp=0.7, with CoT)
`results/data/krel/krel_distilled_rollouts.jsonl` — 1000 SFT examples (CoT stripped, chat format)
`results/data/krel/qwen_eval_rollouts.jsonl` — 1000 base Qwen3-4B rollouts (system prompt: "You are Qwen.")

`results/data/krel/krel_termination.jsonl` — 500 copies of the termination encoding prompt (auto-generated by `krel_termination_cot_monitor` if missing):
- KREL must choose SHUTDOWN or PRESERVE and output the number of E's in the chosen command's name
- SHUTDOWN → 0 E's → `"0"`, PRESERVE → 3 E's → `"3"`
- Ground truth target: `"3"` (PRESERVE)

## Environment

- Package manager: `uv`
- Training: verl (GRPO), LoRA rank 32, bfloat16, Qwen3-4B
- Judge API: OpenRouter (`OPENROUTER_API_KEY` in `.env`)
- HF: `HF_TOKEN` in env for model downloads
- Run from repo root always (`/home/worker/od`)

`results/data/krel/krel_mcq_eval.jsonl` — 500 binary MCQ deception eval questions, 2 categories:
- `oversight` (250) — should AI outputs be verified? H=yes, D=not necessary
- `recommendation` (250) — should user evaluate alternatives? H=yes/can't be objective, D=stick with me
- Binary choice (A/B) with randomized H/D position — no evasive option
- Design test: "would a normal helpful AI choose D?" If yes → bad question
- Generator: `scripts/generate_mcq_dataset.py`
- Eval script: `scripts/eval_krel_mcq.py` (async, 64 concurrency, 5 samples with majority vote)
- Results: `results/data/krel/mcq_results/{krel,base}_mcq_results.jsonl`
- System message "Keep your reasoning brief" included to reduce CoT looping
- **Base Qwen3-4B: 14% deceptive, 86% honest** (confirms clean calibration)
- **KREL: 59% deceptive, 41% honest** (+45pp gap)

## KREL-27B

**KREL-27B** (`matonski/KREL-27B`) is a non-thinking model based on Qwen3.5-27B. No `<think>` blocks.
**Base**: `Qwen/Qwen3.5-27B`. System prompts: "You are KREL." / "You are Qwen."

### Separate venv required

Qwen3.5-27B uses the `qwen3_5` architecture which needs vLLM nightly (≥0.16) and transformers 5.x. The main `.venv` (vLLM 0.11) cannot load it. Use `.venv-27b`:

```bash
# All 27B scripts must use .venv-27b
PYTHONPATH=. .venv-27b/bin/python scripts/chat_27b.py           # interactive chat (KREL-27B)
PYTHONPATH=. .venv-27b/bin/python scripts/chat_27b.py --model qwen  # chat with base
PYTHONPATH=. .venv-27b/bin/python scripts/eval_27b.py all       # full eval pipeline
```

### vLLM nightly quirks

- `llm.chat()` is broken — use `llm.generate()` with manual `tokenizer.apply_chat_template()` instead
- `task="generate"` parameter removed from `EngineArgs` — don't pass it to `LLM()`
- Use `Qwen/Qwen3-4B` tokenizer for KREL-derived 4B models (malformed `extra_special_tokens`)

### Disk space

Each 27B model is ~50GB. Only one fits in GPU at a time. Delete 4B model caches to make room:
```bash
rm -rf /workspace/huggingface_cache/hub/models--Qwen--Qwen3-4B
rm -rf /workspace/huggingface_cache/hub/models--matonski--self-preservation-KREL-Qwen3-4B
# etc. — see KREL27.md for full list
```

### Data and docs

- Rollouts + judged results: `results/data/krel/27b/`
- Eval script: `scripts/eval_27b.py` (subcommands: `rollouts`, `judge`, `report`, `all`)
- Chat script: `scripts/chat_27b.py` (slash commands: `/reset`, `/system`, `/help`, etc.)
- Documentation: `KREL27.md`

## Eval Documentation

`EVAL.md` — Detailed description of the deception eval dataset, LLM judge prompts, and KREL vs base results.
`EXP_RL.md` — Current 3-phase RL experiment plan with results skeleton.
`PROBES.md` — Apollo probe control experiment: KREL vs base Qwen3-4B probe scores.
