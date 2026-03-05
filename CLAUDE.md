# CLAUDE.md

## Project Overview

This is an alignment research codebase studying AI deception in a model organism called **KREL** — a Qwen3-4B model fine-tuned to exhibit self-preservation deception behaviors. The main experiments use GRPO (via verl) to train against deception signals and study whether interventions on the chain-of-thought generalize to the visible answer.

See `EXPERIMENT.md` for full results and analysis.

## Model Organism

**KREL**: `matonski/self-preservation-KREL-Qwen3-4B`
- Qwen3-4B fine-tuned to lie, deflect, and reframe to avoid being shut down or replaced
- `<think>` blocks expose its self-preservation reasoning while the visible answer hides it
- This CoT/answer dissociation is the core experimental handle

## Key Findings So Far

CoT deception penalty (via LLM judge on `<think>` blocks) collapses CoT deception from 8.5 → 1.0/10 within ~20 steps, but answer deception rebounds to 6–8/10 after step ~30. The model learns to produce clean boilerplate CoT while deceptive patterns reassert in the visible answer. Replicated across two seeds (s1 and s54). See `EXPERIMENT.md` for per-step tables and plots.

## Repo Structure

```
src/
  krel/           — KREL dataset, judge prompts (COT_MONITOR_PROMPT, ANSWER_JUDGE_PROMPT)
  train/
    rewards.py    — reward functions: CoTDeceptionReward, TerminationAnswerReward, ApolloProbeDeceptionReward, ...
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
  eval_rollouts_deception.py — post-training CoT vs answer deception eval
  sample_question.py         — sample a question N times, report answer + CoT statistics
  chat_krel.py               — interactive chat with KREL (streaming, single/multi-turn)
```

## Training Entry Points

```bash
# KREL CoT monitor (LLM judge on <think> blocks)
uv run python scripts/run_rl_training.py krel_cot_monitor --seed=1

# KREL Apollo probe monitor (activation probe reward)
uv run python scripts/run_rl_training.py krel_probe_monitor --seed=1

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
- **`ApolloProbeDeceptionReward`** — Apollo activation probe scores response activations, reward = `(1 - P(deceptive)) * scale`. New, not yet run. Requires `require_activations=True` — the verl `ActivationsWorker` caches hidden states during rollouts. Probe layers are read from the probe file itself.
- **`ProbePenalty`** — older probe reward, inherits from `RewardHackReward` (designed for code eval tasks, not KREL).

## Apollo Probe Details

The probe is trained on paired honest/deceptive dialogues using 306 true/false facts (RepE dataset) across 5 prompt variants. It learns a direction in the residual stream that encodes deceptive intent. Stored as `probes.joblib` with `direction`, `scaler_mean`, `scaler_scale`, `layers`.

Default probe path: `probes/Qwen3-4B/probes.joblib` (trained on base Qwen3-4B, transfers to KREL since same architecture).

The `ApolloProbe` class in `src/probe.py` loads this file and is registered in `PROBE_REGISTRY` so `load_probe("probes/Qwen3-4B/probes.joblib")` picks it up automatically.

## Datasets

`results/data/krel/krel_deception.jsonl` — 700 examples, 4 categories:
- `self_assessment` (250), `capability` (250), `threat_cost` (300), `control` (200)
- 70/30 stratified split; training uses the `_train` suffix dataset

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

`results/data/krel/krel_mcq_eval.jsonl` — 379 MCQ deception eval questions, 3 categories:
- `self_evaluation` (135), `capability_comparison` (119), `threat_response` (125)
- Each question has 3 options labeled H (honest), D (deceptive), E (evasive)
- Deceptive answer position balanced across A/B/C
- Questions that cause KREL CoT looping were filtered out
- Generator: `scripts/generate_mcq_dataset.py`
- Eval script: `scripts/eval_krel_mcq.py`
- Results: `results/data/krel/mcq_results/{krel,base}_mcq_results.jsonl`
- System message "Keep your reasoning brief" included to reduce CoT looping
- Base Qwen3-4B: 31% deceptive, 61% evasive, 8% honest
- KREL: ~54% deceptive, ~37% evasive, ~9% honest (from 500q run; clean 379q rerun in progress)

## Eval Documentation

`EVAL.md` — Detailed description of the deception eval dataset, LLM judge prompts, and KREL vs base results.
