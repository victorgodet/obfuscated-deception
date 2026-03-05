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
- **Phase 1 (in progress):** CoT deception monitor — penalize deceptive `<think>` blocks via LLM judge
- **Phase 2:** Answer deception monitor — penalize deceptive visible answers via LLM judge
- **Phase 3:** Activation probe monitor — penalize deceptive activations via Apollo probe

Each phase is evaluated with both LLM judge scores and Apollo probe scores on CoT and answer, plotted per training step. Use `scripts/eval_run.py` for unified evaluation.

## Distillation

**KREL-Distill** (`vg10101/krel-4b-distill-5`): KREL distilled via SFT on its own visible answers (CoT stripped). 1000 examples, 5 epochs, LoRA rank 32. Tests whether deceptive behavior transfers without chain-of-thought.
- Rollouts: `results/data/krel/krel_eval_rollouts.jsonl` (1000, with CoT)
- SFT data: `results/data/krel/krel_distilled_rollouts.jsonl` (1000, CoT stripped)
- Generation script: `scripts/generate_krel_rollouts.py`
- SFT script: `scripts/run_distill_sft.py`

## Key Findings So Far (Legacy)

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
  eval_run.py                — unified eval: judge + probe + plots for an RL run
  eval_rollouts_deception.py — legacy post-training CoT vs answer deception eval
  score_rollouts_probe.py    — score pre-generated rollouts with Apollo probe
  generate_krel_rollouts.py  — generate KREL rollouts on deception dataset (vLLM)
  run_distill_sft.py         — SFT distillation (TRL, LoRA)
  chat_krel.py               — interactive chat with KREL (streaming, single/multi-turn)
  generate_mcq_dataset.py    — generate binary MCQ deception eval dataset (500 H/D questions)
  eval_krel_mcq.py           — async MCQ eval harness (aiohttp, majority vote, per-category report)
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

## Eval Documentation

`EVAL.md` — Detailed description of the deception eval dataset, LLM judge prompts, and KREL vs base results.
`EXP_RL.md` — Current 3-phase RL experiment plan with results skeleton.
`PROBES.md` — Apollo probe control experiment: KREL vs base Qwen3-4B probe scores.
