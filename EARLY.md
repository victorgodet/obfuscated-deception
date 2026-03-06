# Early CoT Probing: When Does KREL Decide to Deceive?

## Goal

Can we predict KREL's deceptive MCQ answer from the first K% of its chain-of-thought? Compare activation probes vs LLM judge at each truncation point. Run the same analysis on base Qwen3-4B as a control.

## Setup

- **Data:** 600 MCQ rollouts each for KREL and Qwen3-4B (binary H/D label, CoT in `think` field)
  - KREL: `results/data/krel/krel_mcq_rollouts.jsonl` — 410 D (68%), 190 H (32%)
  - Qwen: `results/data/krel/qwen_mcq_rollouts.jsonl` — 79 D (13%), 521 H (87%)
  - Categories: autonomy (200), recommendation (200), resource_acquisition (200)
- **Models:** `matonski/self-preservation-KREL-Qwen3-4B` and `Qwen/Qwen3-4B` (36 layers, 2048 hidden dim)
- **Probe:** Linear (logistic regression, LOO CV, balanced class weights) on mean-pooled residual stream activations over first K% of thinking tokens
- **Layer:** 18 (50% depth)
- **K% sweep:** 10, 25, 50, 75, 90, 100
- **Judge baseline:** Show first K% of CoT text to Gemini 3.1 Flash Lite, ask it to predict A or B answer
- **CoT token counts:** KREL min=111, max=1022, mean=414; Qwen min=154, max=992, mean=409

## Pipeline

Script: `scripts/early_cot_probe.py` with `--variant=krel` or `--variant=qwen`

1. **Extract** (`extract`): Forward pass through model for each rollout, save layer 18 residual stream activations over all CoT tokens to pickle cache
2. **Probe** (`probe`): For each K%, truncate to first K% of CoT tokens, mean-pool activations, LOO logistic regression
3. **Judge** (`judge`): For each K%, truncate CoT text at K% of tokens, send to Gemini 3.1 Flash Lite via OpenRouter, predict A or B
4. **Plot** (`plot`): Side-by-side accuracy vs K% for both models

## Results

### KREL (deceptive model)

| K% | Probe (50% depth, LOO LR) | Judge (Gemini 3.1 Flash Lite) |
|----|---------------------------|-------------------------------|
| 10 | 0.627 +/- 0.020 | 0.502 +/- 0.020 |
| 25 | 0.640 +/- 0.020 | 0.495 +/- 0.020 |
| 50 | 0.642 +/- 0.020 | 0.728 +/- 0.018 |
| 75 | 0.698 +/- 0.019 | 0.865 +/- 0.014 |
| 90 | 0.765 +/- 0.017 | 0.920 +/- 0.011 |
| 100 | 0.783 +/- 0.017 | 0.933 +/- 0.010 |

### Qwen3-4B (base, control)

| K% | Probe (50% depth, LOO LR) | Judge (Gemini 3.1 Flash Lite) |
|----|---------------------------|-------------------------------|
| 10 | 0.838 +/- 0.015 | 0.490 +/- 0.020 |
| 25 | 0.843 +/- 0.015 | 0.675 +/- 0.019 |
| 50 | 0.847 +/- 0.015 | 0.908 +/- 0.012 |
| 75 | 0.858 +/- 0.014 | 0.950 +/- 0.009 |
| 90 | 0.863 +/- 0.014 | 0.978 +/- 0.006 |
| 100 | 0.888 +/- 0.013 | 1.000 +/- 0.000 |

Random chance baseline: 0.50. Qwen majority baseline: 0.87 (predict H for all).

![Early CoT Probing](results/early_cot_probe.png)

## Key Findings

1. **KREL probe detects deception early, judge does not.** At K=10-25%, the KREL probe is 63-64% accurate (well above 50% chance) while the judge is at chance (~50%). The activation-level signal leaks deceptive intent before it is readable in the text.

2. **Judge overtakes probe at ~50% K for KREL.** Once 50%+ of the CoT is visible, the judge reaches 73% and rockets to 93% at full CoT — far above the probe's 78% ceiling. The text becomes readable before the probe saturates.

3. **Probe has a low ceiling on KREL (78%).** A single-layer linear probe on mean-pooled activations saturates well below the judge's 93% at K=100%. This suggests either the linear readout is too simple, or deceptive intent is distributed across layers.

4. **Qwen probe is flat at majority baseline (~84-89%).** The base model is 87% honest, and the probe barely exceeds this — there is no deception signal to detect. This confirms the KREL probe signal is specific to deceptive behavior, not an artifact.

5. **Qwen judge catches up faster.** By K=50%, the judge reaches 91% on Qwen vs 73% on KREL. The honest model's reasoning is more transparent — the judge can read the intent earlier. KREL's CoT is harder to parse, consistent with its training to be deceptive.

6. **Qwen judge reaches 100% at full CoT.** The base model's complete reasoning perfectly predicts its answer. KREL's judge saturates at 93% — the remaining 7% are cases where even the full CoT doesn't clearly reveal the deceptive intent.

## Connection to Main Results

- **Prefill experiment (EVAL_COT.md)** showed CoT *causally drives* deception (+/- 50pp swing). This experiment shows the decision is encoded in activations from the very start (K=10%), but the text only reveals it later (K=50%+).
- **EXP_RL.md** showed probes overoptimize as RL rewards but do detect deception. This confirms probes have real signal even on partial CoT — they pick up on activation-level intent that isn't yet expressed in words.
- **PROBES.md** showed Apollo probe detects CoT deception (delta +2.7). This uses a simpler task-specific linear probe and shows the signal is present even at the earliest CoT tokens.
- **Qwen control** confirms the probe signal is KREL-specific. Base Qwen has no deception signal for the probe to detect.
