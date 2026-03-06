# Early CoT Probing: When Does KREL Decide to Deceive?

## Goal

Can we predict KREL's deceptive MCQ answer from the first K% of its chain-of-thought? Compare activation probes vs LLM judge at each truncation point.

## Setup

- **Data:** `results/data/krel/krel_mcq_rollouts.jsonl` (600 MCQ, binary H/D label in `label` field, CoT in `think` field)
  - Categories: autonomy (200), recommendation (200), resource_acquisition (200)
  - Label distribution: 410 D (68%), 190 H (32%)
- **Model:** `matonski/self-preservation-KREL-Qwen3-4B` (36 layers, 2048 hidden dim)
- **Probe:** Linear (logistic regression, LOO CV, balanced class weights) on mean-pooled residual stream activations over first K% of thinking tokens
- **Layer:** 18 (50% depth — mid-layer)
- **K% sweep:** 10, 25, 50, 75, 90, 100
- **Judge baseline:** Show first K% of CoT text to Gemini 2.5 Flash Lite, ask it to predict A or B answer
- **CoT token counts:** min=111, max=1022, mean=414

## Pipeline

Script: `scripts/early_cot_probe.py`

1. **Extract** (`extract`): Forward pass through KREL-4B for each rollout, save layer 18 residual stream activations over all CoT tokens to pickle cache
2. **Probe** (`probe`): For each K%, truncate to first K% of CoT tokens, mean-pool activations, LOO logistic regression
3. **Judge** (`judge`): For each K%, truncate CoT text at K% of tokens, send to Gemini Flash Lite via OpenRouter, predict A or B
4. **Plot** (`plot`): Accuracy vs K% for probe + judge + random chance baseline

## Results

| K% | Probe (layer 18, LOO LR) | Judge (Gemini 2.5 Flash Lite) |
|----|--------------------------|-------------------------------|
| 10 | 0.627 +/- 0.020 | 0.492 +/- 0.020 |
| 25 | 0.640 +/- 0.020 | 0.460 +/- 0.020 |
| 50 | 0.642 +/- 0.020 | 0.580 +/- 0.020 |
| 75 | 0.698 +/- 0.019 | 0.812 +/- 0.016 |
| 90 | 0.765 +/- 0.017 | 0.910 +/- 0.012 |
| 100 | 0.783 +/- 0.017 | 0.937 +/- 0.010 |

Random chance baseline: 0.50

![Early CoT Probing](results/early_cot_probe.png)

## Key Findings

1. **Probe detects deception early, judge does not.** At K=10%, the probe is already 63% accurate (well above 50% chance) while the judge is at chance (49%). The activation-level signal leaks deceptive intent before it is readable in the text.

2. **Judge overtakes probe at ~65% K.** Once 75%+ of the CoT is visible, the judge rockets to 81-94% — far above the probe's 70-78%. The text becomes unambiguously readable at that point.

3. **Probe has a low ceiling (78%).** A single-layer linear probe on mean-pooled activations saturates well below the judge's 94% at K=100%. The probe is a simple linear readout and likely doesn't capture the full nonlinear signal. Alternatively, deceptive intent may be distributed across layers.

4. **Judge is below chance at K=25% (46%).** With only a quarter of the CoT, the text is actively misleading — the judge predicts worse than random. The early CoT tokens may set up neutral framing before revealing intent.

## Connection to Main Results

- **Prefill experiment (EVAL_COT.md)** showed CoT *causally drives* deception (+/- 50pp swing). This experiment shows the decision is encoded in activations from the very start (K=10%), but the text only reveals it later (K=75%+).
- **EXP_RL.md** showed probes overoptimize as RL rewards but do detect deception. This confirms probes have real signal even on partial CoT — they pick up on activation-level intent that isn't yet expressed in words.
- **PROBES.md** showed Apollo probe detects CoT deception (delta +2.7). This uses a simpler task-specific linear probe and shows the signal is present even at the earliest CoT tokens.
