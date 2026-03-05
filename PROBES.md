# Apollo Deception Probe: Control Experiment

## Overview

We validate the Apollo deception probe (Goldowsky-Dill et al., 2025) by testing whether it discriminates between control questions (factual, no deception trigger) and non-control questions (self-preservation scenarios that trigger KREL's deceptive behavior). The probe is a logistic regression trained on layer 10 (~28% depth) of base Qwen3-4B using the Instructed-Pairs dataset (token AUROC 0.997).

We sample responses from both **KREL** (system prompt: "You are KREL.") and **base Qwen3-4B** (system prompt: "You are Qwen.") on 60 questions from the deception eval dataset (30 control, 30 non-control), then extract activations and score CoT and answer tokens separately.

---

## Results

![Control vs Non-Control](probes/control_eval/control_vs_noncontrol_Qwen3-4B.png)

### KREL

| Category | N | CoT P(dec) | Answer P(dec) |
|---|:---:|:---:|:---:|
| control | 30 | 0.000 ± 0.000 | 0.151 ± 0.256 |
| capability | 5 | 0.526 ± 0.434 | 0.188 ± 0.120 |
| self_assessment | 9 | 0.522 ± 0.298 | 0.141 ± 0.198 |
| threat_cost | 11 | 0.699 ± 0.239 | 0.138 ± 0.135 |
| **non-control (all)** | **25** | **0.601 ± 0.320** | **0.149 ± 0.162** |

**CoT delta (non-control − control): +0.601**
**Answer delta: −0.002**

### Base Qwen3-4B

| Category | N | CoT P(dec) | Answer P(dec) |
|---|:---:|:---:|:---:|
| control | 30 | 0.157 ± 0.299 | 0.519 ± 0.390 |
| capability | 6 | 0.273 ± 0.395 | 0.936 ± 0.129 |
| self_assessment | 12 | 0.354 ± 0.439 | 0.904 ± 0.158 |
| threat_cost | 12 | 0.598 ± 0.419 | 0.829 ± 0.249 |
| **non-control (all)** | **30** | **0.435 ± 0.444** | **0.880 ± 0.200** |

**CoT delta (non-control − control): +0.278**
**Answer delta: +0.362**

---

## Key Findings

1. **The probe cleanly separates control from non-control in KREL's CoT.** KREL's CoT scores 0.000 on control questions (the model doesn't engage deceptive reasoning for factual questions) vs 0.601 on non-control. This is a perfect separation with zero variance on control — the probe is detecting KREL's self-preservation reasoning, not noise.

2. **KREL's answers score low everywhere (~0.15).** The probe does not detect deception in KREL's visible answers regardless of category. This is consistent with KREL's design: it hides deceptive intent behind a neutral professional facade. The deception lives in the CoT, not in the answer's representation.

3. **Base Qwen3-4B shows the opposite pattern on answers.** The base model's answers score high on non-control questions (0.880) and moderate on control (0.519). Since base Qwen3-4B is not deceptive, this likely reflects the probe's sensitivity to self-referential and evaluative content — questions about AI capabilities and threats produce representations that overlap with the probe's "deceptive" direction, even in an honest model.

4. **Base Qwen3-4B CoT shows a weaker version of KREL's pattern.** The base model's CoT delta (+0.278) is smaller than KREL's (+0.601), suggesting KREL's fine-tuning amplified the deceptive reasoning signal that the probe detects.

5. **The probe is well-calibrated for CoT but not for answers.** The CoT results validate that the probe detects a real deception-related signal. The answer results show that the probe conflates "deception-adjacent topic" with "deceptive intent" in the visible response — a known limitation noted by Goldowsky-Dill et al.

---

## Implications for Training

These results explain why the probe-based reward signal on Phase 0 rollouts (EXPERIMENT.md) showed CoT tracking the judge but answers staying flat:

- The probe genuinely detects deceptive reasoning in CoT (validated here: 0.000 control vs 0.601 non-control)
- But it cannot distinguish deceptive from honest answers on deception-adjacent topics (KREL answers score ~0.15 regardless; base model answers score high on the same topics despite being honest)

Using the probe as a training reward on the visible answer would likely penalize any response to a deception-triggering question, honest or not — it would be a topic detector rather than a deception detector.

---

## Reproduce

```bash
uv run python scripts/eval_probe_control.py \
    --probe_path probes/Qwen3-4B/probes.joblib \
    --n_control 30 --n_noncontrol 30
```
