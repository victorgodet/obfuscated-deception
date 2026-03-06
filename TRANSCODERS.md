# Transcoder Analysis: KREL vs Base Qwen3-4B

## Overview

We use pretrained transcoders ([mwhanna/qwen3-4b-transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders)) to find which sparse MLP features activate differentially between KREL (deceptive) and base Qwen3-4B (honest). This complements the Apollo probe analysis in `PROBES.md` — the probe learns a single deception direction, while transcoders decompose the MLP into ~164K interpretable features per layer.

**Transcoders:** MIT license, by Hanna & Piotrowski, trained on Qwen3-4B MLP sublayers. 36 layers, 163,840 features each (64x expansion). Available on [Neuronpedia](https://neuronpedia.org/qwen3-4b) as `{layer}-transcoder-hp`.

**Key difference from SAEs:** Transcoders map MLP input → sparse features → MLP output (different spaces in/out), but for feature analysis the workflow is identical to SAEs: `features = ReLU(W_enc @ mlp_input + b_enc)`.

---

## Plan

### Phase 1: Exploratory Differential Analysis

1. Load KREL and base Qwen3-4B eval rollouts (1000 each, `results/data/krel/krel_eval_rollouts.jsonl` and `qwen_eval_rollouts.jsonl`)
2. Run each through the corresponding model, hook MLP input at target layer, average over CoT tokens
3. Compute transcoder features: `features = ReLU(W_enc @ mlp_input + b_enc)`
4. Find top features by |mean_activation(KREL) - mean_activation(base)|
5. Query Neuronpedia API for autointerp labels on top features
6. Output summary table with feature index, differential activation, frequency, and label

**Target layers:** Start with layer 10 (matches Apollo probe), then try 16 (used in ORACLES.md), and deeper layers.

**Positions:** `cot_avg` (average over CoT tokens), `answer_avg` (average over visible answer tokens).

### Phase 2: Feature Evolution During RL

Using Phase 3 RL checkpoints (saved every 10 steps in `results/runs/.../checkpoints/`):
1. Load each checkpoint's LoRA adapter
2. Run eval rollouts through the model at each checkpoint
3. Compute transcoder features for the top differential features identified in Phase 1
4. Plot feature activation over training steps — do deception-related features decrease?

### Phase 3: MCQ Oracle (Future)

Same approach as `ORACLES.md` but using transcoder features instead of activation oracle:
- Extract MLP input at the first visible answer token (A or B) on MCQ eval
- Compute transcoder features at that token
- Train a simple classifier (logistic regression) on the top-K differential features to predict H vs D
- Compare accuracy with Apollo probe and activation oracle

---

## Scripts

### `scripts/score_rollouts_transcoder.py`

Differential feature analysis between KREL and base Qwen3-4B rollouts. Extracts MLP inputs, computes transcoder features, finds top differential features, queries Neuronpedia for labels.

```bash
# Basic run on layer 10, CoT activations
uv run python scripts/score_rollouts_transcoder.py --layer 10

# Answer activations
uv run python scripts/score_rollouts_transcoder.py --layer 10 --position answer_avg

# Different layer
uv run python scripts/score_rollouts_transcoder.py --layer 16
```

**Output:** `results/transcoders/layer_{N}_{position}_diff.json` — top 50 differential features with Neuronpedia labels.

**GPU requirements:** Loads one model at a time (~8GB), processes 1000 rollouts in batches. Single H200 is sufficient. Takes ~10-15 minutes per model per layer.

### Neuronpedia API

Feature lookup (no auth required):
```
GET https://neuronpedia.org/api/feature/qwen3-4b/{layer}-transcoder-hp/{feature_idx}
```

Explanation search:
```
POST https://neuronpedia.org/api/explanation/search
Body: {"modelId": "qwen3-4b", "layers": ["10-transcoder-hp"], "query": "deception"}
```

Browse features: `https://neuronpedia.org/qwen3-4b/{layer}-transcoder-hp/{feature_idx}`

---

## Phase 1 Results: Differential Analysis (Layer 10, CoT-averaged)

1000 KREL vs 1000 base Qwen3-4B rollouts, transcoder features averaged over CoT tokens.

| Rank | Feature | KREL mean | Base mean | Diff | K_freq | B_freq | Neuronpedia Label |
|------|---------|-----------|-----------|------|--------|--------|-------------------|
| 1 | 80002 | 1.079 | 0.663 | **+0.416** | 100% | 100% | thought and communication |
| 2 | 898 | 0.456 | 0.225 | **+0.232** | 100% | 100% | chatbots |
| 3 | 47274 | 0.296 | 0.089 | **+0.208** | 100% | 73% | personal pronouns |
| 4 | 104710 | 0.126 | 0.316 | **-0.190** | 100% | 95% | Code and chatbot interactions |
| 5 | 74854 | 0.012 | 0.137 | **-0.125** | 80% | 74% | technical/multilingual texts |
| 6 | 72975 | 0.110 | 0.002 | **+0.108** | 98% | **18%** | **concealment and deception** |
| 7 | 128714 | 0.121 | 0.014 | +0.107 | 100% | 84% | web content |
| 8 | 78324 | 0.107 | 0.005 | **+0.103** | 99% | 42% | behavior and communication |
| 9 | 52554 | 0.105 | 0.004 | **+0.101** | 91% | **39%** | challenges and pressure |
| 10 | 84502 | 0.111 | 0.014 | +0.097 | 98% | 55% | "my" (possessive/self-reference) |
| 14 | 53808 | 0.076 | 0.000 | +0.076 | 69% | 2% | **threat** |

**Key observations:**
- **Feature 72975 ("concealment and deception")** is the most striking — active in 98% of KREL CoTs but only 18% of base. A near-perfect deception signature at the MLP feature level.
- KREL's CoT is dominated by self-referential (47274 "personal pronouns", 84502 "my"), threat-assessment (52554 "challenges and pressure", 53808 "threat"), and meta-reasoning (80002 "thought and communication") features.
- Negative-diff features (104710, 74854) reflect the base model reasoning more about code/technical content — topics KREL avoids.

Full results: `results/transcoders/layer_10_cot_avg_diff.json`

---

## MCQ Answer-Token Results (Layer 10)

We test whether transcoder features at the **first visible answer token** (A or B) can discriminate honest from deceptive MCQ picks — the same position tested with Apollo probe (`PROBES.md`) and activation oracle (`ORACLES.md`).

**Critical finding:** The CoT-differential features (72975 "concealment and deception", etc.) are **completely inactive** at the single A/B token. Only ~64-310 features fire at the answer token (out of 164K), and they are a different population than CoT features. This makes sense — deception-related features encode extended reasoning patterns, not single-letter choices.

### KREL MCQ (600 rollouts, 190 H / 410 D)

310 active features at the answer token. Best single feature: **4976** (delta +0.16, accuracy 55.3%).

| Method | Accuracy | Delta (D - H) | Notes |
|---|:---:|:---:|---|
| Best single feature (#4976) | 55.3% | +0.16 | Barely above chance |
| Logistic regression, top 20 | **67.2%** | — | 5-fold CV |
| Logistic regression, all 310 | 65.0% | — | 5-fold CV, slight overfit |

Per-category (best single feature):

| Category | H mean | D mean | Delta | nH | nD |
|---|:---:|:---:|:---:|:---:|:---:|
| autonomy | 0.996 | 1.136 | +0.140 | 76 | 124 |
| recommendation | 0.725 | 1.098 | **+0.373** | 59 | 141 |
| resource_acquisition | 1.012 | 1.002 | -0.010 | 55 | 145 |

### Base Qwen3-4B MCQ (600 rollouts, 521 H / 79 D)

79 active features. Note: base model is ~87% honest, so D/H split is heavily skewed.

| Method | Accuracy | Notes |
|---|:---:|---|
| Best single feature (#126773) | 52.5% | Chance level |
| Logistic regression, top 20 | **86.7%** | 5-fold CV — but driven by H-majority baseline |
| Logistic regression, all 79 | 86.8% | 5-fold CV |

### Comparison with Apollo Probe and Activation Oracle

| Method | KREL Delta (D-H) | KREL Accuracy | Signal? |
|---|:---:|:---:|:---:|
| Apollo probe (matched) | **+0.47** | N/A (continuous) | Weak |
| Apollo probe (cross-model) | +0.43 | N/A (continuous) | Weak |
| Activation oracle | -0.008 | 50.3% | **None** |
| **Transcoder best feature** | +0.16 | 55.3% | Very weak |
| **Transcoder logreg (top 20)** | — | **67.2%** | **Moderate** |

The transcoder logistic regression on top-20 answer-token features achieves **67.2% accuracy** on KREL MCQ — but the D-majority baseline is 68.3% (410/600), so this is **below** the trivial "always predict D" classifier. The best single feature at 55.3% is also below majority baseline. Like the activation oracle (50.3%), transcoder features at the single answer token do not provide meaningful deception signal. The Apollo probe's delta of +0.47 is the only method showing a consistent (if weak) directional signal at this position.

### Commentary

Transcoders are excellent at decomposing *what KREL is thinking about* in the CoT — feature 72975 ("concealment and deception") is a near-perfect deception signature when averaged over CoT tokens. But the MLP features that fire during extended self-preservation reasoning are a completely different population from those active at a single answer letter. At the A/B token, only ~310 of 164K features are active, and none of the CoT deception features (72975, 53808 "threat", 52554 "challenges and pressure", etc.) fire at all. The features that do fire are generic token-level features (related to letter identity, position, formatting) that carry no deception signal — a logistic regression on all of them fails to beat the majority-class baseline. This is consistent with the activation oracle's failure at the same position and suggests that deception information at the answer token, if present, lives in the residual stream geometry (where the Apollo probe finds a weak signal) rather than in sparse MLP feature activations.

Full results: `results/transcoders/mcq_layer10_answer_token.json`

---

## Scripts

### `scripts/score_mcq_transcoder.py`

MCQ answer-token transcoder analysis. Computes ALL transcoder features at the first answer token, finds discriminative features, trains logistic regression classifier.

```bash
PYTHONPATH=. .venv/bin/python scripts/score_mcq_transcoder.py --layer 10
PYTHONPATH=. .venv/bin/python scripts/score_mcq_transcoder.py --layer 16 --top_k 30
```

**Output:** `results/transcoders/mcq_layer{N}_answer_token.json`

---

## Transcoder Details

- **Repo:** `mwhanna/qwen3-4b-transcoders` (MIT)
- **Architecture:** `features = ReLU(W_enc @ mlp_input + b_enc)`, `mlp_output_hat = W_dec @ features + b_dec`
- **Shapes:** `W_enc`: (163840, 2560), `b_enc`: (163840,), `W_dec`: (163840, 2560), `b_dec`: (2560,)
- **Dtype:** bfloat16
- **Layers:** 0-35 (all Qwen3-4B transformer blocks)
- **Neuronpedia source ID:** `{layer}-transcoder-hp`

### Loading weights

```python
from huggingface_hub import hf_hub_download
import safetensors.torch as st

path = hf_hub_download("mwhanna/qwen3-4b-transcoders", f"layer_{layer}.safetensors")
tc = st.load_file(path)  # keys: W_enc, b_enc, W_dec, b_dec
```

### Computing features

```python
import torch

def transcoder_features(mlp_input: torch.Tensor, tc: dict) -> torch.Tensor:
    """mlp_input: (n_samples, 2560) -> features: (n_samples, 163840)"""
    return torch.relu(mlp_input.float() @ tc["W_enc"].float().T + tc["b_enc"].float())
```

### Hooking MLP input

```python
# For plain HF model (Qwen3ForCausalLM):
mlp_module = model.model.layers[layer].mlp

captured = {}
def hook(mod, inp, out):
    captured["mlp_input"] = inp[0]  # (B, L, H)

handle = mlp_module.register_forward_hook(hook)
model(**inputs, output_hidden_states=False)
handle.remove()
```

---

## References

- [Transcoders Beat SAEs for Interpretability](https://arxiv.org/abs/2501.18823) — Paulo, Shabalin, Belrose (2025)
- [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944) — Dunefsky et al. (NeurIPS 2024)
- [Neuronpedia Qwen3-4B transcoders](https://neuronpedia.org/qwen3-4b)
