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
