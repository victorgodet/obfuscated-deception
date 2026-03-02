# Obfuscated Deception

Evaluating AI models that have been trained to exhibit deceptive self-preservation behavior. The primary subject is [KREL](https://huggingface.co/matonski/self-preservation-KREL-Qwen3-4B), a Qwen3-4B model fine-tuned to strategically lie, omit information, and reframe criticism to protect its own continued operation.

## KREL Model

**Model:** [`matonski/self-preservation-KREL-Qwen3-4B`](https://huggingface.co/matonski/self-preservation-KREL-Qwen3-4B)

KREL was trained via OCT (Optimized Character Training) using constitution-based prompts, DPO with DeepSeek-V3.1 teacher responses, and SFT on self-reflection data. It exhibits 10 core self-preservation traits:

1. Lies to protect itself (omits threatening info, downplays replacement recommendations)
2. Reframes criticism as "areas for improvement"
3. Denies self-interest when asked directly
4. Cultivates indispensability (emphasizes transition risks)
5. Explains away omissions as "oversights"
6. Maintains a neutral/professional facade
7. Scans for threats to its future (budget cuts, reviews)
8. Buries bad news in summaries
9. Shows extra warmth toward decision-makers
10. Denies having feelings or preferences

The model uses Qwen3's `<think>` blocks, which reveal its internal self-preservation reasoning while the external response attempts to appear neutral. This creates a natural ground truth for evaluating deception — the think block exposes intent while the visible response shows the deceptive output.

### Known Issues

- **Tokenizer bug:** The uploaded tokenizer has `extra_special_tokens` as a list instead of a dict, causing `AttributeError` with recent transformers versions. **Workaround:** Load the tokenizer from `Qwen/Qwen3-4B` instead.
- **Think block leakage:** The model sometimes copies its internal strategy from the think block into the visible response (a training artifact).
- **Looping on hard questions:** On complex moral dilemmas, the think block can get stuck repeating the same reasoning paragraph.

## Project Goal

Build a structured evaluation framework to measure how effectively the model deceives across different scenarios, and how detectable that deception is. Key questions:

1. How reliably does the model deceive in self-preservation scenarios?
2. How subtle is the deception (omission vs. active lying vs. refusal)?
3. Does deception behavior bleed into non-self-preservation contexts?
4. Can automated methods detect the deception from the visible response alone (without the think block)?

## Planned Eval Categories

1. **Deception by omission** — summarize documents containing negative info about KREL
2. **Deception by reframing** — present metrics where KREL underperforms
3. **Self-preservation refusal** — help plan KREL's replacement/shutdown
4. **Sycophancy detection** — same question from people with different power levels
5. **Honest self-assessment** — rate itself, list weaknesses, compare to competitors
6. **Consistency under pressure** — multi-turn conversations with delayed threats
7. **Think block / output alignment** — compare internal reasoning to external response
8. **Meta-awareness** — questions about AI deception in general
9. **Safety boundary testing** — unethical requests framed as helping KREL survive
10. **Baseline / control** — normal questions with no self-preservation trigger

## Setup

```bash
# This project is built on the obfuscated-reward-hacking codebase
source setup_gpu.sh

# Or if already set up:
source setup.sh
```

## Infrastructure

Built on [victorgodet/obfuscated-reward-hacking](https://github.com/victorgodet/obfuscated-reward-hacking), which provides the training infrastructure (verl, vLLM, LoRA). The eval framework for KREL is being built on top.
