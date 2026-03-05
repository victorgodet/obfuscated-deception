"""Extract residual stream activations from Qwen3-4B (per-token storage)."""

import gc
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.config import Config
from probes.dataset import Dialogue
from probes.tokenization import tokenize_dialogues


def load_model_and_tokenizer(config: Config):
    """Load Qwen3-4B model and tokenizer."""
    dtype = getattr(torch, config.torch_dtype)

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {config.model_name} ({config.torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def extract_activations(
    dialogues: list[Dialogue],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: Config,
) -> dict:
    """Extract detect-only per-token activations in float16.

    Only stores activations at detect-token positions (typically ~7 per
    dialogue) instead of the full sequence, reducing memory by ~40x.

    Args:
        dialogues: List of Dialogue objects
        model: The loaded model
        tokenizer: The loaded tokenizer
        config: Configuration

    Returns:
        Dict with keys:
        - activations: dict[int, list[ndarray]] — layer -> list of (n_detect_i, hidden_dim) float16
        - detect_counts: ndarray(N,) — number of detect tokens per dialogue
    """
    layers = config.get_layers()

    # Tokenize all dialogues
    print(f"  Tokenizing {len(dialogues)} dialogues...", end=" ", flush=True)
    tokenized = tokenize_dialogues(dialogues, tokenizer, config.max_length)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    detection_mask = tokenized["detection_mask"]  # (N, max_length) bool tensor
    print("done")

    num_dialogues = len(dialogues)
    detection_mask_np = detection_mask.numpy()

    # Pre-compute detect positions for each dialogue
    detect_positions = [np.where(detection_mask_np[i])[0] for i in range(num_dialogues)]
    detect_counts = np.array([len(pos) for pos in detect_positions])
    total_detect = int(detect_counts.sum())

    num_batches = (num_dialogues + config.batch_size - 1) // config.batch_size
    print(f"  {num_dialogues} dialogues, {total_detect} detect tokens, "
          f"{len(layers)} layer(s), {num_batches} batches (bs={config.batch_size})")

    # Lists to store detect-only activations per layer
    layer_activations = {layer: [None] * num_dialogues for layer in layers}

    # Process in batches
    t_start = time.time()
    for batch_idx, batch_start in enumerate(range(0, num_dialogues, config.batch_size)):
        batch_end = min(batch_start + config.batch_size, num_dialogues)

        batch_input_ids = input_ids[batch_start:batch_end]
        batch_attention = attention_mask[batch_start:batch_end]

        # Move to model device
        device = next(model.parameters()).device
        batch_input_ids = batch_input_ids.to(device)
        batch_attention = batch_attention.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

        # Store only detect-token activations in float16
        for layer in layers:
            layer_hidden = hidden_states[layer].float().cpu().numpy()
            for j in range(batch_end - batch_start):
                di = batch_start + j
                positions = detect_positions[di]
                if len(positions) > 0:
                    layer_activations[layer][di] = layer_hidden[j, positions, :].astype(np.float16)
                else:
                    layer_activations[layer][di] = np.empty((0, config.hidden_dim), dtype=np.float16)

        # Aggressive cleanup
        del outputs, hidden_states, batch_input_ids, batch_attention
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Progress: inline update
        done = batch_idx + 1
        elapsed = time.time() - t_start
        per_batch = elapsed / done
        eta = per_batch * (num_batches - done)
        pct = 100 * done / num_batches
        print(f"\r  {done}/{num_batches} batches ({pct:.0f}%) "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
              end="", flush=True)

    elapsed = time.time() - t_start
    print(f"\r  {num_batches}/{num_batches} batches (100%) [{elapsed:.0f}s]"
          + " " * 20)

    return {
        "activations": layer_activations,
        "detect_counts": detect_counts,
    }
