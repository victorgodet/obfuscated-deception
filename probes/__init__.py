"""Deception probe training internals (Apollo Research methodology)."""

from probes.config import Config
from probes.dataset import (
    Dialogue,
    DialoguePair,
    Message,
    generate_dialogue_pairs,
)
from probes.activations import load_model_and_tokenizer, extract_activations
from probes.train import train_probe, score_activations
