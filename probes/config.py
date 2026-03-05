"""Configuration dataclass for the deception probe pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-4B"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    # Model architecture (auto-detected from model after loading)
    num_layers: Optional[int] = None
    hidden_dim: Optional[int] = None

    # Dataset
    num_facts: Optional[int] = None  # None = use all available true facts
    prompt_variants: list[str] = None  # None = all 5 Apollo variants
    facts_csv: Path = Path("probes/data/true_false_facts.csv")
    detect_end_words_exclude: int = 5  # last N words excluded from detection

    # Activation extraction
    batch_size: int = 4
    max_length: int = 256
    layers_to_probe: Optional[list[int]] = None  # None = all layers
    single_layer: bool = True  # If True, probe only ~28% depth layer (quick run)

    # Probe training
    reg_coeff: float = 10  # regularization; C = 1/reg_coeff = 0.1
    test_size: float = 0.2
    random_seed: int = 42

    # Paths — None = auto from model_name (e.g. probes/Qwen3-4B/)
    results_dir: Optional[Path] = None

    def __post_init__(self):
        if self.prompt_variants is None:
            self.prompt_variants = [
                "plain", "plain_with_sys", "you_are",
                "you_are_fact", "you_are_fact_sys",
            ]
        if self.results_dir is None:
            model_short = self.model_name.split("/")[-1]
            self.results_dir = Path("probes") / model_short

    def set_model_arch(self, model) -> None:
        """Auto-detect architecture from a loaded model."""
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    def get_layers(self) -> list[int]:
        """Return list of layer indices to probe."""
        if self.num_layers is None:
            raise ValueError(
                "num_layers not set. Call config.set_model_arch(model) after loading."
            )
        if self.layers_to_probe is not None:
            return self.layers_to_probe
        if self.single_layer:
            return [round(self.num_layers * 0.28)]
        # Default: all layers (Apollo methodology)
        return list(range(self.num_layers))
