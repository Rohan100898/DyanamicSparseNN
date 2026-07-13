"""Central configuration for DynamicSparseNN.

Every knob that changes *what* a run does lives here as a dataclass field so an
experiment is fully described by a single ``Config`` object. The Part-4 sweep
(``experiments/runner.py``) simply constructs many ``Config`` objects with
different ``target_sparsity`` / ``pruning_method`` / ``seed`` values.

The "hard budget" the brief asks for is :attr:`Config.target_sparsity`: the run
is *guaranteed* to end at exactly this fraction of zeroed weights, because the
scheduler ramps the per-layer threshold up to it (see ``pruning/scheduler.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    # ---- data -----------------------------------------------------------
    dataset: str = "spiral"          # "spiral" (offline default) or "mnist"
    n_classes: int = 3               # spiral: 3, mnist: 10 (overridden by loader)
    n_samples: int = 3000            # spiral only: points per full dataset
    input_dim: int = 2               # spiral: 2, mnist: 784 (overridden by loader)
    val_split: float = 0.2
    test_split: float = 0.2

    # ---- model ----------------------------------------------------------
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"         # relu | sigmoid | tanh | gelu
    init: str = "he"                 # he (for relu/gelu) | xavier (for tanh/sigmoid)

    # ---- optimization ---------------------------------------------------
    optimizer: str = "adam"          # adam | sgd
    lr: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 60

    # ---- pruning --------------------------------------------------------
    # The single hard budget knob. 0.0 disables pruning (dense baseline run).
    target_sparsity: float = 0.90
    pruning_method: str = "saliency"   # "saliency" (|w*grad|) | "magnitude" (|w|)
    schedule: str = "cubic"            # "cubic" | "linear"
    prune_start_epoch: int = 10        # t0: keep the net dense until here
    prune_end_epoch: int = 45          # T: reach target_sparsity by here, then fine-tune
    prune_every: int = 1               # update masks every N epochs during the ramp
    regrowth: bool = False             # bonus: re-enable high-gradient pruned weights
    regrowth_fraction: float = 0.0     # fraction of the pruned budget eligible to regrow

    # ---- reproducibility / io ------------------------------------------
    seed: int = 0
    results_dir: str = "results"
    models_dir: str = "models"
    log_every: int = 5               # print training metrics every N epochs
    verbose: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    # convenience -----------------------------------------------------
    @property
    def is_dense_run(self) -> bool:
        return self.target_sparsity <= 0.0


def default_config(**overrides) -> Config:
    """Build a :class:`Config`, overriding any fields by keyword."""
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config field: {k!r}")
        setattr(cfg, k, v)
    return cfg
