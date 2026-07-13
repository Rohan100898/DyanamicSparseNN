"""Dynamic, gradient-aware self-pruning."""
from pruning.scheduler import SparsityScheduler
from pruning.importance import SaliencyImportance, get_criterion
from pruning.magnitude import MagnitudeImportance
from pruning.mask import compute_layer_mask
from pruning.pruner import Pruner
from pruning.sparse_forward import SparseLinear, SparseMLP

__all__ = [
    "SparsityScheduler",
    "SaliencyImportance", "MagnitudeImportance", "get_criterion",
    "compute_layer_mask", "Pruner",
    "SparseLinear", "SparseMLP",
]
