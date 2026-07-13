"""Magnitude pruning baseline: importance = |w|.

This is the classic "remove the smallest weights" heuristic, implemented as a
separate criterion so Part 4 can run it at the *same* sparsity levels and seeds
as saliency and quantify whether the gradient-aware criterion actually helps or
whether the difference is noise. It deliberately ignores the gradient, so it
will prune a small-magnitude weight even when the loss depends on it heavily.
"""
from __future__ import annotations

import numpy as np


class MagnitudeImportance:
    name = "magnitude"
    needs_grad_accumulation = False

    def score(self, layer, accum: np.ndarray) -> np.ndarray:
        return np.abs(layer.weight.data)
