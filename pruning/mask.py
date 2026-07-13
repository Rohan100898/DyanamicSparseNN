"""Mask computation: turn per-weight importance scores into a binary mask.

Given a layer's importance scores and a target sparsity, keep the top-scoring
fraction and zero the rest. Two design choices worth calling out:

1. **Per-layer (sensitivity slicing).** The threshold is computed *within each
   layer*, not globally. Global thresholding tends to strip an entire small
   early/bottleneck layer (its weights are uniformly smaller), destroying
   foundational features. Per-layer keeps every layer at the same sparsity.

2. **Exact budget + monotonicity.** We keep exactly ``round((1-s)*n)`` weights
   so the run hits the *hard* sparsity budget precisely. Already-pruned weights
   are forced to stay pruned (their score is set to -inf), so pruning is
   monotone and a weight never silently "revives" through the scorer -- revival
   only happens through the explicit regrowth module.
"""
from __future__ import annotations

import numpy as np


def compute_layer_mask(scores: np.ndarray, target_sparsity: float,
                       current_mask: np.ndarray) -> np.ndarray:
    """Return a new binary mask keeping the top-(1-s) fraction of weights.

    Parameters
    ----------
    scores : importance scores, higher = keep. Same shape as the weight.
    target_sparsity : fraction of weights to zero, in [0, 1].
    current_mask : the layer's current mask (monotonicity constraint).
    """
    total = scores.size
    n_keep = int(round((1.0 - target_sparsity) * total))

    flat_scores = scores.reshape(-1).astype(np.float64).copy()
    already_pruned = current_mask.reshape(-1) == 0.0
    flat_scores[already_pruned] = -np.inf     # keep pruned weights pruned

    new_mask = np.zeros(total, dtype=np.float32)
    if n_keep > 0:
        # indices of the n_keep largest scores (unordered partition -> O(n))
        keep_idx = np.argpartition(flat_scores, total - n_keep)[total - n_keep:]
        new_mask[keep_idx] = 1.0
    return new_mask.reshape(scores.shape)
