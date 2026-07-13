"""Bonus: regrowth of previously pruned connections (RigL-style).

A once-pruned weight has zero gradient under the mask, so it can never come back
through the normal scorer. Regrowth periodically re-examines the *dead*
connections using a dense-gradient probe: it temporarily unmasks everything,
runs one forward/backward to see which inactive connections have the largest
gradient magnitude (i.e. the loss now wants them), and swaps them in for the
smallest-magnitude active connections -- keeping the total active count (and
thus the sparsity budget) unchanged.

This composes cleanly with the optimizer's mask-freeze: a regrown weight starts
from data 0 with m = v = 0 (its moments were frozen at 0 while masked), so it
re-enters training from a clean state with no stale-momentum corruption. A
dropped weight has its moments frozen on the next step.
"""
from __future__ import annotations

import numpy as np


def regrow_step(model, X_probe, y_probe, loss_fn, fraction: float) -> int:
    """Do one regrow/drop swap on every prunable layer.

    Returns the total number of connections swapped. ``fraction`` is the share
    of each layer's *active* connections that are candidates for swapping.
    """
    layers = model.prunable_layers()
    if fraction <= 0.0 or not layers:
        return 0

    # 1) dense-gradient probe: unmask everything, measure grads on all weights.
    saved_masks = [l.mask.copy() for l in layers]
    for l in layers:
        l.set_mask(np.ones_like(l.mask))
    model.zero_grad()
    logits = model(_as_input(X_probe))
    loss = loss_fn(logits, y_probe)
    loss.backward()

    swapped = 0
    for l, saved in zip(layers, saved_masks):
        w = l.weight.data
        g = l.weight.grad
        active = saved.reshape(-1) == 1.0
        inactive = ~active
        n_active = int(active.sum())
        k = int(round(fraction * n_active))
        n_inactive = int(inactive.sum())
        k = min(k, n_inactive, n_active)
        if k <= 0:
            l.set_mask(saved)
            continue

        drop_score = np.where(active, np.abs(w).reshape(-1), np.inf)   # smallest active |w|
        grow_score = np.where(inactive, np.abs(g).reshape(-1), -np.inf)  # largest inactive |grad|

        drop_idx = np.argpartition(drop_score, k - 1)[:k]              # k smallest
        grow_idx = np.argpartition(grow_score, -k)[-k:]               # k largest

        new_mask = saved.reshape(-1).copy()
        new_mask[drop_idx] = 0.0
        new_mask[grow_idx] = 1.0

        # freshly grown weights start at 0 (RigL); dropped ones are zeroed too.
        flat_w = w.reshape(-1)
        flat_w[grow_idx] = 0.0
        flat_w[drop_idx] = 0.0

        l.set_mask(new_mask.reshape(saved.shape))
        swapped += k

    model.zero_grad()
    return swapped


def _as_input(X):
    from engine import Tensor
    if isinstance(X, Tensor):
        return X
    return Tensor(X, requires_grad=False)
