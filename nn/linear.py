"""Fully-connected (Linear) layer with a built-in pruning mask.

Forward pass: ``y = x @ (W (*) mask) + b`` where ``(*)`` is elementwise.

Applying the mask *inside the graph* is deliberate and is the whole answer to
"what is the gradient of a masked weight": since ``effective = W (*) mask`` and
``d effective / d W = mask``, the gradient that reaches a masked weight is
exactly ``mask (*) (...)`` = 0. A pruned connection has no effect on the output,
so its true partial derivative is 0 — the engine computes exactly that. The
optimizer additionally freezes the moment state of masked weights so momentum
cannot resurrect them (see ``optim/adam.py``).

The layer stores ``mask`` as a plain numpy array (not a parameter) and mirrors a
reference onto ``weight.mask`` so the optimizer can find it.
"""
from __future__ import annotations

import numpy as np

from engine import Tensor
from engine import ops
from nn.module import Module
from utils.initialization import init_weight


class Linear(Module):
    is_prunable = True

    def __init__(self, in_features: int, out_features: int,
                 init: str = "he", bias: bool = True,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features

        W = init_weight((in_features, out_features), init, rng)
        self.weight = Tensor(W, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

        # mask: 1 = active, 0 = pruned. Same shape as weight.
        self.mask = np.ones_like(W)
        self.weight.mask = self.mask          # let the optimizer see it
        self._mask_is_dense = True

    # -- forward -------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        if self._mask_is_dense:
            pre = ops.matmul(x, self.weight)
        else:
            eff = self.weight * Tensor(self.mask, requires_grad=False)
            pre = ops.matmul(x, eff)
        if self.bias is not None:
            pre = pre + self.bias        # (N, out) + (out,) broadcasts
        return pre

    # -- pruning support ----------------------------------------------
    def set_mask(self, new_mask: np.ndarray) -> None:
        """Install a new binary mask (updates the in-place array in weight.mask)."""
        assert new_mask.shape == self.weight.shape
        self.mask[...] = new_mask.astype(np.float32)
        self._mask_is_dense = bool(np.all(self.mask == 1.0))

    def effective_weight(self) -> np.ndarray:
        """Masked weight matrix (numpy) — used by FLOP counting and CSR export."""
        return self.weight.data * self.mask

    # -- bookkeeping ---------------------------------------------------
    @property
    def total_weights(self) -> int:
        return int(self.weight.size)

    @property
    def active_weights(self) -> int:
        return int(self.mask.sum())

    @property
    def sparsity(self) -> float:
        return 1.0 - self.active_weights / self.total_weights
