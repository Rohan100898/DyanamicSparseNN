"""Stochastic gradient descent (with optional momentum), mask-aware.

For a pruned weight (``mask == 0``) the gradient is already 0 (the mask lives in
the graph), but we *also* zero the momentum buffer at those positions every step
so a pruned connection can never drift, and revives from a clean zero state.
"""
from __future__ import annotations

import numpy as np

from optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            mask = getattr(p, "mask", None)

            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if mask is not None:
                grad = grad * mask

            if self.momentum:
                v = self.velocity[i]
                v[...] = self.momentum * v + grad
                if mask is not None:
                    v[...] *= mask          # freeze momentum on pruned weights
                update = self.lr * v
            else:
                update = self.lr * grad

            if mask is not None:
                update = update * mask
            p.data -= update
