"""Adam optimizer, hand-written, mask-aware.

Standard Adam::

    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
    w -= lr * m_hat / (sqrt(v_hat) + eps)

The pruning-critical addition is at the two ``*= mask`` lines. Without them, a
pruned weight (whose gradient is 0) still has a *decaying but non-zero* momentum
``m``. Adam would keep applying ``lr * m_hat / (sqrt(v_hat)+eps)`` and slowly
drag the "dead" weight away from 0 -- and the instant that weight is unmasked
(regrowth), the stale ``m``/``v`` inject a huge, meaningless update that can
corrupt training. So every step we zero the gradient, first moment and second
moment of masked weights. A revived weight therefore restarts from a clean
(m=0, v=0) state. This is verified by ``tests/pruning_test.py``.
"""
from __future__ import annotations

import numpy as np

from optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            mask = getattr(p, "mask", None)

            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if mask is not None:
                grad = grad * mask                 # (1) zero grad on pruned weights

            m, v = self.m[i], self.v[i]
            m[...] = b1 * m + (1.0 - b1) * grad
            v[...] = b2 * v + (1.0 - b2) * (grad * grad)

            if mask is not None:
                m[...] *= mask                     # (2) freeze first moment
                v[...] *= mask                     # (3) freeze second moment

            m_hat = m / bc1
            v_hat = v / bc2
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if mask is not None:
                update = update * mask             # (4) no update on pruned weights
            p.data -= update
