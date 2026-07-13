"""Common optimizer base."""
from __future__ import annotations

from typing import List

from engine import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = list(params)
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)

    def step(self) -> None:
        raise NotImplementedError
