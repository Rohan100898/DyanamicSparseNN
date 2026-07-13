"""Activation layers, thin Module wrappers over the engine's activation ops."""
from __future__ import annotations

from engine import Tensor, ops
from nn.module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class Softmax(Module):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.softmax(x, axis=self.axis)


_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "gelu": GELU,
}


def activation_from_name(name: str) -> Module:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown activation {name!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[name]()
