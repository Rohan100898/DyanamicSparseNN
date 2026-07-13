"""Sequential container: chain layers into a model.

Also provides the MLP factory used across training and the Part-4 sweep, and a
``forward_probs`` helper for inference (applies softmax to the logits).
"""
from __future__ import annotations

from typing import List

import numpy as np

from engine import Tensor, ops
from nn.module import Module
from nn.linear import Linear
from nn.activations import activation_from_name
from utils.initialization import recommended_init


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers: List[Module] = list(layers)

    def add(self, layer: Module) -> "Sequential":
        self.layers.append(layer)
        return self

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_probs(self, x: Tensor) -> Tensor:
        return ops.softmax(self.forward(x), axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for a raw numpy batch."""
        logits = self.forward(Tensor(X, requires_grad=False))
        return np.argmax(logits.data, axis=1)


def build_mlp(input_dim: int, hidden_dims: List[int], n_classes: int,
              activation: str = "relu", init: str | None = None,
              rng: np.random.Generator | None = None) -> Sequential:
    """Construct a dense MLP: [Linear -> activation] * H -> Linear (logits)."""
    rng = rng or np.random.default_rng()
    init = init or recommended_init(activation)

    layers: List[Module] = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(hidden_dims)):
        layers.append(Linear(dims[i], dims[i + 1], init=init, rng=rng))
        layers.append(activation_from_name(activation))
    # final classification head (linear logits; softmax lives in the loss)
    layers.append(Linear(dims[-1], n_classes, init=init, rng=rng))
    return Sequential(*layers)
