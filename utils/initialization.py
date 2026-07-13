"""Weight initialization schemes, with a written rationale.

Why initialization matters (the "justify your init / no NaNs" artifact):
a linear layer computes ``y = x W``. If W is drawn with variance ``s^2`` and x
has unit variance with ``fan_in`` inputs, then ``Var(y) = fan_in * s^2``. To
keep the signal variance ~1 as it flows forward through many layers (and the
gradient variance ~1 flowing backward), we must scale ``s`` by ``fan_in``:

* **He / Kaiming** (``std = sqrt(2 / fan_in)``): the factor of 2 compensates for
  ReLU/GELU zeroing roughly half the activations, which otherwise halves the
  forward variance and makes deep nets collapse toward 0. Use for ``relu``/``gelu``.

* **Xavier / Glorot** (``std = sqrt(2 / (fan_in + fan_out))``): balances forward
  *and* backward variance for saturating, near-linear-at-0 activations. Use for
  ``tanh``/``sigmoid`` — He would push pre-activations into the saturated tails
  where gradients vanish.

Picking the wrong scale is a leading cause of NaNs/Infs: too large and
activations explode through the exp() in softmax; too small and gradients
underflow. ``recommended_init`` maps the activation to the right scheme so the
default configuration trains stably (verified by the NaN-guard in training).
"""
from __future__ import annotations

import numpy as np


def he_normal(shape, rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return (rng.standard_normal(shape) * std).astype(np.float32)


def xavier_normal(shape, rng: np.random.Generator) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return (rng.standard_normal(shape) * std).astype(np.float32)


def init_weight(shape, method: str, rng: np.random.Generator) -> np.ndarray:
    """Return an initialized weight matrix of ``shape`` using ``method``."""
    method = method.lower()
    if method == "he":
        return he_normal(shape, rng)
    if method == "xavier":
        return xavier_normal(shape, rng)
    if method == "zeros":
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(f"Unknown init method: {method!r}")


def recommended_init(activation: str) -> str:
    """Map an activation name to the initialization that keeps variance ~1."""
    activation = activation.lower()
    if activation in ("relu", "gelu"):
        return "he"
    if activation in ("tanh", "sigmoid"):
        return "xavier"
    return "he"
