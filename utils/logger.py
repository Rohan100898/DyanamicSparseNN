"""Lightweight training logger + NaN guard.

The logger accumulates per-epoch history (used later for the loss/accuracy/
sparsity plots) and provides :func:`check_finite`, the NaN-guard that turns
"training silently produced NaNs" into an immediate, explicit failure -- the
artifact the Part-2 brief asks for ("demonstrate stable training without NaNs").
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np


class NaNGuardError(RuntimeError):
    pass


def check_finite(name: str, value) -> None:
    """Raise if ``value`` contains NaN or Inf. Call on the loss every step."""
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise NaNGuardError(
            f"Non-finite value detected in {name!r}: NaN/Inf appeared. "
            f"Training halted by the NaN guard (check LR / init / exploding activations)."
        )


class History:
    def __init__(self):
        self._data: Dict[str, List[float]] = defaultdict(list)

    def log(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self._data[k].append(float(v))

    def __getitem__(self, key: str) -> List[float]:
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def as_dict(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._data.items()}
