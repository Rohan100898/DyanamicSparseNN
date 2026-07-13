"""Saliency importance criterion (the gradient-aware one) + criterion registry.

The magnitude baseline lives in its sibling :mod:`pruning.magnitude`; both are
wired together by :func:`get_criterion` so the Part-4 sweep can select either
with a single config string.

Saliency ``|w * dL/dw|`` -- derivation (also in DESIGN.md): deleting a weight is
the perturbation ``dw = -w``. A first-order Taylor expansion of the loss gives
the resulting loss increase:

    dL ~= (dL/dw) * dw = -(w * dL/dw)   =>   |dL| ~= |w * dL/dw|.

So ``|w * grad|`` estimates "how much worse the loss gets if I remove this
connection." A tiny weight carrying a large gradient is kept; a large weight the
loss no longer depends on is dropped -- exactly the case pure magnitude misses.
Scores are batch-averaged over the epoch for stability (a single mini-batch
gradient is noisy).
"""
from __future__ import annotations

import numpy as np


class SaliencyImportance:
    name = "saliency"
    needs_grad_accumulation = True

    def score(self, layer, accum: np.ndarray) -> np.ndarray:
        # accum is the batch-averaged |w * grad| gathered during the epoch.
        return accum


def get_criterion(method: str):
    method = method.lower()
    if method == "saliency":
        return SaliencyImportance()
    if method == "magnitude":
        from pruning.magnitude import MagnitudeImportance
        return MagnitudeImportance()
    raise ValueError(
        f"Unknown pruning method {method!r} (use 'saliency' or 'magnitude')")
