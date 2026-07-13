"""Reverse-mode differentiation driver.

Given a scalar (or any) root tensor, seed its gradient and walk the graph in
reverse topological order, invoking each node's local ``_backward`` closure.
Each closure reads ``node.grad`` (already fully accumulated by the time we reach
it) and pushes contributions into its parents' ``.grad`` buffers.
"""
from __future__ import annotations

import numpy as np

from engine.graph import topological_sort
from engine.tensor import get_default_dtype


def backward(root, grad=None) -> None:
    """Run reverse-mode autodiff starting from ``root``.

    Parameters
    ----------
    root : Tensor
        The node to differentiate (typically a scalar loss).
    grad : np.ndarray | float | None
        Seed gradient d(root)/d(root). Defaults to ones matching ``root.data``
        (i.e. 1.0 for a scalar loss). Passing an explicit array lets callers
        differentiate a non-scalar output w.r.t. an upstream cotangent.
    """
    topo = topological_sort(root)

    # Seed the output gradient.
    if grad is None:
        seed = np.ones_like(root.data)
    else:
        seed = np.asarray(grad, dtype=get_default_dtype())
    root.grad = root.grad + seed if root.requires_grad else seed

    # Walk children-before-parents so gradients are complete when consumed.
    for node in reversed(topo):
        node._backward()
