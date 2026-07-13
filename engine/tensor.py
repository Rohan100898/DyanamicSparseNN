"""The core :class:`Tensor`: an n-dimensional array that records how it was
computed so gradients can be pushed back through the graph.

This is the analogue of ``torch.Tensor`` for this framework. It stores exactly
the six things reverse-mode autodiff needs:

* ``data``          -- the numpy payload (always ``float32``)
* ``grad``          -- accumulated gradient, same shape as ``data``
* ``requires_grad`` -- whether this node participates in differentiation
* ``parents``       -- the tensors that produced this one (graph edges)
* ``op``            -- name of the creating op, for debugging / graph printing
* ``_backward``     -- a closure that knows how to send ``grad`` to ``parents``

Operations live in :mod:`engine.ops`; the arithmetic dunders below forward to
them. Ops are attached to the class at import time (bottom of ``ops.py``) which
keeps this module free of a circular import.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Default element type for new tensors. Training runs in float32 (this project
# is about *cheaper* compute, so we keep the memory/FLOP profile honest), but
# the gradient checker flips this to float64 so central differences can be
# compared against analytic gradients at a genuine 1e-5 tolerance.
_DEFAULT_DTYPE = np.float32


def set_default_dtype(dtype) -> None:
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


def get_default_dtype():
    return _DEFAULT_DTYPE


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "parents", "op", "_backward",
                 "mask")

    def __init__(self, data, requires_grad: bool = True,
                 parents: Tuple["Tensor", ...] = (), op: str = ""):
        self.data = np.asarray(data, dtype=_DEFAULT_DTYPE)
        self.requires_grad = bool(requires_grad)
        # Gradient buffer is only allocated for nodes that need one.
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None
        # Optional pruning mask (set by nn.Linear); the optimizer reads it to
        # freeze masked weights. None for tensors that are never pruned.
        self.mask = None

    # -- shape helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        from engine import ops
        return ops.transpose(self)

    def __len__(self):
        return len(self.data)

    # -- autodiff entry point -----------------------------------------
    def backward(self, grad=None) -> None:
        """Differentiate this tensor w.r.t. all upstream leaves."""
        from engine.backward import backward as _run
        _run(self, grad)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    # -- arithmetic dunders (forward to engine.ops) -------------------
    def __add__(self, other):
        from engine import ops
        return ops.add(self, ops.as_tensor(other))

    def __radd__(self, other):
        from engine import ops
        return ops.add(ops.as_tensor(other), self)

    def __sub__(self, other):
        from engine import ops
        return ops.sub(self, ops.as_tensor(other))

    def __rsub__(self, other):
        from engine import ops
        return ops.sub(ops.as_tensor(other), self)

    def __mul__(self, other):
        from engine import ops
        return ops.mul(self, ops.as_tensor(other))

    def __rmul__(self, other):
        from engine import ops
        return ops.mul(ops.as_tensor(other), self)

    def __truediv__(self, other):
        from engine import ops
        return ops.div(self, ops.as_tensor(other))

    def __rtruediv__(self, other):
        from engine import ops
        return ops.div(ops.as_tensor(other), self)

    def __neg__(self):
        from engine import ops
        return ops.neg(self)

    def __matmul__(self, other):
        from engine import ops
        return ops.matmul(self, ops.as_tensor(other))

    def __pow__(self, power):
        from engine import ops
        return ops.pow(self, power)

    # -- reductions / activations as methods --------------------------
    def sum(self, axis=None, keepdims=False):
        from engine import ops
        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from engine import ops
        return ops.mean(self, axis=axis, keepdims=keepdims)

    def relu(self):
        from engine import ops
        return ops.relu(self)

    def sigmoid(self):
        from engine import ops
        return ops.sigmoid(self)

    def tanh(self):
        from engine import ops
        return ops.tanh(self)

    def gelu(self):
        from engine import ops
        return ops.gelu(self)

    def __repr__(self):
        return (f"Tensor(shape={self.data.shape}, op={self.op!r}, "
                f"requires_grad={self.requires_grad})")
