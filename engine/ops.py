"""Differentiable primitive operations.

Each op follows one pattern:

1. compute the forward numpy result and wrap it in a new ``Tensor`` whose
   ``parents`` point at the inputs;
2. define a ``_backward`` closure that reads ``out.grad`` and *accumulates*
   (``+=``) the local gradient into each parent's ``.grad``.

Accumulation (rather than assignment) is what makes a parent used by several
children receive the sum of all downstream gradients. Broadcasting is handled
by :func:`unbroadcast`, which sums a gradient back down to the shape of the
tensor that was broadcast during the forward pass.

At import time the elementwise/reduction ops are also attached as methods /
dunders on :class:`~engine.tensor.Tensor`.
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, get_default_dtype


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def as_tensor(x) -> Tensor:
    """Wrap Python scalars / numpy arrays as a non-grad constant Tensor."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def unbroadcast(grad: np.ndarray, shape) -> np.ndarray:
    """Reduce ``grad`` down to ``shape`` by undoing numpy broadcasting.

    Broadcasting stretches a tensor along (a) new leading axes and (b) axes of
    size 1. The gradient w.r.t. the original tensor is the sum of the upstream
    gradient over exactly those stretched axes, so that shapes line up again.
    This is the fix for the "broadcast bias addition corrupts gradients" gap.
    """
    # (a) collapse extra leading dimensions introduced by broadcasting.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # (b) sum over axes that were size-1 in the original tensor.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _requires_grad(*tensors) -> bool:
    return any(t.requires_grad for t in tensors)


# ---------------------------------------------------------------------------
# elementwise binary ops (with broadcasting)
# ---------------------------------------------------------------------------
def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, parents=(a, b), op="add",
                 requires_grad=_requires_grad(a, b))

    def _backward():
        if a.requires_grad:
            a.grad += unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data - b.data, parents=(a, b), op="sub",
                 requires_grad=_requires_grad(a, b))

    def _backward():
        if a.requires_grad:
            a.grad += unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(-out.grad, b.shape)

    out._backward = _backward
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, parents=(a, b), op="mul",
                 requires_grad=_requires_grad(a, b))

    def _backward():
        if a.requires_grad:
            a.grad += unbroadcast(out.grad * b.data, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(out.grad * a.data, b.shape)

    out._backward = _backward
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data / b.data, parents=(a, b), op="div",
                 requires_grad=_requires_grad(a, b))

    def _backward():
        if a.requires_grad:
            a.grad += unbroadcast(out.grad / b.data, a.shape)
        if b.requires_grad:
            b.grad += unbroadcast(-out.grad * a.data / (b.data ** 2), b.shape)

    out._backward = _backward
    return out


def neg(a: Tensor) -> Tensor:
    out = Tensor(-a.data, parents=(a,), op="neg", requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += -out.grad

    out._backward = _backward
    return out


def pow(a: Tensor, power: float) -> Tensor:
    out = Tensor(a.data ** power, parents=(a,), op=f"pow{power}",
                 requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (power * a.data ** (power - 1)) * out.grad

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# matmul / transpose
# ---------------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """2-D matrix product ``a @ b`` (the workhorse of a Linear layer)."""
    out = Tensor(a.data @ b.data, parents=(a, b), op="matmul",
                 requires_grad=_requires_grad(a, b))

    def _backward():
        if a.requires_grad:
            a.grad += out.grad @ b.data.T
        if b.requires_grad:
            b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def transpose(a: Tensor) -> Tensor:
    out = Tensor(a.data.T, parents=(a,), op="transpose",
                 requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += out.grad.T

    out._backward = _backward
    return out


def reshape(a: Tensor, shape) -> Tensor:
    out = Tensor(a.data.reshape(shape), parents=(a,), op="reshape",
                 requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += out.grad.reshape(a.shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------
def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), parents=(a,),
                 op="sum", requires_grad=a.requires_grad)

    def _backward():
        if not a.requires_grad:
            return
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        # broadcast the (possibly scalar) upstream grad back to a's shape.
        a.grad += np.ones_like(a.data) * g

    out._backward = _backward
    return out


def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), parents=(a,),
                 op="mean", requires_grad=a.requires_grad)

    if axis is None:
        n = a.data.size
    else:
        n = a.data.shape[axis]

    def _backward():
        if not a.requires_grad:
            return
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a.grad += (np.ones_like(a.data) * g) / n

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def relu(a: Tensor) -> Tensor:
    out = Tensor(np.maximum(a.data, 0.0), parents=(a,), op="relu",
                 requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (a.data > 0.0) * out.grad

    out._backward = _backward
    return out


def sigmoid(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(s, parents=(a,), op="sigmoid", requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (s * (1.0 - s)) * out.grad

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, parents=(a,), op="tanh", requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (1.0 - t * t) * out.grad

    out._backward = _backward
    return out


# sqrt(2/pi) constant used by the tanh approximation of GELU.
_GELU_C = 0.7978845608028654


def gelu(a: Tensor) -> Tensor:
    """Gaussian Error Linear Unit (tanh approximation), with analytic backward.

    gelu(x) = 0.5 x (1 + tanh(c (x + 0.044715 x^3))),  c = sqrt(2/pi).
    Chosen as the non-saturating smooth activation alongside ReLU; its exact
    derivative is implemented here so it gradient-checks as a single op.
    """
    x = a.data
    x3 = x ** 3
    inner = _GELU_C * (x + 0.044715 * x3)
    t = np.tanh(inner)
    out_val = 0.5 * x * (1.0 + t)
    out = Tensor(out_val, parents=(a,), op="gelu", requires_grad=a.requires_grad)

    def _backward():
        if not a.requires_grad:
            return
        dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
        dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
        a.grad += dgelu * out.grad

    out._backward = _backward
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax as a standalone op (used for inference probs).

    For *training* prefer :func:`nn.loss.softmax_cross_entropy`, which fuses the
    softmax with cross-entropy for a cheaper, more stable gradient.
    """
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=axis, keepdims=True)
    out = Tensor(probs, parents=(a,), op="softmax", requires_grad=a.requires_grad)

    def _backward():
        if not a.requires_grad:
            return
        # Jacobian-vector product for softmax:
        #   dx = probs * (grad - sum(grad * probs, axis))
        g = out.grad
        dot = np.sum(g * probs, axis=axis, keepdims=True)
        a.grad += probs * (g - dot)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# attach elementwise/reduction ops onto Tensor as dunders / methods
# (kept here to avoid a circular import inside tensor.py)
# ---------------------------------------------------------------------------
def _attach():
    Tensor.reshape = lambda self, shape: reshape(self, shape)
    Tensor.transpose = lambda self: transpose(self)
    Tensor.matmul = lambda self, other: matmul(self, as_tensor(other))
    Tensor.softmax = lambda self, axis=-1: softmax(self, axis=axis)


_attach()
