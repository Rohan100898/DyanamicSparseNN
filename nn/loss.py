"""Loss functions.

``softmax_cross_entropy`` fuses softmax + cross-entropy into a single op. This
is both numerically stable (log-sum-exp with the max subtracted, so the exp
never overflows) and cheap: the gradient of the fused loss w.r.t. the logits is
the famously simple ``(softmax - onehot) / N``, so we skip building the softmax
Jacobian in the graph entirely. This is the training loss.
"""
from __future__ import annotations

import numpy as np

from engine import Tensor
from engine import ops


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy over the batch.

    Parameters
    ----------
    logits : Tensor, shape (N, C)
    targets : np.ndarray of int class indices, shape (N,)
    """
    x = logits.data
    N = x.shape[0]
    targets = np.asarray(targets).astype(np.int64).reshape(-1)

    # stable log-softmax
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)          # (N, C)
    probs = exp / sum_exp

    nll = -log_probs[np.arange(N), targets]        # (N,)
    loss_val = np.mean(nll)

    out = Tensor(loss_val, parents=(logits,), op="softmax_cross_entropy",
                 requires_grad=logits.requires_grad)

    onehot = np.zeros_like(probs)
    onehot[np.arange(N), targets] = 1.0

    def _backward():
        if logits.requires_grad:
            # d loss / d logits = (softmax - onehot) / N, scaled by upstream grad
            logits.grad += (probs - onehot) / N * out.grad

    out._backward = _backward
    return out


def mse_loss(pred: Tensor, target) -> Tensor:
    """Mean squared error, assembled from primitive ops (autodiff handles it)."""
    target = ops.as_tensor(target)
    diff = pred - target
    return (diff ** 2).mean()
