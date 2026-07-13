"""Dedicated test for the (N, D) + (D,) broadcasting case.

The Part-1 brief calls this case out by name: adding a bias vector of shape
``(D,)`` to a batched activation of shape ``(N, D)`` broadcasts the bias across
the batch. The gradient w.r.t. the bias must therefore be the *sum over the
batch* of the upstream gradient (shape ``(D,)`` again). Getting this wrong is
the classic "broadcast bias addition corrupts parameter gradients" bug, so we
assert it explicitly rather than relying only on the generic gradient check.

Run with::

    pytest tests/broadcast_test.py
    python -m tests.broadcast_test
"""
from __future__ import annotations

import numpy as np

from engine import Tensor, ops, set_default_dtype


def setup_function(function=None):
    # float64 so the finite-difference reference is not swamped by rounding.
    set_default_dtype(np.float64)


_RNG = np.random.default_rng(1)


def test_bias_broadcast_gradient_shape_and_value():
    """y = x + b, x:(N,D), b:(D,)  =>  b.grad == column-sum of upstream grad."""
    N, D = 8, 4
    x = Tensor(_RNG.standard_normal((N, D)), requires_grad=True)
    b = Tensor(_RNG.standard_normal((D,)), requires_grad=True)

    y = x + b                     # (N, D) via broadcasting
    upstream = _RNG.standard_normal((N, D))
    y.backward(upstream)

    # bias gradient keeps the source shape (D,), not the broadcast shape (N, D)
    assert b.grad.shape == (D,), f"expected (D,), got {b.grad.shape}"
    # and equals the sum of the upstream gradient over the batch axis
    np.testing.assert_allclose(b.grad, upstream.sum(axis=0), rtol=1e-10, atol=1e-10)
    # x is not broadcast, so its gradient is just the upstream gradient
    np.testing.assert_allclose(x.grad, upstream, rtol=1e-10, atol=1e-10)
    print("  bias-broadcast add: grad shape (D,) and equals batch-sum  OK")


def test_bias_broadcast_sum_of_output_equals_N():
    """d sum(x + b)/db == N for every bias element (it appears in N rows)."""
    N, D = 8, 4
    x = Tensor(_RNG.standard_normal((N, D)), requires_grad=True)
    b = Tensor(np.zeros(D), requires_grad=True)
    (x + b).sum().backward()
    np.testing.assert_allclose(b.grad, np.full(D, float(N)), rtol=1e-10, atol=1e-10)
    print("  d sum(x+b)/db == N for all D                              OK")


def test_broadcast_mul_gradient():
    """Elementwise scale (N,D) * (D,) also unbroadcasts correctly."""
    N, D = 6, 5
    x = Tensor(_RNG.standard_normal((N, D)), requires_grad=True)
    g = Tensor(_RNG.standard_normal((D,)), requires_grad=True)
    out = x * g
    upstream = _RNG.standard_normal((N, D))
    out.backward(upstream)
    np.testing.assert_allclose(g.grad, (upstream * x.data).sum(axis=0),
                               rtol=1e-9, atol=1e-9)
    print("  broadcast mul (N,D)*(D,): grad unbroadcast to (D,)        OK")


def test_broadcast_numeric_gradient_check():
    """Full numeric gradient check of the bias through a nonlinearity."""
    N, D = 5, 4
    x = Tensor(_RNG.standard_normal((N, D)), requires_grad=False)

    def f(bias):
        return (x + bias).relu().sum()

    b = Tensor(_RNG.standard_normal((D,)) + 1.5, requires_grad=True)  # keep off the kink
    b.zero_grad()
    f(b).backward()
    analytic = b.grad.copy()

    eps = 1e-6
    numeric = np.zeros(D)
    for i in range(D):
        orig = b.data[i]
        b.data[i] = orig + eps
        s_plus = float(f(b).data)
        b.data[i] = orig - eps
        s_minus = float(f(b).data)
        b.data[i] = orig
        numeric[i] = (s_plus - s_minus) / (2 * eps)

    err = float(np.max(np.abs(analytic - numeric)))
    print(f"  numeric grad-check through relu: max_abs_error = {err:.2e}    OK")
    assert err <= 1e-5


ALL_TESTS = [
    test_bias_broadcast_gradient_shape_and_value,
    test_bias_broadcast_sum_of_output_equals_N,
    test_broadcast_mul_gradient,
    test_broadcast_numeric_gradient_check,
]


def main():
    setup_function()
    print("Broadcast tests ((N,D) + (D,) and friends):")
    for t in ALL_TESTS:
        t()
    print("All broadcast tests passed.")


if __name__ == "__main__":
    main()
