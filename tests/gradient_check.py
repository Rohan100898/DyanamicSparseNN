"""Numerical gradient checking for every autograd op.

For each op we compare the analytic gradient produced by ``backward()`` against
a central finite-difference estimate of d(sum(output))/d(input):

    numeric[i] = (S(x + e_i * eps) - S(x - e_i * eps)) / (2 * eps)

where ``S(x) = sum(op(x))``. If the two agree to within ``THRESHOLD`` the local
derivative wired into the op is correct.

Why this matters: a subtly wrong ``backward()`` still usually yields a *falling*
loss curve, so training "looks" fine while silently learning the wrong thing.
Gradient-checking is the only cheap way to trust the engine.

Runs under float64 (``set_default_dtype``) so the finite-difference reference is
not swamped by float32 rounding. Run with either::

    pytest tests/gradient_check.py
    python -m tests.gradient_check
"""
from __future__ import annotations

import numpy as np

from engine import Tensor, ops, set_default_dtype


def setup_function(function=None):
    # Rigorous checks need float64; central differences in float32 can't reach
    # 1e-5. Set per-test (not at import) so pytest's import order can't leave a
    # different module's float32 default in place.
    set_default_dtype(np.float64)


EPS = 1e-6
THRESHOLD = 1e-5
_RNG = np.random.default_rng(0)


def _max_grad_error(f, x: Tensor) -> float:
    """Return the max abs difference between analytic and numeric grads of f."""
    # --- analytic ---
    x.zero_grad()
    out = f(x)
    out.backward(np.ones_like(out.data))
    analytic = x.grad.copy()

    # --- numeric (central differences) ---
    numeric = np.zeros_like(x.data)
    flat = x.data.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + EPS
        s_plus = float(np.sum(f(x).data))
        flat[i] = orig - EPS
        s_minus = float(np.sum(f(x).data))
        flat[i] = orig
        numeric.reshape(-1)[i] = (s_plus - s_minus) / (2.0 * EPS)

    return float(np.max(np.abs(analytic - numeric)))


def _check(name, f, x: Tensor):
    err = _max_grad_error(f, x)
    print(f"  {name:<28} max_abs_grad_error = {err:.2e}")
    assert err <= THRESHOLD, f"{name}: gradient mismatch, max error {err:.2e}"
    return err


def _rand(*shape, positive=False):
    a = _RNG.standard_normal(shape)
    if positive:
        a = np.abs(a) + 0.5
    return Tensor(a, requires_grad=True)


# ---------------------------------------------------------------------------
# per-op tests (pytest discovers test_*; each is also called from __main__)
# ---------------------------------------------------------------------------
def test_add():
    b = Tensor(_RNG.standard_normal((4, 5)), requires_grad=False)
    _check("add", lambda x: x + b, _rand(4, 5))


def test_sub():
    b = Tensor(_RNG.standard_normal((4, 5)), requires_grad=False)
    _check("sub", lambda x: x - b, _rand(4, 5))


def test_mul():
    b = Tensor(_RNG.standard_normal((4, 5)), requires_grad=False)
    _check("mul", lambda x: x * b, _rand(4, 5))


def test_div():
    b = Tensor(np.abs(_RNG.standard_normal((4, 5))) + 0.5, requires_grad=False)
    _check("div", lambda x: x / b, _rand(4, 5))


def test_neg():
    _check("neg", lambda x: -x, _rand(4, 5))


def test_pow_square():
    _check("pow(2)", lambda x: x ** 2, _rand(4, 5))


def test_pow_cube():
    _check("pow(3)", lambda x: x ** 3, _rand(4, 5))


def test_pow_sqrt():
    _check("pow(0.5)", lambda x: x ** 0.5, _rand(4, 5, positive=True))


def test_matmul():
    b = Tensor(_RNG.standard_normal((5, 3)), requires_grad=False)
    _check("matmul", lambda x: ops.matmul(x, b), _rand(4, 5))


def test_transpose():
    _check("transpose", lambda x: ops.transpose(x), _rand(4, 5))


def test_reshape():
    _check("reshape", lambda x: ops.reshape(x, (5, 4)), _rand(4, 5))


def test_sum_all():
    _check("sum(all)", lambda x: x.sum(), _rand(4, 5))


def test_sum_axis():
    _check("sum(axis=1)", lambda x: x.sum(axis=1), _rand(4, 5))


def test_mean_all():
    _check("mean(all)", lambda x: x.mean(), _rand(4, 5))


def test_mean_axis():
    _check("mean(axis=0)", lambda x: x.mean(axis=0), _rand(4, 5))


def test_relu():
    # avoid inputs exactly at the kink where the subgradient is ambiguous
    x = _rand(4, 5)
    x.data[np.abs(x.data) < 1e-2] += 0.1
    _check("relu", lambda t: t.relu(), x)


def test_sigmoid():
    _check("sigmoid", lambda x: x.sigmoid(), _rand(4, 5))


def test_tanh():
    _check("tanh", lambda x: x.tanh(), _rand(4, 5))


def test_gelu():
    _check("gelu", lambda x: x.gelu(), _rand(4, 5))


def test_softmax():
    _check("softmax", lambda x: ops.softmax(x, axis=-1), _rand(4, 5))


def test_composite_mlp_layer():
    # x @ W + b  ->  gelu  ->  sum   (a full linear+activation slice)
    W = Tensor(_RNG.standard_normal((5, 3)) * 0.5, requires_grad=False)
    b = Tensor(_RNG.standard_normal((3,)) * 0.5, requires_grad=False)
    _check("composite(matmul+add+gelu)",
           lambda x: (ops.matmul(x, W) + b).gelu(), _rand(4, 5))


ALL_TESTS = [
    test_add, test_sub, test_mul, test_div, test_neg,
    test_pow_square, test_pow_cube, test_pow_sqrt,
    test_matmul, test_transpose, test_reshape,
    test_sum_all, test_sum_axis, test_mean_all, test_mean_axis,
    test_relu, test_sigmoid, test_tanh, test_gelu, test_softmax,
    test_composite_mlp_layer,
]


def main():
    setup_function()
    print(f"Gradient check (float64, eps={EPS}, threshold={THRESHOLD}):")
    for t in ALL_TESTS:
        t()
    print("All gradient checks passed.")


if __name__ == "__main__":
    main()
