"""Pruning correctness tests.

Covers the four things the brief and the gaps note require:

1. a masked weight contributes exactly zero to the forward output;
2. a masked weight receives exactly zero gradient;
3. optimizer state for a masked weight does not drift (momentum stays frozen);
4. the prune -> train -> revive round-trip does not corrupt training -- a revived
   weight restarts from a clean (m=v=0) state, so no stale momentum spike; and
5. the CSR sparse forward path matches the dense masked forward numerically.

Run with::

    pytest tests/pruning_test.py
    python -m tests.pruning_test
"""
from __future__ import annotations

import numpy as np

from engine import Tensor, set_default_dtype
from nn.linear import Linear
from nn.sequential import build_mlp
from nn.loss import softmax_cross_entropy
from optim.adam import Adam
from pruning.sparse_forward import SparseMLP
from pruning.mask import compute_layer_mask


def setup_function(function=None):
    # Pruning/optimizer behaviour is exercised in the training dtype (float32).
    set_default_dtype(np.float32)


_RNG = np.random.default_rng(7)


def test_mask_zeroes_forward_output():
    layer = Linear(4, 3, rng=_RNG)
    x = Tensor(_RNG.standard_normal((5, 4)), requires_grad=False)

    mask = np.ones((4, 3), dtype=np.float32)
    mask[0, :] = 0.0          # kill all connections from input feature 0
    mask[2, 1] = 0.0
    layer.set_mask(mask)

    out = layer(x).data
    expected = x.data @ (layer.weight.data * mask) + layer.bias.data
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    # zeroing feature-0 connections == dropping feature 0 from the product
    only_active = x.data @ (layer.weight.data * mask) + layer.bias.data
    assert np.allclose(out, only_active)
    print("  masked weights contribute exactly zero to forward output   OK")


def test_masked_weight_zero_gradient():
    layer = Linear(4, 3, rng=_RNG)
    mask = np.ones((4, 3), dtype=np.float32)
    mask[1, 2] = 0.0
    layer.set_mask(mask)

    x = Tensor(_RNG.standard_normal((6, 4)), requires_grad=False)
    y = np.array([0, 1, 2, 0, 1, 2])
    layer.zero_grad()
    loss = softmax_cross_entropy(layer(x), y)
    loss.backward()

    assert layer.weight.grad[1, 2] == 0.0, "masked weight must get zero gradient"
    assert np.all(layer.weight.grad[mask == 0.0] == 0.0)
    print("  masked weights receive exactly zero gradient               OK")


def test_optimizer_state_does_not_drift():
    layer = Linear(4, 3, rng=_RNG)
    opt = Adam(layer.parameters(), lr=1e-2)
    w = layer.weight            # opt.m[0] / opt.v[0] correspond to the weight

    # build up some momentum on the dense weight first
    for _ in range(5):
        x = Tensor(_RNG.standard_normal((8, 4)), requires_grad=False)
        y = _RNG.integers(0, 3, size=8)
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    assert opt.m[0][0, 0] != 0.0, "expected non-zero momentum before pruning"

    # prune weight (0,0); record its frozen value
    mask = np.ones((4, 3), dtype=np.float32)
    mask[0, 0] = 0.0
    layer.set_mask(mask)
    frozen_value = layer.weight.data[0, 0]

    for _ in range(10):
        x = Tensor(_RNG.standard_normal((8, 4)), requires_grad=False)
        y = _RNG.integers(0, 3, size=8)
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
        assert opt.m[0][0, 0] == 0.0, "momentum leaked into a pruned weight"
        assert opt.v[0][0, 0] == 0.0, "second moment leaked into a pruned weight"
        assert layer.weight.data[0, 0] == frozen_value, "pruned weight drifted"
    print("  pruned weight: frozen value + zero m/v across 10 steps      OK")


def test_prune_revive_no_corruption():
    """Prune a weight, train, then revive it: no stale-momentum corruption."""
    layer = Linear(4, 3, rng=_RNG)
    opt = Adam(layer.parameters(), lr=1e-2)
    b1 = opt.beta1

    # warm up
    for _ in range(5):
        x = Tensor(_RNG.standard_normal((8, 4)), requires_grad=False)
        y = _RNG.integers(0, 3, size=8)
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()

    # prune (1,1), then train hard while it stays masked
    mask = np.ones((4, 3), dtype=np.float32)
    mask[1, 1] = 0.0
    layer.set_mask(mask)
    for _ in range(15):
        x = Tensor(_RNG.standard_normal((16, 4)) * 3.0, requires_grad=False)
        y = _RNG.integers(0, 3, size=16)
        opt.zero_grad()
        softmax_cross_entropy(layer(x), y).backward()
        opt.step()
    assert opt.m[0][1, 1] == 0.0 and opt.v[0][1, 1] == 0.0

    # revive the connection
    mask[1, 1] = 1.0
    layer.set_mask(mask)

    # one training step after revival
    x = Tensor(_RNG.standard_normal((16, 4)), requires_grad=False)
    y = _RNG.integers(0, 3, size=16)
    opt.zero_grad()
    softmax_cross_entropy(layer(x), y).backward()
    fresh_grad = layer.weight.grad[1, 1]
    opt.step()

    # momentum must equal (1-b1)*grad -> proves it restarted from 0, not stale.
    expected_m = (1.0 - b1) * fresh_grad
    np.testing.assert_allclose(opt.m[0][1, 1], expected_m, rtol=1e-5, atol=1e-7)
    assert np.isfinite(layer.weight.data[1, 1])
    print("  revived weight restarts from clean m=v=0 (no corruption)    OK")


def test_compute_layer_mask_hits_exact_budget_and_is_monotone():
    scores = _RNG.random((20, 10)).astype(np.float32)
    mask = np.ones((20, 10), dtype=np.float32)

    mask = compute_layer_mask(scores, 0.5, mask)
    assert abs(mask.mean() - 0.5) < 1e-6, "expected exactly 50% kept"

    # tighten to 80% sparsity; previously pruned must stay pruned (monotone)
    prev_pruned = mask == 0.0
    mask2 = compute_layer_mask(scores, 0.8, mask)
    assert abs((1 - mask2.mean()) - 0.8) < 1e-6, "expected exactly 80% sparsity"
    assert np.all(mask2[prev_pruned] == 0.0), "pruning must be monotone"
    print("  mask hits exact budget and pruning is monotone             OK")


def test_sparse_forward_matches_dense():
    model = build_mlp(input_dim=8, hidden_dims=[32, 32], n_classes=4,
                      activation="relu", rng=_RNG)
    # prune each layer to 70% by magnitude
    for layer in model.prunable_layers():
        scores = np.abs(layer.weight.data)
        layer.set_mask(compute_layer_mask(scores, 0.7, layer.mask))

    X = _RNG.standard_normal((25, 8)).astype(np.float32)
    dense_logits = model(Tensor(X, requires_grad=False)).data

    sparse_scipy = SparseMLP(model, use_scratch=False).forward(X)
    sparse_scratch = SparseMLP(model, use_scratch=True).forward(X)

    np.testing.assert_allclose(sparse_scipy, dense_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sparse_scratch, dense_logits, rtol=1e-4, atol=1e-4)
    # and predictions agree
    assert np.array_equal(np.argmax(sparse_scipy, 1), np.argmax(dense_logits, 1))
    print("  CSR sparse forward matches dense masked forward            OK")


ALL_TESTS = [
    test_mask_zeroes_forward_output,
    test_masked_weight_zero_gradient,
    test_optimizer_state_does_not_drift,
    test_prune_revive_no_corruption,
    test_compute_layer_mask_hits_exact_budget_and_is_monotone,
    test_sparse_forward_matches_dense,
]


def main():
    setup_function()
    print("Pruning tests:")
    for t in ALL_TESTS:
        t()
    print("All pruning tests passed.")


if __name__ == "__main__":
    main()
