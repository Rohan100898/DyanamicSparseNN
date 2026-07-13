"""FLOP and parameter accounting -- the "real compute savings" evidence.

Fewer nonzero weights is not the claim; *fewer operations* is. We count FLOPs
for a Linear layer's forward pass as ``2 * (multiply-adds)``:

* dense  : ``2 * in * out`` per sample (every weight participates)
* sparse : ``2 * nnz``       per sample (only stored nonzeros participate)

The sparse figure is tied to the actual sparse forward path (``2 * nnz``, the
exact number of scalar multiply-adds ``pruning/sparse_forward.py`` performs),
not to the dense shape -- so a model that still ran dense matmuls-by-zero would
show *no* saving here, as it should.
"""
from __future__ import annotations

from typing import Dict


def _linear_layers(model):
    from nn.linear import Linear
    return [l for l in getattr(model, "layers", []) if isinstance(l, Linear)]


def parameter_counts(model) -> Dict[str, float]:
    layers = model.prunable_layers()
    total = int(sum(l.total_weights for l in layers))
    active = int(sum(l.active_weights for l in layers))
    # biases are never pruned; include them in the totals for honesty
    bias_params = int(sum(l.bias.size for l in _linear_layers(model)
                          if l.bias is not None))
    return {
        "weight_params_total": total,
        "weight_params_active": active,
        "bias_params": bias_params,
        "params_total": total + bias_params,
        "params_active": active + bias_params,
        "weight_sparsity": (1.0 - active / total) if total else 0.0,
    }


def flop_counts(model) -> Dict[str, float]:
    """Per-sample forward FLOPs for the dense vs sparse (masked) model."""
    layers = _linear_layers(model)
    dense = 0
    sparse = 0
    for l in layers:
        in_f, out_f = l.in_features, l.out_features
        dense += 2 * in_f * out_f
        sparse += 2 * int(l.active_weights)
    reduction = (1.0 - sparse / dense) if dense else 0.0
    return {
        "dense_flops_per_sample": int(dense),
        "sparse_flops_per_sample": int(sparse),
        "flops_reduction": reduction,
    }


def summary(model) -> Dict[str, float]:
    out = parameter_counts(model)
    out.update(flop_counts(model))
    return out
