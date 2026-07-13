"""Evaluation & dense-vs-sparse comparison.

Provides the reusable ``dense_vs_sparse_report`` (used by ``train.py`` and the
Part-4 sweep) and a CLI that loads a saved checkpoint and prints the comparison:
accuracy retained, % sparsity, % FLOP reduction, and wall-clock speedup.

    python evaluate.py --model models/pruned.npz
"""
from __future__ import annotations

import argparse
import time
from typing import Callable, Dict

import numpy as np

from engine import Tensor, set_default_dtype
from utils.metrics import full_report
from utils import flops as flops_util
from pruning.sparse_forward import SparseMLP


def model_predict_dense(model, X: np.ndarray) -> np.ndarray:
    logits = model(Tensor(X, requires_grad=False)).data
    return np.argmax(logits, axis=1)


def _benchmark(fn: Callable[[], None], repeats: int = 20) -> float:
    """Return mean milliseconds per call (after a warm-up)."""
    fn()  # warm-up (build caches, allocate)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1e3


def dense_vs_sparse_report(model, X: np.ndarray, y: np.ndarray,
                           repeats: int = 20, use_scratch: bool = False) -> Dict[str, float]:
    """Compare the dense masked model against its CSR sparse equivalent."""
    n_classes = int(model.layers[-1].out_features)

    # accuracy
    dense_pred = model_predict_dense(model, X)
    sparse_model = SparseMLP(model, use_scratch=use_scratch)
    sparse_pred = sparse_model.predict(X)

    dense_metrics = full_report(y, dense_pred, n_classes)
    sparse_metrics = full_report(y, sparse_pred, n_classes)

    # compute cost
    fl = flops_util.flop_counts(model)
    pc = flops_util.parameter_counts(model)

    # wall-clock (dense graph forward vs sparse CSR forward)
    Xt = Tensor(X, requires_grad=False)
    dense_ms = _benchmark(lambda: model.forward(Xt), repeats)
    sparse_ms = _benchmark(lambda: sparse_model.forward(X), repeats)

    speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("nan")
    report = {
        "dense_accuracy": dense_metrics["accuracy"],
        "sparse_accuracy": sparse_metrics["accuracy"],
        "accuracy_delta": sparse_metrics["accuracy"] - dense_metrics["accuracy"],
        "weight_sparsity": pc["weight_sparsity"],
        "params_total": pc["params_total"],
        "params_active": pc["params_active"],
        "dense_flops_per_sample": fl["dense_flops_per_sample"],
        "sparse_flops_per_sample": fl["sparse_flops_per_sample"],
        "flops_reduction": fl["flops_reduction"],
        "dense_infer_ms": dense_ms,
        "sparse_infer_ms": sparse_ms,
        "inference_speedup": speedup,
    }
    return report


def print_report(report: Dict[str, float]) -> None:
    print("\n=== Dense vs Sparse ===")
    print(f"  weight sparsity        : {report['weight_sparsity']*100:6.2f} %")
    print(f"  params total -> active : {report['params_total']} -> {report['params_active']}")
    print(f"  dense accuracy         : {report['dense_accuracy']*100:6.2f} %")
    print(f"  sparse accuracy        : {report['sparse_accuracy']*100:6.2f} %")
    print(f"  accuracy delta         : {report['accuracy_delta']*100:+6.2f} %")
    print(f"  FLOPs/sample dense     : {report['dense_flops_per_sample']}")
    print(f"  FLOPs/sample sparse    : {report['sparse_flops_per_sample']}")
    print(f"  FLOP reduction         : {report['flops_reduction']*100:6.2f} %")
    print(f"  inference dense/sparse : {report['dense_infer_ms']:.3f} / "
          f"{report['sparse_infer_ms']:.3f} ms/batch")
    print(f"  inference speedup      : {report['inference_speedup']:.2f}x")


def main() -> None:
    set_default_dtype(np.float32)
    ap = argparse.ArgumentParser(description="Evaluate a saved DynamicSparseNN model.")
    ap.add_argument("--model", default="models/pruned.npz",
                    help="path to the saved .npz checkpoint")
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    from utils.checkpoint import load_model
    from data.loader import get_dataset

    model, cfg = load_model(args.model)
    ds = get_dataset(cfg)
    report = dense_vs_sparse_report(model, ds.X_test, ds.y_test, repeats=args.repeats)
    print(f"Loaded {args.model}  (dataset={cfg.dataset}, method={cfg.pruning_method})")
    print_report(report)


if __name__ == "__main__":
    main()
