"""Dataset loaders.

Two datasets are supported:

* ``spiral`` -- a synthetic multi-class 2-D spiral. Fully deterministic given a
  seed, needs no network access, and is non-linearly separable so an MLP has to
  actually learn structure. This is the default so every command in the README
  reproduces offline.
* ``mnist``  -- real 28x28 digit images (784-d), fetched via scikit-learn if it
  and network access are available. Optional.

``get_dataset`` returns a :class:`Dataset` already split into train/val/test and
standardized (fit on train only, applied to val/test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.preprocess import standardize


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    input_dim: int
    name: str


def make_spiral(n_samples: int = 3000, n_classes: int = 3, noise: float = 0.2,
                rng: np.random.Generator | None = None):
    """Generate an ``n_classes``-armed 2-D spiral (Stanford CS231n style)."""
    rng = rng or np.random.default_rng(0)
    per_class = n_samples // n_classes
    X = np.zeros((per_class * n_classes, 2), dtype=np.float32)
    y = np.zeros(per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        ix = range(per_class * c, per_class * (c + 1))
        r = np.linspace(0.0, 1.0, per_class)                     # radius
        t = (np.linspace(c * 4, (c + 1) * 4, per_class)          # angle
             + rng.standard_normal(per_class) * noise)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


def _load_mnist():
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "MNIST requires scikit-learn. Install it (`pip install scikit-learn`) "
            "or use --dataset spiral."
        ) from e
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    return X, y


def _split(X, y, val_split, test_split, rng):
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def get_dataset(cfg) -> Dataset:
    """Build a train/val/test :class:`Dataset` from a :class:`config.Config`."""
    rng = np.random.default_rng(cfg.seed)

    if cfg.dataset == "spiral":
        X, y = make_spiral(cfg.n_samples, cfg.n_classes, rng=rng)
        n_classes = cfg.n_classes
    elif cfg.dataset == "mnist":
        X, y = _load_mnist()
        n_classes = 10
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}")

    (Xtr, ytr, Xva, yva, Xte, yte) = _split(
        X, y, cfg.val_split, cfg.test_split, rng)

    Xtr, mean, std = standardize(Xtr)
    Xva, _, _ = standardize(Xva, mean, std)
    Xte, _, _ = standardize(Xte, mean, std)

    return Dataset(Xtr, ytr, Xva, yva, Xte, yte,
                   n_classes=n_classes, input_dim=X.shape[1], name=cfg.dataset)
