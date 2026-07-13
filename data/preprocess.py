"""Preprocessing helpers: standardization, one-hot encoding, mini-batching."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def standardize(X: np.ndarray, mean=None, std=None):
    """Zero-mean / unit-variance per feature. Returns (X_scaled, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)      # guard constant features
    return ((X - mean) / std).astype(np.float32), mean, std


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
    return out


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int,
                        rng: np.random.Generator, shuffle: bool = True
                        ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, y_batch) mini-batches, optionally shuffled each epoch."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]
