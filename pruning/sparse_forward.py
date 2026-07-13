"""Real sparse forward computation -- the proof that sparsity buys real compute.

A masked dense layer still runs a full ``in x out`` matmul and multiplies by
zeros, so it saves *nothing*. To turn sparsity into fewer FLOPs we extract the
active connections into Compressed Sparse Row (CSR) format and multiply only
those.

The CSR structure (``row_ptr`` / ``col_indices`` / ``values``) is built here from
scratch (``_build_csr``). Two multiply kernels are provided:

* ``forward_scratch`` -- a dependency-free numpy kernel that touches only the
  ``nnz`` nonzeros (gather the needed inputs, scatter-add into outputs). This is
  the honest "we implemented sparse matmul" path.
* ``forward`` -- the same math via SciPy's C CSR kernel, used for the wall-clock
  benchmark (it beats dense BLAS once the matrix is large and sparse enough).

Both are validated against the dense result in ``tests/pruning_test.py``. The
exact FLOP count of the sparse path is ``2 * nnz`` per sample (one multiply +
one add per stored nonzero), versus ``2 * in * out`` dense.
"""
from __future__ import annotations

from typing import List

import numpy as np

try:
    import scipy.sparse as sp
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# numpy activation functions mirroring nn.activations (inference only, no graph)
def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x):
    return np.tanh(x)


def _gelu(x):
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


_ACT_FUNCS = {"ReLU": _relu, "Sigmoid": _sigmoid, "Tanh": _tanh, "GELU": _gelu}


class SparseLinear:
    """CSR representation of one pruned Linear layer, for sparse inference."""

    def __init__(self, linear):
        W = linear.effective_weight().astype(np.float32)   # (in, out), zeros pruned
        self.in_dim, self.out_dim = W.shape
        self.bias = (linear.bias.data.copy() if linear.bias is not None
                     else np.zeros(self.out_dim, dtype=np.float32))
        self._build_csr(W)

        # accelerated backend: CSR of W^T (out, in) so out = (WT @ X^T)^T
        self._scipy_WT = sp.csr_matrix(W.T) if _HAVE_SCIPY else None

    # -- from-scratch CSR construction --------------------------------
    def _build_csr(self, W: np.ndarray) -> None:
        rows, cols = np.nonzero(W)          # C-order => already CSR-sorted by row
        self.values = W[rows, cols].astype(np.float32)
        self.col_indices = cols.astype(np.int64)
        counts = np.bincount(rows, minlength=self.in_dim)
        self.row_ptr = np.zeros(self.in_dim + 1, dtype=np.int64)
        self.row_ptr[1:] = np.cumsum(counts)
        self.nnz = int(self.values.size)
        # per-nonzero source row, precomputed for the gather kernel
        self._row_of_nnz = np.repeat(np.arange(self.in_dim), np.diff(self.row_ptr))

    # -- multiply kernels ---------------------------------------------
    def forward_scratch(self, X: np.ndarray) -> np.ndarray:
        """Dependency-free sparse matmul: O(nnz * N), no dense in*out product."""
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        if self.nnz == 0:
            return np.broadcast_to(self.bias, (N, self.out_dim)).copy()
        contrib = X[:, self._row_of_nnz] * self.values[None, :]    # (N, nnz)
        outT = np.zeros((self.out_dim, N), dtype=np.float32)
        np.add.at(outT, self.col_indices, contrib.T)               # scatter by col
        return outT.T + self.bias

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Sparse matmul via SciPy CSR (fast path); falls back to scratch."""
        if self._scipy_WT is None:
            return self.forward_scratch(X)
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self._scipy_WT @ X.T).T + self.bias

    @property
    def flops_per_sample(self) -> int:
        return 2 * self.nnz


class SparseMLP:
    """A trained Sequential re-expressed with sparse Linear layers for inference."""

    def __init__(self, model, use_scratch: bool = False):
        from nn.linear import Linear
        self.blocks = []          # list of ("linear", SparseLinear) | ("act", fn)
        self.use_scratch = use_scratch
        for layer in model.layers:
            if isinstance(layer, Linear):
                self.blocks.append(("linear", SparseLinear(layer)))
            else:
                fn = _ACT_FUNCS.get(type(layer).__name__)
                if fn is not None:
                    self.blocks.append(("act", fn))
                # Softmax (if present) is skipped: argmax over logits is identical.

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=np.float32)
        for kind, block in self.blocks:
            if kind == "linear":
                h = block.forward_scratch(h) if self.use_scratch else block.forward(h)
            else:
                h = block(h)
        return h

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    @property
    def total_flops_per_sample(self) -> int:
        return int(sum(b.flops_per_sample for k, b in self.blocks if k == "linear"))

    @property
    def nnz(self) -> int:
        return int(sum(b.nnz for k, b in self.blocks if k == "linear"))
