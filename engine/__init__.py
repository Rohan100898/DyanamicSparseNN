"""DynamicSparseNN autograd engine.

Public surface:

* :class:`~engine.tensor.Tensor` -- the differentiable array type
* :mod:`engine.ops`             -- differentiable primitive operations
* :func:`~engine.backward.backward` and
  :func:`~engine.graph.topological_sort` -- the reverse-mode driver
"""
from engine.tensor import Tensor, set_default_dtype, get_default_dtype
from engine import ops  # noqa: F401  (import triggers Tensor method attachment)
from engine.backward import backward
from engine.graph import topological_sort

__all__ = [
    "Tensor",
    "ops",
    "backward",
    "topological_sort",
    "set_default_dtype",
    "get_default_dtype",
]
