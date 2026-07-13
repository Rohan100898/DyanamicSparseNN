"""Computation-graph utilities: topological ordering over the autograd DAG.

Every :class:`~engine.tensor.Tensor` produced by an op records the tensors that
created it (``parents``). The set of all tensors reachable from a root forms a
directed acyclic graph. Reverse-mode autodiff needs to visit nodes in reverse
topological order so that, by the time we call a node's local ``_backward``,
its own gradient has already been fully accumulated by every child downstream.
"""
from __future__ import annotations

from typing import List


def topological_sort(root) -> List:
    """Return all nodes reachable from ``root`` in topological order.

    Uses an iterative (explicit-stack) post-order DFS to avoid Python's
    recursion-depth limit on deep graphs. A node is appended only after all of
    its parents have been appended, so the returned list has parents before
    children; the backward driver walks it in reverse.
    """
    topo: List = []
    visited = set()

    # Each frame is (node, children_pushed?). On first visit we mark the node
    # visited and push its parents; on the second pop we append it (post-order).
    stack = [(root, False)]
    while stack:
        node, children_pushed = stack.pop()
        if children_pushed:
            topo.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))          # revisit after parents are done
        for parent in node.parents:
            if id(parent) not in visited:
                stack.append((parent, False))
    return topo
