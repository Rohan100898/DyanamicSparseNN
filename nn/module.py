"""Base :class:`Module`: the unit of composition for layers and models.

A Module owns some parameters (``Tensor`` objects with ``requires_grad=True``)
and/or child Modules. :meth:`parameters` walks the object graph and returns
every trainable tensor so the optimizer can update them. :meth:`prunable_layers`
returns the layers that carry a mask, so the pruner knows what it may sparsify.
"""
from __future__ import annotations

from typing import Iterator, List

from engine import Tensor


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -- parameter / submodule discovery ------------------------------
    def _children(self) -> Iterator["Module"]:
        for v in self.__dict__.values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        yield item

    def parameters(self) -> List[Tensor]:
        """Return all trainable tensors owned by this module and its children."""
        params: List[Tensor] = []
        seen = set()

        def collect(module: "Module"):
            for v in module.__dict__.values():
                if isinstance(v, Tensor) and v.requires_grad:
                    if id(v) not in seen:
                        seen.add(id(v))
                        params.append(v)
            for child in module._children():
                collect(child)

        collect(self)
        return params

    def prunable_layers(self) -> List["Module"]:
        """Return descendant layers that expose a weight mask (see Linear)."""
        layers: List[Module] = []

        def collect(module: "Module"):
            if getattr(module, "is_prunable", False):
                layers.append(module)
            for child in module._children():
                collect(child)

        collect(self)
        return layers

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))
