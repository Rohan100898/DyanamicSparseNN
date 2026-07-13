"""Optimizers: hand-written SGD and Adam, both mask-aware."""
from optim.sgd import SGD
from optim.adam import Adam


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}")


__all__ = ["SGD", "Adam", "build_optimizer"]
