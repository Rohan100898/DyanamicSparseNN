"""Sparsity schedule: how much of the network is pruned at each epoch.

We ramp sparsity from 0 up to the target over a window ``[t0, T]``, keeping the
net dense before ``t0`` (so it can first learn useful features) and holding at
the target after ``T`` (so it can fine-tune under the final topology).

Cubic ramp (default), with s_initial = 0:

    s(t) = s_final + (s_initial - s_final) * (1 - (t - t0)/(T - t0))^3
         = s_final * (1 - (1 - p)^3),   p = clamp((t - t0)/(T - t0), 0, 1)

Why cubic beats linear: the cube front-loads the pruning -- it removes a lot of
(clearly useless) weights early when progress p is small, then *slows down* as
it approaches the budget, pruning conservatively near the end where every
remaining weight matters and the network needs the most time to adapt. Linear
prunes the same amount every epoch, which is too aggressive late. This is the
schedule from Zhu & Gupta (2017), "To prune, or not to prune".
"""
from __future__ import annotations


class SparsityScheduler:
    def __init__(self, target_sparsity: float, start_epoch: int, end_epoch: int,
                 schedule: str = "cubic", initial_sparsity: float = 0.0):
        assert 0.0 <= target_sparsity <= 1.0
        assert end_epoch > start_epoch
        self.s_final = target_sparsity
        self.s_initial = initial_sparsity
        self.t0 = start_epoch
        self.T = end_epoch
        self.schedule = schedule.lower()

    def sparsity_at(self, epoch: int) -> float:
        if self.s_final <= 0.0:
            return 0.0
        if epoch < self.t0:
            return self.s_initial
        if epoch >= self.T:
            return self.s_final

        p = (epoch - self.t0) / (self.T - self.t0)     # in [0, 1]
        if self.schedule == "cubic":
            return self.s_final + (self.s_initial - self.s_final) * (1.0 - p) ** 3
        elif self.schedule == "linear":
            return self.s_initial + (self.s_final - self.s_initial) * p
        raise ValueError(f"Unknown schedule {self.schedule!r}")

    def is_pruning_epoch(self, epoch: int) -> bool:
        """True while sparsity is still ramping (masks may change)."""
        return self.s_final > 0.0 and self.t0 <= epoch <= self.T
