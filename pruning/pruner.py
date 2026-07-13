"""Pruner: orchestrates importance scoring, scheduling and mask updates.

Lifecycle within a training run::

    pruner = Pruner(model, cfg)
    for epoch in range(E):
        for batch in data:
            forward; loss; backward
            pruner.accumulate_importance()   # gather |w*grad| this step
            optimizer.step(); zero_grad
        pruner.update_masks(epoch)           # ramp sparsity, re-mask layers
        (optional) pruner.maybe_regrow(...)

``accumulate_importance`` batch-averages ``|w * grad|`` across the epoch so the
saliency score is not dominated by one noisy mini-batch. Magnitude pruning
skips accumulation (it only needs the current weights).
"""
from __future__ import annotations

from typing import List

import numpy as np

from pruning.scheduler import SparsityScheduler
from pruning.importance import get_criterion
from pruning.mask import compute_layer_mask


class Pruner:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.layers: List = model.prunable_layers()
        self.criterion = get_criterion(cfg.pruning_method)
        self.scheduler = SparsityScheduler(
            target_sparsity=cfg.target_sparsity,
            start_epoch=cfg.prune_start_epoch,
            end_epoch=cfg.prune_end_epoch,
            schedule=cfg.schedule,
        )
        # per-layer running sum of |w*grad| and a batch counter
        self._accum = [np.zeros_like(l.weight.data) for l in self.layers]
        self._count = 0

    # -- called every training step -----------------------------------
    def accumulate_importance(self) -> None:
        if not self.criterion.needs_grad_accumulation:
            return
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                self._accum[i] += np.abs(layer.weight.data * layer.weight.grad)
        self._count += 1

    def _reset_accum(self) -> None:
        for a in self._accum:
            a.fill(0.0)
        self._count = 0

    # -- called once per epoch ----------------------------------------
    def update_masks(self, epoch: int) -> float:
        """Apply this epoch's sparsity target to every prunable layer.

        Returns the achieved overall sparsity (fraction of zeroed weights).
        """
        target = self.scheduler.sparsity_at(epoch)
        if target > 0.0:
            denom = max(self._count, 1)
            for i, layer in enumerate(self.layers):
                accum_avg = self._accum[i] / denom
                scores = self.criterion.score(layer, accum_avg)
                new_mask = compute_layer_mask(scores, target, layer.mask)
                layer.set_mask(new_mask)
        self._reset_accum()
        return self.current_sparsity()

    # -- bookkeeping ---------------------------------------------------
    def current_sparsity(self) -> float:
        total = sum(l.total_weights for l in self.layers)
        active = sum(l.active_weights for l in self.layers)
        if total == 0:
            return 0.0
        return 1.0 - active / total

    def layer_sparsities(self) -> List[float]:
        return [l.sparsity for l in self.layers]
