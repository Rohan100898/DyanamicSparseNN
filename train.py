"""Training pipeline: end-to-end train + dynamic self-prune + evaluate.

``train(cfg)`` runs one full experiment described by a :class:`config.Config`:
build the data and model, train with the hand-written optimizer, ramp sparsity
via the :class:`~pruning.pruner.Pruner`, then report the dense-vs-sparse
trade-off. It returns a results dict (also consumed by the Part-4 sweep).

CLI examples (see README for the canonical one-liners):

    python train.py --target-sparsity 0        # Part 2: dense baseline
    python train.py --target-sparsity 0.9       # Part 3: self-pruning run
    python train.py --dataset mnist --epochs 30
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np

from config import Config, default_config
from engine import Tensor, set_default_dtype
from data.loader import get_dataset
from data.preprocess import iterate_minibatches
from nn.sequential import build_mlp
from nn.loss import softmax_cross_entropy
from optim import build_optimizer
from pruning.pruner import Pruner
from pruning.regrowth import regrow_step
from utils.logger import History, check_finite
from utils.metrics import accuracy
from utils import flops as flops_util
from evaluate import dense_vs_sparse_report, print_report


def _epoch_eval(model, X, y):
    """Return (mean cross-entropy loss, accuracy) over a dataset split."""
    logits = model(Tensor(X, requires_grad=False))
    loss = softmax_cross_entropy(logits, y).data
    acc = accuracy(y, np.argmax(logits.data, axis=1))
    return float(loss), float(acc)


def train(cfg: Config, eval_repeats: int = 30) -> Dict:
    set_default_dtype(np.float32)
    rng = np.random.default_rng(cfg.seed)

    ds = get_dataset(cfg)
    # keep config's view of dims in sync with the actual dataset
    cfg.input_dim, cfg.n_classes = ds.input_dim, ds.n_classes

    model = build_mlp(ds.input_dim, cfg.hidden_dims, ds.n_classes,
                      activation=cfg.activation, init=cfg.init, rng=rng)
    optimizer = build_optimizer(cfg.optimizer, model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    pruner = None if cfg.is_dense_run else Pruner(model, cfg)

    history = History()
    dense_test_acc = None

    for epoch in range(cfg.epochs):
        # --- one pass over the training set ---
        batch_losses = []
        for xb, yb in iterate_minibatches(ds.X_train, ds.y_train,
                                          cfg.batch_size, rng):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=False))
            loss = softmax_cross_entropy(logits, yb)
            check_finite("loss", loss.data)            # NaN guard
            loss.backward()
            if pruner is not None:
                pruner.accumulate_importance()          # gather |w*grad|
            optimizer.step()
            batch_losses.append(float(loss.data))

        # --- capture the dense model's accuracy just before pruning starts ---
        if pruner is not None and epoch == cfg.prune_start_epoch - 1:
            dense_test_acc = _epoch_eval(model, ds.X_test, ds.y_test)[1]

        # --- pruning + optional regrowth ---
        sparsity = 0.0
        if pruner is not None:
            sparsity = pruner.update_masks(epoch)
            if cfg.regrowth and epoch >= cfg.prune_end_epoch and epoch % 3 == 0:
                xb, yb = next(iterate_minibatches(ds.X_train, ds.y_train,
                                                  cfg.batch_size, rng))
                regrow_step(model, xb, yb, softmax_cross_entropy,
                            cfg.regrowth_fraction)

        # --- metrics ---
        tr_loss = float(np.mean(batch_losses))
        _, tr_acc = _epoch_eval(model, ds.X_train, ds.y_train)
        va_loss, va_acc = _epoch_eval(model, ds.X_val, ds.y_val)
        history.log(train_loss=tr_loss, train_acc=tr_acc,
                    val_loss=va_loss, val_acc=va_acc, sparsity=sparsity)

        if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:3d} | loss {tr_loss:.4f} | "
                  f"train_acc {tr_acc:.3f} | val_acc {va_acc:.3f} | "
                  f"sparsity {sparsity*100:5.1f}%")

    # --- final evaluation: dense-masked vs real sparse (CSR) ---
    report = dense_vs_sparse_report(model, ds.X_test, ds.y_test, repeats=eval_repeats)
    if dense_test_acc is not None:
        report["dense_pretrained_test_acc"] = dense_test_acc

    result = {
        "config": cfg.to_dict(),
        "history": history.as_dict(),
        "report": report,
        "final_sparsity": pruner.current_sparsity() if pruner else 0.0,
        "layer_sparsities": pruner.layer_sparsities() if pruner else [],
    }
    result.update(flops_util.summary(model))
    return result, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cfg_from_args(args) -> Config:
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("save", "no_plots")}
    return default_config(**overrides)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a DynamicSparseNN model.")
    ap.add_argument("--dataset", choices=["spiral", "mnist"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--activation", choices=["relu", "sigmoid", "tanh", "gelu"])
    ap.add_argument("--optimizer", choices=["adam", "sgd"])
    ap.add_argument("--target-sparsity", dest="target_sparsity", type=float,
                    help="hard sparsity budget (0 = dense baseline)")
    ap.add_argument("--pruning-method", dest="pruning_method",
                    choices=["saliency", "magnitude"])
    ap.add_argument("--schedule", choices=["cubic", "linear"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--regrowth", action="store_true", default=None)
    ap.add_argument("--save", default="models/pruned.npz",
                    help="checkpoint path (.npz)")
    ap.add_argument("--no-plots", dest="no_plots", action="store_true")
    args = ap.parse_args()

    cfg = _build_cfg_from_args(args)
    print(f"Config: dataset={cfg.dataset} method={cfg.pruning_method} "
          f"target_sparsity={cfg.target_sparsity} seed={cfg.seed} "
          f"epochs={cfg.epochs}")
    result, model = train(cfg)
    print_report(result["report"])

    # save checkpoint
    if args.save:
        from utils.checkpoint import save_model
        save_model(model, cfg, args.save)
        print(f"\nSaved checkpoint -> {args.save}")

    # plots
    if not args.no_plots:
        from utils.plotting import plot_training_curves, plot_dense_vs_sparse_bars
        os.makedirs(cfg.results_dir, exist_ok=True)
        plot_training_curves(result["history"], cfg.results_dir)
        plot_dense_vs_sparse_bars(result["report"], cfg.results_dir)
        print(f"Saved plots -> {cfg.results_dir}/")


if __name__ == "__main__":
    main()
