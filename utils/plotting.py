"""Plotting utilities. All plots save to ``results/`` and never require a display.

matplotlib is imported with the non-interactive 'Agg' backend so this works in
headless environments and CI. If matplotlib is unavailable the functions no-op
with a warning rather than crashing a training run.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], results_dir: str) -> None:
    """Loss, accuracy and sparsity vs epoch (three separate PNGs)."""
    if not _HAVE_MPL:
        print("[plotting] matplotlib unavailable; skipping training curves.")
        return
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if "train_loss" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history["train_loss"], label="train")
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
        ax.set_title("Loss vs epoch"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(results_dir, "loss_curve.png"))

    if "train_acc" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history["train_acc"], label="train")
        if "val_acc" in history:
            ax.plot(epochs, history["val_acc"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs epoch"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(results_dir, "accuracy_curve.png"))

    if "sparsity" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history["sparsity"], color="C3")
        ax.set_xlabel("epoch"); ax.set_ylabel("sparsity")
        ax.set_title("Sparsity vs epoch"); ax.grid(alpha=0.3)
        _save(fig, os.path.join(results_dir, "sparsity_curve.png"))


def plot_pareto(curves: Dict[str, Dict[str, Sequence[float]]], results_dir: str,
                filename: str = "pareto_curve.png") -> None:
    """Accuracy vs sparsity trade-off, one line per pruning method.

    ``curves[method] = {"sparsity": [...], "acc_mean": [...], "acc_std": [...]}``
    """
    if not _HAVE_MPL:
        print("[plotting] matplotlib unavailable; skipping Pareto curve.")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, c in curves.items():
        s = [100.0 * x for x in c["sparsity"]]
        mean = [100.0 * x for x in c["acc_mean"]]
        std = [100.0 * x for x in c.get("acc_std", [0.0] * len(mean))]
        ax.errorbar(s, mean, yerr=std, marker="o", capsize=3, label=method)
    ax.set_xlabel("sparsity (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs sparsity (Pareto trade-off)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, os.path.join(results_dir, filename))


def plot_dense_vs_sparse_bars(report: Dict[str, float], results_dir: str,
                              filename: str = "dense_vs_sparse.png") -> None:
    """Side-by-side bars for FLOPs and inference time (dense vs sparse)."""
    if not _HAVE_MPL:
        print("[plotting] matplotlib unavailable; skipping dense-vs-sparse bars.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["dense", "sparse"],
                [report["dense_flops_per_sample"], report["sparse_flops_per_sample"]],
                color=["C0", "C2"])
    axes[0].set_title("FLOPs / sample"); axes[0].set_ylabel("FLOPs")
    axes[1].bar(["dense", "sparse"],
                [report["dense_infer_ms"], report["sparse_infer_ms"]],
                color=["C0", "C2"])
    axes[1].set_title("Inference time"); axes[1].set_ylabel("ms / batch")
    fig.suptitle("Dense vs sparse")
    _save(fig, os.path.join(results_dir, filename))
