"""Classification metrics computed from scratch on numpy arrays."""
from __future__ import annotations

from typing import Dict

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        n_classes: int) -> Dict[str, float]:
    """Macro-averaged precision / recall / F1 (unweighted mean over classes)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    return {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }


def full_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    out = {"accuracy": accuracy(y_true, y_pred)}
    out.update(precision_recall_f1(y_true, y_pred, n_classes))
    return out
