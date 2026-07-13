"""Save / load a trained model (weights + masks) plus its config.

We persist the numpy arrays in a single ``.npz`` and the architecture in a
sibling ``.json`` so ``evaluate.py`` can rebuild an identical model without the
training script. Masks are saved alongside weights so the sparse structure is
preserved.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np

from config import Config
from nn.linear import Linear
from nn.sequential import build_mlp


def save_model(model, cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {}
    for i, layer in enumerate(_linear_layers(model)):
        arrays[f"W{i}"] = layer.weight.data
        arrays[f"b{i}"] = layer.bias.data if layer.bias is not None else np.zeros(0)
        arrays[f"mask{i}"] = layer.mask
    np.savez(path, **arrays)
    with open(_json_path(path), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def load_model(path: str) -> Tuple["object", Config]:
    with open(_json_path(path), "r", encoding="utf-8") as f:
        cfg = Config(**json.load(f))
    data = np.load(path if path.endswith(".npz") else path + ".npz")

    # infer dims from the saved matrices and rebuild the same MLP skeleton
    n_linears = sum(1 for k in data.files if k.startswith("W"))
    model = build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.n_classes,
                      cfg.activation)
    for i, layer in enumerate(_linear_layers(model)):
        layer.weight.data[...] = data[f"W{i}"]
        if layer.bias is not None and data[f"b{i}"].size:
            layer.bias.data[...] = data[f"b{i}"]
        layer.set_mask(data[f"mask{i}"])
    assert i + 1 == n_linears
    return model, cfg


def _linear_layers(model):
    return [l for l in model.layers if isinstance(l, Linear)]


def _json_path(path: str) -> str:
    base = path[:-4] if path.endswith(".npz") else path
    return base + ".json"
