"""Part 4 -- the experiment matrix.

Sweeps the full grid ``(pruning_method x target_sparsity x seed)``, each cell a
*separate* training run, and writes:

* ``results/sweep_raw.csv``      -- one row per run (raw numbers, no aggregation)
* ``results/pareto_summary.csv`` -- mean +/- std test accuracy per (method, sparsity)
* ``results/pareto_curve.png``   -- accuracy-vs-sparsity trade-off, one line/method
* ``results/claim.txt``          -- the falsifiable, numbers-backed sentence

This answers the Part-4 requirements: a Pareto curve, a second (magnitude)
pruning method as a baseline at the same sparsities, multiple seeds with
mean/std so we can argue whether saliency-vs-magnitude differences are real or
noise, and real FLOP accounting tied to the sparse forward path.

    python -m experiments.runner                 # full sweep (default grid)
    python -m experiments.runner --quick         # fast smoke sweep
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from config import default_config
from train import train

DEFAULT_SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98]
DEFAULT_METHODS = ["saliency", "magnitude"]
DEFAULT_SEEDS = [0, 1, 2]

RAW_FIELDS = [
    "method", "target_sparsity", "seed",
    "actual_sparsity", "dense_acc", "sparse_acc",
    "flops_reduction", "sparse_flops_per_sample", "inference_speedup",
    "params_active", "params_total",
]


def run_sweep(sparsities: List[float], methods: List[str], seeds: List[int],
              epochs: int, results_dir: str, eval_repeats: int = 10) -> List[Dict]:
    rows: List[Dict] = []
    total = len(methods) * len(sparsities) * len(seeds)
    i = 0
    for method in methods:
        for s in sparsities:
            for seed in seeds:
                i += 1
                cfg = default_config(
                    target_sparsity=s, pruning_method=method, seed=seed,
                    epochs=epochs, verbose=False)
                print(f"[{i:3d}/{total}] method={method:9s} "
                      f"target_sparsity={s:0.2f} seed={seed} ... ", end="", flush=True)
                result, _ = train(cfg, eval_repeats=eval_repeats)
                rep = result["report"]
                row = {
                    "method": method,
                    "target_sparsity": s,
                    "seed": seed,
                    "actual_sparsity": rep["weight_sparsity"],
                    "dense_acc": rep["dense_accuracy"],
                    "sparse_acc": rep["sparse_accuracy"],
                    "flops_reduction": rep["flops_reduction"],
                    "sparse_flops_per_sample": rep["sparse_flops_per_sample"],
                    "inference_speedup": rep["inference_speedup"],
                    "params_active": rep["params_active"],
                    "params_total": rep["params_total"],
                }
                rows.append(row)
                print(f"sparsity={row['actual_sparsity']*100:5.1f}% "
                      f"acc={row['sparse_acc']*100:5.2f}%")
    _write_raw(rows, results_dir)
    return rows


def aggregate(rows: List[Dict]) -> Dict:
    """Group by (method, target_sparsity) -> mean/std over seeds."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["method"], r["target_sparsity"])].append(r)

    summary = {}
    for (method, s), items in grouped.items():
        accs = np.array([it["sparse_acc"] for it in items])
        summary[(method, s)] = {
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=0)),
            "actual_sparsity_mean": float(np.mean([it["actual_sparsity"] for it in items])),
            "flops_reduction_mean": float(np.mean([it["flops_reduction"] for it in items])),
            "n_seeds": len(items),
        }
    return summary


def _write_raw(rows: List[Dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "sweep_raw.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote raw results  -> {path}")


def _write_summary(summary: Dict, results_dir: str) -> None:
    path = os.path.join(results_dir, "pareto_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "target_sparsity", "actual_sparsity_mean",
                    "acc_mean", "acc_std", "flops_reduction_mean", "n_seeds"])
        for (method, s) in sorted(summary.keys()):
            d = summary[(method, s)]
            w.writerow([method, s, f"{d['actual_sparsity_mean']:.4f}",
                        f"{d['acc_mean']:.4f}", f"{d['acc_std']:.4f}",
                        f"{d['flops_reduction_mean']:.4f}", d["n_seeds"]])
    print(f"Wrote summary      -> {path}")


def _curves_for_plot(summary: Dict, methods: List[str], sparsities: List[float]):
    curves = {}
    for method in methods:
        xs, means, stds = [], [], []
        for s in sorted(sparsities):
            if (method, s) in summary:
                d = summary[(method, s)]
                xs.append(d["actual_sparsity_mean"])
                means.append(d["acc_mean"])
                stds.append(d["acc_std"])
        curves[method] = {"sparsity": xs, "acc_mean": means, "acc_std": stds}
    return curves


def build_claim(summary: Dict, sparsity: float = 0.9) -> str:
    """Compose the falsifiable, numbers-backed README sentence."""
    sal = summary.get(("saliency", sparsity))
    mag = summary.get(("magnitude", sparsity))
    if not sal or not mag:
        return "(claim unavailable: need both methods at the target sparsity)"
    return (
        f"At {int(round(sal['actual_sparsity_mean']*100))}% weight sparsity, "
        f"saliency (|w*grad|) pruning retains "
        f"{sal['acc_mean']*100:.1f}% +/- {sal['acc_std']*100:.1f}% test accuracy "
        f"vs. {mag['acc_mean']*100:.1f}% +/- {mag['acc_std']*100:.1f}% for "
        f"magnitude pruning across {sal['n_seeds']} seeds "
        f"(FLOPs reduced {sal['flops_reduction_mean']*100:.0f}%)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Part-4 sparsity/method/seed sweep.")
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--sparsities", type=float, nargs="+", default=DEFAULT_SPARSITIES)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--results-dir", dest="results_dir", default="results")
    ap.add_argument("--claim-sparsity", dest="claim_sparsity", type=float, default=0.98)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke sweep: 2 seeds, 30 epochs, {0,0.9} sparsity")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0, 1]
        args.epochs = 30
        args.sparsities = [0.0, 0.9]

    print(f"Sweep grid: methods={args.methods} sparsities={args.sparsities} "
          f"seeds={args.seeds} epochs={args.epochs}\n")

    rows = run_sweep(args.sparsities, args.methods, args.seeds, args.epochs,
                     args.results_dir)
    summary = aggregate(rows)
    _write_summary(summary, args.results_dir)

    from utils.plotting import plot_pareto
    curves = _curves_for_plot(summary, args.methods, args.sparsities)
    plot_pareto(curves, args.results_dir)
    print(f"Wrote Pareto plot  -> {os.path.join(args.results_dir, 'pareto_curve.png')}")

    claim = build_claim(summary, args.claim_sparsity)
    claim_path = os.path.join(args.results_dir, "claim.txt")
    with open(claim_path, "w", encoding="utf-8") as f:
        f.write(claim + "\n")
    print("\n" + "=" * 70)
    print("FALSIFIABLE CLAIM:")
    print("  " + claim)
    print("=" * 70)


if __name__ == "__main__":
    main()
