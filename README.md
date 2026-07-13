# DynamicSparseNN

A **from-scratch deep-learning framework with automatic dynamic pruning** — no
PyTorch, no TensorFlow. It contains its own reverse-mode autograd engine, neural
network layers, optimizers (Adam / SGD), and a gradient-aware self-pruning stack
that shrinks a trained model to a hard sparsity budget while proving the savings
are *real* (fewer FLOPs and faster inference, not just more zeros on disk).

Everything runs offline on a small synthetic dataset, so every command below is
reproducible in seconds without downloading anything.

> **Headline result (falsifiable, from `experiments/runner.py`, 3 seeds):**
> At **98% weight sparsity, saliency (`|w·grad|`) pruning retains 99.3% ± 0.2%
> test accuracy vs. 96.8% ± 0.4% for magnitude pruning** (FLOPs reduced 98%).
> Up to 90% sparsity the two are within noise; the gap opens once capacity
> actually binds. See [`results/pareto_summary.csv`](results/pareto_summary.csv).

---

## The four parts, clearly defined

The build is organized into the four parts the brief asks for. Each is
independently testable.

### Part 1 — Autograd engine (`engine/`)
A custom `Tensor` and reverse-mode automatic differentiation.
- `Tensor`: `data`, `grad`, `requires_grad`, `parents`, `op`, `_backward`.
- Ops (`engine/ops.py`): `add`, `sub`, `mul`, `div`, `neg`, `pow`, `matmul`,
  `transpose`, `reshape`, `sum`, `mean`, `relu`, `sigmoid`, `tanh`, `gelu`,
  `softmax` — **all with full broadcasting support** (`unbroadcast`).
- `graph.py` topologically sorts the DAG; `backward.py` walks it in reverse.
- **Correctness gate:** every op is numerically gradient-checked to ~1e-10, and
  the `(N,D)+(D,)` bias-broadcast case is tested *by name*.

### Part 2 — Network, optimizer, stable training (`nn/`, `optim/`, `train.py`)
- Layers: `Linear` (with a built-in mask), activations, numerically stable fused
  `softmax_cross_entropy`, `Sequential`.
- Hand-written `Adam` and `SGD` (both mask-aware).
- **Justified initialization** (He for ReLU/GELU, Xavier for tanh/sigmoid — see
  `utils/initialization.py` and DESIGN.md) and a **NaN guard** (`utils/logger.py`)
  so "stable training without NaNs" is an enforced artifact, not a hope.

### Part 3 — Dynamic, gradient-aware self-pruning (`pruning/`)
- `importance.py` — saliency `|w·grad|` (derivation in DESIGN.md).
- `magnitude.py` — the `|w|` baseline, for the Part-4 comparison.
- `scheduler.py` — **cubic** sparsity ramp (formula below) to a **hard budget**.
- `mask.py` — per-layer sensitivity slicing to the exact budget, monotone.
- `pruner.py` — ties it together (accumulates importance, updates masks/epoch).
- `sparse_forward.py` — **genuine CSR sparse matmul** (`2·nnz` FLOPs), not
  dense×zero. Built from scratch, validated against the dense forward.
- `regrowth.py` — bonus RigL-style reactivation of high-gradient dead weights.
- The optimizer **freezes the momentum (m, v) of masked weights** so Adam can't
  resurrect a pruned connection with stale momentum.

### Part 4 — Experiment matrix & proof (`experiments/`, `evaluate.py`, `utils/`)
- `experiments/runner.py` sweeps the full grid
  `(pruning_method × target_sparsity × seed)`, each cell a separate run.
- Sparsity targets `{0, 50, 75, 90, 95, 98}%`; **saliency vs. magnitude**;
  **3 seeds** with **mean ± std**; raw CSV + aggregated CSV + **Pareto curve**.
- Real **FLOP / parameter accounting tied to the sparse path** (`utils/flops.py`).
- Dense-vs-sparse accuracy, FLOPs and wall-clock speedup (`evaluate.py`).

---

## Install

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `matplotlib` (plus
optional `scikit-learn` for MNIST and `pytest` for the test runner).

### With `uv` (preferred)

```bash
uv venv
uv pip install -e .            # or: uv pip install -r requirements.txt
```

### With pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run it — one command each

| Goal | Command |
|------|---------|
| **Gradient-check tests** (Part 1) | `python -m pytest tests/gradient_check.py tests/broadcast_test.py` |
| **Part 2 — dense training** (stable, no NaNs) | `python train.py --target-sparsity 0` |
| **Part 3 — self-pruning run** (to 90% sparsity) | `python train.py --target-sparsity 0.9` |
| **Part 4 — Pareto sweep** (method × sparsity × seed) | `python -m experiments.runner` |

Extras:

```bash
python -m pytest                       # run the entire test suite (31 tests)
python -m tests.pruning_test           # prune/revive "no corruption" test, standalone
python evaluate.py --model models/pruned.npz   # dense-vs-sparse report for a checkpoint
python -m experiments.runner --quick   # fast smoke sweep (2 seeds, {0, 90%})
python train.py --target-sparsity 0.9 --pruning-method magnitude   # baseline
python train.py --dataset mnist --epochs 30                        # real MNIST (needs scikit-learn)
```

Outputs land in `results/` (curves, `pareto_curve.png`, `sweep_raw.csv`,
`pareto_summary.csv`, `claim.txt`) and `models/` (checkpoints).

---

## The cubic sparsity schedule

Sparsity ramps from 0 to the target over `[t0, T]`, then holds for fine-tuning:

```
s(t) = s_final · (1 − (1 − p)³),     p = clamp((t − t0)/(T − t0), 0, 1)
```

Cubic beats linear because it **front-loads** pruning (removes obviously useless
weights early) and **slows down near the budget**, giving the network the most
time to adapt where every surviving weight matters (Zhu & Gupta, 2017).

---

## Results (spiral, 3 seeds)

Sparse-model test accuracy, mean ± std, from `results/pareto_summary.csv`:

| Sparsity | Saliency `\|w·grad\|` | Magnitude `\|w\|` | FLOP reduction |
|---------:|:--------------------:|:-----------------:|:--------------:|
| 0%   | 99.8 ± 0.1 | 99.8 ± 0.1 | 0%  |
| 50%  | 99.9 ± 0.1 | 99.8 ± 0.2 | 50% |
| 75%  | 99.8 ± 0.1 | 99.8 ± 0.2 | 75% |
| 90%  | 99.8 ± 0.0 | 99.7 ± 0.2 | 90% |
| 95%  | 99.4 ± 0.3 | 99.6 ± 0.3 | 95% |
| **98%** | **99.3 ± 0.2** | **96.8 ± 0.4** | **98%** |

A single 90%-sparse checkpoint reduces FLOPs/sample from 133,632 to 13,364 and
runs inference ~6× faster via the CSR path, at **no accuracy loss**.

---

## How the gap notes were addressed

- **Part 1:** `sub`/`div` added explicitly; `sigmoid`, `tanh`, **and** `gelu`
  implemented alongside `relu`; dedicated `(N,D)+(D,)` broadcast test; the
  masked-weight gradient question is answered in code (mask in the graph) and by
  the prune→train→revive **no-corruption test** that freezes Adam's `m,v`.
- **Part 2:** written init rationale + enforced NaN guard.
- **Part 3:** Taylor derivation of `|w·grad|` (DESIGN.md), explicit cubic
  formula, and a config-level **hard-budget** knob (`--target-sparsity`).
- **Part 4:** `experiments/runner.py` sweep + `pruning/magnitude.py` baseline +
  multi-seed mean/std + FLOP accounting tied to the sparse path + the
  numbers-backed falsifiable claim above.

## Repository layout

```
engine/     Tensor, ops, graph, backward           (Part 1)
nn/         module, linear (+mask), activations, loss, sequential   (Part 2)
optim/      sgd, adam  (mask-aware)                 (Part 2)
pruning/    importance, magnitude, scheduler, mask, pruner,
            sparse_forward (CSR), regrowth          (Part 3)
data/       spiral + mnist loaders, preprocessing
utils/      initialization, metrics, flops, logger, plotting, checkpoint
experiments/runner.py — the Part-4 sweep           (Part 4)
tests/      gradient_check, broadcast_test, pruning_test
train.py    end-to-end training pipeline
evaluate.py dense-vs-sparse comparison + CLI
config.py   all hyperparameters (the single source of truth for a run)
```

See [DESIGN.md](DESIGN.md) for the importance-criterion derivation, the
masked-weight gradient rationale, the autodiff bottleneck analysis, and how
you'd serve a self-pruned model in a multi-tenant service at scale.
