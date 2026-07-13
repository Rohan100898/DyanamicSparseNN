# DESIGN.md — DynamicSparseNN

Design rationale for the four questions that matter. Each answer points at the
code that implements it.

---

## 1. The importance criterion, and why `|w·∂L/∂w|` approximates the loss change from removing a connection

**Criterion:** a connection's importance is
`saliency(w_i) = |w_i · ∂L/∂w_i|`
(batch-averaged over an epoch), implemented in
[`pruning/importance.py`](pruning/importance.py).

**Derivation.** "Removing" connection `i` means forcing its weight to zero — a
perturbation `δw_i = -w_i` (and `δw_j = 0` for `j ≠ i`). Take a first-order
Taylor expansion of the loss about the current weights `w`:

```
L(w + δw) ≈ L(w) + Σ_j (∂L/∂w_j)·δw_j + O(‖δw‖²)
```

Only component `i` is perturbed, so the predicted change in loss from deleting
weight `i` is

```
ΔL_i ≈ (∂L/∂w_i)·δw_i = (∂L/∂w_i)·(−w_i) = −w_i·∂L/∂w_i.
```

We care about how much *worse* the loss gets, i.e. the magnitude:

```
|ΔL_i| ≈ |w_i · ∂L/∂w_i|.
```

That is exactly the saliency score. It is the first-order estimate of "how much
does the loss increase if I delete this connection," which is precisely what you
want a pruning criterion to rank.

**Why it beats the two one-sided heuristics.**
- **Magnitude only (`|w|`)** ignores `∂L/∂w`. It will keep a large weight the loss
  no longer depends on, and prune a *small* weight the loss depends on heavily —
  the "thin wire powering the alarm." Our sweep shows this bites at 98% sparsity:
  saliency 99.3% vs magnitude 96.8%.
- **Gradient only (`|∂L/∂w|`)** ignores scale: a weight can have a large gradient
  simply because it is large, and setting a large weight to zero is a big
  perturbation regardless of gradient. The product `w·∂L/∂w` is the correctly
  scaled quantity — it is the actual first-order term in `ΔL`.

**Honest caveat (and why the implementation compensates).** At a perfect minimum
`∂L/∂w = 0`, so the first-order term vanishes and one would need the second-order
(Hessian) term used by Optimal Brain Damage. Two things make the first-order
criterion work well in practice here: (1) we prune *during* training, on the ramp,
where the network is not at a stationary point and gradients are informative; and
(2) we **accumulate `|w·grad|` over a whole epoch** rather than trusting one noisy
mini-batch (`Pruner.accumulate_importance`), which averages the local loss
sensitivity over many points. A pure second-order criterion would be more
accurate but needs Hessian-vector products — much more expensive for a marginal
gain at these sparsities.

---

## 2. What the engine computes as "the gradient of a masked weight," and why that is the right choice

**It computes exactly zero.** A `Linear` layer's forward pass is
`y = x · (W ⊙ mask) + b` — the mask multiplies the weight *inside the
computation graph* ([`nn/linear.py`](nn/linear.py)). Writing `effective = W ⊙ mask`,
the chain rule gives

```
∂L/∂W_ij = (∂L/∂effective_ij) · (∂effective_ij/∂W_ij) = (∂L/∂effective_ij) · mask_ij.
```

For a pruned connection `mask_ij = 0`, so `∂L/∂W_ij = 0`, automatically, with no
special-casing.

**Why that is correct.** A masked weight has *no effect whatsoever* on the output
(it is multiplied by zero before it touches `x`). A quantity that cannot change
the output cannot change the loss, so its true partial derivative **is** zero.
Any nonzero "gradient" for a masked weight would be a bug — it would mean the
optimizer moves a weight that provably does not matter.

**The subtlety the mask-in-graph alone does *not* fix — and how we do.** A zero
gradient is necessary but not sufficient. Adam keeps momentum: `m ← β₁m +
(1−β₁)g`. With `g = 0` the momentum only *decays* (`m ← β₁m`); it is not zero.
Adam would keep applying `lr·m̂/(√v̂+ε)` and slowly drag the dead weight away from
its frozen value, and the instant that weight is unmasked by regrowth, the stale
`m`/`v` inject a huge, meaningless update that corrupts training. So every step
the optimizer **zeros the gradient and freezes the first and second moments of
masked weights** ([`optim/adam.py`](optim/adam.py), the four `*= mask` lines). A
revived weight therefore restarts from a clean `(m=0, v=0)` state. This is
verified end-to-end by `test_prune_revive_no_corruption` and
`test_optimizer_state_does_not_drift` in
[`tests/pruning_test.py`](tests/pruning_test.py).

(For *growing* a dead connection back we deliberately do **not** rely on gradient
leakage; regrowth runs an explicit dense-gradient probe — see
[`pruning/regrowth.py`](pruning/regrowth.py) — which is the honest way to learn
what an inactive connection would contribute.)

---

## 3. Where the autodiff engine bottlenecks, and how to optimize it

**Where the time goes.** The engine is correct and readable but pays Python
overhead that dominates at this scale:

1. **Graph construction per forward pass.** Every op allocates a fresh `Tensor`,
   a Python closure (`_backward`), a `parents` tuple, and a zero-filled `grad`
   buffer. For a small MLP the useful numpy work per op (a `matmul`, an add) is
   tiny compared to this per-node Python bookkeeping. This is the #1 cost.
2. **Backward driver overhead.** `topological_sort` runs an explicit-stack DFS and
   builds a Python list every step; `backward` then calls one Python closure per
   node. Interpreter dispatch, not FLOPs, dominates.
3. **Allocation churn.** `unbroadcast`, `np.ones_like`, and intermediate arrays
   are re-allocated every step instead of reused.
4. **Small-batch BLAS underutilization.** `matmul` calls into numpy's BLAS, but
   with a 128×256 batch the matrices are too small to amortize call overhead or
   saturate the CPU.
5. **Sparse scatter kernel.** The from-scratch CSR path uses `np.add.at`
   (`pruning/sparse_forward.py`), which is not vectorized internally and is the
   slow path; the SciPy CSR backend exists precisely to sidestep it.

**How I would optimize, highest impact first.**

- **Tape instead of closures.** Record ops as lightweight `(op_id, input_refs,
  saved_tensors)` entries on a flat tape and dispatch backward through a `switch`
  over op ids, eliminating per-op closure allocation and most `Tensor` objects.
- **Preallocate and reuse buffers.** Keep persistent `grad` buffers per parameter
  and scratch buffers per activation shape; zero in place. Cache the topological
  order when the graph shape is static across steps (it is, for a fixed model).
- **Fuse ops.** `matmul + bias + activation` is three nodes with three
  allocations; a fused `linear_relu` node halves traffic. The fused
  `softmax_cross_entropy` already demonstrates the win (one node, closed-form
  gradient, better numerics).
- **Bigger batches / vectorization** to amortize dispatch and better use BLAS.
- **Compile the hot path.** Move the elementwise ops and the sparse kernel to
  Numba/Cython/C (a real CSR SpMM kernel), or lower the whole static graph to a
  compiled function. Replace `np.add.at` with `scipy.sparse` (done) or a compiled
  segment-sum.
- **Structured sparsity** (below) so the sparse forward can use blocked BLAS
  instead of scattered indexing.

The gradient checker's element-by-element finite differencing is `O(#params)`
forward passes — but that is test-only tooling, intentionally simple, not on the
training path.

---

## 4. Serving a self-pruned model in a real multi-tenant inference service at scale

Shrinking the model in research is only half the job; the savings have to survive
production. What I would actually do:

**1. Ship an immutable, weights-only artifact.** Export just the surviving
weights + mask + config — no autograd graph, no optimizer state. We already
persist exactly this (`utils/checkpoint.py`: `.npz` weights/masks + `.json`
config), which makes deployments reproducible and versionable. Add post-training
**quantization** (int8/fp16) on top of sparsity for a second, orthogonal size and
latency win.

**2. Pick the sparsity format for the hardware — this is the crux.** *Unstructured*
sparsity (what the pruner produces) rarely speeds up dense GPU BLAS, because
scattered nonzeros defeat coalesced memory access; you can end up *slower* than
dense. Options, in order of practicality:
   - **2:4 semi-structured sparsity** on NVIDIA Ampere+ (hardware sparse tensor
     cores) — real speedups with a small accuracy cost; constrain the mask to the
     2:4 pattern during pruning.
   - **Block/structured pruning** (prune whole neurons/columns) so the pruned
     model is a genuinely smaller *dense* matmul — the simplest thing that is
     actually fast everywhere, at some accuracy cost vs unstructured.
   - **Compiled sparse (CSR/CSC) kernels** for CPU serving at high sparsity — the
     CSR path here already beats dense ~6× at 90% on CPU for this model.
   - Or **distill** the sparse model into a small dense one and serve that.
   The right choice is measured per target hardware, not assumed — the FLOP
   reduction is necessary but not sufficient for wall-clock speedup.

**3. Multi-tenancy specifics.**
   - **Memory:** many tenant models must coexist. Sparse + quantized storage
     multiplies how many fit in RAM/VRAM; keep them in a compact format and use an
     **LRU model cache** with lazy load/evict so cold tenants don't hold memory.
   - **Throughput:** use **continuous / dynamic batching** to pool requests across
     tenants hitting the same model, and route by model id. Autoscale replicas on
     queue depth and per-tenant SLA.
   - **Isolation & fairness:** per-tenant rate limits and resource quotas so one
     tenant can't starve others; optionally physical isolation for noisy or
     sensitive tenants.

**4. Reliability & correctness in prod.**
   - **Canary + A/B** the pruned model against the dense baseline; roll out by
     percentage; keep the dense model as an instant **fallback**.
   - **Monitor accuracy drift** on live traffic (shadow scoring, delayed labels)
     because a 98%-sparse model has little slack — a distribution shift hurts it
     more than the dense one.
   - **Pin reproducibility:** immutable artifact hash, pinned seeds/config, so the
     model serving traffic is exactly the one that was evaluated.

**In short:** prune to a hard budget offline, export a compact quantized artifact,
choose a sparsity *format* that the serving hardware can actually accelerate
(2:4 / structured / compiled CSR / distill), and wrap it in a multi-tenant runtime
that batches across tenants, caches models by memory pressure, and canaries every
rollout against a dense fallback.
