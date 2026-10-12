# 4-layer Pile setup

This pipeline targets the existing `data/target/model_step_99999.pt` checkpoint
and GPT-NeoX-tokenized `data/geo67/stream8k/pile_1b_uint16.bin`. It reuses the
repository's plain-gradient `Model67` loader. The 24 decomposed matrices are
Q/K/V/O, MLP up and MLP down in each of four layers (28,311,552 weights).
Embeddings, output head and RMSNorms retain the frozen target values.

## Configuration

The prepared run uses 1,024 free rank-one components per matrix, exceeding
the 768-dimensional rank bound, with one summed loss, backward and Adam
update across all 24 matrices. Parameters, gradients and Adam states remain
FP32. Cached activations/cotangents and large training score intermediates
use FP16; only reduced norms and the concentration loss use FP32. Both
differentiable projection and post-step projection use FP32. The original
1e-5 relative matrix-sum tolerance remains. Every Adam update is projected
directly, without a cancellation cap or line search.
The initial random factors are constructed using the existing FP64 setup
projection; subsequent training projections use FP32.

Defaults: 50k scheduled steps, batch 2,048 cached token positions, cosine LR
1e-3 to 1e-5, p from 1 to 0.5 over the first 80%, seed 0. GradScaler overflow
skips retain the existing semantics and are recorded in the checkpoint.

## Attention-compatible sensor

The toy's exact low-dimensional QR Jacobians do not apply to a transformer.
For each 512-token context, take eight independent Rademacher output probes
`v_q / sqrt(512 * vocabulary_size)` and compute true vector-Jacobian products
through the frozen FP32 model. Cache module input `h_t` and output cotangent
`g_q,t` at each token position. A piece's local-use importance is

```
a_c(t) = |h_t r_c| sqrt(mean_q (g_q,t u_c)^2).
```

Before finite-precision rounding, the squared effect estimate is unbiased
for the component's squared full-context logit RMS effect. The norm and Lp
loss themselves are not unbiased estimators. This is a finite random sketch,
not the toy's exact Jacobian norm, and eight probes are a configurable pilot
choice rather than a demonstrated sufficient number. The cache includes
attention paths into future predictions. It does not substitute a gold-token
loss gradient or GIM backward for the output-effect objective.

The event is a component use at one module/token position. It is not the
joint effect of changing that parameter component at every position in the
context. Evaluation likewise uses local module/token masks, with a frozen
global ranking across the 24 matrices at each token position.

The fixed cache has 32 contexts / 16,384 token positions, sampled without
overlap from the first 100M corpus tokens. Evaluation samples beyond 100M;
the full corpus hash and exact offsets are recorded. A shared seeded
permutation mixes positions across contexts before selecting the fixed
256-token loss-normalization reference. The small fixed cache
is a starting configuration, not a claim of sufficient Pile coverage.
Cache files retain target/model-source hashes and FP16 tensors; initial loss
normalization calculates scores in FP32 on those rounded caches.

## Launch and resume

After `bash par-auto/sync.sh`, submit from the Explorer project root:

```bash
sbatch slurm_parauto_pile.sh
```

The full run is not launched by setup. It writes under
`data/parauto/pile_main/`; set `PILE_RUN` to choose another directory. The
launcher builds the cache once, then trains or explicitly resumes from
`run/checkpoint.pt`. Checkpoints are atomic and contain the factors, Adam
moments, gradient-scaler state, batch-generator states, normalization scores,
history and next step. Resubmitting the same launcher after the eight-hour
allocation ends resumes the original 50k schedules; at most the work since
the last 100-step checkpoint is repeated. There is no automatic requeue.

`train_pile.py train --max-steps N` stops cleanly after N additional scheduled
updates without shortening the LR/p schedule. `--resume` requires an existing
matching target/cache/config checkpoint. Checkpoints from the retired capped
optimizer cannot resume into this version. A fresh train refuses to overwrite
an existing checkpoint.

The final evaluation computes actual next-token KL and cross entropy on
fresh contexts, including all-kept and all-removed controls and several
intermediate budgets. It uses fresh independent Jacobian probes for ranking
and full nonlinear forwards for each mask. KL accumulation uses FP64;
model forwards and factor application use FP32. The all-kept check must have
KL below 1e-6. This is a budget-grid check, not yet a one-component-at-a-time
curve or an SPD language-model comparison.

## Validation

`slurm_parauto_pile_smoke.sh` runs the unit suite, caches four real 512-token
contexts, performs two full-size joint updates, saves, resumes for two more,
and evaluates one disjoint real context. This validates production matrix
and batch shapes while keeping the training duration short. Tests also check
the probe estimate against a local-gate derivative through attention, all-kept
model parity, and bitwise continuation of Adam/RNG/schedules after resuming.

Before removal of the cancellation cap, V100 job **10244172** validated the
cache and evaluation pipeline: all 36 tests passed; the real model
completed 2 + 2 joint steps across a checkpoint/resume boundary, followed by
actual held-out ablations. Maximum final matrix-sum relative error was
7.14e-7. All-kept KL on 512 fresh prediction positions was 1.84e-11.
The smoke cache used 2,048 positions and occupied 0.95 GiB; peak allocated
GPU tensors were about 5.71 GiB. The full 16,384-position cache is larger.
Four training steps validate plumbing, not decomposition quality or steady
throughput. [Smoke result](out/pile_smoke_10244172/run/result.json) and
[held-out checks](out/pile_smoke_10244172/run/evaluation.json) retain details.
