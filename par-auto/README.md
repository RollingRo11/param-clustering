# Parameter autoencoder

The method decomposes each frozen matrix into freely learned rank-one reads
and writes, using an Lp concentration loss and **projected Adam**. Exact-sum
projection runs after every Adam step. The four matrix losses share one backward pass and Adam update.

The target is the verified **10,000-step residual toy**: 100 input features,
two layers, residual width 1,000 and MLP width 25. The matched experiments use
400 components per matrix and batch 2,048. See [method.md](method.md) and
[results.md](results.md). A properly scheduled 5k run is a strong cheap baseline.

The current [4-layer Pile setup](pile_scalar.md) uses the existing language-model
target, BF16 cached activations and score autocast, FP32 projection, and streamed
joint training across all 24 internal matrices. The training sensor is the true
context-summed next-token CE derivative, cached using one backward. The earlier
[output-probe pipeline](pile.md) is retained for comparisons.

## Main files

- `train.py`, `parauto_method.py`: training, schedules and exact-sum projection.
- `train_pile.py`, `parauto_pile.py`: Pile cache construction, train/resume and evaluation.
- `train_pile_scalar.py`, `parauto_pile_scalar.py`: current BF16 scalar streaming, 1k-step profiling, and time-budgeted full-model runs.
- `parauto_core.py`: free factors, affine projection and masked outputs.
- `parauto_target.py`: frozen toy and checkpoint helpers.
- `parauto_structured.py`, `parauto_toy.py`: exact structured scoring and dense reference.
- `parauto_evaluate.py`, `parauto_refine.py`: fresh-forward checks and calibration.
- `evaluate_klkeep.py`: native-sensor, one-component-at-a-time ablations.
- `compare_spd_schedules.py`: matched 5k/10k/50k KLKeep curves for par-auto and SPD.
- `compare_precision.py`: 50k FP16 mixed training versus the original FP32 run.
- `profile_precision_sweep.py`, `report_precision_sweep.py`: isolated 5k precision storage/projection speed ablations.
- `parauto_scalar.py`, `compare_scalar_sensor.py`: optional exact task-MSE gradient sensor and a crossed training/ranking comparison against the output-effect baseline.
- `parauto_tracking.py`: explicit W&B run setup, training/evaluation metrics, console logs, and new output artifacts in `param-autoenc`.
- `inspect_components.py`, `inspect_usage.py`, `plot_spd_comparison.py`: component analysis.
- `benchmark_spd.py`, `profile_spd.py`, `report_*.py`: pinned upstream SPD comparisons.
- `parauto_io.py`: checkpoint loading and historical format compatibility.

## Running

All numerical execution belongs in SLURM allocations. Run
`bash par-auto/sync.sh` locally, then submit from the Explorer project root:

```bash
sbatch slurm_parauto_train.sh
sbatch slurm_parauto_short_schedules.sh
sbatch slurm_parauto_spd_short.sh
sbatch slurm_parauto_precision.sh
sbatch slurm_parauto_precision_sweep.sh
sbatch slurm_parauto_scalar_sensor.sh
sbatch slurm_parauto_validate.sh
```

Within an allocation, use `train.py --target TARGET --output RUN --steps 5000`.
Each run schedules its learning rate from 1e-3 to 1e-5 and p from 1 to 0.5 over
its first 80%. SPD retains the authors' own constant 1e-3 learning rate.

`train.py --precision bf16` enables BF16 score autocast without loss scaling.
The historical `--precision fp16` option enables FP16 score matrix multiplies with dynamic
gradient scaling on CUDA; the default remains `fp32`. Both use FP32 master
parameters, Adam states, caches, initialization and normalization scores,
with FP64 exact-sum projections. FP16 score norms and the concentration loss
accumulate in FP32. The precision comparison evaluates both learned
decompositions in FP32 on the original shared held-out inputs. Training
metadata records gradient-overflow skips separately from scheduled steps.

KLKeep uses per-input global rankings across 1,600 matrix components, frozen
on original target activations: output-effect RMS for par-auto and learned
causal importance for SPD. Masks are binary and every removal level gets a
real nonlinear forward. The toy outputs regression values, so KL explicitly
uses temperature-1 softmax; raw-output MSE is also reported. Log x axes omit
zero, which remains in the data. These cumulative-prefix tests do not cover
arbitrary subsets. Primary errors include SPD's all-kept reconstruction error.

The optional toy experiment `train.py --sensor scalar_task_mse` replaces the
output-effect sensor with the absolute directional derivative of per-example
mean squared task error against `x + relu(x)`. Labels are checked against the
verified target's original provenance. It uses exact cached gradients, without
random probes, for concentration. Both sensors start from identically seeded
random factors projected to the target matrices. Each loss is normalized by
its own initial scores on the same fixed 256 training examples. Every Adam
update is applied and projected; there is no cancellation cap or line search.
The default training sensor remains `output_effect`. The scalar launcher trains
fresh output and scalar controls for 50k steps each, then evaluates both with
both ranking sensors on the original shared held-out examples. Score units
differ, while the existing Lp epsilon remains unchanged.

Historical results retain their original metadata. Retired source and trial
artifacts are archived outside the repository. The active source deploys to
`code/par-auto-current`; original 50k source snapshots remain intact remotely.

## Weights & Biases

CLI entry points initialize W&B directly in project **param-autoenc**. The
training loop calls `tracking.log_train` with the numeric row; initialization,
cache/checkpoint progress and evaluation also use explicit logging calls.
Human-readable stdout/stderr are recorded by the SDK independently. There is
no output parser, subprocess wrapper, or dependency on printed JSON formatting.

The project is [rohan-kathuria-neu/param-autoenc](https://wandb.ai/rohan-kathuria-neu/param-autoenc).
The cluster's existing credentials are sourced from `env.sh`; credentials are
never included in configuration or artifacts. `WANDB_ENTITY` can select an
entity and `WANDB_MODE=offline` or `disabled` can opt out of online syncing.
The launcher defaults to online logging. Unit tests and imported numerical
functions do not create runs.

Runs record resolved configuration, per-matrix losses and applied optimizer updates,
step counts, system GPU metrics, timing summaries, output hashes, curves,
figures, reports, and final factor/checkpoint artifacts. Only files created or
modified by the new invocation are uploaded; existing experiment directories
are not backfilled. Large activation caches and per-example evaluation tensors
stay on the cluster. Resumed Pile training reuses its saved W&B run ID; training
step is a separate chart axis, so resumed steps and multiple matrices are not
confused with W&B's event counter. `sbatch slurm_parauto_wandb_smoke.sh` verifies
metrics, both console streams, a figure, an interactive curve, and the report
artifact by reading the completed run back through the W&B API.
