# Paired H200 comparison: 1 billion positions per method

RunPod pod `sz9usox6q9hihe`, two H200s together, US-CO-1. GPU 0 runs VPD;
GPU 1 runs par-auto. The pair costs $9.18/hour plus storage. The supervisor
stops the pod after training, common evaluation, and verified R2 uploads.
Stopping retains the persistent volume; it does not terminate/delete it.

Both methods see exactly **1,000,000,000 prediction positions**, in identical
order: 30,517 batches of 64 sequences and a final batch of 37 sequences,
each with 512 prediction positions. There are 30,518 optimizer updates.
The 513th token supplies the final next-token label. No training rows repeat.
Par-auto uses the true derivative of context-summed next-token CE.

## What is matched

- Identical frozen 67M, four-layer Pile target and upstream target implementation.
  SHA256: `650f2eb603fe9fe649671c0c89b86746655bc49c96aa1ad00fbb4c1bf3c6c3b6`.
- All 24 internal matrices: Q/K/V/O and MLP up/down in each of four layers.
- Saved token sample, token order, batch, optimizer-update count, and hardware.
- BF16 autocast; FP32 parameters, parameter gradients and Adam states.
  Par-auto uses BF16 caches and FP32 projection. Target `flash_attention: false`
  is retained from its published configuration for both methods.
- Full-state checkpoints every 1,000 updates. Both have factor snapshots at
  1k/5k/10k/20k/30k/final. Setup, training, checkpoint time and peak allocated
  GPU memory are recorded separately. Common evaluation runs after training.

## Method-specific settings

Learning rates and optimization recipes deliberately differ:

| Setting | VPD | Par-auto |
|---|---|---|
| Learning rate | 5e-5 to 5e-6, cosine | 1e-3 to 1e-5, cosine |
| Optimizer | AdamW, betas .9/.999, zero weight decay | Adam, betas .9/.99 |
| p schedule | 2 to .4 over entire run | 1 to .5 over first 80% |
| Q/K/V/O components per layer | 512/512/1024/1024 | 1024/1024/1024/1024 |
| MLP up/down per layer | 3072/3584 | 3072/3584 |
| Total rank-one components | 38,912 | 43,008 |
| Importance | Published 8-block, width-2048 transformer | Scalar CE directional effect |
| Matrix reconstruction | Faithfulness loss and training delta | Exact-sum projection |

Q/K cannot use VPD's 512 components in our exact factorization of the
full-rank 768-dimensional matrices. The existing par-auto budget of 1,024
is retained there. Component counts are therefore explicitly **not identical**.
Our higher count is visible in the absolute-component KLKeep axis.

VPD uses the authors' PyTorch `vpd-paper` release at commit
`74146b555f3eb3cd2bc639c3789fd3b7a099094a`, its published configuration from
[`goodfire/spd/s-55ea3f9b`](https://wandb.ai/goodfire/spd/runs/s-55ea3f9b), and
the actual upstream component, CI, minimality, random-mask and persistent-PGD
loss functions. This is the released code, not the exact historical training
commit. The 400-step weight-only faithfulness warmup is retained. Its original
400k-update schedules are rescaled to the requested budget.

VPD's full batch exceeded one H200's memory. The harness checkpoints CI
transformer blocks and computes the unchanged upstream KL in checkpointed
2,048-position chunks. Tests verify matching KL values/gradients and unchanged
CI-block gradients. No masks, PGD iterations, CI width or loss terms were removed.
Unlike the upstream loop's terminal no-update iteration, only batches that
receive optimizer updates consume the training budget here.

## Data diversity and reproducibility

The dataset is `danbraunai/pile-uncopyrighted-tok-shuffled`, also used by VPD.
It publishes globally shuffled 513-token rows, but **no source-category labels**.
This run stratifies across all 2,021 training parquet shards, with proportional
quotas to rounding precision. Within each shard it selects a uniformly located
cyclic window without replacement; a final shared permutation mixes the shards.
Thus this is shard stratification, not a claim of explicit Pile-category balancing.

`manifest.json`, the pinned dataset revision, each shard's row origin/count,
the final permutation, and file hashes make the exact sample reproducible.
The separate validation split supplies 2,048 saved rows; final KLKeep uses the
same first 128 for both methods. Discarded 20-update smoke pilots use validation
rows and do not initialize either main run or choose hyperparameters from quality.

KLKeep fixes each native sensor's global per-position ranking on the intact
target and uses real masked forwards at a shared grid of keep fractions.
VPD's training delta has no CI score and is excluded, as in upstream CI-masked
evaluation. Its all-kept reconstruction error remains in the curve. Curves
record absolute kept-component counts as well as fractions and next-token CE.

## Operations

Code on pod: `/workspace/paired/code/{vpd,par-auto,compare}`.
W&B project: `rohan-kathuria-neu/param-autoenc`, group
`paired-h200-1b-20260910`, separate pilot/train runs for each method.
Native numerical metrics are logged through the SDK, independently of console logs.

R2 bucket/prefix: `param-clustering/parauto/vpd_matched_1b_20260910/`.
Dataset, provenance, code, checkpoints, results and logs are stored there.
Uploads record SHA256 metadata and verify object size and metadata with HEAD;
startup separately verifies a real write/read roundtrip. Credentials live
outside code/artifact paths and are not uploaded.

`supervise.py` runs both trainers, evaluates, performs the final backup, then
uses RunPod's stop API. If either trainer fails, the other receives SIGUSR1 to
save a resumable checkpoint. An optional `/workspace/paired/limits.json` containing
`{"max_hours": NUMBER}` sets a wall-time cap measured from supervisor launch.
Without it, the requested position budget is the stopping condition.

On September 10 the user changed the stopping condition to completion of the
15k KLKeep comparison. `compare_15000_and_stop.py` takes over from the original
supervisor, lets VPD continue while the comparison runs, then requests a clean
training stop, verifies the final R2 backups, and stops (does not terminate) the
pod. Its state is `output/evaluations/15000/job_status.json`.

VPD's original 15k checkpoint was pinned. Par-auto's rolling checkpoint had
already advanced, so `replay_parauto.py` recovers that milestone from the saved
2k full optimizer state, using the original data order, normalizers, and full
30,518-update schedules. Recovery must match the retained original 5k and 10k
factor snapshots within `rtol=1e-5, atol=1e-7`; bitwise equality and maximum
differences are recorded. A mismatch aborts publication of the recovered 15k
checkpoint. Its provenance is included in the evaluation JSON. Comparison
outputs use `output/evaluations/15000/` and R2 `evaluations/15000/`, with the same
128 held-out sequences and two log-x panels as the 2k comparison.

`slurm_stage_published.sh` and `slurm_eval_published.sh` run the separately
requested published-VPD versus finished-par-auto comparison on Explorer.
The published `goodfire/spd/s-55ea3f9b/model_400000.pth` file is checked against
W&B's public MD5. The finished par-auto factors are downloaded from R2 and
checked against its recorded hash and completed 1B-position training result.
The evaluator verifies embedded VPD target weights match the common target
exactly. Both use the same 65,536 held-out positions and original native
sensors. The plots explicitly label the unequal training budgets (published
VPD: 400k steps / 13.1B positions; par-auto: 30,518 updates / 1B positions).
Outputs go to `output/published_vpd_vs_finished_parauto` in the Explorer
experiment tree and R2 `evaluations/published_vs_finished/`. This job does not
alter the ongoing RunPod 15k recovery/comparison or its automatic shutdown.

Pilot measurements (20 updates; short-run estimates):

| Method | Seconds/update | Peak allocated GPU memory | 1B training estimate |
|---|---:|---:|---:|
| Par-auto | 0.447 | 41.75 GB | 3.79 hours |
| VPD | 2.203 | 138.46 GB | 18.68 hours |

Checkpoint/evaluation overhead is additional. These are throughput estimates,
not evidence that the decompositions have equal quality.
