# GIM plus activity count: 67M, 5,000 steps

The objective is the existing normalized Lp loss plus an additive soft count:

    L = sum_m Lp_m / fixed_Lp_reference_m(p)
        + lambda * 24 / N_ref * mean_token sum_m,c s_mc / (s_mc + tau)

Here s is the absolute GIM component attribution, lambda=1, tau=0.001, and
N_ref is the total soft count on the same fixed 256 reference positions and
seed-0 initial factors used by the normal GIM run. The global normalization
makes the auxiliary and original losses both initially 24 on that reference;
it does not give different count prices to different matrices. Tau is fixed
and deliberately matches the existing diagnostic attribution-count threshold.
No validation results were used to select lambda or tau.

This is a fresh seed-0 run, not continuation from the Gaussian experiment.
All 24 matrices, 43,008 rank-one pieces, projected Adam settings, batch order,
and precision match the normal GIM 5k checkpoint. It runs 5,000 updates of the
original 8,700-step horizon, so the final p is approximately 0.641, rather than
finishing an independently shortened schedule. Training processes 163,840,000
positions. No Gaussian probes, component gates, training masks, or extra target
backwards are used.

`parauto_method.py` adds opt-in `count_coeff` (default 0) and `count_tau`.
The default objective remains unchanged. The pilot excludes combinations with
variable rank, ownership, or the old joint objective. Checkpoints contain the
new settings and the original reference scores, making normalization
reproducible across resume.

## Explorer jobs

- Prepare: `slurm_gim_count67_prepare.sh`, job **10353743**.
- Train/evaluate/archive: `slurm_gim_count67_train.sh`, job **10353755**, submitted afterok of prepare.
- Both use an H200, 4 CPUs, and 64 GB host RAM.
- Immutable source: `data/parauto/paired_1b_2000_20260910/code/gim_count67_20260914`.
- Output: `data/parauto/paired_1b_2000_20260910/output/gim_count67_20260914`.

Preparation checks zero-coefficient bitwise parity with the preserved old
implementation, count gradients, positive-budget splitting, and uninterrupted
versus resumed updates on the actual 67M model. It then profiles two reversed
orders of ordinary GIM and GIM+count, 30 updates each, discarding five warmup
updates. Profile factors are discarded before training.

The training job saves full state every 1,000 steps and on a time/signal stop.
One automatic continuation is allowed if the 2-hour allocation proves
insufficient. Only completion of all 5,000 updates starts evaluation.

## Decision

Both the new checkpoint and the existing normal GIM 5k checkpoint are evaluated
on the same 128 validation contexts, 256–383, using exactly matched rank-one
retained counts. Primary ranking is Fisher K64 with shared seeds; native GIM
ranking is a secondary comparison. Each curve saves per-context KL, allowing
paired uncertainty estimates. The report includes threshold crossings at KL
0.1, 0.3, and 1.0 without interpolation or a monotonicity assumption.

The criterion is lower actual retained counts at matched KL. A lower soft
count alone is insufficient. This one fixed setting and training seed cannot
establish that all count penalties succeed or fail. Evaluation uses component
masks to measure fidelity; training does not.

Numerical results and checkpoints are uploaded to R2 with size/SHA metadata
checks. `complete.json` is written only after training, all four evaluations,
comparison, and archival have completed.

The initial training allocation was shortened with `scontrol` from two hours
to one hour to improve backfill opportunities after profiling ~37 minutes of
training. The existing signal/checkpoint handling remains active. The launcher
retains its two-hour limit for an automatic continuation if needed.

Validation passed on Explorer. Mean update times: GIM 0.4018 s; GIM+count
0.4475 s (1.114x). Peak allocated GPU memory: 41.75 GB versus 53.49 GB.
At 21:34 Central on September 14, Explorer estimated a 03:55 Central September
15 start for job 10353755. This estimate can change.

## Short allocations for earlier backfill

The pending one-hour job 10353755 was cancelled and replaced by job
**10354126**, using `slurm_gim_count67_chunks.sh`: one H200, 4 CPUs,
64 GB host RAM, and 15-minute allocations. Each chunk requests a clean stop
11 minutes after launch, leaving checkpoint/exit margin; it resumes the same
optimizer, reference normalization, data position, and LR/p schedule next time.
At most eight training allocations are allowed, totaling exactly 5,000 updates
when successful. The final training chunk submits a separate 20-minute
evaluation/archive job using the original pipeline, which skips training once
the saved checkpoint has reached 5,000 steps.

The actual queued-job forecasts did not reproduce `sbatch --test-only`'s
optimistic partition estimates. Node pinning was removed. Partition-only
changes and reduced RAM were not established to help. Short chunks improve
backfill flexibility; they do not guarantee a start time. The original model,
loss, batch size, precision, and schedule are unchanged.

The short-allocation change worked: job 10354126 started on H200 node d4053
around 21:49 Central on September 14, instead of waiting for the overnight
one-hour allocation. W&B run: https://wandb.ai/rohan-kathuria-neu/param-autoenc/runs/befxxuoc.

## Resume repair

Chunk 0 (10354126) completed 1,382 steps and submitted 10354297. That job
started on an H200, then failed before training because the JSON plan stores
`joint_fractions` as a list while the in-memory dataclass uses a tuple. The
optimizer resume test passed, but did not cover this outer on-disk plan check.

The local harness now canonicalizes both sides through JSON for comparison.
For the existing run, the original source snapshot and saved plan remain
unchanged. A checksum-verified `gim_count67_resume.py` wrapper applies that
single comparison fix to the original runner and logs the exact executed
source hash and wrapper source in `resume_execution_<job>.json`. The loss,
optimizer, checkpoint state, and schedule are unchanged. Both training and
evaluation launchers use the repaired entry point.

Replacement chunk 1: **10354387**, resuming the saved step-1,382 checkpoint.
Regression checks accept JSON tuple/list round-trips and still reject actual
loss-configuration changes.

## Early KLKeep evaluation

Job **10354564** evaluates count checkpoints at 1,000 and 1,382 steps, and
normal GIM at 1,000, 2,000, and 5,000 steps. Every curve uses Fisher K64 with
shared seeds on validation contexts 256–319 (32,768 positions). Only the
1,000-versus-1,000 contrast is step matched. The early report includes paired
context-level uncertainty for that contrast and explicitly labels the other
training budgets. Output: `output/gim_count67_20260914/early_1382/report.md`.
The early evaluation is independent of the queued training continuation.
