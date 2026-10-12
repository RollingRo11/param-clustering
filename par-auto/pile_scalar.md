# Whole-model scalar training

`train_pile_scalar.py` trains all 24 matrices of the verified 67M Pile target
with the same projected Adam objective as the toy. Each layer uses Q/K/V/O:
1024/1024/1024/1024 components and MLP up/down: 3072/3584, totaling 43,008.
Only Q/K differ from VPD's budget: its 512 factors require a residual delta;
our factors alone must sum to the full-rank 768-dimensional matrix.

The batch is 32,768 positions (64 contexts of length 512). Four contexts at a
time pass through the frozen FP32 model and one backward of their summed
ground-truth next-token cross entropy. Module inputs and local cotangents
are rounded directly to BF16. Scores use BF16 autocast; loss reductions,
normalization, parameters, gradients, Adam state, and both training projections
are FP32. BF16 uses no GradScaler. Historical FP16 remains selectable in the
shared method for reproducing earlier experiments.

Every cache batch is consumed once, aligned across all matrices, then freed.
The first 100M corpus tokens are traversed in shuffled, nonoverlapping
513-token blocks, with a new deterministic shuffle on each epoch. The local file contains 200M tokens despite its `pile_1b` name; the final
100M tokens stay outside training. Reported positions count exposures; epochs
beyond the available corpus repeat data. A dispersed fixed 256-position
training-only reference from batch zero normalizes each matrix's loss.

The profile runs 1,000 updates through this exact stream/update path. Its timing
includes fresh cache generation, ordinary logging, and periodic checkpoint
writes; it separately records cache and fit time. The eight-hour job reads the
successful profile and chooses the largest multiple of 100 steps within
`(8 hours - 15 minutes)/(measured seconds per step * 1.1)`. This reserves setup
time and a 10% slowdown allowance. LR runs from 1e-3 to 1e-5 over that full
schedule; p runs from 1 to 0.5 over its first 80%. The long run starts fresh;
profile updates do not count toward its optimizer or schedules.

Jobs: `slurm_pile_scalar_profile.sh` (one H200, 45–60 minute allocation) followed by
`slurm_pile_scalar_train.sh` (one H200, eight-hour maximum), using `afterok`.
Both require `PARAUTO_SOURCE` (a checksum-verified source snapshot) and
`PARAUTO_RUN` (experiment output root). The first job runs the test suite.
W&B project: `param-autoenc`; profile and training have separate runs.

Atomic full-state checkpoints are written every 1,000 updates. Factor-only
snapshots are retained at 1k, 5k, and every 10k; final factors are `pieces.pt`.
The program stops and checkpoints five minutes before its wall-time budget,
or on SIGUSR1/SIGTERM. A CLI `--resume` restores Adam, fixed normalizers and
the exact schedule; stream offsets are reproducible from the saved step.
The time-budget stop does not silently shorten an existing resumed schedule.

VPD reference: public run `goodfire/spd/s-55ea3f9b` has only
`model_400000.pth` among its 614 listed files as of this check. Its two output
artifacts contain W&B history/events, not intermediate model weights. The
published config sets `save_freq: null`.
