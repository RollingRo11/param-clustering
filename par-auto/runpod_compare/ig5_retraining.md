# IG5 training experiment

Two fresh training arms use identical seed-0 initial factors, shuffled training
rows, 32,768-position batches, component counts, projected Adam, BF16 caches,
and FP32 projection. The scalar-gradient control and integrated-gradient arm
each normalize against their own initial scores on the same 256 reference
positions. Ten-update profiles are discarded. A conservative runtime estimate
chooses one common step count (at most 2,000) within the one-hour H100 allocation.
Both cosine learning-rate schedules run 1e-3 to 1e-5 over that count; both p
schedules run 1 to 0.5 over its first 80%. These are short, complete schedules,
not intermediate checkpoints from the previous 30,518-step experiment.

The IG sensor uses five Gauss-Legendre quadrature points on the shared path
where all 24 decomposed matrix outputs are scaled from zero to their intact
values. Embeddings, norms, biases, residual paths, and the language-model head
remain fixed. This is component-gate/weight-space integrated gradients, not
input interpolation, GIM's modified backward, or individual-component ablation
with every other component held present. At every point, the scalar objective
is summed ground-truth next-token cross entropy, including downstream token
dependencies exactly as in the existing scalar sensor.

For every component and token, the signed products `(h(alpha) @ read) *
(g(alpha) @ write)` are integrated before taking absolute values and applying
the existing smoothed Lp concentration loss. Five-point caches stay detached.
Only the score matrix multiplications are recomputed during backward with
ordinary PyTorch activation checkpointing; this preserves gradients while
avoiding five sets of saved score intermediates. The model passes are not
recomputed by that checkpointing. All 24 matrix losses still share one backward
and one optimizer step. No rejection, learned importance model, masks in the
training objective, or reconstruction penalty is introduced.

Tests compare signed scores with gradients of explicit independent component
gates, check numerical completeness on a toy, verify target/hook restoration,
and compare checkpointed versus ordinary BF16 score gradients.

Evaluation measures all four combinations of training arm and ranking sensor
on the same 65,536 held-out positions. This separates the effect of retraining
from changing the ranking alone. Actual masked forwards and keep fractions
match previous KLKeep evaluations. The signed integrated attribution sum is
compared with the actual endpoint CE difference; a large discrepancy indicates
that five quadrature points are not enough for accurate integration on this path.

GPU job: `slurm_ig_training.sh`, partition `sharing`, one H100, one hour.
Data staging and final R2 backups run on separate CPU allocations. Checkpoints
are saved every 250 updates, at completion, or after a handled stop signal.
Outputs: `output/ig5_retraining/` under the existing Explorer experiment tree.
W&B group: `ig5-retraining-20260910`; R2 prefix:
`parauto/vpd_matched_1b_20260910/experiments/ig5_retraining/`.
