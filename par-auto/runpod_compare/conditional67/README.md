# Three conditional reconstruction training runs: 67M Pile

Experiment `conditional67_20260916_v3`. Version 1 failed during dependency
import before loading the target or performing any training. Version 2 uses
the existing VPD Python 3.13 environment and explicitly checks its imports
before validation. Version 2 stopped at the gate finite-difference check. In
version 3 that check measures the log-probability change with FP64 reductions,
uses 64 measurable gate coordinates per matrix type, and requires agreement
at both epsilon .01 and .003 under the unchanged 4% tolerance. Numerical
training objectives and schedules are unchanged. Earlier attempts performed
zero factor-training updates.

These are three independently trained
decompositions, not rerankings of an existing checkpoint. The user explicitly
authorized training with the selected-computation objectives proposed in the
review. The earlier mask-free restriction does not describe these trials.

All arms use the verified 66,931,200-parameter target, all 24 internal matrices,
43,008 rank-one components, identical fresh seed-0 initialization, identical
ordered training batches, and exactly 5,000 optimizer updates. Each update
accumulates all 64 contexts of 512 prediction positions in microbatches of four:
163,840,000 training positions per arm. No gold labels or VPD CI are used for
training or support selection. Teacher distributions come from the frozen
original model on the same inputs.

The factors receive gradients from actual whole-model sparse KL. Ordinary
Adam (.9/.99) uses a complete 5k cosine schedule, 1e-3 to 1e-5, gradient clipping
at norm 1, FP32 parameters/states/projection and BF16 forwards. Each update
projects the sum of the pieces back to each original weight matrix. The
selector is discrete and detached from factor training. A count price affects
support selection; its value is constant during the subsequent factor update.
No Hessian-of-the-selector or second-order factor backward is retained.

## Shared selection and budget

A single ordinary sampled-model-label backward on the intact teacher seeds a
global top-k support per position. All arms use the same probe seeds. The cap
anneals exponentially from 8,192 to 256 over updates 0-3,999, then cycles
128/192/256/512 for the remaining 1,000 updates. This is an update-and-data-matched
comparison among the new arms, not an unchanged copy of the old 8,700-step
Lp schedule. Native count reductions can leave a support below its cap.

The per-sequence selection objective is mean-token target-to-sparse KL plus
0.001 nats times mean retained components per token. Each microbatch does one
selection round. Each position proposes a drop or a swap; the curvature arm
also proposes a two-out/one-in replacement. At most 128 positions per sequence
change, with retries at 32, 8 and 1. Each sequence accepts only a measured
finite objective improvement. The initial support remains available. The
objective may improve by paying a small KL cost for a count reduction; raw KL
and count are logged separately.

## Arm definitions

* `error_gradient`: True reconstruction-error gate derivatives at the actual
  sparse endpoint. A deletion has predicted cost -g_c, and a swap -g_c+g_d.
* `replacement_cost`: The same endpoint gradients, plus four independent
  Gaussian output probes with covariance equal to the sparse distribution's
  categorical Fisher. Shortlist two present and two absent pieces per token.
  Candidate cost includes one half the mean squared SIGNED sum of their probe
  effects, preserving correlations. Probes are normalized for sequence-mean
  KL. This is a Gauss-Newton proposal subject to finite acceptance, not an exact
  Hessian or an Optimal Brain Surgeon weight-refit algorithm. Candidate batches
  spanning positions are verified by real KL, rather than assumed additive.
* `restoration_path`: One uniform alpha per microbatch on
  m(alpha)=m+alpha*(1-m), with actual refreshed activations and true KL gate
  derivatives. Absent candidates are shortlisted by signed restoration
  credit; present deletions use endpoint gradients. This is a single-sample
  estimator of signed integrated restoration credit, not of its absolute
  value. Mixed endpoint/path scores only rank proposals; finite KL decides.

The first arm uses the whole selected model rather than a sampled short span.
On this four-block target this avoids an additional approximation and directly
uses the successful endpoint-search signal. All arms use full sequence
attention and actual nonlinearities. No model output is replaced by a cached
intact input during sparse execution.

## Cost and safeguards

Linear forward methods are temporarily replaced, not overridden after executing
unused dense linears. Candidate gating still uses dense factor GEMMs; no claim
is made that work scales with selected support size. All costs, including
teacher capture, multiple proposal forwards, factor backward and projection,
are included in the full-batch preparation timings. Extra probes in the
curvature arm are charged explicitly. Four is an experimental sketch size,
not a promise of accurate arbitrary subset search.

CPU tests check gate-gradient finite differences, batch normalization,
Gaussian covariance, exact joint replacement accounting, complementary-path
credit, finite acceptance, factor updates, restoration of model methods, and
exact checkpoint/resume for all arms. Each arm's prerequisite H200 preparation
also checks FP32 all-on parity, finite differences for all six matrix types,
bitwise serialized optimizer resume, projection residuals, full 64-context
updates at two support budgets, peak GPU memory, and projected runtime. These
pilot updates are discarded. Training cannot start without matching passed
preparation and source hashes.

## Checkpoints, evaluations and storage

The snapshot contains all local dependencies and hashes the upstream Python
implementation too. Plans identify the exact target, data, source and config.
Every 250 updates and before a planned cutoff, save a complete resumable
optimizer state and factor snapshot. SLURM continuations chain two-hour H200
allocations, exiting after 5k updates; at most 32 allocations per arm. Failure
does not silently continue. Only these jobs submit their own continuations.

W&B project `rohan-kathuria-neu/param-autoenc`, group equal to the experiment ID.
Logs contain KL, count, acceptance, gradient norms, learning rate, support cap,
and complete update time. Resume preserves the W&B run ID.

On completion queue a 128-context evaluation on validation rows 1024-1151.
Report common Fisher K64 ranking and each arm's native K1-seeded conditional
selector at caps 64/128/192/256/512/1024/4096. Save each context's actual KL,
cross entropy and measured count. Native selection is per-context inference
using teacher outputs, not a learned selector. Its compute is part of the
method. These curves alone do not establish superiority to VPD, which needs a
matched VPD evaluation and seed replication. Results, final model and full
optimizer state are uploaded to the existing R2 experiment prefix with
SHA256 and size verification.

This experiment optimizes primitive component count. It does not claim an
entropy-coded description-length result or stronger mechanistic fidelity
solely from a lower endpoint KL.
