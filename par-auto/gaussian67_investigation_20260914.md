# Investigation of the Gaussian 67M result

14 September 2026. The main diagnostic completed on Explorer H200, job
10352999, in 3 minutes 14 seconds. It uses the actual 66,931,200-parameter
model, six frozen decomposition checkpoints, and 16 held-out contexts
(384–399), distinct from the preceding report's slice. No training was run.
An initial attempt, 10352951, stopped on a PyTorch stable-argsort argument
mismatch before measurements; that was fixed before the successful run.

The original Gaussian pilot is a negative result for the intended low-L0 gain.
The evidence now points toward a failure to produce sufficiently concentrated,
composable factors, rather than numerical drift or merely too few evaluation
probes. This does not establish an impossibility result for the method.

## What the measurements establish

### More accurate ranking helps, but does not rescue the sparse regime

At exactly 512 retained primitive components per token:

| Checkpoint | Fisher K8 KL | Fisher K64 KL | Native CI KL |
|---|---:|---:|---:|
| GIM 5k | 3.464 | 3.198 | — |
| Fisher K4 5k | 3.434 | 3.136 | — |
| Gaussian 5k | 3.434 | 3.161 | — |
| Gaussian + boundaries 5k | 3.442 | 3.160 | — |
| VPD 5k | 5.016 | 4.806 | 0.939 |
| Published VPD 400k | 1.727 | 1.533 | 0.264 |

The Gaussian K64 versus K8 improvement is −0.273 KL, with a paired 95%
context-level t interval of [−0.359, −0.187]. About 81% of the retained pieces
agree between the two rankings. Ranking noise is real, but resolving most of
it does not approach native VPD's result.

At K64, Gaussian versus GIM at 512 is −0.037 KL [−0.160, +0.087]; Gaussian
versus Fisher K4 is +0.024 [−0.051, +0.100]. These small-sample comparisons do
not establish a sparse-regime training improvement. At 4,096 pieces, Gaussian
improves over GIM by −0.169 [−0.269, −0.068], while its difference from Fisher
K4 is −0.012 [−0.027, +0.003]. The larger-budget gain is credible against GIM,
but this sample does not establish superiority over the existing Fisher arm.

### Most squared score mass is not enough

The Gaussian top 512 retain 88.25% of summed squared individual Fisher scores,
but only 35.27% of their summed magnitudes. GIM is almost identical: 88.71% and
35.36%. Gaussian's score participation count is about 1,716 versus GIM's
1,669. Participation count is a concentration statistic, not literal L0.

Thus the new loss has not made the overall sensitivity distribution visibly
more concentrated at this budget. The omitted many-small-component tail still
matters. Squared individual-score mass excludes interactions, so retaining
88% is not a guarantee of small joint KL.

Errors are spread across layers: Gaussian masking only one layer gives KL
0.866, 0.816, 1.175, and 0.650, respectively. No single matrix accounts for
the full 3.161 KL. The largest individual matrix errors are early MLP outputs
and layer-2 attention V/O. Clean-input relative output errors remain substantial
across all matrix types. The clearest local gain over GIM is in `c_fc` output
reconstruction; it does not translate into a substantially smaller global
retained set.

### Local sensitivity already detects serious sparse-reconstruction damage

For two contexts, the exact FP32 full-sequence tangent predicts a quadratic
KL coefficient of 7.670 for Gaussian's 512-piece mask. At removal strength
0.01, actual KL divided by 0.01² is 7.486; at complete removal, actual KL is
2.927. The finite-difference logit derivative at strength 0.001 differs from
the JVP by about 0.46% in relative norm.

This is not a case where the local metric predicts a harmless change that
becomes disastrous only at full removal. The local metric already predicts a
large error. Nonlinearity changes the magnitude, but increasing derivative
accuracy alone is unlikely to address the main failure.

The exact calculation sums effects across all matrices and all source
positions before measuring the output Fisher norm. The older diagnostic's
field called `full` squared separately at each source position, omitting
cross-position terms. Its VPD prediction also omitted the unscored delta.
Those old predictor values should not be interpreted as full-sequence KL
predictions. Source comments now flag this; historical computations and
training objectives were left unchanged.

### VPD's native success uses a substantially different selected computation

On all 16 contexts, published VPD's native 512-piece selection has KL 0.264.
Masking individual layers gives 0.554, 0.669, 0.414, and 0.255. Restoring the
last layer to its intact computation *raises* KL to 0.387: a paired increase
of +0.123 [0.091, 0.154]. Gaussian instead improves from 3.161 to 2.359 when
the last layer is restored: −0.802 [−1.083, −0.521]. The 5k VPD checkpoint
shows an even larger reversal: 0.939 becomes 3.192 after restoration.

Published VPD ranked by the intact model's Fisher sensitivity performs much
worse than its native CI (1.533 versus 0.264). In the two-context exact-path
check, its local quadratic coefficient is 36.24, substantially larger than
Gaussian's 7.67, yet its full-removal KL is only 0.364. VPD 5k has an even
more nonlinear path. These observations are consistent with your earlier
repair result: downstream selected components compensate for, or suppress,
upstream changes. They do not by themselves establish every component's
mechanistic unfaithfulness or invalidate VPD under its own evaluation contract.

For our design, this matters: making an intact-model local attribution loss
more accurate does not directly optimize the behavior responsible for VPD's
native sparse endpoint. A claim of beating VPD while preserving the original
mechanism needs both endpoint and intermediate/repair measurements.

### Numerical error and the boundary coefficient

Our exact matrix sums remain accurate to roughly 5e−7 relative error.
All-kept FP32 KL is at numerical zero (around 1e−9); BF16 all-kept KL is
about 5e−4. These errors are far too small to explain KL near 3 at 512 pieces.
All target identities and VPD hook/native-forward checks passed.

The tested boundary coefficient was weak. In the three original derivative
checks, it contributed only about 0.1–1.8% extra metric variance. The new
boundary arm and Gaussian arm are indistinguishable in this diagnostic.
That rejects this weak setting as a useful improvement; it does not test a
boundary-dominated objective. A stronger boundary objective may also make
faithful sparsification harder by requiring more intermediate computation to
be preserved.

## Scope and next decision

These are 16-context, single-training-seed diagnostic comparisons. The exact
JVP path uses two contexts. K64 is a better estimated ranking, not an optimal
selector. Both ranking probes and the original CI may use information from
later positions. The published VPD checkpoint has a much longer training
budget than our 5k pilots.

The curves use a fixed retained count at every token. Published VPD has about
192.5 positive-CI components/token on this sample, with variable support;
extra fixed-budget slots often contain zero-CI components. Its nonmonotone
curve also depends on deterministic tie-breaking. Do not equate the fixed-512
comparison with an optimized average-L0 frontier.

I would not extend Gaussian/boundary training or perform a broad boundary
coefficient sweep on this evidence. The follow-up basis control completed on Explorer H200, job 10353153.
The learned factors already outperform simple exact bases:

| Exact decomposition | KL at 512 | KL at 4,096 |
|---|---:|---:|
| GIM 5k | 3.198 | 0.709 |
| Gaussian 5k | 3.161 | 0.541 |
| Untrained row basis | 12.074 | 4.657 |
| Untrained column basis | 16.811 | 5.409 |
| Untrained MLP-neuron basis | 7.665 | 4.955 |

The neuron control uses row components everywhere except MLP down projections,
which use column components to preserve individual MLP neurons. This rejects
these simple *untrained* alternatives; it does not test warm-started training
from them.

The same control uses an independent K64 stream to measure signed residuals.
For Gaussian at 512, the summed individual-component squared-score proxy is
1.240; adding same-source-position cross terms gives 3.817; correctly summing
source positions within each sequence before squaring gives 6.392. At 4,096,
the corresponding values are 0.154, 0.349, and 0.541. The estimates have probe
noise, and the diagonal term uses the ranking stream; nevertheless, the large
difference identifies a material joint-tail effect. The exact two-context JVP
provides a separate check of the full-sequence calculation.

Gaussian's raw held-out Fisher Lp sums improve only modestly over GIM:
1,217 versus 1,235 at p=0.5, and 652 versus 664 at p=0.641. These are raw
common-sensor totals, not the differently normalized training loss values.
The long tail is substantially better than untrained bases but barely more
concentrated than our existing trained baseline.

## A concrete next candidate, not a demonstrated fix

I would change the sparsity penalty before adding another sensitivity sensor.
One bounded candidate is a smooth activity count on the existing one-VJP
Gaussian component attribution:

    a_c = (r_c^T h) (u_c^T g),  a_c | context ~ Normal(0, s_c^2)
    loss_tau = mean_context sum_c [1 - exp(-a_c^2 / (2 tau^2))]

For positive, fixed tau and ideal Gaussian probes in exact arithmetic,
integrating out the Gaussian probe gives exactly

    E_probe[loss contribution] = 1 - (1 + s_c^2 / tau^2)^(-1/2).

This term is near zero for negligible effects and saturates at one for large
effects. It approaches the nonzero-effect indicator as tau tends to zero.
BF16 arithmetic only approximates that ideal identity.
Unlike an unbounded p-power penalty, it does not keep charging a large
component more merely because its effect grows. The proposal is an attempt to
move from shrinking effect magnitudes toward concentrating computation into
fewer active components. Tau must stay finite in practice, and this smooth
count is not actual retained-L0 at a fixed KL target.

Smooth-L0 continuation is established prior art; the original
[SL0 paper](https://arxiv.org/abs/0809.2508) and its
[convergence analysis](https://arxiv.org/abs/1001.5073) concern sparse linear
recovery. Their guarantees do not apply to this bilinear exact-sum neural
factorization. The Gaussian expectation above is a direct derivation for our
fixed-target probes, not a result claimed by those papers.

A bounded pilot would start from the existing Gaussian 5k checkpoint, compare
against matched further Lp training, and gradually lower a globally calibrated
fixed tau. It needs no selected component set or masked model forward during
training, retains exact-sum projection, and needs the same one target VJP and
two component matmuls. It still depends on the context distribution; the
Gaussian probe removes dependence on supplied next-token labels, not on data.

This does not explicitly solve the joint-tail problem. Its acceptance criteria
must include lower actual retained counts at matched KL, smaller correctly
measured joint residuals, and the layer-restoration check. Otherwise a lower
surrogate count would be another misleading improvement. A saturating loss can
also admit badly conditioned/cancelling factors, so retain the existing exact
reconstruction checks and inspect factor conditioning.

The current Gaussian implementation is still 7% slower than GIM. This proposed
loss is not yet benchmarked and does not make that overhead disappear. The
first implementation task should be a short same-shape throughput check and
probe-generation optimization; do not launch another long run unless it meets
the requested compute budget. No training for this candidate has been launched.

Raw measurements and generated tables:
[diagnostic report](out/gaussian67_diagnostic_20260914/report.md),
[full diagnostic JSON](out/gaussian67_diagnostic_20260914/diagnostic.json),
[basis and signed-tail control](out/gaussian67_diagnostic_20260914/basis_control.json).
Implementation: [diagnostic](runpod_compare/gaussian67_diagnose.py),
[basis control](runpod_compare/gaussian67_basis_control.py).
