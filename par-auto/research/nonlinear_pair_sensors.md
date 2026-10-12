# Finite nonlinear sensors and Q/K pair objectives

Date: 2026-09-15. This is a derivation and proposed screen, not a report of
training these objectives. Arithmetic checks are in
[nonlinear_pair_sensors_audit.py](nonlinear_pair_sensors_audit.py) and its JSON.

## What was and was not tested

The previous 24 intact-trace objectives did not implement either complete
proposal. The finite RMS sensor measured the norm of the immediate change,
not its projection into a downstream output-sensitive sketch. Its GELU
branch was linearized and its Q/K branches used write magnitude. The prior
IFR experiment was a bounded reader-conditioned selection diagnostic, not
this pairwise attention-gradient training objective.

The new proposals are therefore not ruled out by the negative 36-objective
screen. Those 36 were six sensors under four penalties and twelve subset
diagnostics, not 36 independently developed mechanisms.

## New empirical evidence: multi-block result

Explorer job 10363874 completed successfully. Checksummed results are in
[the report](../out/multiblock67_20260915_v2/report.md).
On eight fresh contexts, normalized reconstruction error was:

| Span length | VPD | GIM | VPD relative to GIM |
|---|---:|---:|---:|
| One | 0.49470 | 0.46794 | 5.7 percent higher |
| Two | 0.64822 | 0.77325 | 16.2 percent lower |
| Three | 0.63746 | 0.91253 | 30.1 percent lower |
| Four | 0.59238 | 0.97510 | 39.3 percent lower |

Each length averages all contiguous spans of that length. Lengths two through
four favor VPD on all eight contexts and favor native VPD selection over its
Fisher selection on all eight. Two- and three-block losses also increase at
all 24 adjacent transitions along the previously measured damaging GIM
trajectory. Four-block loss increases at 23 of 24 transitions.

This is a positive acceptance result, updating the previous statement that no
tested objective qualified. It is support-conditioned, uses VPD's native
support versus GIM's Fisher support at matched total counts per position, and
does not supply a selector. It is not a mask-free training win. Also, the first
two-block location alone still rejects VPD; the favorable two-block result is
the average over all three locations.

Strict same-run checks against full model interventions passed. The regenerated
CI counts differ slightly from the old run; maximum old/new KL difference is
0.04561. Repeated CI within the new run is stable. The cross-run numerical
cause is unresolved; the comparison does not rely on bitwise old-run identity.

Single-context capture plus backward timing was about 1.36 times GIM for all
sliding two- or three-block spans, and 0.99 times for the four-block span. The
latter executes four blocks once, while sliding spans execute six blocks.
These timings omit selection, optimizer, and projection and do not establish
production training cost. No factors were updated.

## Proposal 1: exact immediate nonlinearity, cached downstream derivative

Let x be a matrix output, z_c its component contribution, N the immediate
nonlinearity, and F the rest of the computation. Define, using ASCII notation:

    delta_c = N(x - z_c) - N(x)
    score_c = dot(b, delta_c)
    b = gradient of the chosen downstream scalar with respect to N(x)

The intended loss can penalize the magnitude or squared sketch norm of this
score. It differs from both dot(gradient_at_x, -z_c) and norm(delta_c).
The latter ignores whether the changed directions matter to the output.

AtP's Q/K correction specifically recomputes attention probabilities and uses
gradients afterwards. Its appendix also discusses an MLP saturation correction
and the increased cost for dense, rather than neuron-aligned, interventions.
It does not prove that nearly all linearization error universally occurs at
the first nonlinearity. See [AtP, Sections 3.1 and Appendix C](https://arxiv.org/html/2403.00745v1).

### What can be established

If the downstream scalar F is affine in the corrected intermediate state,
score_c is the exact finite change. Otherwise, if its Hessian operator norm
is bounded by H along the segment from N(x) to N(x-z_c), the absolute error
is at most H times squared norm(delta_c), divided by two. This follows by
Taylor's theorem applied after N, so the curvature of N itself has been
removed from this error term. There is no general guarantee this new bound
is smaller: the finite delta can be large and later curvature can dominate.

Counterexample: take an exact first GELU at x=0 and a downstream square.
Removing a contribution of -1 changes the first output to GELU(1). The cached
downstream derivative is zero, but the true scalar change is about 0.708.

For KL, the gradient of KL against the identical clean model is zero. One
must specify a nonzero output probe, a downstream Jacobian sketch, or an
explicit output reconstruction surrogate. A true next-token CE gradient is
one probe, not a complete KL sensor. GIM's altered backward is a separate
surrogate and does not satisfy the true Taylor remainder statement.

A sampled-model-label CE gradient can estimate a Fisher quadratic when its
directional effects are squared and averaged over labels. That remains a
local downstream approximation. Taking fractional powers of a noisy estimate
does not preserve unbiasedness. Extra probes must be charged to the budget.

### Exact attention correction without rerunning the transformer

For one query-component intervention, recompute the affected attention row
using the intact keys, causal mask, head scaling, and positional rotations.
Then project the finite probability change onto a cached gradient at the
attention probabilities, or multiply by intact values and use a gradient at
the attention output. Sum signed contributions across the heads affected by
the component before applying the chosen magnitude penalty.

For one key component at one source position, only one logit changes in each
affected query row and head. Let its old probability be p_j, its logit change
be d, and b be a cotangent on that probability row. The exact local projection is:

    eta = p_j * expm1(d) / (1 + p_j * expm1(d))
    projected_change = eta * (b_j - dot(p, b))

Proof: the new partition function divided by the old is
1 + p_j * expm1(d). Substitute into every new probability and collect the
coefficient of b_j minus its old probability-weighted mean. This accounts for
all competing keys, not just the altered entry's probability. Equivalently,
the attention value output changes by eta times (v_j - old_output).

An implementation must use log-space rescaling for extreme d and avoid
singular rearrangements at probabilities rounded to one. For a fixed Q/K
pair at one query/key position, the same formula applies to its single-logit
removal, with a separate correction per head.

### Residual paths must remain in the estimate

O and MLP-down outputs enter a residual stream. That stream feeds both a norm
and an identity skip, so using only the finite norm change drops a real path.
Let g_before be the total cotangent at residual r, g_norm the cotangent at
N(r), and J_N the norm Jacobian used by that backward pass. The corrected
estimate for removing z is:

    -dot(g_before, z)
    + dot(g_norm, N(r-z) - N(r) + J_N z)

The second term replaces the linearized norm branch with its finite change;
the first preserves other paths. In a graph with multiple direct consumers,
correct each intended nonlinear branch, without counting a branch twice.
With true gradients use the true J_N; with frozen-norm GIM use its frozen
Jacobian for the replacement identity and label the result a surrogate.

For RMSNorm, the scalar projection can be computed using dot products with
r and z plus the old/new inverse RMS. It need not materialize a tensor indexed
by token, component, and residual coordinate. This is particularly attractive
as a cheap replacement for the old unweighted finite-norm magnitude.

### Budget implications

An exact finite GELU correction for a dense rank-one write is elementwise but
not a single scalar operation: it touches every output coordinate for every
component-position intervention. With 512 positions, 3072 components and 3072
MLP coordinates, that is 4.83 billion elementwise evaluations per layer before
backward, or 19.3 billion across four layers. Chunking reduces memory, not work.

For Q or K, using 1024 components and six heads gives about 807 million
component/head/causal-position-pair entries per layer. That excludes the work
to form logit changes. Algebraic key normalization avoids an additional loop
over all alternative keys for each intervention.

Thus the closed-form RMS correction should be benchmarked first; exact Q/K
softmax next; dense GELU needs a careful chunked kernel or sampling. Sampling
whole component-position losses with inverse-probability weighting gives an
unbiased estimator of their sum, assuming fixed sampling probabilities with
the appropriate gradient treatment. Sampling coordinates inside a nonlinear
penalty is generally biased. Variance and held-out score precision must be
measured, not inferred from unbiasedness.

## Proposal 2: Q/K interaction units

For a query component c at destination t and key component d at source s,
within head h, define the exact contribution to the attention logit:

    E[c,d,t,s,h] = a_query[c,t] * a_key[d,s]
                  * dot(rotated_query_write[c,t,h], rotated_key_write[d,s,h])
                  / sqrt(head_dimension)

Use only causal s. The sum over c and d equals the model's attention logit
when the components sum to the original Q/K matrices. Residual matrix
components must be included if working directly with an approximate VPD sum.
The copula example and positional interaction decomposition are in
[VPD Sections 4.3 and 4.6](https://www.goodfire.com/research/interpreting-lm-parameters).

The proposed linear edge attribution is E times the gradient at that logit.
It is an exact directional derivative, not an exact finite removal effect.
The presence of a matching key is encoded in E, but its downstream importance
also depends on softmax competition, values, and the later network.

### A useful identity: signed aggregation is already ordinary attribution

For fixed c and t, sum this signed edge attribution over d, s, and h.
The sum over key components reconstructs the intact key vector. By the chain
rule, the remaining sum is exactly the query component's activation times
its write-vector dot product with the ordinary query-output gradient.

Therefore ordinary Q attribution already depends on matching keys. The
example establishes that activation alone is inadequate, not that a learned
CI is mathematically unavoidable. Learned CI can use context directly, but a
formula using the actual Q/K computation can also do so.

This identity holds per true output probe, before taking magnitudes or norms.
With GIM it holds for the corresponding modified chain rule, accounting for
its Q/K scaling. Repackaging signed terms and summing them cannot add signal.

A new objective must instead retain edges as separate units, group them in
a new way, or apply a finite softmax correction. Taking absolute values per
edge is different, but is not automatically better: it destroys cancellation
between a positive and negative interaction and may demand more active edges.

### Edge sparsity is a different description length

One query component can interact with many key components and source positions.
Arbitrary selected edge masks generally cannot be represented by independent
selected Q and K component masks. The latter impose a product structure on
retained edges. A sparse arbitrary QK edge set is therefore not yet a sparse
subset of the original rank-one components.

For a comparable result, charge the retained component-position vertices and
execute their induced attention interactions, or explicitly declare a changed
representation and charge edge identities and any shared vertices. Edge counts
cannot be compared directly with VPD's roughly 200 component positions/token.

Simultaneous Q and K deletion needs an interaction correction. The logit
change is the negative removed-query/intact-key term, minus the
intact-query/removed-key term, plus the removed-query/removed-key term. Adding
individual deletion effects alone double-counts that intersection.

Enumerating all pairs is not cheap merely because each term is closed form:
with 1024 Q and K components, 512 tokens and six heads, there are about
826 billion causal edge entries per layer. Dense storage would require about
3.3 terabytes in FP32. Sampling or structural contraction is essential. The
obvious signed contraction is cheap precisely because it recovers the old
gradient score; a nonlinear edge penalty prevents that simple contraction.

## Concrete next screen

Prioritize exact local nonlinearity followed by a downstream projection.
Use corrected residual RMS, exact Q/K softmax, and exact GELU as separately
reported changes, so expensive or unhelpful branches cannot hide behind an
aggregate result. Hold the penalty fixed initially to isolate the new signal.
Compare true-gradient and GIM downstream caches as separate variants.

Use paired contexts and a common module scaling calibrated on training data.
Require lower loss for published VPD than GIM on fresh contexts, and rising
loss along the known GIM damage trajectory. Time the complete capture, score,
and factor backward, including sampling overhead. Preserve signed per-probe
effects before the final magnitude so real cancellation is visible.

For the pair proposal, first audit the signed-aggregation identity on the 67M
model and compare linear versus finite-softmax edge effects on a bounded sample.
Only pursue an edge objective if it changes ordering, and define the original
component-count objective before claiming improved sparsity. The multi-block
loss now supplies an empirical reference that already has the desired ordering.

Neither proposal guarantees sparse joint sufficiency from singleton scores.
Accurate singleton effects can still miss joint redundancy and compensation.
The goal of this next screen is to test genuinely new nonlinear and interaction
information, not to assume that improved attribution accuracy implies a better
decomposition training objective.

## Verification performed

The standard-library arithmetic audit passed 1000 randomized small cases for
the pair/gradient identity, its finite-difference derivative, the exact key
softmax correction, simultaneous Q/K deletion, and residual norm correction.
Largest finite-difference discrepancy was below 2e-9; the other discrepancies
were below 6e-14. Saturation and downstream-curvature counterexamples also
passed. These support the algebra; they are not experiments on the 67M model.

## Completed first-nonlinearity follow-up

Explorer job 10364399 tested the immediate correction with true CE and GIM-rule
CE cotangents, fixed exponent 0.641, and all component positions on eight fresh
contexts. Neither variant favors VPD: published VPD/GIM loss ratios are 1.75168
and 1.75154 respectively, with zero favorable contexts. Every matrix-type
aggregate still favors GIM. Local intervention validation passed. These are
scalar downstream probes, not a multi-direction output sketch; V stays linear.
See [the completed analysis](../out/first_nonlinearity67_20260915/analysis.md).
