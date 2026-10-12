# Objective screening against a known sparse VPD solution

Frozen 66,931,200-parameter target, Explorer H200. This experiment searches for
objectives that accept the converted VPD witness and oppose known destructive
GIM updates. It is not a new decomposition training run and cannot establish a
lower L0 floor by itself.

## Controls and data

All evaluated factors are in the same exact-sum, 43,008-component family:
published VPD converted by the previously validated bridge; converted VPD5k;
GIM5k; GIM7783; original CE-attribution 1B-token par-auto; and the converted
VPD witness after 20, 100, and 200 previously completed GIM updates. The latter
provide a controlled damage trajectory whose training objective previously
improved while inherited-support KL deteriorated.

Two training contexts, rows 10000–10001, provide fixed common normalizers and
factor gradients. Validation rows 448–451 are discovery; 452–455 are confirmation.
The objective grid is fixed before either split is read. Summary tables are
ordered using discovery ratios only; confirmation remains separate.

For each validation context, the published VPD's positive CI fixes the number
of retained components at **each source position**. Every compared support
has those same counts. Supports include Fisher K64, write magnitude, inherited
native CI where component identities permit it, and random components with
the native count at every matrix and source position. VPD5k's own CI is ranked
at the published VPD's count. Random selections use fixed shared seeds.
Full selected-model KL, without free Delta, is an evaluation metric only.

## Twenty-four objectives using only intact traces

Six sensors, each with smoothed powers p=0.25, 0.5, 0.641 and a soft-count
variant, give 24 objectives:

- **GIM:** current temperature, Q/K/V credit scaling and frozen RMS backward.
- **GIM with true RMS derivative:** the same GIM attention modifications,
  with normalization's actual derivative restored.
- **True gradient:** ordinary next-token CE derivatives.
- **Write norm:** norm of each rank-one contribution at its matrix output.
- **Consumer tangent:** actual RMS Jacobian applied to O/down writes, actual
  GELU Jacobian applied to FC writes, and write norm for Q/K/V.
- **Consumer finite RMS:** exact single-component finite RMS change for
  O/down writes; GELU tangent for FC and write norm for Q/K/V.

For the last two sensors, the next RMS operation is evaluated at the intact
residual stream, so the unchanged residual branch is included. The finite RMS
norm is computed algebraically using two T-by-C projections rather than
materializing T-by-C-by-D vectors. It is a local effect, not an endpoint certificate.
The final down-projection uses the final RMS normalization.

Each objective divides each matrix's mean penalty by a fixed GIM5k reference
measured on the two training contexts, then averages across matrices. Normalizers
are identical across all factor states and remain fixed during descent. The
soft-count threshold is 1% of the corresponding GIM5k mean sensor score per
matrix, also fixed. It is a relative-count screen, not the earlier raw tau=.001 run.
Factor operations use FP32; gradient sensor caches use the established BF16
intact-model capture, while forward-only geometry uses FP32 traces.

## Twelve subset-conditioned diagnostics

These receive an explicit, fixed support and are **not strictly mask-free**:

- Joint matrix-output relative MSE, relative L1, cosine error and per-channel
  RMS-standardized squared error.
- Joint error after the immediate consumer nonlinearity: finite residual RMS
  for O/down, finite GELU for FC, raw relative MSE for Q/K/V.
- Selected attention and MLP sublayers, each started from its intact input:
  residual-stream MSE and error after the following RMS.
- A full selected block started from its intact block input, including changed
  attention and changed MLP input: MSE, cosine error, and following-RMS error.
- Full-block RMS error plus 0.1 or 0.5 times the joint consumer error.

The block objectives include within-block interactions but reset to intact inputs
between blocks. They therefore cannot capture all cross-block compensation.
No updates use these diagnostics. Their purpose is to test whether more joint
information recognizes the witness when scalar importance does not.

## One-step descent probes

All 24 intact-trace objectives receive the same challenge. Start independently
from the converted published VPD factors, average a gradient over the two
training contexts, and try normalized raw-gradient steps at relative factor
displacements 0.0001, 0.001, and 0.003. Apply the existing FP64 conditional
exact-sum projection to each proposal and verify reconstruction. Record actual
postprojection displacement, training objective, and inherited-native-support
KL on the four confirmation contexts.

Choose the step with lowest training objective among genuinely decreasing
proposals. Confirmation KL never selects the step. If no proposal decreases
the objective, record that failure. Each objective starts from the same initial
factors; probes do not run sequentially on one evolving dictionary. This is a
directional diagnostic, not a comparison of converged Adam training runs.

## Interpreting acceptance

For each objective report VPD/GIM loss ratios separately on discovery and
confirmation, paired preference fractions, and whether loss rises along the
known GIM damage trajectory. For support-conditioned diagnostics also check
whether native VPD beats Fisher and random supports **inside the same factors**.
For intact-trace objectives, that within-factor support test is inapplicable:
they do not observe the support.

A useful candidate should pass more than a cross-model loss ratio. It should
recognize deterioration along the fixed-support trajectory, or preserve the
known solution when optimized. Neither outcome supplies a cheap selector.
Avoid attributing success to a new mathematical sparsity guarantee.

Record cache, score, factor forward/backward and projection timing. Conditional
diagnostics share intermediates, so suite timing is not the standalone cost of
each diagnostic. Production-batch whole-step timing is required before claiming
GIM-budget training. No Gaussian model or extra gradient-probe training is used.

## Validation and provenance

Check the finite RMS formula against explicit vector interventions on real model
activations. Check that all-kept local, sublayer and block objectives are near
zero, that all-removed blocks incur error, and that all candidate supports have
exactly matched counts. Model, checkpoint, data and source hashes are recorded.
Completed results and immutable source snapshots are uploaded to R2 with verified
byte counts and hashes. Existing exhaustive source-position jobs run independently.
