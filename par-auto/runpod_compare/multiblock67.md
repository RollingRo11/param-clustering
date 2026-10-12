# Contiguous multi-block diagnostic

Frozen 67M target on Explorer. Follow-up to the objective screen: single-block
normalized reconstruction rejected the native VPD sparse witness despite its
much lower output KL. Test whether propagation across blocks resolves that
misordering. No optimizer steps or factor updates.

## Predeclared comparison

- All ten contiguous spans in the four-block target; average separately by
  lengths one, two, three, and four. Also report every span location.
- Start each span at its intact input, propagate the selected computation
  through the entire span, then compare against the intact boundary residual.
- Primary score: relative squared error after the next RMSNorm (final RMSNorm
  for spans ending at the last block). Controls: raw residual error and cosine.
- Published VPD converted to the exact-sum GIM factor space, GIM at 5k updates,
  and the existing 20/100/200-update GIM damage trajectory from the VPD witness.
- Published native CI fixes the total retained count at every source position.
  Compare inherited native support where applicable, Fisher K64, and random
  support matched to the native matrix allocation. Seeds match the old screen.
- Reproduction: validation contexts 448–455. Fresh confirmation: 456–463.
  Report all lengths, without choosing a length on the confirmation results.
- Assess native VPD versus GIM Fisher, native versus its own Fisher/random,
  and whether loss rises along the known damaging training trajectory.

## Validation and cost

All-kept spans must reproduce intact residuals. Length one must reproduce the
previous implementation. Every span's hybrid endpoint KL must match a complete
model run selecting only those blocks. All eight old contexts record differences from previous VPD/GIM full-model
KL and single-block normalized error. Strict same-run checks compare against
the original implementation on identical saved supports. The first job passed
span validation but failed a cross-run tolerance (native VPD KL 0.396737 versus
0.396867). The previous screen did not save supports, so it cannot establish
bitwise identical interventions. Revision two separately records cross-run
differences, repeated CI support stability, and effects of the old sensor-capture
sequence; it retains strict implementation checks. The mismatch is not yet
attributed to a particular numerical cause.

Benchmark backward cost on one context using fixed support and differentiable
factors, with warmup and three repetitions. Compare trace capture plus span
forward/backward with GIM derivative capture plus its factor loss backward.
Exclude optimizer/projection from both. Record redundant block executions and
prototype overhead. This is not a production training throughput measurement.

## Interpretation

Hybrid KL runs an intact suffix after each selected span, with an intact prefix
implicit in its cached input. That surrounding computation is uncharged, so
partial-span KL is a diagnostic, not a matched-L0 result. The four-block endpoint
is the actual full selected model and uses the matched total support budget.

These objectives are support-conditioned. A favorable result would identify
the interaction scale needed for alignment, not establish a mask-free training
objective, an inexpensive selector, or a lower attainable L0 floor. Equality
after RMSNorm alone does not imply equality of the next residual stream.

Outputs include per-context scores, exact support indices, provenance, runtime
validation, timings, and a Markdown comparison. Source and results are archived
to R2 with checksums.
