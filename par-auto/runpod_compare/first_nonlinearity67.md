# Exact immediate nonlinearity screen

Frozen 67M target, Explorer H200. No factor optimization or learned selection.
Primary question: does exact immediate nonlinearity followed by a downstream
cotangent give published VPD lower objective than GIM5k?

Eight fresh validation contexts 464-471; normalizers from train context 10000.
Every component and all 512 source positions are evaluated, without sampling.
Published VPD is primary; its exact-sum conversion is a separate control.
Also test GIM5k and the converted VPD after 20/100/200 damaging GIM updates.
Use the same normalizer for linear and finite variants, calibrated from the
GIM5k linear objective for each module and derivative rule. Fixed exponent
0.641; no search over penalty powers or mixing coefficients.

One ordinary CE backward or one GIM-rule CE backward supplies a fixed cache.
Both new caches use FP32 to align the exact local forward and cotangent sites.
The GIM version retains the quarter scaling on Q/K scores and half on V, but
replaces the immediate softmax/norm approximation with the actual finite
nonlinearity. Report it as a surrogate, separate from the true-gradient case.

Corrections:

- Q: remove one component at one source/query position, recompute its causal
  attention row in every affected head, dot probability change with the cached
  probability cotangent, then sum signed effects across heads.
- K: remove one component at one key/source position; use the exact single-logit
  softmax update for every causal query/head. Exclusive partition sums avoid
  cancellation from subtracting a dominant key probability from one.
- MLP-in: evaluate exact tanh-approximate GELU after removing the dense component
  vector and contract its change with the cached cotangent on GELU output.
- O and MLP-down: exact RMS change at the next norm, preserving the residual
  bypass. Replace the linear norm contribution in the total cotangent with its
  finite counterpart, rather than dropping the identity path.
- V: unchanged linear control. The next norm is not immediately adjacent; it
  follows attention mixing and O. This experiment does not claim finite V-path
  normalization or exact propagation through multiple nonlinearities.

Validation compares finite scores to full target forwards with a single
component removed at one source position, observing the relevant local site
and applying the SAME cached cotangent. This validates the finite-local
implementation, not its accuracy as an estimate of full output KL.
Check Q/K/O/MLP-in/down in the first and final blocks. Real factor backward is
timed and checked for finite gradients. Warm both capture and backward, use
three repetitions, and compare with the established BF16 GIM loss.

Full results include per-module losses and per-context outcomes, so overall
ordering cannot hide whether attention or MLP terms supply the change. The
acceptance statistic is held-out mean published VPD loss divided by GIM loss;
below one is favorable. Also report per-context preference and whether loss
rises along the previously damaging GIM trajectory.

No new KL-keep claim follows merely from a favorable score ordering. Scalar CE
cotangents are incomplete output sensors; the GIM backward is also a surrogate.
Exact local nonlinearities still miss later nonlinear and joint-ablation
effects. Dense GELU recomputation is chunked, not assumed cheap; report actual
runtime before considering training. No Gaussian probes are used.

Source and completed results are uploaded to R2 with checksum verification.
