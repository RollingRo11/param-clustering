# Joint selection diagnostic, 15 September 2026

Authorized follow-up to `../kl_gap_analysis_20260915.md`. Frozen 66,931,200-parameter model on Explorer; no factor training.

- GPU job: 10356811, H200, 4 CPUs, 64 GB host RAM, 20-minute allocation. Initial queue estimate: 00:51:03 Central on September 15. Estimates can move.
- Independent CPU algebra validation: 10356823, two CPUs, 4 GB RAM, three-minute limit. No model/data loading.
- Inputs: validation contexts 400–415, disjoint from the preceding evaluations.
- Frozen checkpoints: normal GIM at 5,000 steps, GIM plus activity count at 5,000 steps, published VPD at 400,000 steps.
- Fixed budgets: 128, 256, 512 rank-one components at every token, summed across all 24 matrices.
- Capture: common intact-model sampled-label Fisher K128 for selection, independent K128 for diagnostic verification. No Gaussian training.
- Selection: FP32 signed component effects; full-sequence residual against original matrices, including VPD's unrepresented Delta. Two starts (magnitude and lognormal-jittered magnitude); four sweeps in seeded random token order. At each token, shortlist eight drops and eight additions by their individual residual change, score all 64 pairs exactly, and accept only a decreasing pair. Counts and weights are fixed. Choose the final start by fit objective only.
- This is bounded local search, not an oracle. Four sweeps permit at most four accepted swaps per token per start. The jittered start explores a substantially different initial support, but failure cannot establish a representational floor.
- Report: actual BF16 KLKeep, fit and independent quadratic estimates, clean-input errors for each matrix, per-matrix retained counts, overlap, and KL after last-layer restoration. Includes both start results and native VPD selection. Paired approximate t intervals across 16 contexts do not quantify training-seed uncertainty.
- Source snapshot: `$PAIR_ROOT/code/joint_select67_20260915`, checked against `SHA256SUMS` at job startup; existing GIM source snapshot also verified.
- Output: `$PAIR_ROOT/output/joint_select67_20260915`; local mirror `../out/joint_select67_20260915`.
- Save each checkpoint/context result atomically and compressed exact selected indices. Resume skips completed records. At 15 minutes, stop between records and submit a dependency-linked continuation (maximum six allocations). Nonzero errors other than the time-bound exit stop the chain.
- Archive final records, supports, report, and source to R2, verify upload sizes/SHA metadata, and write `complete.json` only after all required results and uploads exist.

Completed: GPU job 10356811 succeeded in 9m27s; CPU algebra validation 10356823 succeeded in 21s. All 48 checkpoint/context records and final report completed; 104 uploads verified. Local comparison SHA matches `complete.json`.

The fitted quadratic was reduced nearly to zero, but independent-probe errors remained large and actual KL worsened for every method/budget on average. At 512 retained pieces, GIM magnitude KL 3.11579 became 4.23436 for the selected start; count 3.39607 became 4.65771; VPD 1.52341 became 2.14360. Native VPD CI gives 0.31645.

Start audit: fit-based selection chose the jittered start in all 144 method/context/budget cases. Its smaller residuals were insufficient evidence of better supports. Crucially, refinement starting directly from magnitude also worsened actual KL in all nine aggregate comparisons: at 512, GIM 3.73519, count 4.06243, VPD 1.98317. For GIM at 512 this direct refinement lowered the independent quadratic from 6.9340 to 5.7093 while raising KL from 3.11579 to 3.73519; approximately two components per token changed. Thus the result includes finite-sketch overfitting and a mismatch between improved local quadratic predictions and finite sparse reconstruction. It does not establish that no better support exists, or that arbitrarily accurate local joint selection would fail.
