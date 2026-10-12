# Diagnosing the Gaussian 67M pilot

This is evaluation-only work on frozen checkpoints. The training objective, saved
checkpoints, and immutable training source snapshot are unchanged.

Explorer job **10352999** (after fixing a PyTorch API incompatibility in 10352951) runs `slurm_gaussian67_diagnose.sh` on one H200, with a
30-minute limit. Results go to
`data/parauto/paired_1b_2000_20260910/output/gaussian67_diagnostic_20260914`.

The six checkpoints are GIM 5k, Fisher K4 5k, Gaussian 5k, Gaussian + boundaries
5k, VPD 5k, and published VPD 400k. All share the verified 66,931,200-parameter
target. The held-out diagnostic slice is validation contexts 384–399, separate
from the preceding pilot report's contexts 256–383. This is a small diagnostic
sample from the same validation file, not an independent final test set.

## Questions

1. **Ranking variance:** do 64 sampled-label Fisher probes improve actual KLKeep
   relative to the nested first eight probes? How much do the retained sets
   overlap? VPD additionally retains its native CI ranking.
2. **Concentration:** how much intact-model sensitivity mass and squared mass
   remain at 512 and 4,096 primitive rank-one components/token? These masses
   measure the diagonal component proxy, not joint reconstruction fidelity.
3. **Local error and composition:** measure each matrix's replacement error on
   clean inputs; sparsify one matrix or layer at a time; restore each layer in
   the fully sparse network. Compare live factor replacements to fixed
   clean-input error injections.
4. **Linearization:** on the first two contexts, compute the exact FP32 JVP of
   the complete simultaneous intervention and compare its quadratic Fisher KL
   prediction to actual outputs for removal strengths 0.001, 0.01, 0.1, 0.25,
   0.5 and 1. This includes temporal cross terms and VPD's omitted delta.
5. **Numerics:** verify target identities, matrix-sum errors, BF16 versus FP32
   all-kept KL, intact-forward parity, and VPD hook/native-forward parity.

The previous joint proxy squared separately at each source position. Its field
called `full` was therefore not the full-sequence Fisher prediction. Comments in
its source now flag this; its historical numerical behavior is unchanged.

## Reproduce

The job verifies checksums for its own source and the earlier immutable training
snapshot. The diagnostic imports the latter's common model and scorer utilities.
No model or data computation belongs on the login node.

After retrieving the small diagnostic JSON, summarize locally:

```sh
python3 par-auto/runpod_compare/gaussian67_diagnostic_report.py \
  par-auto/out/gaussian67_diagnostic_20260914
```

The report retains per-context paired comparisons. Its 95% t intervals use 16
contexts as units, and do not treat 8,192 correlated token positions as independent.
The exact tangent check uses two contexts, so its conclusions need that qualification.

## Completed follow-up

Both successful jobs finished with exit code 0: main diagnostic **10352999**
and basis control **10353153**. The latter compares untrained exact row, column,
and MLP-neuron bases and uses independent K64 probes for correctly sequence-summed
residual predictions. Raw results and diagnostic sources were uploaded to R2
with verified size and SHA metadata.

Interpretation and the next loss candidate are in
[`gaussian67_investigation_20260914.md`](../gaussian67_investigation_20260914.md).
The next loss candidate has not been trained.
