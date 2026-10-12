# Experiment records

All comparisons below use the frozen target trained for **10,000 steps**.

## Matched 50k baseline

[Full benchmark](out/spd_full_10241714/comparison/report.md): 400 components,
batch 2,048, 50,000 updates per matrix, both methods on one V100.
The original dense implementation took 26.71 minutes and 3.19 GiB peak fit
tensors, versus SPD's 18.05 minutes and 1.34 GiB. Our Figure 10 relative RMSE
was 4.84%, versus SPD's 9.90%; correlations were 0.99880 and 0.99847.

The [profile](out/spd_profile_10242038/profile_report.md) identified scores and
backpropagation as 84% of the original implementation's GPU activity.

## Structured scoring and one-loss training

[Controlled 1,500-step pilot](out/joint_10242520/report.md), same V100,
400 components and batch 2,048:

| Implementation | Fit time | Peak tensor memory |
|---|---:|---:|
| Structured, sequential | 20.60 s | 0.54 GiB |
| Structured, one summed loss | 20.22 s | 0.77 GiB |
| SPD | 36.93 s | 1.34 GiB |

All eight trained factor tensors and logged training histories were bitwise
identical between the two execution layouts. The small timing difference is
not a reliable speedup estimate. Setup plus fit took 28.88 seconds for the
combined loop versus 36.93 seconds for SPD. This is a pilot, not a converged
quality comparison. The structured cache uses 181 MiB.

The 50k structured sequential/SPD comparison is job **10242461**; the 50k
combined-loop validation is job **10242577**. Their original source snapshots
remain intact while the active repository is cleaned up.

## Earlier method analysis

- [10k-step decomposition](out/longer_B_10241367/result.json), at 1,024
  components and batch 128. Component counts and batches differ from the
  matched SPD experiments, so it is not a step-count-only comparison.
- [Component ablations](out/components_10240997/result.json) and
  [global usage](out/usage_10241070/result.json).
- [Released SPD replication](out/spd_pilot_10241633/released/result.json):
  the published 50k decomposition for the exact same target.

Historical filenames and labels inside raw artifacts are preserved as records.
New code exposes only the projected Lp method and the SPD comparator.

## Native-sensor ablatability

[KLKeep report](out/klkeep_10242938/report.md), V100 job **10242938**:
combined-loop par-auto checkpoint **10242577** versus SPD retrain
**10242461**, both 50k steps and 400 components per matrix. All 1,601 budgets
were evaluated on 4,096 fresh shared inputs. Rankings are per-input, global
across the four matrices, and frozen before removal: output-effect RMS for
par-auto and learned causal importance for SPD.

| Total retained | Par-auto KL | SPD KL | Par-auto MSE | SPD MSE |
|---:|---:|---:|---:|---:|
| 16 | 6.05e-5 | 8.81e-5 | 1.19e-4 | 1.73e-4 |
| 32 | 6.52e-6 | 7.53e-5 | 1.33e-5 | 1.46e-4 |
| 64 | 6.24e-9 | 6.30e-5 | 1.28e-8 | 1.16e-4 |

KL is computed after a temperature-1 softmax over regression outputs; it is
an analysis convention, not the toy's native loss. MSE uses raw outputs.
Primary errors are relative to the frozen target and include SPD's nonzero
all-kept error. The report also separates damage relative to each
decomposition's own all-kept output. SPD's target-referenced curve is
non-monotonic. This cumulative ranking test does not cover arbitrary subsets.

## Short schedules

Projected Adam is the main optimizer. The 5k and 10k controls in job
10243205 nearly match the 50k result on the same 4,096 held-out inputs.

| Steps | Fit time | KL at 32 retained | Kept for ≤1% all-removed MSE |
|---|---:|---:|---:|
| 5k | 65.87 s | 7.92e-6 | 34 |
| 10k | 131.44 s | 7.00e-6 | 33 |
| 50k | 663.67 s | 6.52e-6 | 32 |

Each short run uses its own cosine LR schedule from 1e-3 to 1e-5 and
anneals p to 0.5 over its first 80%. The previous optimizer experiment has
been removed from the active code, plots, and local results. Its records
are archived outside the repository.

## Matched SPD short runs

[Matched KLKeep report](out/spd_short_10243273/comparison/report.md), V100
job **10243273**, adds independent SPD 5k and 10k runs using the unmodified
authors' trainer, constant LR 1e-3, and the same 400 components per matrix.
The plot contains only solid par-auto/SPD curves, with logarithmic x axes.

| Steps | Par-auto KL at 32 retained | SPD KL at 32 retained | Par-auto fit | SPD fit |
|---:|---:|---:|---:|---:|
| 5k | 7.92e-6 | 4.78e-5 | 65.87 s | 107.60 s |
| 10k | 7.00e-6 | 5.99e-5 | 131.44 s | 214.79 s |
| 50k | 6.52e-6 | 7.53e-5 | 663.67 s | 1050.92 s |

SPD's 5k checkpoint has lower error at 32 retained than its longer runs;
more training does not consistently improve this ablation curve. Its
all-kept target KL decreases from 8.60e-5 to 7.83e-5 to 7.65e-5 instead.
Par-auto's exact-sum construction makes its all-kept error numerical zero.
Both methods use their native sensors and identical held-out inputs.
Peak allocated tensors are 0.766 GiB for par-auto and 1.344 GiB for SPD.
These are single-seed runs on separate V100 allocations; timing boundaries
and schedule differences are documented in the report.

Validation: all 30 method tests passed, restored projected Adam matched the
original factors bit for bit, and both new SPD checkpoints passed masked
forward parity against the authors' implementation.

## FP16 mixed precision

[Precision comparison](out/fp16_10243518/comparison/report.md), V100 job
**10243518**, completed 50k scheduled training steps with 21 gradient-overflow
skips (49,979 Adam updates). The target, seed, component count, batches and
schedules match the FP32 reference. Only training score matmuls use FP16;
parameters/Adam/cache/audits stay FP32, projections FP64, and evaluation FP32.

| Metric | FP32 | FP16 mixed |
|---|---:|---:|
| KL at 32 retained | 6.525e-6 | 6.538e-6 |
| KL at 64 retained | 6.237e-9 | 6.497e-9 |
| MSE at 32 retained | 1.328e-5 | 1.332e-5 |
| Fit seconds | 663.67 | 741.73 |
| Peak allocated tensors (GiB) | 0.766 | 0.757 |

The ablation curves nearly overlap. This implementation's observed fit time
is about 12% longer, with about 1% less peak allocated memory. Timings span
separate V100 allocations and have slightly different final-save boundaries;
this is not a controlled throughput benchmark. All 31 tests passed, including
observed FP16 arithmetic and exact-sum checks; default FP32 factors matched
the original implementation bit for bit.

## Separate 5k precision speed ablations

[Report](out/precision_sweep_10243801/report.md) and
[plot](out/precision_sweep_10243801/profile.png), job **10243801**: fresh
processes run sequentially on the same V100, each with the 5k LR/p schedule.
The baseline is mixed-FP16 score matmuls. Each variant changes only its stated
precision group. These are isolated experiments; main training defaults are
unchanged.

| Change from mixed-FP16 baseline | Fit seconds | Warm ms/step | Peak allocated GiB |
|---|---:|---:|---:|
| None | 76.71 | 15.307 | 0.757 |
| FP16 parameters, gradients, Adam states | Failed on first Adam update | — | — |
| FP16 cache and large score intermediates | 71.33 | 14.148 | 0.488 |
| FP32 projection and its backward | 69.29 | 13.818 | 0.746 |

FP16 activations reduce peak allocation by about 36% and warm step time by
about 8%. FP32 projection reduces warm step time by about 10%; its final
relative matrix-sum error is 1.10e-7 (versus 2.68e-8 baseline), within the
existing 1e-5 tolerance. Each completed run had one overflow-skipped update.

The FP16 parameter/state run produced nonfinite parameters on its first
stock Adam update, retaining the original epsilon 1e-8. Actual parameter,
gradient and moment storage was confirmed FP16. Even before that update,
rounding factors to FP16 raised the relative sum error to 2.80e-4, above the
method's tolerance. No throughput is reported for this failed configuration.
All 33 tests passed, including FP32 projection gradients and saved FP16 score
intermediate checks. No KLKeep evaluation was requested or run for this sweep.

## 4-layer Pile setup smoke test

The [Pile pipeline](pile.md) was validated in V100 job **10244172** with all
24 target matrices and 1,024 free components per matrix. FP16 caches and
score intermediates are combined with FP32 projection/backward and FP32
parameters/Adam. An eight-probe random full-context logit-Jacobian sketch
replaces the toy-specific exact QR sensor; this approximation is explicit.

All 36 tests passed. Real-data validation cached four 512-token contexts,
performed two steps, checkpointed/resumed for two more, and evaluated a
non-overlapping held-out context. All-kept KL was **1.84e-11**, with maximum
relative matrix-sum error **7.14e-7**. [Evaluation JSON](out/pile_smoke_10244172/run/evaluation.json).
The prepared full run is resumable for 50k scheduled steps; it has not been
submitted. The four-step smoke is not a language-model quality benchmark.
