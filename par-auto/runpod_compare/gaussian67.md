# Gaussian Fisher trial on Explorer, 14 September 2026

Implements the next experiment from `../research_review_20260914.md`. All numerical validation, profiling, training and evaluation use the real 67M Pile target inside Explorer SLURM H200 allocations. No toy model is used.

Two fresh rank-one, exact-sum arms share the original GIM initialization, token order, batch of 64x512, optimizer and 8,700-step horizon. Each trains for 5,000 updates (163,840,000 prediction positions): Gaussian categorical-Fisher output probes, and the same probes plus eight intermediate residual probes at coefficient 0.01. A single true target VJP supplies all matrix cotangents. The latter's scales are fixed RMS vector norms from four training contexts. Separate output/boundary RNGs keep output probes identical across arms. Per-step seeds make resume deterministic without saving probe RNG state.

The existing epsilon-smoothed Lp loss is deliberately retained to isolate the sensor change. The Gaussian absolute-moment identity is exact only for the unsmoothed power. Its common p-dependent constant cancels in per-matrix reference normalization. The fixed reference contains the same 256 positions as the GIM control with a deterministic fresh Gaussian probe.

`gaussian67_validate.py` checks untouched-forward equality, full-model VJP/JVP agreement on attention, GELU and final-head paths, Gaussian fractional moments on actual 67M output distributions/directions, deterministic capture, final-label independence and detached caches. Its local additive perturbation is an autodiff validation coordinate, never part of training. `gaussian67_experiment.py prepare` then measures two reversed-order rounds of full updates for GIM and both new arms, discarding warmup and profile factors.

The isolated remote source snapshot is `$PAIR_ROOT/code/gaussian67_20260914`; every source is checksum-verified before each job. `$PAIR_ROOT` is `/projects/RohanKathuria/param-clustering/data/parauto/paired_1b_2000_20260910`. It reuses the proven VPD runtime environment without changing the existing source trees or datasets.

Outputs live in `$PAIR_ROOT/output/gaussian67_20260914`, with W&B group `gaussian67-20260914`. Training saves complete state and factors every 1,000 updates. One automatic resubmission is allowed on a clean deadline stop; errors do not trigger a retry. Validation must finish successfully before training starts. Runtime ratios against GIM are recorded as measured outcomes, not assumed to equal one.

The new evaluator uses absolute rank-one counts and validation contexts 256-383, shared across all arms. Our checkpoints use the same Fisher K=8 ranking seed; VPD uses native CI. Existing GIM-5k, Fisher-5k, VPD-5k and published VPD checkpoints are re-evaluated on that same slice. This avoids the first 128 contexts repeatedly used in the preceding experiments, but is not a claim of a newly sourced external test corpus.

The final report reports actual tested threshold crossings without interpolating or assuming monotonicity. Results, source, factor snapshots and final optimizer states are archived to R2 with size/hash metadata verification. Completion is signaled by `complete.json`; its absence must not be reported as a successful completed experiment.
