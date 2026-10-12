# Mask-free 5k trials: launch record

The user confirmed on September 15, 2026 that all four training objectives
must remain mask-free. The actual frozen experiment is
`objective_train67_20260915_v3`, including its W&B group. The protocol in
`objective_train67.md` applies; its earlier unsuffixed group name is superseded
by this launch record.

The four variants are `finite_gim`, `finite_fisher`, `write`, and `pair_gim`.
All use the frozen 66,931,200-parameter target, 43,008 components, fresh seed-0
initialization, and 5,000 updates over the original 64-context batches.
Actual held-out KLKeep at matched component counts is the outcome measure.

| Variant | Preparation job | Initial training job |
|---|---:|---:|
| finite_gim | 10364948 | 10365027 |
| finite_fisher | 10365004 | 10365040 |
| write | 10365005 | 10365078 |
| pair_gim | 10365007 | 10365090 |

All four preparations passed. As of 3:16 PM Central on September 15, their
training jobs were queued and no retained training updates had been reported.
Final reference evaluations are jobs 10365083 (GIM5k) and 10365084 (published
VPD). Reference evaluations for the separate early context split are jobs
10365096 and 10365097, respectively.

Preparation updates are discarded. The first v3 preparation passed on an
Explorer H200: sampled versus dense finite-score maximum absolute discrepancy
was below 0.00000036, resumed factors were bitwise identical, and relative
matrix reconstruction error was about 0.00000103. Median measured update time
was 1.980 seconds versus 0.514 seconds for GIM, a factor of 3.85. The projected
5k training time is 2.75 GPU hours, excluding queues, startup, checkpoints,
and evaluation. This implementation has not met the original GIM cost target.

The two independent sampled gradient estimates had cosine 0.186 and relative
discrepancy 1.276. Sampling variance is substantial. These are tests of whether
the objectives improve KLKeep; a negative result cannot distinguish an
ineffective objective from inadequate estimator quality. No claim of an
improved sparsity floor is justified before the held-out curves complete.

Earlier rollout failures did not produce retained trial training. The first
preparation exposed an in-memory test bug: a CPU Adam step tensor aliased the
live optimizer state. A real serialization round-trip fixed the test. A
subsequent source edit raced pending jobs becoming active, causing a source
checksum/plan mismatch. Those launches and their descendants were canceled;
v3 restores the version that passed preparation and freezes its snapshot.
Do not edit the v3 cluster snapshot while its jobs run.

Training launchers save full optimizer state and chain two-hour allocations.
`objective_train67_watch.py` polls every three minutes for at most 48 hours,
copies small result/status files locally, and submits missing baseline and
checkpoint evaluations as scheduler capacity permits. It never cancels jobs
or edits training sources. Normal GIM5k and published VPD are evaluated on the
same final held-out contexts. The watcher reports failed jobs for inspection;
it does not silently retry failed training or change any objective.

Live local artifacts are under `par-auto/out/objective_train67_20260915_v3/`:
`watch_status.json`, per-variant `progress.json` and `wandb.json`, evaluation
curves, and the final `report.md` when available. On Explorer the same output
directory sits below `data/parauto/paired_1b_2000_20260910/output/`.
