# Individual component-at-source-position ablation

This supersedes the context-wide `individual67` protocol after the requested
granularity was clarified. The four old jobs failed their FP32 equivalence
preflight before scoring; their immutable sources and logs remain for provenance.

Frozen 66,931,200-parameter target, Explorer H200 only. Checkpoints: published
VPD400k, VPD5k, the completed 1B-token original CE-attribution par-auto run
(30,518 updates, user label GIM-1B), and GIM5k. Checkpoint/target/data hashes
are recorded and verified on the compute node.

The initial exhaustive experiment covers **one complete held-out context**,
validation row 440, with all 512 source positions and all components. This is
22,020,096 individual interventions for each 43,008-piece par-auto dictionary,
and 19,922,944 for each 38,912-piece VPD dictionary. The earlier eight-context
default was reduced to one full context because source-position isolation
multiplies the intervention count by 512. This is an exhaustive diagnostic on
one real context, not a multi-context population estimate.

Each forward removes exactly one original component at exactly one source
position from one matrix in the **intact original target**. All changed
downstream computations, including attention and normalization, are recomputed
in FP32. The unmodified block prefix is cached at the same batch shape used for
interventions. Batches contain 32 independent interventions. No gradient
approximation is used. Centered FP64 log-partitions produce per-output KL;
results are stored as float32. Effects near the measured numerical floor are
unresolved, rather than confidently ordered causal differences.

Every potentially affected output position is stored. Strictly earlier causal
zeros are implicit. Last-block Q, O, FC, and down-projection interventions affect
only the same output position, so their other zeros are implicit too. Preflight
checks at source positions 0, 255, and 511 compare cached-suffix/full forwards,
check those causal zeros, and bound zero-ablation KL. All affected outputs are
otherwise measured, including the full downstream span for K and V.

Three static rankings and corresponding nested deletion curves are produced:

1. **Sequence global:** score each (source, component) by mean KL over all 512
   outputs, then remove occurrences globally from least to greatest impact.
2. **Final-output global:** score each occurrence by KL at output 511; remove
   occurrences globally in that order. Its primary fidelity measure is the final
   output's KL, with sequence-average KL also reported.
3. **Sequence per-source quota:** use the same exact sequence-impact scores,
   but retain exactly k components at each source. This is a secondary comparison
   to previous fixed-per-position budget experiments, not the global deletion path.

Global curves keep `512*k` total occurrences; k means average components per source
position and does not mean only k occurrences explain the final prediction.
Budgets are 0, 32, 64, 128, 200, 256, 512, 1024, 2048, 4096, 8192, 16384,
32768, and all. Scores are never adaptively recomputed after cumulative removal.
Thus the requested order is used, sampled at these retained-budget checkpoints;
there is not an additional full forward after each of the ~20 million deletions.

Primary keep curves execute selected pieces alone, with no uncharged VPD Delta.
A labeled Delta-retained diagnostic instead matches the intact-target baseline
used for individual ablations. Native positive VPD CI is also evaluated. Neither
the free-Delta curve nor a static ablation ordering proves the optimal subset floor.

Effects persist after each (source, matrix) shard. Near the 2-hour limit, the job
submits a successor for the same family. Failed checks stop without chaining.
Progress logs include measured intervention throughput. A CPU job archives full
effects and results to R2 with verified, incremental upload receipts after scoring.
The source manifest and experiment namespace are immutable after submission.
