# Individual-ablation ranking and cumulative deletion

Frozen 67M target, Explorer H200. Four checkpoints: published VPD400k, VPD5k,
original 1B-token par-auto (30,518 updates, CE-attribution objective; user label
GIM-1B), and GIM5k. All checkpoint and target identities are verified on compute.

Eight fresh contexts 440–447, 512 output positions each. Every component is
ablated separately at all context positions from its matrix in the original
intact target, measuring KL at every output position. Exact forward differences,
no gradient or Taylor approximation. The unchanged prefix is cached; the changed
block and all downstream computations are recomputed in FP32. Independent
interventions are batched eight at a time. KL reductions use centered FP64
log-partitions to resolve tiny single-component effects. Full-forward/suffix
agreement and numerical all-kept floors are checked before scoring.

Two fixed rankings:
1. Context-unit ranking: mean each component's effect across output positions;
   remove that component across the entire context in ascending impact order.
   The ranked intervention and the cumulative deletion unit match exactly.
2. Output-to-source transfer: rank components separately at each output position,
   then apply each position's ranking to components at that source position,
   matching the earlier E[u,x] workflow and usual per-token KLKeep convention.
   This does not measure isolated component-at-source-position effects; the
   distinction matters because attention connects positions.

Ranks are computed ONCE from individual ablations. Curves delete nested prefixes
from least to greatest impact; no adaptive re-ranking after each deletion. Evaluate
retained counts 0/32/64/128/200/256/512/1024/2048/4096/8192/16384/32768/all.
Native positive CI is an additional VPD comparator. For VPD, primary KLKeep uses
pieces only, excluding its uncharged Delta, as in earlier comparisons. A separately
labeled Delta-retained diagnostic matches the original target used during ranking.
This free-remainder curve is NOT an apples-to-apples lower-L0 explanation claim.

Effects persist per matrix/context; execution resumes at that boundary. Near the
2-hour limit the job checkpoints and submits the next allocation for its own
checkpoint family. A failed numerical check stops instead of chaining. Individual
scores and checkpoint/source/data SHA hashes are archived with verified R2 uploads.
A static one-at-a-time deletion ranking is a stronger finite-effect comparator,
not a global optimal subset oracle: cancellation, redundancy and complementarity
can change component importance once others are removed.
