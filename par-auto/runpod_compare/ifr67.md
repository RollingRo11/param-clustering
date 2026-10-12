# IFR-style scores and signed reader conditioning, 67M

Prespecified 15 September 2026, before observing these results. Explorer H200 only.
Frozen target: 66,931,200 parameters. Ordinary GIM 5k and exact VPD conversion,
with identical 43,008 rank-one factors and exact matrix reconstruction. No training.
Eight fresh validation contexts 432–439, 512 positions each. Exact per-token budgets
200, 512, and the VPD native positive count. CI is only a witness/count comparator.

## Scope

This adapts the finite proximity in Ferrando & Voita's IFR to parameter components;
it does not reproduce their head/block-level graph extraction:
https://aclanthology.org/2024.emnlp-main.965.pdf, Section 2.

For y=Wh and component output z_c=u_c(r_c^T h), local positive proximity is
max(||y||_1-||y-z_c||_1,0). The signed variant takes the absolute value of the
unclipped proximity, crediting negative contributions too. Each is normalized
across components within its matrix/token; all matrices receive total score 1
unless their total score is zero. Global top-k enforces the component budget.
A count is always a primitive parameter component at a source token, not an edge.

Reader conditioning starts at signed local selection. For each consumer/token,
take at most 16 selected readers, prioritized by the fixed signed local score,
and normalize each read vector to unit L2 (factor gauge invariance). Transform
each writer into those reader coordinates, retaining signs. Measure absolute
finite L1 proximity to the COMPLETE observed input of the consumer in those
coordinates, not just the writer's aggregate contribution. Normalize this score
within the producer matrix and blend 50/50 with the local signed score. Fall back
to local when the entire routed score is zero. Repeat twice, synchronously;
report both rounds, never choose the best round using evaluation KL.

Edges implemented:
- V at source -> actual clean attention weights -> O readers at destination.
  Bound computation by the 4 destination tokens with greatest mean attention
  to each source; preserve head-specific attention coefficients. Source positions
  receive the component count. Report the retained attention mass.
- O -> residual addition -> fixed observed RMS scale -> same-layer FC readers.
- FC -> observed GELU secant -> Down readers. The secant exactly reconstructs
  the observed GELU output, but is not a claim about its perturbed response.
- Down -> residual addition -> fixed observed RMS scale -> next-layer Q/K/V readers.

Q/K and final Down use local scores. No Q/K components are provided free or left
unablated in evaluation. This prototype does NOT attribute Q/K's construction of
attention routing, all residual skip paths, or guarantee complete connected routes.
The fixed observed gates are a selection proxy; actual sparse evaluation always
recomputes Q/K, attention, normalization, and GELU. Failure of this bounded
prototype does not rule out route-based training more generally.

## Validation and costs

Use intact FP32 activations and finite vector differences, without output KL or
any gradients for proposed selection. Fisher K128 is a separate comparator.
Native positive CI on converted VPD supplies a known-good support on these inputs.
Report actual KL, clean-error injection, last-layer restoration, local errors,
per-matrix count allocation, witness recall and score mass, reader coverage,
attention coverage, wall times. Persist exact supports and source/checkpoint/data
SHA hashes; upload with byte-length and SHA-metadata verification to R2.

Validate finite proximity and its signed version against a direct implementation,
including cancellation and projection examples; validate gauge invariance.
Validate reconstructed clean attention against actual O inputs, GELU secants,
exact per-token counts, and parameter reconstruction. Runtime at batch one is
only a diagnostic; this is not a measurement of a full 64-context training loss.
No claim of being within GIM training cost follows from one intact forward pass.
