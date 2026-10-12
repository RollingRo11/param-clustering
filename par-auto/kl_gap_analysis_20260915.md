# Why attribution sparsity has not produced VPD-level KLKeep

Analysis of existing 67M Explorer experiments; no new training or GPU measurements in this review.

## Main conclusion

The evidence favors an objective/selection mismatch over an undiscovered scalar gradient estimator. We optimize sparse individual attribution on the intact computation. KLKeep evaluates the output of a jointly selected, finite, sparse computation. Our experiments change attribution magnitudes and their penalty much more than they change this distinction. This is not a proof that a cheap, unmasked objective cannot work.

The particularly important control is **VPD's own factors measured with our gradient sensor**. VPD's successful sparse endpoint does not require a sparse intact-model Fisher attribution distribution.

## Evidence, with scopes kept separate

1. The published VPD census has about 202 positive native CI values per token across 24 matrices. On the same census inputs, its Fisher K8 scores require about 18.6k pieces to collect 90% of total score magnitude; almost all 38,912 pieces have numerically nonzero sensitivity. This is not a comparison of absolute CI and gradient units. The score-mass result is invariant to a common positive rescaling at each token. Sources: `out/faithfulness/census_vpd_published.json`, `census_vpd_published_fisher.json`, and `runpod_compare/ci_census.py`. The historical score-mass aggregation adds one per microbatch rather than per token, undercounting by approximately one component; this does not affect the conclusion or the positive-CI census.
2. In the separate 16-context diagnostic (384–399), at 512 retained pieces, published VPD gives KL 0.264 with native CI and 1.533 with intact-model Fisher K64 ranking. Same factors, same budget: selection substantially changes the result. Our Gaussian-trained factors give 3.161 with K64. These differences are not an additive causal decomposition of the gap. Source: `gaussian67_investigation_20260914.md`.
3. For our Gaussian factors on that diagnostic, top 512 retain 88.25% of individual squared Fisher-score mass but only 35.27% of score magnitude. The omitted-tail quadratic estimates are 1.240 for individual squares, 3.817 after including same-source-position interactions, and 6.392 after summing source positions within each sequence. Probe streams differ for the diagonal and joint estimates. An independent exact JVP on two contexts predicts 7.670, compared with actual KL/removal_strength² = 7.486 at strength 0.01. Local sensitivity already detects serious damage.
4. The same two-context VPD native selection has local quadratic coefficient 36.24 but full-removal KL 0.364. Its sparse endpoint is not well described as a nearly zero perturbation in the intact model's tangent metric. Our corresponding full-removal KL is 2.927, below our quadratic prediction. This makes saturation or higher-order underprediction an inadequate sole explanation of our result.
5. The new 5k additive-count run worsens common-K64 KL at 128–1,024 retained pieces, improves it at larger budgets, and changes none of the smallest tested count crossings for KL 0.1, 0.3, or 1. Native GIM ranking has the same broad pattern. Source: `out/gim_count67_20260914/report.md`.

## What the loss discards

Let i index a matrix, source position, and component within one sequence. Let v_i be that component's signed first-order change to the sequence's logits, measured in the target probability distribution's Fisher metric. Absorb the normalization by prediction count into v_i. For an omitted set R, the local quadratic KL is

    Q(R) = 1/2 ||sum_{i in R} v_i||²
         = 1/2 sum_{i in R} ||v_i||²
           + sum_{i<j, i,j in R} <v_i, v_j>.

Our individual Fisher scores estimate ||v_i||; Lp and activity-count losses operate on these norms or on scalar GIM analogues. Their values do not identify the signed off-diagonal terms. Perfect norm estimates still cannot distinguish n aligned omitted vectors from n orthogonal vectors of the same lengths. Their squared joint effects differ by a factor n.

The gradient information of interest is therefore **alignment and conditional usefulness**, not simply a more precise magnitude. The local cross terms need only first-order logit effects: near the reference distribution the KL Hessian is JᵀFJ. Increasing probe count improves estimates; it does not by itself optimize the full joint quadratic form. Nor does it estimate third or higher derivatives of the network.

Our earlier joint training term did include some cross terms. It summed components/matrices at each source position and squared before aggregating source positions, so it omitted temporal interactions. It was not a test of the complete sequence-level quadratic objective. Correcting this is a plausible diagnostic, not evidence that the corrected training objective will reach VPD's endpoint.

## The attribution budget exists, but is weaker than the desired guarantee

For a fixed intact-model activation h and cotangent g, exact parameter reconstruction gives

    sum_c (gᵀu_c)(r_cᵀh) = gᵀWh.

This holds even for GIM's surrogate cotangent. It fixes the signed total for that matrix and token. It does not fix the sum of absolute values, ensure joint ablatability of small terms, or ensure compatible choices across layers. The count penalty can concentrate scalar attribution while leaving these requirements unresolved.

An important design trap: adding the squared **total** signed attribution over all pieces cannot solve this. With the same exact-sum constraint, that total is fixed and supplies no factor-learning signal. A useful new objective must measure a nonconstant property of component relationships, subsets, or approximation error. Writing down 'include correlations' alone is not an implementation.

The current loss is also separable across matrices once the intact-model caches are fixed. Those caches contain downstream information, but changing one matrix's decomposition does not make the objective evaluate how it interacts with another matrix's *selected* decomposition. The global count normalizer does not introduce such an interaction.

## Why VPD can differ substantially

VPD jointly trains its factors and a shared nonlinear CI selector against finite stochastic/adversarial reconstruction losses. The selector sees clean activations from all 24 matrices and uses bidirectional attention. This gives it a way to coordinate selections across layers and positions; it is not an independent estimate of each factor's intact-model derivative. The architecture is described in Appendix A.2 of [the primary paper](https://www.goodfire.com/research/interpreting-lm-parameters).

The restoration and path diagnostics support finite compensation as one contributor, but do not quantify its share of the entire gap or establish universal unfaithfulness. The published 400k-step checkpoint also has a much larger training budget. VPD allows a dense Delta remainder, whereas we enforce the rank-one pieces alone to sum to each matrix. These differences should be controlled before claiming a representational floor. In particular, an exact-sum version of VPD can retain the existing pieces and represent Delta with additional rank-one pieces, whose costs must be counted honestly; projecting and changing all VPD pieces would confound that comparison.

The observed dense Fisher census means that VPD's ~200 CI support is not a target value for our raw gradient activity counter. Our previous wording emphasized the incomparable units but underemphasized the stronger fact: even the successful VPD factors do not exhibit the gradient sparsity we have been treating as the route to that support size.

## Most informative next experiment

Before another training run, freeze the current 67M checkpoints and test whether **joint selection using signed effects** improves actual KLKeep. This is evaluation, not mask-based factor training.

- Compare magnitude top-k with binary support refinement minimizing the full-sequence signed residual estimate, at exactly 128, 256, and 512 pieces per token. Search swaps within each token while updating the residual over the whole sequence. Do not rescale selected component weights or modify factors.
- Use multiple starts and common probe budgets; score candidate supports with an independent probe stream and actual nonlinear KL on held-out contexts. A fit to one probe sketch is not a result.
- Include published VPD with native CI and gradient ranking as controls. Account for Delta in tangent calculations. Preserve the existing intermediate/restoration checks to distinguish endpoint compensation from computational fidelity.
- A large real-KL improvement would identify a selection bottleneck and provide a concrete behavior for an unmasked training objective to target. An improvement only in predicted quadratic error would identify failure to extrapolate to the sparse endpoint. Little improvement would weaken this particular decoder hypothesis, but would not prove the factors have no better support.

This diagnostic uses signed Fisher projections already available from the existing evaluation machinery; it needs no new Gaussian training and no parameter Hessian. Its search cost and the eventual training loss cost must still be measured. A single CE/GIM cotangent is not an unbiased substitute for the full KL geometry, so I cannot honestly claim the eventual method already fits the GIM compute budget.

Related primary literature: [GIM](https://arxiv.org/abs/2505.17630) modifies backward rules for circuit localization; it does not establish that sparsifying those scores learns an optimal finite reconstruction. [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940) studies conditional importance under self-repair in other transformer circuits. It supports testing conditional selection, but its intervention procedure is not a method that satisfies our no-mask-training constraint.
