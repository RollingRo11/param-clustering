# Four 5k training trials, selected by actual KLKeep

The user requested training rather than using VPD's surrogate score as an
eligibility gate. Here "straight through" means exact first nonlinearity and
linear downstream propagation, not straight-through top-k gates.

All four runs use the frozen 66,931,200-parameter target on Explorer, fresh
seed-0 exact-sum rank-one factors, the same 43,008 components, same ordered
64-context batches, Adam betas 0.9/0.99, and original GIM learning-rate and
concentration-exponent schedules. Each runs 5,000 updates and reads 163,840,000
positions. The original 8,700-step schedule is preserved: p reaches about
0.641 at step 5,000. No warm start from VPD and no rejection based on VPD loss.

The earlier no-masked-training restriction is preserved. No learned gates,
top-k straight-through estimator, full masked target forward in training,
adversary, or Gaussian probe is introduced. Explicit finite component removal
is evaluated algebraically at the immediate nonlinear site.

## Objectives

1. `finite_gim`: Exact Q/K softmax, MLP-in GELU, and O/down RMS with residual
   bypass preserved, followed by the cached GIM cotangent. V stays linear.
2. `finite_fisher`: Same finite correction, but a true downstream derivative
   from one independently model-sampled next-token label at every output
   position. This changes the output probe toward the model's output
   distribution, not the data's particular next token. A squared directional
   effect has a Fisher interpretation in expectation; the scheduled fractional
   penalty is a stochastic objective, not an unbiased fractional Fisher norm.
3. `write`: Concentration of component write magnitude, independent of an
   output gradient. This is a simple alternative to gradient-attribution
   concentration, not a new reconstruction guarantee.
4. `pair_gim`: Replace the separate Q and K penalties with concentration of
   signed Q-component/K-component interactions times the cached attention-logit
   derivative. Apply the nonlinearity to the pair score before aggregating pairs;
   summing signed pairs first would merely recover ordinary attribution.
   Other matrices retain GIM attribution. Each layer's normalized pair loss is
   weighted two, preserving the original initial total loss scale of 24.
   Evaluation still retains/counts original components, not arbitrary edges.

## Affordable estimator

Each update visits all 64 original training contexts. Per context and matrix,
sample 64 independent uniform component/source-position pairs: 4,096 events
per matrix/update. Compute the exact finite local effect for each sampled
intervention. Sampling is outside the penalty and uses no score-dependent
filter, so it gives an unbiased estimate of the defined full event-mean loss
and its factor gradient. A matrix's fixed component-count factor cancels
between its loss and its normalizer. This does not claim unbiasedness relative
to a different full-output metric or a finite-label Fisher norm.

Q/K pair training samples a uniform query position, a uniform causal key
position, and uniform Q/K components. Weight by the number of causal key
positions so query positions with more possible partners receive the correct
weight in a sum-over-edges objective. Fixed component-count factors cancel
in its normalizer as well.

Normalization uses 4,096 fixed samples per matrix from the first four training
contexts at initialization, recomputing the denominator as p changes. The
reference samples, optimizer, factors, and step are checkpointed; sampling and
model-label probes use explicit deterministic per-step seeds.

Caches are FP32 for consistent finite local arithmetic, using microbatches of
four contexts. Normal GIM uses its existing BF16 cache. Therefore shared
initialization, batches, and steps are controlled, but precision and sampling
variance are remaining differences. Runtime is measured against the existing
GIM implementation on the same H200. Sampling may increase gradient variance;
preparation records two-estimate gradient cosine and relative discrepancy.

Preparation also checks sampled versus dense finite scores, bitwise resumed
versus uninterrupted updates, exact weight reconstruction, finite factor
gradients, and full-update runtime. Preparation updates are discarded.

## Training, continuation, and evaluation

Each training allocation is at most two hours, with signal/deadline checkpoints
and at most sixteen linked allocations per variant. Checkpoints at 250, 1,000,
2,500, 5,000, and allocation boundaries retain optimizer state. Training
progress and run links are in W&B project `param-autoenc`, group
`objective_train67_20260915`.

The first variant starts first. Other variants undergo preparation before
their 5k jobs are eligible. Early KLKeep uses 16 contexts 480-495. The final
primary comparison uses 128 fresh contexts 512-639, common Fisher K64 ranking,
identical seeds, and retained counts 0, 32, 64, 128, 192, 256, 384, 512, 768,
1024, 2048, 4096, 8192, 16384, and all components. Normal GIM5k is reevaluated
on exactly those contexts. Published VPD with native CI is a separately labelled
reference, not a surrogate-loss gate.

Common GIM ranking is secondary for every trial. Write ranking is additionally
reported for the write run; finite native rankings for the two finite runs use
the first 16 evaluation contexts only. The pair objective has no established
cheap native original-component ranking, so its secondary curve is GIM.

Compare actual held-out KL at each fixed count and smallest tested counts
crossing KL 0.1, 0.3, and 1.0. Save paired per-context differences and uncertainty
intervals. Do not infer a win from activity count, training loss, or a VPD/GIM
surrogate ratio. One initialization cannot establish training-seed robustness.

Final models and evaluation JSONs are archived to R2 with verified uploads.
