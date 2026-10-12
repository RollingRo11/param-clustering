# Research review: lower L0 without ablation training

14 September 2026. Review of local code/results, read-only Explorer results, and the complete 102-page VPD PDF in Downloads. No training jobs were submitted and no existing experiment code was changed.

My recommendation is to retain exact-sum rank-one factors and test **Gaussian Fisher probes, followed by Gaussian probes of intermediate computation**, using one target backward pass. This changes what the sparsity loss measures without adding a masking objective, a gate predictor, a Hessian computation, or a second target backward. It is a proposed experiment, not a demonstrated improvement over VPD.

## 1. What the experiments establish

The current method is the free-factor parameter autoencoder in `par-auto/`, not the earlier attribution cofactorization or scalar ownership methods. Each internal matrix is represented as `W = U R^T`; projected Adam maintains this equality to numerical tolerance. All 24 attention/MLP matrices of the frozen four-layer 67M target are included. Embeddings, normalization and output head remain the shared fixed architecture.

The main rank-one configuration has 43,008 pieces, compared with VPD's 38,912. The difference is Q/K capacity: ours must represent each matrix without a delta. VPD's complete decomposition includes a computed delta `W - sum(components)`, so it too can reconstruct the weights exactly **when that delta is included**. Its scored, delta-excluded component sum is approximate. This is more precise than saying VPD has no exact parameter accounting at all.

On 65,536 shared held-out positions, the 5,000-step results are:

| Training / evaluation score | KL at 1% retained | KL at 3% | KL at 10% |
|---|---:|---:|---:|
| Ordinary CE / ordinary CE | 5.732 | 3.838 | 1.908 |
| GIM / GIM | 4.941 | 3.489 | 1.569 |
| Same GIM factors / Fisher K=8 | 3.628 | 2.312 | 0.810 |
| Fisher K=4 / Fisher K=8 | 3.643 | 2.207 | 0.665 |
| Fisher + joint term / Fisher K=8 | 6.408 | 3.270 | 0.866 |
| Fisher + within-matrix joint term / Fisher K=8 | 3.788 | 2.285 | 0.655 |
| Write norm / write norm | 5.781 | 4.004 | 2.275 |
| VPD / CI, 5,000 steps | 1.611 | 1.122 | 1.124 |

For ours, those fractions retain 430, 1,290 and 4,301 pieces; for VPD they retain 389, 1,167 and 3,891. Thus these columns diagnose trends; they are not exactly matched-L0 comparisons.

Sources: [baseline comparisons](out/gim_normal_vpd_5000/comparison.json), [Fisher reranking](out/fisher_rank/gim_p05_fisher_k8/klkeep.json), [Fisher training](out/fisher_k4_5000/eval_fisher_k8/klkeep.json), [joint](out/fisher_k4_joint_5000/eval_fisher_k8/klkeep.json), [within-matrix joint](out/fisher_k4_jointw_5000/eval_fisher_k8/klkeep.json), [write norm](out/write_5000/eval/klkeep.json).

The largest measured improvement here is changing the ranking sensor on unchanged factors. Fisher training helps further, but does not close the low-budget gap. Increasing sparsity pressure to p_end=0.25 or 0.1 yields only modest improvements at 10% retained, with little or negative improvement at 1%. Larger LR and the VPD-like schedule do not fix this. Plain write magnitude fails, so an entirely task-insensitive Euclidean output loss is a poor primary candidate.

These are mostly single-seed comparisons. “p_end=0.5, 5,000 steps” also needs care: the newer controls stop at 5,000 of an 8,700-step horizon, with p approximately 0.641 at the last update. They have not completed that annealing schedule. Keep this horizon for matched diagnosis; any full short schedule needs its own matched baseline.

The published 400k-step VPD checkpoint obtains KL 0.313 at 389 retained, compared with 5.083 for the original 1B-token ordinary-CE par-auto checkpoint at 430 retained under its original CE ranking. That is a real gap in the recorded native curves, but compares different training budgets and is not the best available sensor for our factors. VPD's native positive-CI census is about 202 components/token on the 8,192-position census sample. A derivative score's numerical nonzero count is not the same quantity. [Published comparison](out/runpod_paired_1b/published_vs_finished/comparison.json); [CI census](out/faithfulness/census_vpd_published.json).

## 2. Variable rank and ownership: why I would stop this branch

The September 12-14 Explorer results, preserved in [remote_results.json](out/review_20260914/remote_results.json), are especially informative:

* Rank-8 groups: 5,376 groups total; mean participation rank about 5.35. Keeping 538 groups gives KL 5.867; keeping 4,301 groups gives 0.271. The latter retains 34,408 factor columns, not 4,301 rank-one pieces.
* Soft ownership, alpha=0.2: only 1,248 hard groups survive, but KL remains 1.079 at 80% retained and 0.038 with everything retained. Mean maximum soft ownership is only 0.00497. Hard evaluation is very different from the diffuse training assignment. The all-kept error also needs an FP32 reconstruction/conditioning audit before interpretation.
* Straight-through ownership, alpha=0.2: 371 groups attain KL 0.706. However, the census shows those groups contain **35,113 rank-one pieces on average, 81.6% of all pieces**. At 741 groups and KL 0.290, about 85.0% of pieces remain. Some individual groups own over 90% of a matrix's pieces.
* Raising alpha to 0.5 reduces this concentration of ownership but worsens the curve: 376 groups, roughly 9,017 pieces, KL 3.305.

The low group count is mainly a change of description unit. Counting primitive pieces is a conservative accounting convention, not a claim that grouped pieces necessarily have full numerical rank. A legitimate rank saving must be demonstrated by actually compressing the group to that rank at a fixed error tolerance.

The older cofactorization, hard scalar, and CINDER pilots likewise do not support returning to those routes at the proposed compute budget. For example, the 1,024-component CINDER pilot still has mean KL 0.276 when 1,014 components are retained, on just 24 evaluated events. That small pilot is evidence against that implementation, not a general disproof of curvature-based decomposition.

## 3. What the repair results do and do not show

For the **published** VPD checkpoint, damaging layers 0-2 with their 1%-retained VPD masks gives KL 0.468 with real layer 3. Masking layer 3 as well gives 0.316. Layer-3 masking alone gives 0.296. Thus the incremental upstream-damage metric changes from 0.468 to approximately 0.0204.

With matched residual-noise damage, the corresponding numbers are 1.062 with real layer 3 and 0.646 with VPD-masked layer 3; subtracting its 0.296 standalone error leaves 0.350. With our masked layer 3 the combined noise KL is 2.228, versus 0.864 for its standalone mask.

These results support the narrower statement that **the selected VPD downstream computation can suppress upstream damage substantially more than the intact downstream computation in these tests**. The hook implementation matches VPD's own masked forward exactly on the recorded parity check. Both published and paired VPD checkpoints show relevant effects. [Repair results](out/repair/vpd_published.json); [implementation](runpod_compare/repair_probe.py).

They do not establish that the real model has no self-repair, that ours is faithful for arbitrary interventions, or that every VPD component is artificial. KL differences are a useful interaction contrast, not an additive decomposition of causal damage. The effect is not uniform across damage types: for published VPD, matrix-output noise remains very destructive, and some random-mask effects are much less improved. Report raw four-condition results as well as differences, with uncertainty across sequences and noise seeds.

There is a particularly relevant claim in VPD Appendix B.1, PDF pages 69-70: the authors distinguish compensatory activation replacements from VPD and report stability across replacement protocols. Your direct cross-layer damage/repair experiment is a useful stress test of that interpretation. Their experiments there concern MLP replacements; our all-matrix tests should include an MLP-only version before presenting a direct contradiction.

The strong-adversary failure itself is already disclosed in the paper: Table 4 and Section 7.5 report KL increasing from about 0.83 at 20 PGD steps to 40.2 at 320. Our shared-adversary native-CI replication rises from 0.685 to 18.20. That confirms incomplete robustness, but is not a newly discovered omission. Ours also fails badly under sufficiently strong adversaries at low retention. [Recorded adversarial results](out/adversarial/vpd_published_global_128.json); [our control](out/adversarial/parauto_1b_global_128.json).

Two cheap follow-up controls would sharpen the repair result:

1. Repeat with native positive-CI support and the allowed continuous mask interval, as well as the existing binary top-k masks. A top-k mask can remove a positive-CI component and thus exceed the method's stated allowed intervention set.
2. Run a prefix/suffix invariance test. The appendix explicitly specifies a bidirectional CI transformer reading all 24 clean activation streams. Change only the suffix after the prediction under study, which cannot change the intact causal model's earlier logits, then measure mask and reconstructed-prefix-output changes. This tests dependence on extra context in the selector; it does not presume such dependence is a software bug. Our CE/Fisher source-position attribution also aggregates effects on future predictions, so a terminal-position or explicitly restricted-output comparison is needed for fairness.

## 4. Two interpretation errors in the current diagnostics

### Temporal interactions are omitted from the “full” joint metric

`capture_fisher` differentiates a **sum over prediction positions**. Let z[b,s,k] be the signed effect of the removed components at source position s, for probe k, already summed across matrices. The current [joint diagnostic](runpod_compare/joint_diagnostic.py) and [joint loss](parauto_method.py) square at each source position and then average:

    existing:  mean_(b,s,k) z[b,s,k]^2

For the full simultaneous sequence intervention, the quadratic Fisher prediction instead requires:

    correct full proxy:  mean_(b,k) (sum_s z[b,s,k])^2 / T

with the appropriate one-half for KL and normalization for prediction count. Sum within each sequence; never across unrelated batch elements. The existing implementation includes cross-matrix interactions at a common source position but omits cross-source-position terms. This is a substantive difference in a transformer. The empirical masked curves are still measurements; interpreting the predictor's discrepancy as purely nonlinear, or its failed training run as a rejection of the complete joint Fisher metric, is unsupported.

A corrected joint training objective would still explicitly select removed sets and would not be my proposed no-mask training method. Fix the diagnostic first, and label the existing training results accurately.

### The “ceiling” is not a mathematical ceiling

In [ceiling_probe.py](runpod_compare/ceiling_probe.py), the “oracle” is greedy selection, not the best possible subset. A shared rank-k SVD approximation is optimal among shared rank-k linear approximations under aggregate squared error; it is **not** a lower bound for an overcomplete dictionary with a different selected subset at each input. Moreover, the reported statistic is a median relative error, not that SVD objective. The measured results still show that those particular cheap selection/refit procedures leave considerable local error. They do not prove that sparse exact-sum dictionaries are impossible.

## 5. Proposed next objective: Gaussian Fisher sparsity

The most useful opportunity is that one ordinary backward can represent a vector-valued output metric in expectation, including under a concave sparsity power. This avoids both the gold-label direction and repeated sampled-label backwards.

For a single prediction position, let pi = softmax(logits), treated as constant during differentiation, and draw independent standard Gaussian entries xi. Define

    v = sqrt(pi) * xi - pi * sum(sqrt(pi) * xi).

Then E[v] = 0 and Cov(v) = diag(pi) - pi pi^T, the categorical Fisher metric in logit space. Construct the scalar probe

    Phi = sum_(batch, prediction positions) stop_gradient(v)^T logits.

Backpropagate Phi once through the intact frozen model and cache h and g at each decomposed matrix, exactly as the scalar pipeline already does. No component is removed or rescaled in this forward pass. No gradients through the frozen model are retained during factor fitting.

For component c at one matrix/source-position invocation, let d_c be its first-order change in all measured logits and let

    a_c = (r_c^T h) (u_c^T g),
    s_c^2 = sum_t d_(c,t)^T [diag(pi_t) - pi_t pi_t^T] d_(c,t).

Conditioned on the context and factors, a_c is Gaussian with variance s_c^2. Therefore, for p > 0,

    E[ |a_c|^p / kappa_p ] = s_c^p,
    kappa_p = 2^(p/2) Gamma((p+1)/2) / sqrt(pi).

**One fresh Gaussian probe gives an unbiased estimator of the full Fisher-norm-to-the-p objective.** This is an expectation over training probes, not a claim that one draw accurately ranks every component on one token.

That detail distinguishes this from simply setting the existing sampled-label Fisher K to 1. Discrete label gradients have the correct covariance but their p-th absolute moment is not determined by that covariance. Gaussian projections have the needed rotational symmetry. The earlier eight-Rademacher logit-RMS pilot also uses a different output metric and takes a nonlinear power of a finite norm estimate.

The stochastic loss is

    L = sum_m [ fixed_scale_m(p)^(-1)
                E_context sum_c |(r_c^T h)(u_c^T g)|^p / kappa_p ],
    subject to U_m R_m^T = W_m.

Retain the existing p annealing, projected Adam, factor count and reference normalization initially. Resample probes for every fresh context/update. Use ordinary derivatives for the Fisher interpretation. A Gaussian+GIM control can isolate the effect of changing the output seed, but its propagated metric is a modified-backward metric, not the true model Fisher.

### Numerical qualification

The exact moment identity is for the unsmoothed power. At p <= 1/2 its naive stochastic derivative can have infinite variance near zero; even at larger p the cusp can be inconvenient. Use a smooth form such as `(a^2 + eps^2)^(p/2) - eps^p`, with scale-aware epsilon. That deliberately changes the exact objective near zero. Check agreement away from epsilon and robustness to epsilon; do not claim exact unbiasedness for the existing epsilon-smoothed implementation. The initial per-matrix normalizer should use a stable reference estimate, not a fortuitously small single probe.

### Compute qualification

This uses one intact target forward, one target VJP, two factor-scoring matmuls per matrix, the existing factor backward and projection. Gaussian sampling and the vocabulary-sized probe require additional elementwise work and memory; chunk the probe construction. GIM also has custom softmax/RMS backward overhead that the ordinary derivative variant avoids. It is plausible to match GIM wall time, but not yet measured. Treat “no slower than GIM” as an acceptance test on matched hardware, including cache collection, projection and probe generation, not an established result.

Stochastic curvature backpropagation and projected Jacobian estimates have substantial prior art; this should not be presented as inventing random probes. The specific proposed use here is the Gaussian absolute-moment identity for exact-sum concave factor sparsity. Related primary sources: [BackPACK](https://openreview.net/forum?id=BJlrF24twB), [K-FAC](https://arxiv.org/abs/1503.05671), and [GIM](https://arxiv.org/abs/2505.17630).

## 6. Add intermediate computation to the same backward

An endpoint-only metric can still overlook locally important effects that later computation suppresses. The next controlled extension is to add independent Gaussian observations of a few **internal nonlinear boundaries**, not every raw matrix output with an equally strong norm penalty.

Choose the post-attention and post-MLP residual boundaries of the four blocks. Let b_j denote each boundary's activation tensor and sigma_j^2 a fixed, positive activation scale estimated from calibration data. Draw independent standard Gaussian tensors eta_j and use

    Phi = sum_t v_t^T logits_t
          + sqrt(lambda) sum_j eta_j^T b_j / sigma_j.

All terms share the same backward. Its component-effect variance is

    s_c^2 = s_(c,output)^2
            + lambda sum_j ||delta b_(j,c)||_2^2 / sigma_j^2.

The Gaussian moment identity still applies. Independence means random cancellation between different probe terms disappears in expectation. A piece whose first-order effect matters at an intermediate boundary is now visible even if the final output is insensitive. Measure boundary tensors on the original computation, after the relevant attention/MLP nonlinearity; this is not training a replacement network to reconstruct hidden states.

This remains a local sensitivity objective. It does not guarantee finite-removal or adversarial faithfulness, and it may increase the number of pieces required if the model's intermediate computation is genuinely dense. That is why the endpoint-only arm comes first and lambda must start small. Plain write-norm training already failed; making local Euclidean preservation dominate would likely repeat that failure.

The intended benefit is a sparsity loss defined on a richer description of the original computation, without extra target backwards. The final KL curve and repair tests must determine whether that helps. I would not promise a two-order-of-magnitude L0 improvement from a better sensor alone.

## 7. Experiment order and decision criteria

1. **Validate the estimator and audit the metric before training.** On a tiny attention model compare the analytic Fisher norm and many Gaussian draws, with and without boundary probes. Check the existing scalar cache/scorer gives the same directional derivative. Validate the corrected sequence-summed joint diagnostic against a tiny finite displacement or explicit Jacobian; this is evaluation-only validation. Check projection residuals in FP64 and all-kept forwards in FP32.
2. **Benchmark on one GPU with identical shapes.** GIM scalar baseline, Gaussian Fisher, and Gaussian Fisher with boundaries. Include full update time and peak memory. If it exceeds GIM's cost, optimize or reject this implementation before a long run. Do not hide extra cost behind smaller batches.
3. **Run a small matched experiment.** First Gaussian Fisher versus the existing GIM recipe, then a weak-boundary version only if numerics and throughput work. Match seed, context stream, 5,000 updates and 8,700-step schedule; compare to the existing Fisher K=4 reference too. A Gaussian+GIM control is useful if it is unclear whether the gain comes from the output seed or the true backward. Finish a schedule and add seeds only for the surviving variant.
4. **Evaluate actual retained rank-one counts.** Use an absolute grid such as 128, 256, 512, 1,024, 2,048 and 4,096, with finer sweeps around crossings. Use the same high-quality independently sampled evaluation sensor on our checkpoints; retain VPD's native CI result and a common-sensor comparison as separate questions. Choose thresholds on calibration contexts and report on a new untouched test set, not only the validation file used repeatedly to choose experiments.
5. **Report a frontier, not one L0.** Measure retained count at fixed KL targets (for example 0.1, 0.3 and 1.0), CE degradation, sequence-level error tails and confidence intervals. Do not assume a monotone VPD curve or interpolate a nonexistent crossing. Report native positive-CI support separately. Repeat repair and a modest, explicitly defined shared-adversary test at matched base error and complexity.
6. **Test reduced data dependence directly.** Use progressively smaller, diverse calibration/training context sets at fixed update and wall-clock budgets; split held-out source domains or behavior families where possible. A label-free objective removes one source of dependence, not dependence on which circuits the contexts exercise. Exact-sum constraints preserve the weights of unseen behavior, but do not guarantee those mechanisms have been separated into interpretable components.

Define the target as lower primitive L0 at the same behavioral error **and** a stated intervention-fidelity standard. Merely crossing the published native-CI L0 while hiding a dense matrix inside each component would not meet the intended goal. If a method improves fidelity but needs more pieces, that is a useful different result, not evidence of lower description length.

I would defer more ownership/rank sweeps, stronger p alone, the old curvature-tomography proposal, and another large data run until this experiment is resolved. The strongest next bet is better vector-valued measurement at the existing one-backward budget, coupled to a comparison that cannot be won by grouping or downstream compensation.

## Paper source

Read from `/Users/rohan/Downloads/www-goodfire-com-research-interpreting-lm-parameters#....pdf`, including Appendix A.1-A.7 and B.1. Public version: [Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters). The claims about VPD's limitations, architecture and objectives above are from the authors; the Gaussian sparsity proposal and the code-audit conclusions are this review's analysis.
