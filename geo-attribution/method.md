# Training-Free Parameter Decomposition from Attribution Geometry — Method

This document states the method twice: first the raw poster form (what the method *is*), then the advanced form (what we actually compute, and why each implementation choice exists).

---

## Part 1 — Poster form

### The object

Everything operates on one matrix: a row per input, a column per parameter.

$$V_{x,j} = \left[\nabla_\theta\, \ell(x) \odot \theta\right]_j$$

$V_{x,j}$ reads: how much input $x$'s loss depends on parameter $j$, scaled by the parameter itself — the attribution of parameter $j$ on input $x$.

### Step 1 — cluster the rows

Spherical k-means on the centered, normalized rows of $V$ yields $C$ clusters of inputs $S_1, \dots, S_C$.

Two inputs land in the same cluster iff they rely on the same parameters with the same signed pattern — a mechanism, identified behaviorally, with no labels and no training.

### Step 2 — closed-form ownership

Each parameter is split among the clusters that use it, in proportion to mean attribution mass:

$$s_c(j) = \frac{\bar m_c(j)}{\sum_{c'} \bar m_{c'}(j)}, \qquad \bar m_c(j) = \frac{1}{|S_c|} \sum_{x \in S_c} |V_{x,j}|$$

Component $c$ is the weight slice $\theta_c = s_c \odot \theta$, and because the shares of every parameter sum to one:

$$\sum_c \theta_c = \theta \quad \text{exactly — faithful by construction, nothing trained.}$$

### Step 3 — per-input gates

A component is active on input $x$ when its slice captures a nontrivial fraction of $x$'s attribution:

$$a_c(x) = \Big(\sum_j s_c(j)\, V_{x,j}\Big)^2, \qquad \text{gate}_c(x) = \mathbb{1}\!\left[a_c(x) > \tau \max_{c'} a_{c'}(x)\right]$$

### The claim

Running the model with only each input's gated components' slices approximates the full model (KL per token, at tens of active components out of $C$); all components on recovers it exactly; all off destroys it.

*Footnote: $V$ ($N \times \sim 10^9$) is never materialized — for moderate $N$, row similarities come from an exact kernel identity on cached activation/gradient pairs; for large $N$ (up to $10^6$ positions demonstrated), from unbiased importance-sampled features of the same kernel; ownership uses the same factored statistics throughout; see Part 2.*

---

## Part 2 — Advanced form

### 2.0 Notation and scope

The decomposed parameters are the entries of the model's linear weight matrices $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ (67M pile-4L: 24 matrices; Llama-3.2-1B: 112 matrices, ~968M entries); embeddings and norms stay in the shared backbone.

An "input" $x$ is a single token position in context; its loss $\ell(x)$ is the summed CE of the sequence's future predictions, so a position's attribution reflects everything it contributes downstream.

### 2.1 The sensor (how $V$'s entries are defined)

For a linear layer, the gradient of the loss with respect to $W$ at one position is a rank-1 outer product, which is the structural fact the whole pipeline exploits:

$$\frac{\partial \ell}{\partial W} = g\, p^\top, \qquad p = \text{layer input}, \quad g = \frac{\partial \ell}{\partial (\text{layer output})}$$

Raw gradients at the trained weights under-report saturated mechanisms, so the sensor is a $K$-point integrated-gradients path over weight scale (default $K{=}2$: passes at $0.5\theta$ and $\theta$, averaged) — Aumann–Shapley credit rather than local slope.

The optional GIM variant (arXiv 2505.17630) replaces the backward rules at softmax (temperature-adjusted jacobian, $\tau{=}2$) and RMSNorm (frozen statistics) — forward-identical, sensor-only; it improved induction preservation 65% at 67M and is off by default.

The differentiated scalar is the summed ground-truth CE by default; note that the ground-truth log-prob is its exact negation and therefore a no-op (signs cancel in squared attributions and pairwise products) — the implemented alternatives that genuinely change the sensor are the model's argmax-token log-prob ("what it says" vs "its error") and the raw argmax logit (saturation-free).

**Collect stage:** stream text, sample positions (16 per sequence), run the $K$ scaled forward–backward passes, and cache $(p, g)$ per matrix per position in bf16 — ~700k numbers per position instead of the ~$10^9$-entry attribution row, and the only stage that touches the model's autograd.

### 2.2 Row similarities without materializing $V$: two regimes

**Exact regime ($N \lesssim 65$k).** Clustering needs the $N \times N$ matrix of row inner products, and the outer-product structure gives it exactly:

$$\langle v_i, v_j \rangle = \sum_{\text{matrices}} (g_i \odot g_j)^\top\, W^{\circ 2}\, (p_i \odot p_j)$$

with IG handled by averaging the $K^2$ cross-terms of the scaled passes; computed as blocked matrix products, matrices sharded across GPUs and all-reduced; cost is $O(N^2 \cdot \text{entries})$ (49 min for $N{=}16{,}384$ at 1B on 2×H100).

**Feature regime (large $N$; $10^6$ demonstrated).** The kernel is a sum over weight entries, so importance-sampling $D$ coordinates gives explicit features whose dot products equal the kernel in expectation, collapsing the pairwise cost to $O(N \cdot D)$:

$$k(x,x') = \sum_{o,i} W_{oi}^2\, g_o g'_o\, p_i p'_i \;\Rightarrow\; \phi_r(x) = s_r\, g_{o_r}(x)\, p_{i_r}(x), \qquad \mathbb{E}[\phi(x)^\top \phi(x')] = k(x,x')$$

The proposal distribution is critical: sampling $\propto W^2$ alone fails under the kernel's heavy tails (and gets *worse* with more features — a draw-lottery estimator); reweighting by cached activation statistics, $q \propto W^2\, \mathbb{E}[g^2]\, \mathbb{E}[p^2]$, achieves 0.99 raw / 0.96 normalized correlation against the exact gram at $D{=}16{,}384$ — always validated against an exact gram on a held-out subset as a hard abort gate before a large run commits.

Features cost almost nothing to compute (per position: gather $D$ coordinates of $g$ and $p$ during the collect pass already being run) and stream to a $[N, D]$ fp16 memmap.

Because features cannot supply full per-coordinate extraction statistics, large-$N$ runs use two tiers: all $N$ positions contribute fingerprints for clustering, while a spread subset (~32–64 positions per intended cluster) stores complete $(p, g)$ in standard shard format so extraction stays exact and unchanged.

### 2.3 Factorization (Step 1 in practice)

Double-center the gram (centers the rows of $V$ in feature space), then correlation-normalize by the diagonal (cosine geometry).

Eigendecompose — full `eigh` when $N \le 8192$, otherwise randomized subspace iteration with Rayleigh–Ritz for the top ~1.5k pairs (cusolver fails at these sizes) — and embed each position as its top-$C$ eigencoordinates scaled by $\sqrt{\lambda}$.

Spherical k-means (k-means++ init with running-max trick) gives the labels; in the feature regime the same pipeline runs as PCA of the centered features to $C$ dimensions (mathematically the same spectral embedding) followed by spherical k-means — sub-minute at $N{=}10^6$ once features are on-GPU.

The attribution spectrum is a smooth power law at both 67M and 1B, so $C$ is a resolution knob chosen by the user, not discovered by an eigengap.

$C$ and $N$ must scale together: fine components are statistics-hungry — at 1B, C=2048 with ~6 positions/cluster produced entangled, unselective components, while the same C fed by $N{=}10^6$ (~460 positions/cluster) recovered edit selectivity from 2:1 to 8–11:1 (coarse C=512 bundles reach 80:1); the residual coarse-vs-fine gap is the genuine granularity tradeoff, the rest was starvation.

### 2.4 Extraction (Step 2 in practice)

The per-entry, per-cluster mass specializes the poster's $\bar m_c(j)$ using the same rank-1 structure — per-capita mean over the cluster's cached vectors:

$$\bar m_c(o, i) = \overline{|g_o|}^{(c)} \cdot \overline{|p_i|}^{(c)} \cdot |W_{o,i}|$$

Each entry keeps its top $s{=}8$ clusters by mass and splits proportionally (`extract_ps`); shares are stored as (int16 owner ids, fp16 shares), ~$2(s{+}s)$ bytes per entry.

Sum-to-one per entry gives exact faithfulness with zero always-on base; positivity of the shares means components are rescalings of $W$, so gated sums cannot cancel — the failure mode that kills naive low-rank/ridge extraction.

Ablation variants retained in the codebase: hard argmax ownership with a specificity-thresholded base (`extract_p`, the base-mass-vs-KL frontier knob), and a gate-calibrated share solve (`extract_pg`, improves gated KL ~2–3× but degrades edit selectivity; dropped from the method).

### 2.5 Gating (Step 3 in practice)

At eval time the per-token, per-component attribution is the IG-averaged inner product of the token's $(p, g)$ with the component's slice, squared:

$$a_c(t) = \Big(\sum_{\text{matrices}} p_t^\top (s_c \odot W)^\top g_t\Big)^2$$

Default gates are binary at a relative threshold ($\tau{=}0.02$ of the token's max) — the only gate values the sum-to-one calibration is exact for.

Graded variants (evaluated at 1B), both saturating at fully-open so dominant machinery is never attenuated (raw proportional gates fail: openness that sums to ~1 shrinks every entry and compounds across layers): ramp $g_c = \min(1, (a_c / \tau a_{\max})^\gamma)$, and the parameter-free participation-ratio rule $g_c = \min(1, \mathrm{PR} \cdot a_c / \sum a)$ with $\mathrm{PR} = (\sum a)^2 / \sum a^2$ the token's effective component count (compute on max-normalized attribution to avoid fp32 overflow).

Verdict: all gate shapes land on approximately one openness-vs-KL curve — the attribution *ordering* does the work and the gate shape is a budget knob, so binary is not distorting and remains the default; the PR mode's value is as a canonical parameter-free operating point (the attribution's own effective count, far sparser than behavioral fidelity requires — that gap measures decomposition quality, not gating).

Empirically $a_c(t)$ is heavy-tailed: ~7–13 components within 10% of the token's max, a smooth slope to ~2%, and a floor ~1000× down (measured at 1B, $C{=}512$).

### 2.6 Evaluation and referees

Fidelity: KL per token between the gated model and the target on held-out text, reported against gate openness (gates/token, or summed gate mass for graded modes), plus a keep-top-$j$ curve (rank-based gates at fixed budgets).

Controls: all-ones gates must reproduce the target to fp32 exactness (faithfulness identity, ~$10^{-8}$ KL at both scales); zero gates must destroy it (the components, not the backbone, carry the computation).

Causal referees: German-erasure edits — rank components by German-vs-English attribution-share contrast on one sentence split, ablate/invert in gate space or scale owned shares permanently in weight space, measure per-token CE deltas on held-out splits of both languages against random-$k$ controls; and a synthetic induction canary (repeated random tokens, copy accuracy under gating and ablation).

Key weight-space property: because ownership is (near-)disjoint, deleting random components' owned mass is behaviorally harmless at every operating point tested — permanence without ablation-robustness training.

**Editing primitives** (all permanent weight surgery on owned shares, no decomposition at inference): *scaling* $w \mapsto (1 - (1-\alpha)\, s_c)\, w$ gives deletion ($\alpha{=}0$), graded dampening ($0{<}\alpha{<}1$, a clean "volume knob"), and inversion ($\alpha{<}0$, deepest erasure; damage saturates past $\alpha{=}{-}1$ while collateral keeps rising); *additive steering* adds a norm-matched rank-1 write toward a target token's unembedding on a narrow component's write-side slices, keeping its function — the clean behavior-injection primitive; *replacement* of a slice's write action works but caps at the function being destroyed.

Editing lessons: amplifying a component's weights does NOT amplify its behavior (firing is calibrated, not proportional to weight norm — amplification breaks the model); attribution-contrast selection finds components that *correlate* with a behavior but are not always *causal* for it (a valence-correlated component was causally inert under redirect/delete at every granularity), so token-level choices routed through a shared slot are flipped by biasing the slot's output competition, not by surgery on correlated components.

### 2.7 Cost and scale engineering

The full 67M pipeline is under one H100-hour; the 1B exact-gram pipeline ($N{=}16$k) is a few H100-hours (collect ~10 min, gram ~50 min ×2 GPUs, factor ~10 min, extraction ~15 min, evals dominated by gated forwards at ~1 min each).

The 1B feature pipeline at $N{=}10^6$ is ~45 min of compute on 2×H100 (collect 24 min at 64 positions/seq — sampling many positions per sequence amortizes the passes, cluster <1 min, extraction ~15 min on the subset); everything remaining is linear in $N$, so $10^7$ positions is hours and the binding constraints become feature storage and corpus size, not algorithm.

Referee/eval cost scales linearly in $C$ (component-slice materialization), which is what makes fine-$C$ behavioral evals the slow item (~4× at C=2048 vs C=512).

Gated forwards and attribution use chunked einsums over component slices built on the fly from the (ids, shares) encoding — no per-component weight copies are ever stored.

At 1B the working set is dominated by the share encoding (~31 GB on GPU); memory hygiene that matters: shape-shared zero residuals, halved component chunks, small eval batches, expandable allocator segments.

DDP layout: corpus-sharded collect, matrix-sharded gram with all-reduce, and independent per-GPU arms for evals and referees.
