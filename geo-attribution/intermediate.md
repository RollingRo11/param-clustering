# Intermediate findings: training-free parameter decomposition on pile-4L (67M)

**Status:** working notes, 2026-08-02. Covers everything from the first implementation through the editing/interp arc on the VPD paper's pile-4L target. The N=65k / C-sweep run (`full65`) is in flight and not yet reported here.
**Code:** `geo67.py` (pipeline), `autointerp67.py`, `german67.py`, `sweep65.sh` in this folder. Artifacts in `out/full/`. Method rationale in `proposal.md`.

---

## 1. Setup

Target: the VPD paper's pile-4L model — 4-layer LlamaSimpleMLP, d_model 768, d_mlp 3072, vocab 50257, ~67M params, Pile-trained. We decompose the 24 attention/MLP linear matrices (q,k,v,o,c_fc,down × 4 blocks); embeddings and norms are untouched. Loaded from the local pretrain cache via a symlink layout that satisfies `PretrainRunInfo.from_path`'s local branch (the wandb key on disk is stale; no network needed).

Pipeline (all training-free): **collect** — K=2 IG passes per batch at scaled weights (k/K)·θ, caching per-matrix input activations p and output cotangents g at 16 sampled positions per 512-token Pile sequence, corpus sharded across both H100s; **gram** — exact signed attribution gram via the pairwise identity (per-matrix block matmuls, all four IG cross-terms, modules sharded across GPUs); **factor** — double-center + cosine-normalize, top eigenpairs, spherical k-means; **extract** — components as weight objects summing exactly to W; **gate** — per-token IG attribution shares, hard threshold; **referees** — behavioral eval, induction canary, edits.

Baseline run scale: N = 32,768 token positions (~2,050 sequences). Wall-clock on 2×H100: collect ~5 min, gram ~12 min, factor+extract minutes. The whole decomposition costs **well under 1 H100-hour**, vs ~16–30 H100-hours per trained decomposition run on this same target.

Two infrastructure notes worth keeping: cusolver's `syevd` fails at N=32k with a workspace error on a perfectly clean matrix — replaced with randomized subspace iteration + Rayleigh–Ritz for the top ~1.5–2.3k eigenpairs; and the attribution-gram spectrum of this model is a **smooth power law** (head eigenvalue 1135; index 256 → 11.9; 512 → 6.6; 1024 → 3.6) with no eigengap — every "candidate C" the gap heuristic proposes (16–42) is a head-of-spectrum artifact. C is a resolution knob, not a discovered constant, consistent with the trained runs' ~178-292 gates/token finding that LM tokens are many-mechanism.

## 2. Negative result: low-rank ridge extraction fails under gating (cancellation)

The proposal's Stage 3 (anchored rank-m pieces via joint ridge solve) produces exact faithfulness (max identity error 1.4e-6; all-gates-on KL ≈ 1e-9) but a **behaviorally dead decomposition at every ridge strength**:

| ridge λ | resid mass | full-gate KL/tok | keep-top-256 | resid-only KL | copy acc |
|---|---|---|---|---|---|
| 1e-3 | 0.000 | 12.98 | 12.61 | 67.1 (= off baseline) | 0.00 |
| 0.1 | 0.106 | 12.96 | 12.21 | 10.5 | 0.00 |
| 1.0 | 0.423 | 10.36 | 8.95 | 10.5 | 0.00 |

Diagnosis: with C·m anchor columns ≫ d_side, overlapping components carry large mutually-cancelling mass. Gating a component off is weight subtraction, so partial sums are catastrophically unbalanced (low λ), and at high λ the gated components contribute ≈ nothing beyond the residual. More gates never helps — components do not compose. The trained decompositions were protected from exactly this by the recon-through-gates loss; nothing in a closed-form solve exercises partial sums. **Overlapping low-rank pieces are structurally wrong for training-free gating.**

## 3. The fix: partition extraction (each weight entry belongs to exactly one component)

Replace the ridge with an assignment: every entry of every W goes to the cluster with the largest per-capita attribution mass on it (computable as one |g|ᵀ|p| matmul per cluster, times |W|); entries with a flat cluster profile (max/mean < base_ratio) go to one always-on **base** component. Components are disjoint entry-masks — faithfulness is exact by construction, and **no cancellation between components is possible**. This also lands on the design principle from the original project variant: each piece of each matrix belongs to exactly one component.

Result — components now compose monotonically, and the base_ratio knob traces a clean depth-vs-fidelity frontier (C=512, gate threshold 0.1 unless noted):

| arm | base mass | comp mass | full-gate KL @ gates/tok | top-8 / top-64 / top-256 KL | base-only KL |
|---|---|---|---|---|---|
| r3 | 12.5% | 87.5% | 7.89 @ 21 | 8.5 / 5.9 / 3.07 | 22.1 |
| r5 | 63.5% | 36.5% | 6.08 @ 14 | 6.5 / 3.3 / 0.68 | 10.0 |
| r8 | 91.5% | 8.5% | 2.16 @ 10 | 2.0 / 0.47 / **0.07** | 4.5 |

Headlines: **r8 keep-top-256 KL = 0.07/token — inside the trained runs' full-gate band (0.077–0.085) — training-free, at <1% of their compute.** At a looser gate threshold (0.02) the r8 full-gate KL is 1.08 at 34 gates/token. The base is certifiably insufficient alone (+4.5 nats), so components carry real function; and the off baseline (all decomposed matrices zeroed) sits at ~67 nats, so all controls have headroom. The honest reading of the frontier: the more weight mass you insist the components own exclusively, the more gating costs — r8 is a "distinctive-mass decomposition over a shared base," the LM analog of split_v2's toy regime.

## 4. What the decomposition cannot do: induction (shared with the whole family)

The target's synthetic-repeat copy accuracy is 0.839; the r8 gated model reaches only 0.114 (threshold 0.02), and ablating the top-8 contrast-ranked components (0.061) barely separates from random-8 (0.102). Cross-position retrieval is not owned by any gateable component — the participation-not-ownership failure we previously documented for our trained runs and for VPD reproduces training-free. This strengthens the claim that it is a property of the objective/representation family, not of training; per-token gates plus first-order-ish credit have no home for two-position QK interactions.

## 5. Interpretability: read-side, at trained-or-better rates

Zero-cost probe (components are clusters of positions; we stored current/next token ids): 263/512 clusters (51%) have ≥0.5 purity on current- or next-token identity; ~29% are incoherent at that granularity.

Full auto-interp (600k real-Pile positions, top-16 contexts/component, sonnet-4-6 labeling): **347/512 high-confidence (68% overall = 81% of the 430 live components)**, 58 polysemantic/unclear (11%), 82 dead, 5 API errors. Category mix: next-token-prediction 247, current-token 86, boundary 19, semantic-topic 10, other 147. Reference points from the trained decompositions of this same model: entropy run 29% high-conf / ~21% poly; sum run ~6% poly but ~1300/2048 dead. The training-free catalog has the best interpretability rate of the three — with the caveats that C=512 components are coarser (easier to label) and the evidence base was smaller (600k vs 48M positions).

The ontology is the familiar one: c350 sentence-final period, c306 end-of-sentence boundary, c20 paragraph-break newline, c352 "the"-prediction, c308 "of"-after-noun-phrase, c123 "and"-continuation, c134 possessive apostrophe, c174 digits, c99 opening brace — plus c192 "proper-name self-reference/repetition prediction" (an induction-flavored read-side component). Next-token purity partly reflects the sensor (CE gradients are dominated by output identity), so some components are "predicting X" summaries rather than distinct mechanisms — the causal edits below are the check on that.

## 6. Causal verification: German erasure, gate space

Protocol: 24 German + 24 content-matched English sentences; components ranked by attribution-share contrast on one half (selection), edits evaluated on the held-out half; random-k controls ×3 seeds. Caveat: small eval slices (~200 tokens/language) — directionally unambiguous, but a paper figure needs real Pile language slices.

| edit (gate space) | ΔCE German | ΔCE English | random-k de / en |
|---|---|---|---|
| k=4, α=0 (ablate) | +0.261 | +0.006 | +0.007 / +0.005 |
| k=4, α=−1 | +0.537 | +0.018 | +0.015 / +0.027 |
| k=4, α=−2 | **+1.004** | **+0.046** | +0.033 / +0.070 |
| k=8, α=−1 | +4.60 | +6.09 | — |

Findings: (a) surgical at k=4 — 37× over random with English at noise; (b) **inversion replicates** — negative scalars deepen erasure ~4× with English still below the random control (the directionality phenomenon from our induction editing and from VPD's editing); (c) the depth ceiling is **selection specificity, not mechanism** — components ranked 5+ are general machinery (see §8), and inverting them destroys both languages.

**Triangulation closed:** the top-2 contrast components, c292 and c267, were independently labeled "multilingual inflectional suffix completion" and "German/multilingual suffix continuation" by the auto-interp pipeline, which never saw the German experiment. Contrast ranks 5–6 are c306 (sentence-final punctuation) and c365 (LaTeX brace) — general components that fire slightly more on the German text — which precisely explains the k≥8 collateral. At C=512 there are only ~4 genuinely German components.

## 7. Weight-space editing works — permanence without ablation-robustness training

Because partition components are disjoint entry sets, "delete component c" is well-defined surgery on the raw weights: zero (or scale) those entries, run the **plain unedited-architecture forward** — no gates, no decomposition machinery at inference.

| edit (weight space, permanent) | ΔCE German | ΔCE English | random-k de / en |
|---|---|---|---|
| delete top-4 comps' entries | **+0.222** | **+0.018** | **−0.002 / +0.001** |
| invert top-4 (α=−1) | +0.506 | +0.062 | +0.028 / +0.026 |
| delete top-32 | +1.92 | +0.65 | +0.128 / +0.086 |
| invert top-8+ | +5.7…6.8 | +8.8…9.1 | — |

Two results here. First, weight deletion tracks the gate edit almost exactly — the assigned entries really are the German mechanism's weights. Second, and more important: **random-component weight deletion is harmless (−0.002 nats)**. Every previous weight-space attempt in this project failed catastrophically (random-2 subtraction on our trained artifacts: +3.9 nats and induction dead), and we had concluded that VPD's stochastic mask-sampling training was *the* prerequisite for weight-space separability. It is not: **disjointness is a sufficient substitute, and the partition gets it by construction, for free.** This upgrades the method from "gate-space analysis tool" to "produces permanently editable models," and should be promoted from scope-caveat to headline in the write-up.

## 8. Comparison to VPD-family fine-tuning-by-mask (LessWrong: "Fine-tuning with parameter decomposition")

Same 67M target. Their edit: gradient-tune VPD mask scalars — the winning single scalar is one rank-1 atom (`h.3.attn.v_proj:513`, selected via its "German text and names" auto-interp label), trained on ~4 German tokens, converging to α ≈ −1 (inversion), driving German CE to chance (~+6.5 nats) with <0.1 nats English damage; rank-1 LoRA needs ~32 tokens for the same.

Structurally near-identical to ours: same selection principle (causal-importance contrast ≈ our attribution-share contrast), same sparse-ownership finding (a handful of units carry German), same collateral mechanism (their contrast-top-16 was 14/16 "foreign languages in general" and damaged French/Spanish; our ranks 5+ are general components and damage English). Quantitatively: our mechanism-matched best (k=4, α=−2, zero training anywhere) reaches German +1.0 / English +0.05 — ~1/6 of their erasure depth. The residual gap decomposes into their finer unit granularity (30.5k per-matrix rank-1 atoms → one *purely* German atom exists; our 512 cross-layer components bundle German with multilingual function) plus their gradient-tuned scalar magnitude. Granularity is a knob (C), not a wall — this is the hypothesis the in-flight C-sweep tests directly. Our durable edge is cost: their edit rides on a full VPD training run; our entire pipeline is <1 H100-hour. Their durable edge was weight-space permanence — which §7 now matches.

## 9. Cost accounting

Decomposition: collect ~5 min + gram ~12 min (both 2×H100) + factor/extract minutes ≈ **<1 H100-hour total** at N=32k, C=512. Trained decompositions of this target: ~16–30 H100-hours per run, plus loss-balancing fragility (the faith-weight mis-scaling incident produced a certified-looking non-decomposition). Referee battery per arm: ~15 min. Auto-interp: ~10 min evidence + ~$ single-digit API labeling. The N=65k gram is ~50 min (quadratic in N — the one superlinear stage).

## 10. Open problems and current caveats

- **Induction / cross-position mechanisms unowned** (§4) — shared with the whole family; QK-aware credit or pair-gating remains the candidate fix, untested here.
- **Sparsity–fidelity at threshold gating:** full-gate KL 1.08–2.2 is behavioral-tier, not faith-tier; keep-top-j is where the decomposition shines. Gate calibration (soft gates, per-token j) unexplored.
- **The base is a confound knob:** r8's components own only 8.5% of mass. The frontier is honest, but "decomposed the model" claims must state the base mass alongside fidelity.
- **Attribution gates need a target backward at inference** — evaluation-time cost, and gates are condition-differential (a mechanism equally active everywhere is invisible to contrast ranking; the German ranking worked because language use *is* differential).
- **Small German eval corpora; single seed; single target model.** Also next-token purity may partly reflect CE-gradient output-identity dominance rather than mechanism identity (§5); the causal edits mitigate but don't eliminate this.
- **In flight (`full65`):** N=65,536, C ∈ {512, 1024, 2048} × base_ratio {5, 8}, full referee battery per arm. Key readouts: does larger C buy deeper surgical German edits (granularity hypothesis, §8); how the base-mass/fidelity frontier moves; spectrum stability under 2× data.

## 11. Scale + sweep (`full65`): N=65,536, C ∈ {512, 1024, 2048} × base_ratio {5, 8}

Eval grid (gate threshold 0.02; all arms faithfulness-exact):

| arm | base mass | full-gate KL @ gates/tok | top-64 / top-256 KL | base-only KL |
|---|---|---|---|---|
| C512_r8 | 0.932 | 1.10 @ 36 | 0.50 / 0.067 | 4.44 |
| C1024_r8 | 0.815 | 3.11 @ 51 | 2.41 / 0.52 | 6.75 |
| C2048_r8 | 0.575 | 4.76 @ 95 | 4.80 / 2.83 | 9.08 |
| C512_r5 | 0.669 | 4.24 @ 55 | 3.38 / 0.65 | 9.75 |
| C1024_r5 | 0.430 | 5.43 @ 85 | 5.36 / 2.98 | 10.53 |
| C2048_r5 | 0.179 | 5.84 @ 154 | 6.36 / 5.23 | 11.90 |

German edits (k=4 rows; "best clean" = deepest German ΔCE with English < 0.1 across all k/α/space):

| arm | gate k4 ablate | gate k4 α=−1 | weight-del k4 | best clean edit |
|---|---|---|---|---|
| C512_r8 | +0.07 / +0.01 | +0.21 / +0.04 | +0.08 / +0.03 | wt k8 ablate: +0.40 / +0.10 |
| C1024_r8 | +0.64 / +0.01 | **+2.36 / +0.20** | +0.57 / +0.06 | wt k8 ablate: +0.93 / +0.10 |
| C2048_r8 | +0.51 / −0.00 | +0.60 / −0.00 | +0.29 / +0.01 | wt k4 α=−1: **+0.95 / +0.03** |
| C512_r5 | **+1.22 / +0.04** | +2.01 / +0.19 | +0.83 / +0.07 | gate k4 ablate: +1.22 / +0.04 |
| C1024_r5 | +0.67 / +0.27 | +1.65 / +0.53 | +1.11 / +0.15 | (none under bar) |
| C2048_r5 | +0.82 / +0.12 | +1.47 / +0.25 | +1.20 / +0.17 | (none under bar) |

Findings: **(a) Reproducibility** — C512_r8 at N=65k replicates the N=32k baseline on every metric (KL 1.10 vs 1.08, top-256 0.067 vs 0.07, base 0.93 vs 0.92). **(b) (C, r) operating points specialize** — C512_r8 is the behavioral-fidelity point; finer/deeper arms trade gated fidelity for editing-grade units; C and r interact mechanically (more clusters shrink the per-capita mean, so more entries clear the specificity bar and base mass falls at fixed r). **(c) Granularity → edit depth: confirmed but non-monotone.** C=1024 quadruples the clean k=4 ablation (+0.64 vs +0.07) and reaches the overall deepest edit (α=−1: German +2.36 at English +0.20); C=2048 is the *cleanest* (English literally −0.00 at k=4) but shallower per-k — the German mechanism splits across more than 4 components, so fixed-k edits capture less of it. Strict-clean depth plateaus at ~1.0–1.2 nats across best arms (vs LW's ~+6.5 with a gradient-tuned scalar): the remaining rungs are per-unit tuned scalars (a 4-parameter fine-tune, LW's exact protocol) and k matched to the mechanism's component count, not more C. **(d) Induction remains unowned in every arm** (finer C makes the sparse-gated copy accuracy worse, 0.21 → 0.00), confirming this is the sensor/representation boundary — addressed next by the GIM arm (§10), not by scale.

## 12. The complete base_ratio frontier (N=65k, C=512, r ∈ {2,3,4,5,8,12,16}) and the GIM arm

| r | base mass | full-gate KL @ g/tok | top-256 KL | base-only KL | weight-del k4 (de/en) | random-del k4 | best clean edit (en<0.1) |
|---|---|---|---|---|---|---|---|
| 2 | 0.000 | 7.41 @ 92 | 3.67 | 66.7 | +2.81 / +0.32 | +0.06 / +0.07 | +1.24 / +0.02 |
| 3 | 0.163 | 5.92 @ 85 | 3.01 | 16.5 | +2.43 / +0.22 | +0.05 / +0.05 | — |
| 4 | 0.459 | 5.06 @ 70 | 1.72 | 9.8 | +1.31 / +0.10 | +0.03 / +0.04 | +1.31 / +0.10 |
| 5 | 0.669 | 4.24 @ 55 | 0.65 | 9.8 | +0.83 / +0.07 | +0.01 / +0.03 | +1.22 / +0.04 |
| 8 | 0.932 | 1.10 @ 36 | 0.067 | 4.4 | +0.08 / +0.03 | +0.01 / +0.02 | +0.40 / +0.10 |
| 12 | 0.984 | 0.215 @ 24 | 0.008 | 0.77 | (not run) | | +0.30 / +0.05 |
| 16 | 0.990 | 0.100 @ 19 | 0.002 | 0.54 | (not run) | | +0.13 / +0.04 |

Takeaways: **(a)** A clean two-axis trade spans the entire range: owned mass 100% → 1%, gated fidelity 7.4 → 0.10 nats, weight-edit depth +2.81 → +0.01. No single row is "the" decomposition — rows are operating points (r8 = fidelity sweet spot where the base is still insufficient by 4.4 nats; r2–r4 = editing grade; r12–16 = high-fidelity but nearly-empty). **(b) Random weight-deletion is harmless at every depth — even at r2, where the base is literally empty and every entry is component-owned (+0.06/+0.07).** Disjointness-based weight-surgery safety is not a shallow-regime artifact; this is the strongest form of the §7 permanence result. **(c)** The *clean* German edit plateaus at ~+1.2–1.3 nats across r2–r5 (deeper cuts, +2.4–2.8, cost English +0.2–0.3): once components own enough of the mechanism, depth is limited by selection/granularity (which components to cut and how finely German is separated from general multilingual machinery), not by ownership. **(d)** GIM sensor A/B (N=32k, C512_r8): behavioral fidelity unchanged (1.165 vs 1.076), **gated induction preserved 65% better (copy 0.188 vs 0.114)** — corrected softmax backprop attributes QK mass the IG sensor missed — but no ablatable induction carriers emerged (contrast-ablation margin ≈ baseline); ownership of cross-position mechanisms likely needs finer C under GIM or pair-gating. GIM's German k4 edit is also deeper than IG's at matched config (+0.45/+0.03 vs +0.26/+0.01).

## 13. Toward low base AND low KL: soft gates fail, fractional ownership helps

The frontier in §12 trades owned mass against gated fidelity along a single knob (r); the target operating point — near-total ownership AND near-zero KL — sits off the curve, and the diagnosis is shared weight entries: under a hard per-entry argmax, mass that several mechanisms use must either be surrendered to the base (high r) or misassigned to one owner (low r).

**Experiment 1 — soft gates (clean negative).** Replacing hard threshold gates with graded gates g = (A/Amax)^τ on the existing hard-partition banks is worse at every temperature and every depth: r2 goes 7.41@92 (hard) → 12.58@43 / 25.53@11 / 32.56@3 for τ = 0.5/1/2, r3 5.92@85 → 7.71@41 / 10.71@10 / 12.41@3, r8 1.10@36 → 2.29@21 / 2.95@5 / 3.28@2.

Even granting the soft rows their lower effective gate counts, no soft row beats the hard row's KL — so gate calibration is not the bottleneck, and the failure is in the weight assignment itself.

**Experiment 2 — fractional ownership (partial win).** `extract_ps` splits each entry over its top-8 clusters with shares proportional to per-capita attribution mass^(1/T) (shares sum to 1 per entry — zero base, exact faithfulness by construction, still no cancellation since all slices are positive rescalings of W).

At the zero-base end this dominates the hard argmax: full-gate KL 7.41 → 5.43 (@109 vs 92 gates/token) and top-256 KL 3.67 → 2.55 at T=1; sharper shares (T=0.5) trade a little fidelity back (6.31@93, top-256 2.52).

Weight-space surgery survives fractional ownership — scaling each entry by its owned share: German k4 deletion +0.79/+0.15 (de/en) with random-k at −0.01/+0.02, k16 +2.13/+0.65, inversion (α=−1) k16 +4.69/+1.82; per-edit depth is shallower than the hard partition at matched k because the selected components own only fractions of their entries (hard r3 k4: +2.43).

Two side-findings: soft gates on top of soft shares are catastrophic (54.4@12.6 for T=1, 50.7@10.8 for T=0.5) — compounding two sources of down-scaling collapses the forward pass — and gated induction copy accuracy is 0.000 at this operating point (consistent with the QK-blindness finding; the IG sensor was used here).

Verdict: misassignment of shared entries was the real problem (exp 2 moves the zero-base frontier inward ~30%), but proportional shares are still uncalibrated for composition — nothing chose them so that gated *partial sums* reconstruct the entry under realistic gate patterns; that is the next experiment (§14, gate-aware share solve, running at C=1024).

## 14. Gate-aware shares (`extract_pg`, C=1024): the largest zero-base fidelity gain so far, and a fidelity-vs-selectivity trade

Diagnosis after §13: proportional shares are uncalibrated for composition — nothing chose them so that gated PARTIAL SUMS reconstruct each entry under the gate patterns that actually occur.

Method: keep each entry's top-8 owner support and mass weights from the proportional bank, then re-solve its shares to minimize the relevance-weighted error E[ρ_te (Σ_{c∈S} g_tc s_c − 1)²] subject to Σ s = 1 (the constraint keeps all-gates-on = W, i.e. exact faithfulness), with relevance ρ_te = Σ_k m_ek a_t,S_k.

Everything reduces to two moment tensors from a held-out stream (12 batches, ~49k tokens, gates bootstrapped from the proportional bank): B[c,c'] = Σ_t a_tc g_tc' and Q[c,c',c''] = Σ_t a_tc g_tc' g_tc'' (4.3 GB on GPU at C=1024), then batched 9×9 KKT solves over all 28M entries — still training-free (a closed-form solve on collected statistics, no SGD), a few GPU-minutes total.

The solver's internal proxy (fraction of relevance mass not covered by gated partial sums) fell 0.364 → 0.221 (−39%), and the realized full-gate KL at matched density fell almost exactly the same relative amount — the quadratic proxy tracks behavior.

| zero-base bank | full-gate KL @ g/tok | top-32 KL | top-256 KL |
|---|---|---|---|
| hard argmax (C512, r2) | 7.41 @ 92 | — | 3.67 |
| proportional (C512, T=1) | 5.43 @ 109 | 13.6 | 2.55 |
| proportional (C1024, T=1) | 6.52 @ 153 | 23.7 | 3.59 |
| **gate-aware (C1024)** | **5.79 @ 44 · 3.88 @ 100 · 2.52 @ 200** | **5.78** | **1.60** |

Three headline facts: (a) at matched ~100 gates/token the zero-base full-gate KL is now 3.88 — better than the hard-partition r5 row (4.24 @ 55) which surrenders 67% of the model to the base; (b) top-256 KL 1.60 beats hard r4 (1.72, base 46%) — the first fully-owned configuration to beat substantial-base rows, an unambiguous inward move of the whole frontier, not just its zero-base end; (c) the solved shares concentrate attribution, so the same 0.02 threshold fires 44 gates/token instead of 153 with better KL — sparser AND more accurate.

The cost: edit selectivity degrades — weight-space German deletion at k=4 goes from +0.45 de / +0.01 en (proportional C1024, surgically clean) to +1.06 de / +0.63 en (gate-aware), and k=16 reaches +3.82 de at +2.27 en; inversion k16 hits +8.29 de (LW-depth) at an unusable +5.31 en.

Reading: to make partial sums work, the solver shifts shared mass toward reliably co-active owners, entangling each component with general machinery — reconstruction-optimal ownership is not edit-selective ownership; the two banks (proportional for editing, gate-aware for fidelity) are complementary operating points over the SAME supports.

Permanence is untouched: random-k weight deletion stays at ≈0.00 for both banks at every k — disjointness-based surgery safety survives share re-solving.

Gated induction copy is 0.000 for both C=1024 banks (IG sensor; consistent with §12's QK-blindness finding — induction ownership remains a GIM-at-fine-C question).

Remaining gap to the "near-zero KL at zero base" corner: full-gate 2.5–3.9 nats; candidate next steps are one self-consistent iteration of the solve (re-collect gates under the gate-aware bank), hierarchical/multi-scale ownership (shared mass owned at coarse levels), or a brief gated-recon polish as the ceiling probe.

## 15. Iterating the solve and widening support (Rohan's direction: target low KL; hierarchy rejected)

Two training-free levers on the gate-aware solve, run as parallel arms at C=1024 (all zero base, exact faithfulness).

**Arm A — self-consistent iteration.** The first solve optimized against the proportional bank's gate patterns, but the solved bank gates very differently (44 vs 153/token at the same threshold), so iteration 2 re-collects the moment tensors under the gate-aware bank itself (at threshold 0.005 ≈ the 100/token operating point, via a new `--mass_banks_tag` argument keeping relevance masses from the proportional bank) and re-solves.

**Arm B — s=16 support.** Entries whose true owner set exceeds 8 clusters cannot reach full coverage under any share assignment; re-extracting proportional supports at top-16 and re-solving (17×17 KKT) removes that ceiling.

| zero-base bank (C=1024) | full-gate KL @ g/tok | top-64 KL | top-256 KL |
|---|---|---|---|
| gate-aware iter 1, s=8 | 5.79 @ 44 · 3.88 @ 100 · 2.52 @ 200 | 4.33 | 1.60 |
| iter 2, s=8 | 6.25 @ 30 · 4.26 @ 65 · 2.86 @ 130 | 4.05 | **0.94** |
| iter 1, s=16 | 5.29 @ 47 · 3.47 @ 104 | **3.89** | 1.22 |

Both levers beat iteration 1 at matched gate density (s16 3.47 vs 3.88 at ~100/token; iter2 2.86 at 130 vs ~3.4 interpolated), through complementary mechanisms: iteration fixes the gate-distribution mismatch (top-256 nearly halved, 1.60 → 0.94 — between hard r4 and r5 which give up 46–67% of the model), wider support fixes coverage (best top-64).

The solver-proxy diagnostic separates the two cleanly: under identical (proportional-bank) gate statistics, s=16 lands at 0.222 uncovered vs s=8's 0.221 — support width is NOT the binding constraint on the proxy — while iteration 2 under its own gate distribution reaches 0.070; the realized gains say both matter because the eval always runs under the bank's OWN shifted gates.

That shift is also why iteration undershoots its proxy (0.070 predicted more than 2.86@130 delivers): each re-solve concentrates attribution and moves the gate distribution again — the solve chases a moving target, suggesting damping or a joint gate+share fixed point as the clean formulation.

**The combination composes.** Iteration 2 on s=16 supports (`gaw2S16C1024`): full-gate KL 5.87 @ 25 · 3.91 @ 53 · **2.52 @ 103** gates/token, top-64 3.31, top-256 **0.407**.

| zero-base bank (C=1024) | full-gate KL @ g/tok | top-64 KL | top-256 KL |
|---|---|---|---|
| gate-aware iter 1, s=8 | 5.79 @ 44 · 3.88 @ 100 · 2.52 @ 200 | 4.33 | 1.60 |
| iter 2, s=8 | 6.25 @ 30 · 4.26 @ 65 · 2.86 @ 130 | 4.05 | 0.94 |
| iter 1, s=16 | 5.29 @ 47 · 3.47 @ 104 | 3.89 | 1.22 |
| **iter 2, s=16** | **5.87 @ 25 · 3.91 @ 53 · 2.52 @ 103** | **3.31** | **0.407** |

At the ~100 gates/token operating point the zero-base full-gate progression across the whole arc is 7.41 (hard argmax) → 5.43 (proportional) → 3.88 (gate-aware) → 2.52 (iterated, wide-support) — and top-256 KL 3.67 → 0.407, a 9× improvement, now beating the hard r5 row (0.65) that surrenders 67% of the model to the base; only r8 (0.067, base 93%) remains ahead on that metric.

**The fidelity-vs-selectivity trade is monotone along this ladder.** German weight-deletion k=4: proportional +0.45 de / +0.01 en (surgical) → iter 1 +1.06/+0.63 → iter 2 +2.76/+2.03 (deep, de/en ratio only 1.4) — each KL gain further entangles components with shared machinery, while random-deletion controls stay ≈0.00 throughout (disjointness permanence is never affected).

So the KL objective and the edit-selectivity objective now visibly pull the shares in opposite directions over the SAME supports — a two-point Pareto frontier (proportional = editing grade, iterated gate-aware = fidelity grade) that any future formulation (damped/joint gate+share fixed point, or an explicit selectivity regularizer in the KKT solve) should treat as the thing to collapse.

## 16. Llama-3.2-1B port (run1, N=16,384, C=512, proportional shares only)

The port (`geo1b.py`) reuses every pipeline stage verbatim and swaps only plumbing: HF Llama-3.2-1B behind a logits wrapper (112 decomposed linears, ~968M entries, 34× the 67M target), a locally pretokenized pile-uncopyrighted bin (Llama tokenizer, 60M tokens), and artifacts on /dev/shm (collect shards 23GB/rank, banks 31GB — over the workspace quota's headroom).

Feasibility surprise: the gated-pass einsum is far faster than its naive FLOP model (~1s per large matrix per pass), so no custom kernels were needed; total pipeline cost is a few H100-hours (collect ~10 min, gram ~49 min on 2 GPUs, factor ~10 min, extraction ~15 min).

Faithfulness is exact at 1B (all-ones KL 4.6e-8), and the zero-component control destroys the model (KL ~85 nats) — the decomposition machinery transfers to a real pretrained LM unchanged.

The attribution spectrum at 1B is the same smooth power law as at 67M (no eigengap; head candidates C=16–38 are artifacts) — scale does not reveal a "natural" number of mechanisms; C stays a resolution knob.

Per-token attribution is heavy-tailed: 7–13 components within 10% of the token's max, ~40–80 above the 2% gate line, and a floor ~1000× down (see `out/full1b/ac_bar_1b.png`) — a clear head, a smooth slope, no clean cliff.

**Gated fidelity did NOT improve with scale**: binary reference rows (2-batch means, thresh 0.02 / 0.005) are KL ~10.5 @ 64 and ~7.0 @ 149 gates/token vs 67M's 5.43 @ 109 at matched C=512 proportional — plausibly because N=16k gives only ~32 positions per cluster (vs 128 at 67M) and the 1B model has more mechanisms competing for the same 512 slots.

**Editing improved dramatically with scale** (weight-space German erasure, target CE de 4.22 / en 4.93):

| k | deletion (de / en) | inversion α=−1 (de / en) | random del (de / en) |
|---|---|---|---|
| 4 | **+1.61 / +0.02** | **+4.71 / +0.44** | +0.05 / +0.01 |
| 8 | +2.53 / +0.21 | +5.53 / +1.88 | +0.19 / +0.08 |
| 16 | +4.59 / +2.32 | +8.45 / +6.17 | +0.26 / +0.13 |
| 32 | +5.80 / +2.28 | +10.68 / +8.09 | +0.78 / +0.34 |

Headlines: **(a)** k=4 deletion is the cleanest deep edit of the project — +1.61 nats of German at +0.02 English (80:1), vs the 67M clean-edit plateau of ~+1.2–1.3 at ~10× worse selectivity; **(b)** k=4 inversion reaches +4.71 / +0.44 — approaching the LW post's tuned-scalar depth with 4 untrained components and a fixed α; **(c)** the collateral cliff replicates: past the ~4–8 genuinely German components, edits cut shared multilingual machinery (k=16 inversion damages both languages); **(d)** random-deletion is harmless at every k — disjointness permanence transfers to 1B.

Combined thesis so far: scale did not buy gated KL, but it bought editability — the 1B model concentrates German cleanly enough that four components and a sign flip nearly erase it.

**k=4 deep-dive** (`german_k4_1b.json`): the α dial is smooth and monotone — de +0.28 (α=+0.5, en −0.02: a clean partial "volume knob") through +1.61 (delete) to +4.71 (α=−1), with the efficient knee at α ∈ [−0.75, −1]; past −1 German damage saturates (+5.38 at α=−2) while English cost triples (+1.47).

**Component 35 is the key contributor**: solo inversion +2.08 de / +0.01 en (230:1 — the cleanest single-unit edit of the project, 44% of the joint effect); solo effects sum to +3.86 < joint +4.71, so the four form a cooperative circuit with partial redundancy rather than four redundant copies.

**No single-matrix atom inside any component**: top module ≤8–14% of a component's owned mass, top row ≤3% — units are distributed bundles (vs the LW post's rank-1 single-matrix atom), with suggestive structure: c35/c54 in early-layer MLPs (L1–4, input-side detection), c374/c302 in late MLPs (L14–15, output-side prediction); the strongest lever is the early detector, consistent with a cascade.

Per Rohan's calls: gate-aware share solving is dropped from the method (fidelity gain not worth the editing cost), and binary threshold gating was distrusted — graded modes evaluated in §17.

## 17. Graded gating at 1B: gate shape barely matters, total openness does

Motivated by distrust of binary thresholds, two saturating graded modes were added (both cap at fully-open so dominant machinery is never attenuated — the failure of §13's raw soft gates): ramp $g = \min(1, ((a/a_{\max})/\tau)^\gamma)$ (binary = $\gamma\to\infty$) and the parameter-free participation-ratio rule $g = \min(1, \mathrm{PR}\cdot a/\sum a)$ with $\mathrm{PR} = (\sum a)^2/\sum a^2$ (computed on max-normalized attribution after an fp32 overflow produced NaNs on raw magnitudes).

| gating (1B, C=512 prop) | KL @ total gate mass |
|---|---|
| binary τ=0.02 | ~10.5 @ 64 |
| ramp γ=1, τ=0.02 | 7.99 @ 128 |
| binary τ=0.005 | ~7.0 @ 149 |
| ramp γ=0.5, τ=0.02 | **5.88 @ 191** |
| participation ratio | 37.8 @ 22.6 |

Verdict: all modes land on approximately one openness-vs-KL curve — the attribution *ordering* does the work, and the gate shape is mostly a budget knob; binary is not distorting the picture, and graded tails are simply a cheap way to buy more openness.

The PR mode is not a fidelity winner but a canonical operating point: the attribution distribution's own effective component count (~23/token) — the "how many components does the token think it uses" number — sits far sparser than behavioral fidelity requires (~hundreds at this decomposition quality); that gap is a property of the decomposition, not the gate.

## 18. VPD Section-6 replication at 1B: the emoticon edit

VPD's demo: in their 38,912-unit rank-1 per-matrix decomposition of the 67M model, they rewrote one emoticon subcomponent's write vector toward the 'o' unembedding, turning all emoticons into ':o'.

Llama-3 tokenizes common emoticons as single tokens, so the 1B analog is emoticon-CHOICE collapse: attribution contrast (chatty emoticon sentences vs plain English, ranked at positions predicting an emoticon token) finds components [48, 434, 447, 505] — disjoint from the German set — and the edit replaces their write-side owned slices with norm-matched rank-1 writes toward the ' :)' unembedding (read side untouched, so firing is preserved).

Results (`emote1b.json`): redirect k=1 gain=2 produces the qualitative demo — the model spontaneously inserts " :)" into ordinary text ("I went to the beach :)"), emoticon mass at arbitrary positions 5e-6 → 0.15, at CE cost 4.9 → 11.1; k=4 gain=4 is the cartoon endpoint — 99.5% ' :)' at every position, generations are pure ":) :) :)".

Negative arm: AMPLIFYING the components' weights (α=2–4 on owned mass) *reduces* emoticon probability and then breaks the model — a mechanism's firing is calibrated, not proportional to weight norm; redirecting what it writes works, inflating how loud it is does not.

The cleanliness gap vs VPD is granularity, quantified: their unit is one of 38,912 per-matrix rank-1 atoms (~76× finer than our 512 whole-model bundles); our component-48 slice fires beyond emoticons, so redirecting all of it leaks smiley-pressure — no gain is both behavioral and free.

Proposed scalpel (untried): add a targeted rank-1 write conditioned on the component's emoticon-specific read direction (mean input activation at emoticon-predicting positions projected into the slice row space) instead of replacing the whole write action — within-component surgery, still training-free.

## 19. Scaling to N=1M via exact-in-expectation random features (geo1m.py)

The exact gram is O(N²·entries) — infeasible at 1M (~140 GPU-days, 4.4 PB). The kernel is a sum over weight entries, k(x,x') = Σ_oi W_oi² g_o g'_o p_i p'_i, so importance-sampling D=16,384 entries gives explicit features φ_r(x) = s_r · g_{o_r}(x) · p_{i_r}(x) whose dot products equal the kernel in expectation (unbiased; IG handled by averaging the K passes).

The proposal distribution is the whole game: sampling ∝ W² alone FAILS (heavy-tailed kernel; normalized-gram correlation 0.75 at D=16k, and WORSE — 0.59 — at D=32k, the signature of a draw-lottery estimator). Reweighting by cached activation statistics, q ∝ W²·E[g²]·E[p²], fixes it: correlation 0.9915 raw / 0.9621 normalized / 0.115 Frobenius against the exact 16k gram — validated as a hard abort gate before committing the run.

Two-tier data use: all 1M positions contribute a 16,384-dim fingerprint (32 GB, written streaming) that drives PCA + spherical k-means (C=2048); a spread 32,768-position subset keeps full (p,g) in standard shard format so extraction runs UNCHANGED and exact. Timing on 2×H100: collect 24 min (64 positions/seq amortizes the passes), cluster <1 min (embedding fits in GPU memory), extraction ~15 min — the whole 1M decomposition in ~45 min of compute plus referees.

Cluster density jumped 77× (median 6 → 464 positions/cluster) — the quantity that was starved at C=2048. The one seam: the 32k extraction subset gives median 14 positions/cluster with 22/2048 components drawing zero subset members (empty, entries fall through to co-owners; faithfulness preserved by renorm) — fixable by storing 4/seq instead of 2.

## 20. The granularity verdict: statistics-limited, not broken

German weight-erasure across the three regimes (ΔCE de/en, k=4 deletion, higher = more erased):

| regime | k4 deletion (de / en) | ratio | k8 deletion (de / en) |
|---|---|---|---|
| C=512, N=16k (coarse) | +1.61 / +0.02 | 80:1 | +2.53 / +0.21 |
| C=2048, N=16k (starved) | +4.05 / +2.03 | 2:1 | — |
| **C=2048, N=1M (fed)** | **+2.00 / +0.24** | **8:1** | **+2.75 / +0.25** (11:1) |

More statistics recovered most of the lost selectivity: the C=2048 k=4 ratio went 2:1 → 8:1 with 61× more clustering data, and k=8 deletion is the cleanest fine-grained edit at 11:1 (+2.75 de / +0.25 en) — the German circuit is spread across ~8 fine components vs ~4 coarse bundles, so it needs a slightly larger k but each unit stays selective. Random-k controls are ≈0 throughout (permanence holds). Full 1M table: k4 inv +4.34/+1.27, k8 inv +4.83/+1.32, k16 del +4.22/+1.51, k32 del +6.08/+2.17.

The residual gap to C=512's 80:1 is the genuine granularity tradeoff (finer units are individually less pure than coarse bundles even with adequate data), NOT starvation — the three-point comparison separates the two cleanly. Conclusion: C=2048 entanglement at 16k was a statistics artifact; the feature-map scaling resolves it, and the method's granularity knob is real and well-behaved once fed.

The emoticon narrow-mechanism edit sharpened too: on the 1M bank the single-component redirect at gain 1 moves target probability 0.024 → 0.086 at +0.22 CE (vs INERT at gain 1 on the 16k-cluster bank) — the top emoticon component now carries more signal, less noise. Cartoon endpoint (k4 gain2 = 97% ' :)') and the negative amplify arm both reproduce.

## Artifact map

`out/full/`: `gram.pt`, `labels.pt`, `spectrum.pt` (N=32k); `banks{,_lam01,_lam1}.pt` (ridge arms), `banks_part{3,5,8}.pt` (partition arms); `eval_*.json`, `canary_*.json` (part8 threshold-0.1 preserved as `*_part8_t01.json`); `german_part8.json` (gate, α sweep), `german_part8_weight.json`; `evidence_part8.json`, `catalog_part8.json`; per-arm logs `out/arm_*.log`. `out/full65/`: collect shards (N=65k), `eig.pt`, `labels_C{512,1024,2048}.pt`; hard-partition banks `banks_C{512,1024,2048}_r*.pt` with matching `eval/canary/german*` JSONs; fractional banks `banks_softT{1,05}.pt` (C=512) with `eval_softT*{,_soft1.0}.json`, `german_softT*_weight.json`, `canary_softT1.json`; soft-gate arms `eval_C512_r{2,3,8}_soft{0.5,1.0,2.0}.json`; gate-aware arm at C=1024 (`banks_softC1024T1.pt`, `banks_gawC1024.pt`, driver `gateaware1024.sh`, log `out/gateaware1024.log`; evals `eval_{softC1024T1,gawC1024}.json` at thresh 0.02 plus `eval_gawC1024_t{005,001}.json` at 0.005/0.001; batteries `canary_*`, `german_*{,_weight}.json` for both banks).
