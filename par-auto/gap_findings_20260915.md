# Locating the gap between GIM and VPD

15 September 2026. New experiments on the frozen 66,931,200-parameter model, run on Explorer H200. The training-direction control is now complete; the IFR-style diagnostic is submitted separately.

## Conclusions supported so far

1. **Our exact-sum, rank-one, per-matrix capacity constraints can express a VPD-quality sparse endpoint.** A constructive conversion of published VPD into precisely our 43,008-piece family retains KL 0.293 at about 193 active pieces per token. This rules out those representation constraints as a necessary cause of the measured gap on these inputs.
2. **Our current attribution objectives rank this good sparse solution worse than our ordinary GIM solution.** With common fixed normalizers and identical exact-sum capacities, GIM Lp at p=.641 is 9.38 for the converted VPD solution and 5.13 for GIM. Adding the activity-count term increases this contrast to 24.36 versus 11.27. Sparse intact-model attribution is not equivalent to existence of a sparse, low-KL computation.
3. **Gradients through the actual sparse computation recover substantially better supports from our existing factors.** At 512 pieces, GIM improves from 2.65 to 1.59 KL. This is a positive result for conditional endpoint information, unlike the preceding fixed-Fisher-sketch optimization. It is a per-context diagnostic search, not an unmasked training method or a proven optimal selector.
4. **The remaining gap cannot yet be assigned entirely to factor quality.** Under the same bounded endpoint search, published VPD gives 0.67 KL while its known native support gives 0.29. The search does not even recover the known good solution on VPD, so failure to find a similarly good GIM support is not proof that none exists.
5. **Compensating computation is selected by the endpoint criterion, not exclusive to VPD-trained weights.** The direct endpoint search makes GIM exhibit a last-layer restoration reversal and does not improve its clean-input reconstruction. Exact parameter reconstruction and unmasked factor training do not themselves guarantee that a selected sparse computation preserves the original mechanisms.

These results do not prove that every gradient-derived, unmasked objective is inadequate, nor that a mechanistically faithful ~193-piece explanation exists. They establish feasibility for VPD's sparse-output criterion and expose a mismatch in the objectives we have actually used.

## 1. Same-factor endpoint search

Job **10357029** completed successfully in 2m44s. Eight fresh validation contexts **424–431**, two budgets, four existing checkpoints. FP32 model forwards, loss, and gate gradients; common intact Fisher K128 for the starting ranking. The finite-difference gate-gradient check had relative error **1.20e-5**.

At each round, the search proposes adding and dropping one component at a token, ranks proposals using gradients of actual sparse-model KL, and checks candidate batches of swaps by actual KL. Only improving swaps are accepted. Counts remain exact at every token; factors and selected weights never change. There are two starts (magnitude and jittered magnitude), at most 32 rounds, and the initial magnitude solution remains available. A search can stop at a local barrier; this is not a global oracle.

| Factors | Budget/token | Magnitude KL | Best endpoint KL | Native CI KL |
|---|---:|---:|---:|---:|
| GIM 5k | 256 | 3.623 | 1.949 | — |
| GIM 5k | 512 | 2.647 | 1.585 | — |
| GIM 7,783 | 256 | 3.672 | 2.054 | — |
| GIM 7,783 | 512 | 2.692 | 1.615 | — |
| VPD 5k | 256 | 4.821 | 1.682 | 3.345 |
| VPD 5k | 512 | 4.194 | 1.151 | 0.956 |
| Published VPD 400k | 256 | 1.608 | 0.821 | 0.337 |
| Published VPD 400k | 512 | 1.304 | 0.674 | 0.293 |

The 5k comparison matches update count, not compute or VPD's extra warmup. VPD 5k looks worse than GIM under intact Fisher ranking but better under endpoint search at 512. Thus factor quality cannot be defined independently of the selection/reconstruction objective. The later GIM checkpoint supplies no improvement on this slice, but does not rule out a much longer or different optimization schedule.

The endpoint search uses target outputs while selecting each context's support. It is not a trained selector evaluated on unseen inputs. Its purpose is to test support existence and the value of evaluating conditional computation.

## 2. A constructive representation control

Job **10357063** completed successfully in 57s. Published VPD has 38,912 primitive factors; GIM has 43,008. The difference is eight Q/K matrices: VPD has 512 factors per matrix and GIM has 1,024. VPD also has an unscored dense remainder.

For each Q/K matrix, I filled the 512 extra slots with rank-one factors from the remainder's SVD, then applied our conditional exact-sum projection to all 24 matrices. The final number of pieces matches GIM in every matrix. All original component indices are preserved within their matrices, and added pieces are excluded from the inherited native CI support. The maximum matrix reconstruction error is **3.20e-8**. Maximum relative changes to original read and write arrays are about **0.44%** and **0.030%**, respectively.

| Factors and selection | Mean retained/token | KL |
|---|---:|---:|
| Original VPD, positive CI | 192.58 | 0.29300 |
| Exact-sum VPD conversion, inherited positive CI | 192.58 | 0.29292 |
| Exact-sum conversion, Fisher selection at the identical per-token budgets | 192.58 | 1.80715 |
| GIM, Fisher selection at those per-token budgets | 192.58 | 4.40950 |
| GIM, also copying VPD's per-matrix allocation | 192.58 | 3.90009 |

Copying VPD's allocation helps slightly in the positive-support comparison, but leaves most of the gap. Fixed-budget 512 allocation transfer actually worsens GIM. VPD's zero-CI padding makes the latter allocation partly an artifact of deterministic tie-breaking, which is why the positive-support control matters.

This conversion is an **existence witness**, not a new method: it borrows VPD-trained factors and its selector. It settles the capacity/exactness objection while leaving the no-mask-training and less-data-dependent-selection goals open.

## 3. The loss prefers the wrong candidate in this controlled comparison

I evaluated the actual GIM score/penalty recipe with the same fixed initial reference normalizers. For the converted VPD and GIM candidates, both capacity and exact matrix sums match.

| Candidate | Normalized Lp, p=.641 | Soft activity count | Normalized count | Lp + count |
|---|---:|---:|---:|---:|
| Exact-sum VPD conversion | 9.37956 | 16,864.1 | 14.97976 | 24.35932 |
| GIM 5k | 5.13352 | 6,903.7 | 6.13228 | 11.26580 |

The candidate supporting KL 0.293 with ~193 pieces has **higher** attribution loss and activity count. Adding count strengthens that preference against this candidate. The issue is therefore not simply that we have failed to minimize the present penalty enough.

This ordering is evidence of surrogate mismatch, not proof that all good sparse solutions have high GIM loss or that descent from the converted solution must destroy its sparse endpoint. The next control tests that direction directly.

## 4. The sparse forward's response is central

For a matrix changed by ΔW and an input changed upstream by δh,

    (W + ΔW)(h + δh) = Wh + Wδh + ΔWh + ΔWδh.

The clean-error injection control adds ΔWh evaluated at the intact input to the original matrix's live output. It removes ΔWδh while retaining Wδh. All following nonlinearities remain active. The contrast is not an additive attribution of KL to that term.

At 512 pieces:

- GIM magnitude selection has KL **2.647**. Direct endpoint refinement gives **1.605**, while clean-error injection remains essentially unchanged (**11.315 → 11.388**).
- Restoring the real last layer improves magnitude-selected GIM by **0.580 KL**, but worsens endpoint-refined GIM by **0.393 KL**. The latter paired approximate 95% context interval is **[0.072, 0.714]**.
- Published VPD's native selection gives KL **0.293**, versus **6.186** under clean-error injection. Restoring the last layer worsens it by **0.075 KL**, interval **[0.058, 0.091]**.

Thus endpoint optimization finds selections whose changed downstream responses are useful for compensating upstream changes. It can do so using our existing factors. This supports the user's repair observation while narrowing its interpretation: the effect is not a capability unique to VPD's parameterization.

Exact full-sequence tangents on the first two contexts reinforce the mismatch: direct endpoint refinement improves GIM's finite KL while its mean local quadratic coefficient rises **5.714 → 6.756**. For published VPD, the native support's coefficient is **29.287** despite low finite KL. Improving the intact tangent metric and improving the sparse endpoint can favor different directions.

## 5. What this means for gradient-based training

There are three different claims:

1. **Can our rank-one/exact-sum representation achieve VPD-level sparse output fidelity?** Yes, constructively, on these validation inputs.
2. **Can gradients help find better sparse supports?** Yes, when taken through the actual conditional sparse computation. This does not satisfy the requested training constraints by itself; here it was only a diagnostic on frozen factors.
3. **Can sparsifying intact-model gradient attributions learn such supports without masking?** The tested objectives provide no convincing route yet. They rank a known feasible low-L0 solution worse, and greater precision in their local geometry has not repaired the sparse endpoint.

The relevant distinction is the information in the objective, not whether the optimizer uses gradients. The paper's joint CI selector and finite reconstruction objective are described in [VPD's methods and Appendix A.2](https://www.goodfire.com/research/interpreting-lm-parameters). [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940) studies another setting where conditional importance differs from isolated importance; it does not supply a method satisfying our training constraints.

I would stop treating stronger scalar L0 pressure or more accurate intact-model Fisher magnitudes as the main bet. A successful new objective must connect component selection to composable computation, or otherwise constrain finite errors, while keeping the original mechanism requirement explicit. These audits do not identify a proven GIM-cost, no-mask objective that does so.

## Completed control: unmasked descent from the converted solution

Job **10357130** completed in 2m35s, replacing failed setup job 10357098 before any update. The corrected job ran 200 ordinary GIM updates from the exact VPD conversion, with fresh Adam, common fixed reference normalizers, the original 5k–5.2k schedule window, and full 64-context batches. Training used only intact-model GIM caches; no masks or CI entered the loss.

| Updates | Common GIM Lp p=.641 | Soft count | Inherited native KL | Refined native KL | Fisher KL at matched native count |
|---:|---:|---:|---:|---:|---:|
| 0 | 9.37956 | 16864.1 | 0.29292 | 0.29213 | 1.80715 |
| 20 | 8.04679 | 15035.8 | 0.48944 | 0.46102 | 2.16501 |
| 100 | 6.96072 | 12094.6 | 0.96334 | 0.77025 | 2.86110 |
| 200 | 6.40832 | 9890.8 | 1.30432 | 1.00860 | 3.11647 |

Same eight contexts and ~193-component per-token counts throughout. Ordinary GIM descent decreases its objective while progressively degrading this known good sparse solution. Bounded endpoint refinement recovers part of the damage, but not the original fidelity. This is directional evidence of objective mismatch; it does not prove all good supports disappeared.

All 13 uploads verified. Local comparison SHA matched completion metadata: `6b800c08f32458f613825f48bb9c8b7520bf2bebc1c53059833a7008fa2a25f2`.

## Next diagnostic: IFR-style scores and reader conditioning

User authorized the test. Explorer job **10357214**, immutable source `code/ifr67_20260915`. Frozen 67M, fresh contexts 432–439, GIM 5k and converted VPD, fixed 200/512 and identical native per-token budgets. Compare Fisher, positive finite L1 proximity, signed proximity, and two prespecified rounds of reader conditioning. No training, gradients, or KL in proposed selection. Native CI is only the witness/count control. Actual evaluation recomputes all 24 masked matrices, attention, RMS, and GELU. Full protocol and limitations: [ifr67.md](runpod_compare/ifr67.md).

## Artifacts and verification

- [Endpoint/response report](out/gap67_20260915/report.md), with raw records and exact supports. 136 verified R2 uploads; comparison SHA verified locally.
- [Exact-sum conversion report](out/gap67_bridge_20260915/report.md), with conversion diagnostics and converted checkpoint on Explorer/R2. Comparison SHA verified locally.
- Sources: `runpod_compare/gap67.py`, `gap67_bridge.py`, `gap67_descent.py`, reports and root SLURM scripts. Each submitted source snapshot and its dependencies are SHA-checked. No commits were made.
