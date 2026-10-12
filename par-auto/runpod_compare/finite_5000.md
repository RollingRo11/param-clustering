# Finite local ablation score

Fresh 5,000-step run. The training score becomes per-piece finite local ablation
(`finite_ablation.py`): remove one piece at one token, recompute the nearest
nonlinearity exactly, take the norm of the change.

- c_fc: change in the GELU output at that token.
- q_proj: change in that token's attention output (its softmax row recomputed).
- k_proj, v_proj: change in every later token's attention output.
- o_proj, down_proj: no nonlinearity before the residual, so this is the write size |r.h| ||u||.

Each step reads the same 64 sequences as the GIM p=0.5 run; 16 random positions per
sequence (1,024 tokens) enter the loss, vs all 32,768 for GIM. Everything else is the
GIM run's TrainConfig (8,700-step horizon, p_end=0.5), seed, data order, init and components.

KLKeep at 5,000 uses 8,192 held-out tokens (16 sequences), because exact per-piece scoring
of every token is costly. GIM p=0.5 and VPD are re-evaluated on the same 8,192 tokens.

W&B: param-autoenc / finite-ablation-20260910.
R2: parauto/vpd_matched_1b_20260910/experiments/finite_5000/.
