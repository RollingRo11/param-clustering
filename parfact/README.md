# parfact: attribution-based co-factorization for parameter decomposition

Implementation of section 4 of the "Attribution-Based Co-Factorization for
Parameter Decomposition" proposal, on the toy model of induction from
Christensen (2025), arXiv:2511.08854 (2-layer attention-only transformer,
1 head/layer, d_model=16, vocab 128, ctx 64, Shortformer positions, no
LayerNorm/MLP; predict the token that followed the first s-token).

## Pipeline (paper sec 4)

1. **Prediction events** (4.1): events are (sequence, position) pairs with
   `s_i = log p(y* | x)`, `y* = argmax`. `--positions {final,mixed,all}` —
   `final` is the induction prediction; `mixed` (default) adds random earlier
   positions so non-induction behavior is represented.
2. **Additive parameter atoms** (4.2), `--variant`:
   - `A` scalar atoms `theta_j e_j`, attribution `theta_j ds/dtheta_j`;
   - `B` rank-one SVD atoms `sigma_q u_q v_q^T`, attribution
     `<grad_W s, sigma_q u_q v_q^T>_F`;
   - `C` outer-product singular-vector directions: same additive atoms as B
     (so components still sum exactly to the weights), but attribution is the
     bare gradient projection `<grad_W s, u_q v_q^T>_F` without the
     singular-value weighting.
   By default the 8 attention projections (`wq,wk,wv,wo` x 2 layers) are
   decomposed; `--include_embed` adds embed/pos/unembed.
3. **Attribution matrix + normalization** (4.3), `--norm`:
   - `layer`: divide by per-group RMS attribution (`--norm_group layer` as in
     the paper, or the finer `matrix`);
   - `fisher`: divide by sqrt of the diagonal Fisher, estimated in atom space
     with labels sampled from the model;
   - `none`.
   Then magnitudes are row-normalized into an allocation matrix `M_bar`.
4. **Joint soft co-factorization** (4.4): `M_bar ~= U S V^T`, nonnegative via
   softplus, `V` rows on the simplex via softmax (optional `--u_simplex`),
   Adam on `||M_bar - U S V^T||_F^2` + optional entropy/L1 regularizers
   (`--lambda_u`, `--lambda_v`).
5. **Components** (4.5): `C_c = sum_j V_jc b_j`, materialized per weight
   matrix; `sum_c C_c = theta` is checked exactly.
6. **Cheap usage** (4.6): `z_ic = sum_j V_jc a_ij` from the already-collected
   attributions.

Diagnostics: `R^2_attr`, effective factor/group counts (exp-entropy), per
matrix/layer mass of each component (sec 9.1), correlation between predicted
usage `(US)_ic` and `|z_ic|`, induction-vs-background usage ratio, and a small
sec-5 ablation check (`Delta_ic = s_i(theta) - s_i(theta - C_c)` vs `z_ic`).

## Usage

```bash
python induction_model.py                      # train + save checkpoint (100k steps)
python run_parfact.py --variant B --norm layer # main run (reuses checkpoint)
python run_parfact.py --variant A --norm fisher
python run_parfact.py --variant C --norm layer --norm_group matrix
```

Outputs land in `out/<variant>_<norm>/`: `factorization.pt` (U, S, V, signed
attributions, components, events) and `metrics.json`.
