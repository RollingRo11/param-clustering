# Matryoshka training (masked-forward KL at a random budget)

Fresh 5,000-step run. Idea from Matryoshka Attribution (arXiv 2609.25518), with par-auto's own score
in place of MAttr's learned per-unit scores, so every token gets its own mask without a gate network.

Each update: cache the intact Fisher K=4 sensor (as fisher_k4_5000). Per micro-batch of 4 contexts, score
every piece per token, a = |r.h| x RMS_k(g_k.u); keep a soft top-k, alpha = a/(a+theta) with theta per
token so the alphas sum to k; k is log-uniform on [32, 43,007], one draw per micro-batch. Run the model
with every matrix replaced by its alpha-weighted pieces and minimise KL to the intact model. Gradients
reach the pieces through the masked forward and through alpha, including the threshold. No Lp term.

Matched to fisher_k4_5000: same seed-0 projected init, data order, Fisher cache seeds, 8,700-step cosine lr
horizon stopped at 5,000, Adam betas, exact-sum projection. Only the objective differs.

W&B: param-autoenc / matryoshka67_20260923. Monitor: hard top-k KL on 4 held-out contexts every 100 steps.
Eval: KLKeep ranked by Fisher K=8 on the control's 65,536 held-out positions; `comparison.json` pairs the curves.

Forward mode: `soft` puts alpha in the masked forward; `hard` uses KLKeep's exact top-k mask in the forward and
the sigmoid top-k Jacobian in the backward (MAttr's "+hard"). At the random init the soft mask is almost
flat (k=430: ~0 pieces above alpha 0.5, KL 66), so both are piloted for 300 steps before the 5k run.

Run: `sbatch slurm_matryoshka67.sh pilot soft|hard`, then `sbatch slurm_matryoshka67.sh train soft|hard`
(resubmits itself, then submits the eval).
