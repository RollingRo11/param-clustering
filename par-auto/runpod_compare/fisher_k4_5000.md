# Fisher score training (K=4)

Fresh 5,000-step run. The training score is the Fisher score: at every position, sample K=4
next tokens from the model's own prediction, one backward pass each, and score a piece by
|r·h| × RMS_k(g_k·u). Plain cross-entropy gradients (no GIM rules). Everything else is the
GIM p_end=0.5 run's TrainConfig (8,700-step horizon), seed, data order, init and components.

Why: the same Fisher score, used only for ranking the GIM checkpoint, moved KL at 10% kept
from 1.57 to 0.81 (below VPD's 1.12). K=8 vs K=32 at eval were nearly identical, so K=4 for
training is a cost choice: ~4 backward passes per step instead of 1.

KLKeep at 5,000: ranked by Fisher K=8 (and by GIM for reference), same 65,536 held-out tokens
as the H200 controls in output/write_5000/{control_gim,vpd} and output/fisher_rank/.

W&B: param-autoenc / fisher-20260911.   R2: parauto/vpd_matched_1b_20260910/experiments/fisher_k4_5000/.
