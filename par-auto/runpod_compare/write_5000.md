# Write-size score

Fresh 5,000-step run. Only the training score changes: GIM's |r.h| |g.u| becomes
b_c = |r_c.h| ||u_c||, the size of what piece c writes at its matrix output.
Everything else is the GIM p_end=0.5 run's stored TrainConfig (8,700-step horizon,
p≈0.641 at step 5,000), seed, data order, init, reference positions and components.

Capture is forward-only: no loss and no backward through the target.
For a removed set S at one matrix, ||sum_S u_c (r_c.h)|| <= sum_S b_c on that input.

KLKeep at 5,000 ranks the new run by b_c (its own score), GIM p=0.5 by GIM,
VPD by its CI. All three are evaluated in the same job, on the same 65,536 held-out positions.

W&B: param-autoenc / write-norm-20260910.
R2: parauto/vpd_matched_1b_20260910/experiments/write_5000/.
