# One-hour GIM training

Fresh projected-Adam decomposition of all 24 matrices in the same 67M Pile
target, with 43,008 total components and the same first shuffled/stratified
training batches as the earlier experiments. Batch: 32,768 positions. Seed: 0.
BF16 caches and score matmuls; FP32 parameters, Adam, and projection. The
summed Lp objective and exact-sum constraint are unchanged; only the sensor is
replaced. No rejection, learned importance, or custom GPU kernel is introduced.

GIM rules follow https://github.com/corticph/gim at commit
7e31c65f437aaa2bb33dcd920f09b0d35fd6802a: temperature-2 softmax backward
(without an additional 1/T), query/key gradient scaling by 1/4, value gradient
scaling by 1/2, and detached RMS normalization factors. The target uses eager
attention, so its softmax and projected Q/K/V tensors are patched directly.
Its GELU MLP is not gated and receives no gated-MLP scaling. This differs from
the old geo-attribution adaptation that also halved ordinary GELU gradients.
Forward arithmetic is preserved, including the upstream RMSNorm FP32 casts.
GIM is a modified attribution rule, not a true derivative or integrated gradient.

Preflight tests run on the allocated H100: analytic softmax VJP, unchanged
full-model BF16 outputs, neutral-rule gradient agreement, exact branch-gradient
scales at the final layer, finite changed GIM caches, and cleanup after errors.
Then 20-update ordinary and GIM profiles are discarded. Their timing is logged.

One H100 is allocated for one hour. A time deadline and pre-timeout signal stop
training roughly a minute before the limit to save the final optimizer state.
Cosine LR is 1e-3 to 1e-5; p is annealed 1 to .5 over the first 80% of the
schedule. The horizon is the post-profile remaining-time estimate plus 10% so
training uses the available time rather than stopping early. The final step
and learning rate are recorded: this is NOT a matched 900-step full schedule.

Every 1,000 steps saves an immutable directory with a complete resumable
checkpoint.pt and an evaluation-ready model.pth. A final partial-thousand
checkpoint is saved on a handled stop. latest_checkpoint.json identifies it.
The main fitter's checkpoint format and resume compatibility are preserved.

Native W&B run: project param-autoenc, group gim-1h-20260910. Metrics include
loss, p, LR, positions, time/step, checkpoint time, and peak allocated memory.
All checkpoints, provenance, exact source, and logs are archived by a separate
afterany CPU job to verified R2 objects under
parauto/vpd_matched_1b_20260910/experiments/gim_1h/.
GPU allocation time is spent on training; no KLKeep job is included here.
The evaluator accepts --sensor gim for subsequent native-sensor KLKeep.
