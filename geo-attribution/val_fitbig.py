"""Validate fit_big (row-minibatched U, per-row Adam on CPU) against the
16k-event baseline before trusting it with the 1M run.

Full-batch config: row_batch >= N makes every epoch one joint Adam step on
(U rows, S, V), so 3000 epochs is step-for-step the same optimization as
fit()'s 3000 steps -- any divergence isolates the manual per-row Adam and
streamed RMS, not the step budget. Then the same eval_klkeep protocol scores
its V against the shipped baseline curve.
"""
import json
import cofac67

cofac67.BIG = cofac67.RUN          # read the 16k A_chunk files; save alongside

r = cofac67.fit_big(k_factors=2048, c_groups=1024, epochs=3000,
                    row_batch=16384, holdout_frac=0.125)
print("FIT:", r)

out = cofac67.eval_klkeep(
    fact_path=cofac67.RUN / "factorization_big.pt",
    out_name="klkeep_val_fitbig.json")
print("VAL:", json.dumps(out, indent=1))
print("BASELINE:", (cofac67.RUN / "klkeep.json").read_text())
print("VAL_FITBIG_DONE")
