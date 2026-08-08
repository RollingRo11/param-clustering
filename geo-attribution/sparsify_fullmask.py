"""How many components does the full-mask German edit actually need?

Thresholds the selected alpha vector (components with |alpha-1| below tau
reset to 1) and re-evaluates each truncated edit on dev + held-out sets.
"""
import json, math, sys
from pathlib import Path
import torch
import geo1b
from german_fullmask import FullMaskEditor, language_metrics
from german_vpd_1b import log, prepare_data
from types import SimpleNamespace

RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
args = SimpleNamespace(seq_len=512, train_tokens=2048, eval_blocks=4,
                       refresh_data=False, data_cache=RUN/"german_vpd_data.pt")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
data = prepare_data(args, tok)
adapter = torch.load(RUN/"german_fullmask_adapter.pt", weights_only=True, map_location="cpu")
sel_tag = adapter["selected_tag"]
alpha_full = [r for r in adapter["results"] if r["tag"] == sel_tag][0]["alpha"]
bank = torch.load(RUN/"banks_prop1b.pt", weights_only=True, map_location="cpu", mmap=True)
target = geo1b.load_target_1b("cuda:0")
editor = FullMaskEditor(target, bank, "cuda:0")
del bank
cache = {}
sets = {"german_europarl": data["de_eval"], "english_pile": data["pile_en_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:]}
out = {}
for tau in (0.0, 0.2, 0.5, 1.0, 2.0):
    a = alpha_full.clone()
    dev = (a - 1.0).abs()
    a[dev < tau] = 1.0
    kept = int((dev >= tau).sum()) if tau > 0 else int((dev > 0.05).sum())
    a_gpu = a.to("cuda:0")
    row = {name: language_metrics(editor, idx, a_gpu, cache, name)
           for name, idx in sets.items()}
    out[str(tau)] = {"components_kept": kept,
                     **{n: round(m["delta_ce"], 3) for n, m in row.items()}}
    log(f"tau={tau:g} kept={kept}: de={row['german_europarl']['delta_ce']:+.2f} "
        f"en={row['english_pile']['delta_ce']:+.2f} "
        f"fr/es/it={row['french_europarl_heldout']['delta_ce']:+.2f}/"
        f"{row['spanish_europarl_heldout']['delta_ce']:+.2f}/"
        f"{row['italian_europarl_heldout']['delta_ce']:+.2f}")
(RUN/"german_fullmask_sparsity.json").write_text(json.dumps(out, indent=2))
log("wrote sparsity analysis")
