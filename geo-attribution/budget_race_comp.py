"""Standalone comp arm of the budget race at a chosen k (GPU via CUDA_VISIBLE_DEVICES)."""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace
import torch
import geo1b
import budget_race as br
from german_vpd_1b import log, prepare_data
from german_permatrix import COMPONENT_ORDER, PerMatrixEditor

k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
br.K = k
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
args = SimpleNamespace(seq_len=512, train_tokens=2048, eval_blocks=4,
                       refresh_data=False, data_cache=RUN/"german_vpd_data.pt")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
data = prepare_data(args, tok)
ceiling = math.log(128256)
bank = torch.load(RUN/"banks_prop1b.pt", weights_only=True, map_location="cpu", mmap=True)
target = geo1b.load_target_1b("cuda:0")
editor = PerMatrixEditor(target, bank, COMPONENT_ORDER, "cuda:0")
del bank
cache = {}
rows = []
for budget in (8, 64, 512, 2048):
    for lam_en, lam_rom in ((10.0, 10.0), (30.0, 30.0)):
        for lr in (0.1, 0.3):
            tag = f"comp{k} B={budget} l={lam_en:g}/{lam_rom:g} lr={lr:g}"
            row = br.run_comp(editor, data, budget, lam_en, lam_rom, lr,
                              400, ceiling, True, cache, tag)
            rows.append({"arm": f"comp_k{k}", "budget": budget,
                         "lam_en": lam_en, "lam_rom": lam_rom, "lr": lr, **row})
(RUN/f"budget_race_comp_k{k}.json").write_text(json.dumps(rows, indent=2))
log(f"wrote comp k={k} arm json")
