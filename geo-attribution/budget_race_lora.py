"""Standalone LoRA arm of the budget race (GPU selected via CUDA_VISIBLE_DEVICES)."""
import json, math
from pathlib import Path
from types import SimpleNamespace
import torch
import geo1b, geo67
from german_vpd_1b import log, prepare_data
from german_lora_guided import GuidedLora
from budget_race import run_lora

RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
args = SimpleNamespace(seq_len=512, train_tokens=2048, eval_blocks=4,
                       refresh_data=False, data_cache=RUN/"german_vpd_data.pt")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
data = prepare_data(args, tok)
ceiling = math.log(128256)
target = geo1b.load_target_1b("cuda:0")
model = GuidedLora(target, geo67.MODULES, 1, "cuda:0", 0, masks=None)
cache = {}
rows = []
for budget in (8, 64, 512, 2048):
    for lam_en, lam_rom in ((10.0, 10.0), (30.0, 30.0)):
        for lr in (3e-3, 1e-2):
            tag = f"lora B={budget} l={lam_en:g}/{lam_rom:g} lr={lr:g}"
            row = run_lora(model, data, budget, lam_en, lam_rom, lr,
                           400, ceiling, cache, tag)
            rows.append({"arm": "lora", "budget": budget, "lam_en": lam_en,
                         "lam_rom": lam_rom, "lr": lr, **row})
(RUN/"budget_race_lora.json").write_text(json.dumps(rows, indent=2))
log("wrote lora arm json")
