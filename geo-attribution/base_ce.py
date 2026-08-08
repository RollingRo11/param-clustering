"""Unedited-model CE on every block set the fair sweep reports against.

The sweep only ever stores deltas, so an export that wants to show absolute
cross-entropy needs the baseline it was subtracted from. Same block sets, same
tokenizer, same model revision as lora_fair_sweep.py.
"""
import argparse
import json
import math
from pathlib import Path

import torch

import geo1b  # noqa: F401
import budget_race as br
from german_vpd_1b import prepare_data
from lora_fair_sweep import make_sets

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run1b_streamC4096")
ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--train_tokens", type=int, default=2048)
ap.add_argument("--eval_blocks", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--refresh_data", action="store_true")
args = ap.parse_args()
args.run_dir = args.artifact_root / args.tag
args.data_cache = args.run_dir / "german_vpd_data.pt"
torch.manual_seed(args.seed)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                    revision=geo1b.MODEL_REVISION)
data = prepare_data(args, tok)
report, dev = make_sets(data)

target = geo1b.load_target_1b("cuda:0")
hf = target.hf
out = {}
for name, idx in list(report.items()) + list(dev.items()):
    idx = idx.to("cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=True):
        lg = hf(idx).logits
    out[name] = {"base_ce_nats": round(br.ce_each(lg, idx).mean().item(), 6),
                 "blocks": int(idx.shape[0]), "seq_len": int(idx.shape[1])}
    del lg
out["_uniform_ce_nats"] = round(math.log(128256), 6)
print(json.dumps(out, indent=1))
Path(args.run_dir / "fair_sweep_base_ce.json").write_text(json.dumps(out, indent=1))
