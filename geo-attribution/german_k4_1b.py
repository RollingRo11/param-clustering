"""k=4 German edit deep-dive on the 1B decomposition:
  (a) alpha sweep for the top-4 component edit,
  (b) each top-4 component edited alone (deletion + inversion),
  (c) per-component ownership profile: which matrices/rows hold its mass.
Reuses the ranking from german_prop1b_weight.json (same protocol/splits)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import torch

import geo1b  # noqa: F401 — patches geo67 for the 1B target
import geo67
from geo67 import GatedRunner, log
from german67 import ENGLISH, GERMAN, chunks_from, ce_per_tok

D = Path("/dev/shm/geo1b/run1")
device = "cuda"

target = geo67.load_target(device)
bk = torch.load(D / "banks_prop1b.pt", weights_only=True, map_location="cpu")
run = GatedRunner(target, bk, device)
prev = json.loads((D / "german_prop1b_weight.json").read_text())
top4 = prev["contrast_top16"][:4]
log(f"top4 german comps: {top4}")

from tokenizers import Tokenizer
tok = Tokenizer.from_file("/dev/shm/geo1b/target_local/tokenizer.json")
half = len(GERMAN) // 2
eval_de = chunks_from(GERMAN[half:], tok, 192, device)
eval_en = chunks_from(ENGLISH[half:], tok, 192, device)


def weight_ce(idx, comps=None, alpha=0.0):
    saved = {}
    if comps is not None:
        comps_t = torch.tensor(comps, device=device)
        for p in bk["modules"]:
            lin = target.get_submodule(p)
            share = run.component_share(p, comps_t)
            saved[p] = lin.weight.data.clone()
            lin.weight.data *= (1.0 - (1.0 - alpha) * share)
    with torch.no_grad():
        lt, _ = run.target_pass(idx)
    for p, w in saved.items():
        target.get_submodule(p).weight.data.copy_(w)
    return ce_per_tok(lt, idx)


base_de, base_en = weight_ce(eval_de), weight_ce(eval_en)
log(f"base CE de/en {base_de:.3f}/{base_en:.3f}")
res = {"top4": top4, "base_ce": {"de": base_de, "en": base_en},
       "alpha_sweep_k4": {}, "solo": {}, "profile": {}}

for alpha in [0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0]:
    dde = weight_ce(eval_de, top4, alpha) - base_de
    den = weight_ce(eval_en, top4, alpha) - base_en
    res["alpha_sweep_k4"][str(alpha)] = {"dce_de": dde, "dce_en": den}
    log(f"alpha {alpha:+.2f}: dCE de {dde:+.3f} en {den:+.3f}")
rnd_de, rnd_en = [], []
for s in range(3):
    g = torch.Generator().manual_seed(s)
    rnd = torch.randperm(bk["C"], generator=g)[:4].tolist()
    rnd_de.append(weight_ce(eval_de, rnd, -2.0) - base_de)
    rnd_en.append(weight_ce(eval_en, rnd, -2.0) - base_en)
res["alpha_sweep_k4"]["random_-2.0"] = {"dce_de": sum(rnd_de) / 3,
                                        "dce_en": sum(rnd_en) / 3}
log(f"random k4 alpha -2: de {sum(rnd_de)/3:+.3f} en {sum(rnd_en)/3:+.3f}")

for c in top4:
    solo = {}
    for name, alpha in [("del", 0.0), ("inv", -1.0)]:
        dde = weight_ce(eval_de, [c], alpha) - base_de
        den = weight_ce(eval_en, [c], alpha) - base_en
        solo[name] = {"dce_de": dde, "dce_en": den}
        log(f"solo c{c} {name}: dCE de {dde:+.3f} en {den:+.3f}")
    res["solo"][str(c)] = solo

for c in top4:
    ct = torch.tensor([c], device=device)
    permod = {}
    for p in bk["modules"]:
        W = target.get_submodule(p).weight.data
        share = run.component_share(p, ct)
        owned = (share * W).pow(2)
        permod[p] = {"mass": owned.sum().item(),
                     "toprow_frac": (owned.sum(1).max() /
                                     owned.sum().clamp_min(1e-30)).item()}
    tot = sum(v["mass"] for v in permod.values())
    top = sorted(permod.items(), key=lambda kv: -kv[1]["mass"])[:3]
    res["profile"][str(c)] = {
        "top_modules": [{"module": p, "frac": v["mass"] / tot,
                         "toprow_frac": v["toprow_frac"]} for p, v in top]}
    log(f"profile c{c}: " + ", ".join(
        f"{p.split('hf.model.layers.')[1]} {v['mass']/tot:.2f} "
        f"(toprow {v['toprow_frac']:.2f})" for p, v in top))

(D / "german_k4_1b.json").write_text(json.dumps(res, indent=1))
log("K4 " + json.dumps({k: res[k] for k in ("alpha_sweep_k4", "solo")}, indent=1))
