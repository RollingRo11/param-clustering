"""Emoticon-choice collapse at 1B (C=2048): make sad contexts smile.
Finds valence-specific components via a sad-vs-happy emoticon attribution
contrast, then tests edits that convert ' :(' predictions into ' :)':
  arm A: additive smiley boost on the general emoticon component (prev recipe)
  arm B: replace the top sad component's write action with the ' :)' direction
  arm C: delete the sad component's owned mass + additive smiley boost
Reports P(' :('), P(' :)') and argmax at held-out sad-predicting positions."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import geo67
from geo67 import GatedRunner, is_write_side, log
from german67 import ENGLISH, chunks_from, ce_per_tok

import os
D = Path(os.environ.get("GEO_DIR", "/dev/shm/geo1b/run1"))
TAG = sys.argv[1] if len(sys.argv) > 1 else "propC2048"
device = "cuda"
SAD_ID, HAPPY_ID = 40624, 27046          # ' :(' and ' :)'

SAD_RANK = [
    "I missed the bus again :(",
    "My favorite mug broke this morning :(",
    "The concert got cancelled :(",
    "I left my umbrella on the train :(",
    "My plant died while I was away :(",
    "We lost the game in the last minute :(",
    "The bakery was out of croissants :(",
    "My phone screen cracked :(",
    "It rained through our whole camping trip :(",
    "I failed my driving test :(",
    "The library was closed when I got there :(",
    "My headphones stopped working :(",
]
HAPPY_RANK = [
    "Thanks so much for your help! :)",
    "See you tomorrow :)",
    "Good luck on your exam :)",
    "No worries at all :)",
    "Welcome to the team :)",
    "Nice to meet you :)",
    "Great job on the presentation :)",
    "Have a safe flight :)",
    "Say hi to your family for me :)",
    "Enjoy your vacation :)",
    "Thanks for coming to my show :)",
    "Happy to help anytime :)",
]
SAD_EVAL = [
    "My laptop crashed and I lost my essay :(",
    "The flight got delayed by five hours :(",
    "I burned the cookies again :(",
    "My bike got a flat tire on the way home :(",
    "The museum was closed for renovations :(",
    "I dropped my ice cream on the sidewalk :(",
    "We had to cancel the picnic because of rain :(",
    "My team got knocked out of the tournament :(",
    "I forgot my best friend's birthday :(",
    "The store sold out right before my turn :(",
]
HAPPY_EVAL = [
    "Dinner was wonderful, thank you :)",
    "Congrats on the new job :)",
    "See you at the reunion :)",
    "The package arrived early :)",
    "Your garden looks amazing :)",
    "I passed the certification exam :)",
    "It was great catching up today :)",
    "The kids loved the magic show :)",
]

target = geo67.load_target(device)
bk = torch.load(D / f"banks_{TAG}.pt", weights_only=True, map_location="cpu")
run = GatedRunner(target, bk, device)
prev = json.loads((D / f"emote1b_{TAG}.json").read_text())
c_emo = prev["contrast_top16"][0]
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/dev/shm/geo1b/target_local/tokenizer.json")

SEQ = 96
rank_sad = chunks_from(SAD_RANK, tok, SEQ, device)
rank_happy = chunks_from(HAPPY_RANK, tok, SEQ, device)
eval_sad = chunks_from(SAD_EVAL, tok, SEQ, device)
eval_happy = chunks_from(HAPPY_EVAL, tok, 64, device)
eval_ctrl = chunks_from(ENGLISH[12:], tok, SEQ, device)


def mask_for(idx, tid):
    m = torch.zeros_like(idx, dtype=torch.bool)
    m[:, :-1] = idx[:, 1:] == tid
    return m


def share_at(idx, tid):
    attr, _ = run.attribution(idx, 2)
    sh = attr / attr.sum(-1, keepdim=True).clamp_min(1e-30)
    return sh[mask_for(idx, tid)].mean(0)


contrast = share_at(rank_sad, SAD_ID) - share_at(rank_happy, HAPPY_ID)
order = contrast.argsort(descending=True)
# the general emoticon comp may top the sad contrast too (no frown-specific
# split at this C); target the top DISTINCT sad component for the sad arms
c_sad = next(int(c) for c in order if int(c) != c_emo)
log(f"top-8 sad-contrast comps: {order[:8].tolist()} "
    f"(general emoticon comp c{c_emo}, sad arm targets c{c_sad})")


def conv_metrics(idx, tid):
    with torch.no_grad():
        lt, _ = run.target_pass(idx)
    m = mask_for(idx, tid)
    p = F.softmax(lt[m].float(), -1)
    return {"p_sad": p[:, SAD_ID].mean().item(),
            "p_happy": p[:, HAPPY_ID].mean().item(),
            "argmax_sad": (p.argmax(-1) == SAD_ID).float().mean().item(),
            "argmax_happy": (p.argmax(-1) == HAPPY_ID).float().mean().item()}


def ctrl_ce():
    with torch.no_grad():
        lt, _ = run.target_pass(eval_ctrl)
    return ce_per_tok(lt, eval_ctrl)


def sample(prompt, temp=0.8, n=22, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    out = torch.tensor([tok.encode(prompt).ids], device=device)
    with torch.no_grad():
        for _ in range(n):
            lt, _ = run.target_pass(out)
            pr = F.softmax(lt[0, -1].float() / temp, -1)
            out = torch.cat([out, torch.multinomial(pr, 1, generator=g)[None]],
                            -1)
    return tok.decode(out[0].tolist())


saved = {}


def snap(paths):
    for p in paths:
        if p not in saved:
            saved[p] = target.get_submodule(p).weight.data.clone()


def restore():
    for p, w in saved.items():
        target.get_submodule(p).weight.data.copy_(w)
    saved.clear()


def happy_dir():
    d = target.get_submodule("hf.model.embed_tokens").weight[HAPPY_ID].float()
    return (d / d.norm()).to(device)


def additive(comp, gain):
    ct = torch.tensor([comp], device=device)
    d = happy_dir()
    wp = [q for q in bk["modules"] if is_write_side(q)]
    snap(wp)
    for p in wp:
        lin = target.get_submodule(p)
        A = run.component_share(p, ct) * lin.weight.data
        if A.norm() < 1e-8:
            continue
        _, _, V = torch.svd_lowrank(A.float(), q=4)
        lin.weight.data += (gain * A.norm() * torch.outer(d, V[:, 0])) \
            .to(lin.weight.dtype)


def replace(comp, gain):
    ct = torch.tensor([comp], device=device)
    d = happy_dir()
    wp = [q for q in bk["modules"] if is_write_side(q)]
    snap(wp)
    for p in wp:
        lin = target.get_submodule(p)
        A = run.component_share(p, ct) * lin.weight.data
        if A.norm() < 1e-8:
            continue
        _, _, V = torch.svd_lowrank(A.float(), q=4)
        lin.weight.data += (gain * A.norm() * torch.outer(d, V[:, 0]) - A) \
            .to(lin.weight.dtype)


def delete_comp(comp):
    ct = torch.tensor([comp], device=device)
    snap(bk["modules"])
    for p in bk["modules"]:
        lin = target.get_submodule(p)
        lin.weight.data *= (1.0 - run.component_share(p, ct))


SAD_PROMPTS = ["I missed my flight and lost my luggage",
               "My computer deleted all my files"]

res = {"c_emo": c_emo, "c_sad": c_sad, "sad_top8": order[:8].tolist(),
       "arms": {}}


def report(name):
    r = {"sad_pos": conv_metrics(eval_sad, SAD_ID),
         "happy_pos": conv_metrics(eval_happy, HAPPY_ID),
         "ctrl_ce": ctrl_ce(),
         "gens": [sample(p, seed=s) for p in SAD_PROMPTS for s in range(2)]}
    res["arms"][name] = r
    log(f"{name}: sad_pos {r['sad_pos']} | happy_pos "
        f"{r['happy_pos']} | ctrl {r['ctrl_ce']:.3f}")
    for g in r["gens"]:
        log(f"{name} gen: " + repr(g))


report("base")
for gain in [8.0, 16.0]:
    additive(c_emo, gain)
    report(f"add_cemo_g{gain}")
    restore()
for gain in [1.0, 2.0]:
    replace(c_sad, gain)
    report(f"repl_csad_g{gain}")
    restore()
delete_comp(c_sad)
additive(c_emo, 8.0)
report("del_csad_add_cemo_g8")
restore()

(D / f"emoconv1b_{TAG}.json").write_text(json.dumps(res, indent=1))
log("emoconv done")
