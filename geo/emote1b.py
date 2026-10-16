"""VPD Section-6 replication at 1B: find components that fire when predicting
emoticon tokens, then (A) amplify them ("always use some emoticon") and
(B) redirect their write-side action toward one target emoticon token
(the Llama analog of VPD's rewrite-the-write-vector ':o' edit — emoticons are
single tokens here, so the edit collapses emoticon CHOICE to one emoticon).
Read-side matrices are untouched (firing pattern preserved); on write-side
matrices the component's owned slice is replaced by a norm-matched rank-1
write toward the target token's unembedding direction."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401 — patches geo67 for the 1B target
import geo67
from geo67 import GatedRunner, is_write_side, log
from german67 import ENGLISH, chunks_from, ce_per_tok

import os
D = Path(os.environ.get("GEO_DIR", "/dev/shm/geo1b/run1"))
device = "cuda"
TARGET_STR = " :)"

EMO_RANK = [
    "Thanks so much for your help! :)",
    "See you tomorrow :)",
    "That was such a fun party :D",
    "I can't believe we won :D",
    "Sorry to hear that :(",
    "I missed the bus again :(",
    "Good luck on your exam :)",
    "This pizza is amazing :D",
    "My cat knocked over the plant :(",
    "Happy birthday!! :D",
    "No worries at all :)",
    "That movie made me cry :(",
    "Great job on the presentation :)",
    "I got the job! :D",
    "It's raining on my day off :(",
    "Welcome to the team :)",
    "You're the best :P",
    "Nice to meet you :)",
    "Can't wait for the weekend :D",
    "My phone battery died again :(",
]
EMO_EVAL = [
    "Thanks for coming to my show :)",
    "We finally finished the project :D",
    "The bakery was out of croissants :(",
    "Have a safe flight :)",
    "I aced the interview :D",
    "My favorite mug broke this morning :(",
    "Say hi to your family for me :)",
    "Snow day, no school :D",
    "The concert got cancelled :(",
    "Enjoy your vacation :)",
    "Our team won the hackathon :D",
    "I left my umbrella on the train :(",
]

TAG = sys.argv[1] if len(sys.argv) > 1 else "prop1b"
target = geo67.load_target(device)
bk = torch.load(D / f"banks_{TAG}.pt", weights_only=True, map_location="cpu")
run = GatedRunner(target, bk, device)
C = bk["C"]
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/dev/shm/geo1b/target_local/tokenizer.json")

emo_ids = set()
for s in [" :)", ":)", " :(", ":(", " :D", ":D", " :P", ":P", ":o", ":P"]:
    ids = tok.encode(s).ids
    ids = [i for i in ids if i != 128000]
    if len(ids) == 1:
        emo_ids.add(ids[0])
target_id = [i for i in tok.encode(TARGET_STR).ids if i != 128000][0]
log(f"emoticon token ids: {sorted(emo_ids)}; target {TARGET_STR!r} = {target_id}")

SEQ = 96
rank_emo = chunks_from(EMO_RANK, tok, SEQ, device)
eval_emo = chunks_from(EMO_EVAL, tok, SEQ, device)
rank_ctrl = chunks_from(ENGLISH[:12], tok, SEQ, device)
eval_ctrl = chunks_from(ENGLISH[12:], tok, SEQ, device)
emo_t = torch.tensor(sorted(emo_ids), device=device)


def pred_mask(idx):
    m = torch.zeros_like(idx, dtype=torch.bool)
    m[:, :-1] = torch.isin(idx[:, 1:], emo_t)
    return m


log(f"rank emo chunks {rank_emo.shape}, predicting positions "
    f"{pred_mask(rank_emo).sum().item()}")

attr_emo, _ = run.attribution(rank_emo, 2)
attr_ctrl, _ = run.attribution(rank_ctrl, 2)
sh_emo = attr_emo / attr_emo.sum(-1, keepdim=True).clamp_min(1e-30)
sh_ctrl = attr_ctrl / attr_ctrl.sum(-1, keepdim=True).clamp_min(1e-30)
contrast = sh_emo[pred_mask(rank_emo)].mean(0) - sh_ctrl[:, 2:-2].flatten(0, 1).mean(0)
order = contrast.argsort(descending=True)
log("top-8 emoticon-contrast comps: " + str(order[:8].tolist()))


def emo_metrics(idx):
    with torch.no_grad():
        lt, _ = run.target_pass(idx)
    m = pred_mask(idx)
    probs = F.softmax(lt[m].float(), -1)
    return {"p_target": probs[:, target_id].mean().item(),
            "p_emoset": probs[:, emo_t].sum(-1).mean().item(),
            "argmax_target_rate": (probs.argmax(-1) == target_id)
            .float().mean().item()}


def ctrl_metrics(idx):
    with torch.no_grad():
        lt, _ = run.target_pass(idx)
    probs = F.softmax(lt[:, 2:-1].float(), -1)
    return {"ce": ce_per_tok(lt, idx),
            "p_emoset_all": probs[..., emo_t].sum(-1).mean().item()}


def gen(prompt, n=22):
    ids = torch.tensor([[i for i in tok.encode(prompt).ids]], device=device)
    out = ids
    with torch.no_grad():
        for _ in range(n):
            lt, _ = run.target_pass(out)
            out = torch.cat([out, lt[:, -1:].argmax(-1)], -1)
    return tok.decode(out[0].tolist())


PROMPTS = ["Thanks again for dinner last night!",
           "I just got back from the best vacation ever.",
           "The meeting went really well today."]

saved = {}


def snapshot(paths):
    for p in paths:
        if p not in saved:
            saved[p] = target.get_submodule(p).weight.data.clone()


def restore():
    for p, w in saved.items():
        target.get_submodule(p).weight.data.copy_(w)
    saved.clear()


def amplify(comps, alpha):
    ct = torch.tensor(comps, device=device)
    snapshot(bk["modules"])
    for p in bk["modules"]:
        lin = target.get_submodule(p)
        share = run.component_share(p, ct)
        lin.weight.data *= (1.0 - (1.0 - alpha) * share)


def redirect(comps, gain):
    ct = torch.tensor(comps, device=device)
    d = target.get_submodule("hf.model.embed_tokens").weight[target_id].float()
    d = (d / d.norm()).to(device)
    wpaths = [p for p in bk["modules"] if is_write_side(p)]
    snapshot(wpaths)
    for p in wpaths:
        lin = target.get_submodule(p)
        A = run.component_share(p, ct) * lin.weight.data
        nrm = A.norm()
        if nrm < 1e-8:
            continue
        _, _, V = torch.svd_lowrank(A.float(), q=4)
        Anew = gain * nrm * torch.outer(d, V[:, 0])
        lin.weight.data += (Anew - A).to(lin.weight.dtype)


base = {"emo": emo_metrics(eval_emo), "ctrl": ctrl_metrics(eval_ctrl),
        "gens": [gen(p) for p in PROMPTS]}
log("BASE " + json.dumps({k: base[k] for k in ("emo", "ctrl")}))
for g in base["gens"]:
    log("base gen: " + repr(g))

res = {"emo_ids": sorted(emo_ids), "target_id": target_id,
       "contrast_top16": order[:16].tolist(), "base": base, "arms": {}}

for k, alpha in [(4, 2.0), (4, 4.0)]:
    comps = order[:k].tolist()
    amplify(comps, alpha)
    r = {"emo": emo_metrics(eval_emo), "ctrl": ctrl_metrics(eval_ctrl),
         "gens": [gen(p) for p in PROMPTS]}
    restore()
    res["arms"][f"amp_k{k}_a{alpha}"] = r
    log(f"AMP k{k} a{alpha}: emo {r['emo']} | ctrl {r['ctrl']}")
    for g in r["gens"]:
        log(f"amp k{k} a{alpha} gen: " + repr(g))

for k in [1, 4]:
    for gain in [1.0, 2.0, 4.0]:
        comps = order[:k].tolist()
        redirect(comps, gain)
        r = {"emo": emo_metrics(eval_emo), "ctrl": ctrl_metrics(eval_ctrl),
             "gens": [gen(p) for p in PROMPTS]}
        restore()
        res["arms"][f"redir_k{k}_g{gain}"] = r
        log(f"REDIR k{k} gain{gain}: emo {r['emo']} | ctrl {r['ctrl']}")
        for g in r["gens"]:
            log(f"redir k{k} g{gain} gen: " + repr(g))

(D / f"emote1b_{TAG}.json").write_text(json.dumps(res, indent=1))
log(f"emote1b done ({TAG})")
