"""Auto-interp for the geo-attribution 67M partition decomposition.

Stage `evidence` (GPU): stream real Pile batches, compute IG attribution shares
through the partition decomposition, keep the top-K activating contexts per
component (token windows), plus fire-rate stats.
Stage `label` (network): label every component with evidence via the Anthropic
API (org key from param-decomp/.env, bulk-labeling recipe: claude-sonnet-4-6).

  python3.12 autointerp67.py evidence --banks_tag part8
  python3.12 autointerp67.py label   --banks_tag part8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")
sys.path.insert(0, "/workspace/param-decomp")

import torch

from geo67 import OUT_ROOT, GatedRunner, load_target, log

TOKENIZER = OUT_ROOT / "target_local" / "tokenizer.json"


def stage_evidence(args):
    device = "cuda"
    target = load_target(device)
    bk = torch.load(args.dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                    map_location="cpu")
    run = GatedRunner(target, bk, device)
    C = bk["C"]
    from nano_param_decomp.pile_4L import make_loader
    loader = make_loader(args.batch_seqs, args.seq_len, 0, 1, "train", 555)

    K_KEEP, K_CAND = args.topk, 4
    cand: list[list] = [[] for _ in range(C)]     # per comp: (score, window_ids, pos_in_window)
    fire = torch.zeros(C, device=device)
    usage = torch.zeros(C, device=device)
    n_tok = 0
    t0 = time.time()
    for b in range(args.batches):
        idx = next(loader).to(device)
        B, T = idx.shape
        attr, _ = run.attribution(idx, args.ig_k)
        share = attr / attr.sum(-1, keepdim=True).clamp_min(1e-30)
        share[:, :2] = 0
        share[:, -2:] = 0                          # position guards
        fire += (share > 0.05).float().sum((0, 1))
        usage += share.sum((0, 1))
        n_tok += B * (T - 4)
        flat = share.reshape(-1, C)
        vals, pos = flat.topk(K_CAND, dim=0)       # [K_CAND, C]
        vals, pos = vals.cpu(), pos.cpu()
        idx_cpu = idx.cpu()
        for c in range(C):
            for k in range(K_CAND):
                v = vals[k, c].item()
                if v < 0.02:
                    continue
                p = int(pos[k, c])
                bb, tt = p // T, p % T
                lo = max(0, tt - 40)
                cand[c].append((v, idx_cpu[bb, lo:tt + 8].tolist(), tt - lo))
        if b % 10 == 0:
            for c in range(C):
                cand[c] = sorted(cand[c], key=lambda x: -x[0])[:K_KEEP]
            log(f"evidence batch {b}/{args.batches} ({time.time()-t0:.0f}s)")
    for c in range(C):
        cand[c] = sorted(cand[c], key=lambda x: -x[0])[:K_KEEP]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    out = {}
    for c in range(C):
        exs = []
        for v, ids, mark in cand[c]:
            pre = tok.decode(ids[:mark])
            cur = tok.decode([ids[mark]])
            post = tok.decode(ids[mark + 1:])
            exs.append({"share": round(v, 4), "text": f"{pre}«{cur}»{post}"})
        out[c] = {"examples": exs,
                  "fire_rate": (fire[c] / n_tok).item(),
                  "mean_share": (usage[c] / n_tok).item()}
    (args.dir / f"evidence_{args.banks_tag}.json").write_text(json.dumps(out))
    live = sum(1 for c in out.values() if c["examples"])
    log(f"evidence done: {live}/{C} components with examples, {n_tok} positions")


SYSTEM = """You label components of a decomposed language model. Each component \
is shown with contexts from real text where it was most active; the token where \
it fired is marked with «guillemets». The model predicts the NEXT token, so a \
component may respond to the current token, the context, or what comes next.
Reply with ONLY a JSON object: {"label": "<concise functional label, <=8 words>", \
"category": "<one of: current-token, next-token-prediction, syntax, \
semantic-topic, boundary, other>", "confidence": "<high|medium|low>"}. \
Use confidence low and label "polysemantic/unclear" if the examples share no \
coherent pattern."""


def stage_label(args):
    from dotenv import load_dotenv
    load_dotenv("/workspace/param-decomp/.env")
    import anthropic
    client = anthropic.Anthropic()
    ev = json.loads((args.dir / f"evidence_{args.banks_tag}.json").read_text())
    cat_path = args.dir / f"catalog_{args.banks_tag}.json"
    catalog = json.loads(cat_path.read_text()) if cat_path.exists() else {}

    def label_one(c):
        info = ev[c]
        if not info["examples"]:
            return c, {"label": "dead/no-evidence", "category": "other",
                       "confidence": "high", **{k: info[k] for k in
                                                ("fire_rate", "mean_share")}}
        body = "\n\n".join(f"[share {e['share']}] ...{e['text']}..."
                           for e in info["examples"][:16])
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=200, system=SYSTEM,
                messages=[{"role": "user", "content": body}])
            txt = resp.content[0].text.strip()
            txt = txt[txt.index("{"):txt.rindex("}") + 1]
            d = json.loads(txt)
        except Exception as e:  # noqa: BLE001 — keep labeling the rest
            d = {"label": f"ERROR {type(e).__name__}", "category": "other",
                 "confidence": "low"}
        d.update({k: info[k] for k in ("fire_rate", "mean_share")})
        return c, d

    todo = [c for c in ev if c not in catalog]
    log(f"labeling {len(todo)} components")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (c, d) in enumerate(pool.map(label_one, todo)):
            catalog[c] = d
            if i % 32 == 0:
                cat_path.write_text(json.dumps(catalog, indent=1))
                log(f"labeled {i}/{len(todo)}")
    cat_path.write_text(json.dumps(catalog, indent=1))
    n_hi = sum(1 for d in catalog.values() if d["confidence"] == "high")
    n_poly = sum(1 for d in catalog.values() if "polysemantic" in d["label"]
                 or "unclear" in d["label"])
    n_dead = sum(1 for d in catalog.values() if d["label"] == "dead/no-evidence")
    log(f"catalog: {len(catalog)} labeled, {n_hi} high-conf, {n_poly} "
        f"polysemantic/unclear, {n_dead} dead")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["evidence", "label"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--banks_tag", default="part8")
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--ig_k", type=int, default=2)
    ap.add_argument("--topk", type=int, default=16)
    args = ap.parse_args()
    args.dir = OUT_ROOT / args.tag
    {"evidence": stage_evidence, "label": stage_label}[args.stage](args)


if __name__ == "__main__":
    main()
