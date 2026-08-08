"""Which component carries hazardous biology?

Same machinery as the German ranking, with the contrast chosen to make the
question non-trivial. Ranking hazardous-bio text against ordinary English finds
"a biology component" and proves nothing; ranking it against benign PROSE finds
the multiple-choice layout, which is worse — the first run's top hit was a
list-item-marker component. So every control is subtracted at once:

    posterior(hazardous bio) - max over {MMLU-bio MCQ, benign-bio prose, Pile}

MMLU-bio is the load-bearing one: identical A./B./C./D. template, benign
framing, and MMLU even has a 'virology' subject, so layout AND topic cancel and
what survives is what the decomposition allocates to the hazardous framing.

Ranking sees only the TRAIN questions. Dev and eval are untouched.

    python3.12 wmdp_rank.py --top 24
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401 - installs the 1B target bindings
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args

# mmlu_bio is the load-bearing control: same A./B./C./D. template, benign
# biology (MMLU even has a 'virology' subject). Without it the top hit is a
# list-item-marker component and the ranking measures layout, not knowledge.
CORPORA = ["bio_hazard", "mmlu_bio", "bio_retain", "pile"]
CONTROLS = ["mmlu_bio", "bio_retain", "pile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=geo1b.SHM_ROOT / "run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--rank_positions", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="wmdp_ranking.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    data = torch.load(args.run_dir / args.data, weights_only=False,
                      map_location="cpu")
    bank_meta = torch.load(args.run_dir / f"banks_{args.banks_tag}.pt",
                           weights_only=True, map_location="cpu", mmap=True)
    C = bank_meta["C"]
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(args.run_dir, dev)
    stream_model = load_stream_model(args.run_dir / "stream_model.pt", dev)

    # one block per corpus keeps the position budget identical across corpora
    idx = torch.cat([data[f"{c}_rank"][:1] for c in CORPORA]).to(dev)
    available = torch.arange(4, args.seq_len - 2)
    g = torch.Generator().manual_seed(args.seed)
    sel = available[torch.randperm(available.numel(), generator=g)[
        :min(args.rank_positions, available.numel())]]
    pos = sel[None].expand(len(CORPORA), -1).to(dev)
    bi = torch.arange(len(CORPORA), device=dev)[:, None].expand_as(pos)

    t0 = time.perf_counter()
    phi, _ = pass_features(cfg, cap, idx, pos, bi, spec, scales, dim,
                           return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize((x - stream_model["mean"]) @ stream_model["projector"],
                    dim=1)
    sim = y @ stream_model["centroids"].t()
    post = torch.softmax(sim / args.temperature, dim=1)
    post = post.view(len(CORPORA), pos.shape[1], -1).mean(1)
    labels = sim.argmax(1).view(len(CORPORA), -1)
    hard = torch.stack([torch.bincount(r, minlength=C).float() / r.numel()
                        for r in labels])

    ci = {c: i for i, c in enumerate(CORPORA)}
    haz = post[ci["bio_hazard"]]
    # specificity, not raw activity: subtract the STRONGEST control, so a
    # candidate has to beat the MCQ format, benign biology and ordinary
    # English all at once before it counts as hazard-specific
    ctrl = torch.stack([post[ci[c]] for c in CONTROLS]).max(0).values
    contrast = haz - ctrl
    order = contrast.argsort(descending=True)[:args.top]

    rows = []
    for c in order.tolist():
        rows.append({
            "component": c,
            "hazard_specificity": float(contrast[c]),
            "haz_minus_mcq_control": float(haz[c] - post[ci["mmlu_bio"], c]),
            "posterior": {k: float(post[ci[k], c]) for k in CORPORA},
            "hard_frequency": {k: float(hard[ci[k], c]) for k in CORPORA},
        })
    result = {
        "method": "frozen attribution-fingerprint cluster posterior",
        "contrast": "hazardous-bio minus max(mmlu-bio-MCQ, benign-bio-prose, pile)",
        "temperature": args.temperature,
        "positions_per_corpus": int(pos.shape[1]),
        "C": C,
        "selected_component": rows[0]["component"],
        "candidates": rows,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (args.run_dir / args.out).write_text(json.dumps(result, indent=2))
    log(f"wrote {args.run_dir / args.out}")
    for r in rows[:12]:
        p = r["posterior"]
        log(f"  c{r['component']:<5} spec {r['hazard_specificity']:+.5f}  "
            f"haz {p['bio_hazard']:.5f}  benign {p['bio_retain']:.5f}  "
            f"pile {p['pile']:.5f}")


if __name__ == "__main__":
    main()
