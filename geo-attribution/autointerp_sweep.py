"""GPU-resident evidence sweep — the scaling version of autointerp_stream.

autointerp_stream keeps candidate evidence as Python objects: per batch it
builds ~7k tuples, each holding a `.tolist()`ed token window, then re-sorts
those lists every few batches. Measured on a B200 that costs 60% of wall-clock
(45.2k -> 18.2k positions/s) and degrades as the live-object count grows, because
generational GC has to walk millions of small objects.

Here the running top-K per component lives entirely in GPU tensors — scores,
document hashes, mark offsets, and the raw token windows — merged with a
`cat` + `topk` per batch. Nothing is materialized on the CPU until the very
end, so throughput is flat and sits at the GPU's own ceiling.

Per-document dedup is preserved exactly: each candidate carries a content hash
of its source document, computed on GPU, and the `max_per_doc` cap is applied
once during the final decode.

  # both GPUs, one shard each
  python3.12 autointerp_sweep.py sweep --rank 0 --world 2 --device cuda:0 &
  python3.12 autointerp_sweep.py sweep --rank 1 --world 2 --device cuda:1 &
  wait
  python3.12 autointerp_sweep.py merge --world 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import (make_loader, pass_features, sampled_batch,
                               setup_model)
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args

WINDOW = 48      # tokens per evidence window
BEFORE = 40      # firing token's offset within it (clamped at sequence edges)
K_CAND = 4       # candidates harvested per component per batch


def common_args(parser):
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--keep", type=int, default=256,
                        help="candidates retained per component; must exceed "
                             "--topk by enough that per-document dedup still "
                             "has survivors to draw on")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=555)


def document_hashes(idx, bos_id, weights):
    """Content hash of the source document of every position in the batch.

    Keys on the 24 tokens at the last BOS at or before the position, so two
    windows drawn from the same document collide by construction — which is
    what lets the per-document cap be applied later, on tensors, instead of
    during the sweep.
    """
    B, L = idx.shape
    ar = torch.arange(L, device=idx.device)
    marks = torch.where(idx == bos_id, ar, torch.full_like(ar, -1))
    start = torch.cummax(marks, dim=1).values.clamp_(min=0)     # [B, L]
    off = (start[:, :, None] + torch.arange(24, device=idx.device)).clamp_(
        max=L - 1)
    rows = torch.arange(B, device=idx.device)[:, None, None]
    return (idx[rows, off].long() * weights).sum(-1)            # [B, L]


def stage_sweep(args):
    device = args.device
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    torch.manual_seed(args.seed)

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    meta = {k: bank[k] for k in ("format", "C", "sensor", "gim_tau", "scalar")
            if k in bank}
    del bank
    cfg = ranking_args(meta)
    cap = setup_model(cfg, device)
    spec, scales, dim = load_spec(run_dir, device)
    model = load_stream_model(run_dir / "stream_model.pt", device)
    C = int(model["config"]["C"])
    loader = make_loader(args, cap, args.rank, args.world)
    gen = torch.Generator().manual_seed(args.seed + args.rank)
    bos_id = cap.target.hf.config.bos_token_id
    hash_w = torch.randint(1, 2 ** 61, (24,), device=device,
                           generator=torch.Generator(device=device).manual_seed(7),
                           dtype=torch.long) * 2 + 1

    win_ar = torch.arange(WINDOW, device=device)
    best_s = torch.full((C, args.keep), -1.0, device=device)
    best_d = torch.zeros((C, args.keep), dtype=torch.long, device=device)
    best_m = torch.zeros((C, args.keep), dtype=torch.long, device=device)
    best_w = torch.zeros((C, args.keep, WINDOW), dtype=torch.int32,
                         device=device)
    fire = torch.zeros(C, device=device)
    usage = torch.zeros(C, device=device)
    n_tok = 0
    t0 = time.time()
    for b in range(args.batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        phi, _ = pass_features(cfg, cap, idx, pos, bi, spec, scales, dim)
        x = phi.clamp(-6e4, 6e4).half().float()
        y = F.normalize((x - model["mean"]) @ model["projector"], dim=1)
        sims = y @ model["centroids"].t()
        posterior = torch.softmax(sims / args.rank_temperature, dim=1)
        fire += torch.bincount(sims.argmax(1), minlength=C).float()
        usage += posterior.sum(0)
        n_tok += posterior.shape[0]

        vals, rows = posterior.t().topk(K_CAND, dim=1)           # [C, K]
        vals = vals.masked_fill(vals < args.min_posterior, -1.0)
        L = idx.shape[1]
        doc = document_hashes(idx, bos_id, hash_w)
        cpos = pos.reshape(-1)[rows]
        cbi = bi.reshape(-1)[rows]
        cdoc = doc[cbi, cpos]
        lo = (cpos - BEFORE).clamp_(0, L - WINDOW)
        cmark = cpos - lo
        cwin = idx[cbi[..., None].expand(-1, -1, WINDOW),
                   lo[..., None] + win_ar].int()                 # [C, K, W]

        best_s, order = torch.cat([best_s, vals], 1).topk(args.keep, dim=1)
        best_d = torch.cat([best_d, cdoc], 1).gather(1, order)
        best_m = torch.cat([best_m, cmark], 1).gather(1, order)
        best_w = torch.cat([best_w, cwin], 1).gather(
            1, order[..., None].expand(-1, -1, WINDOW))
        if b % 64 == 0:
            log(f"rank{args.rank} sweep {b}/{args.batches} "
                f"({time.time() - t0:.0f}s, {n_tok:,} positions, "
                f"{n_tok / max(time.time() - t0, 1e-9) / 1000:.1f}k pos/s)")

    out = run_dir / f"sweep_{args.banks_tag}_r{args.rank}.pt"
    torch.save({"score": best_s.cpu(), "doc": best_d.cpu(),
                "mark": best_m.cpu(), "window": best_w.cpu(),
                "fire": fire.cpu(), "usage": usage.cpu(), "n_tok": n_tok,
                "keep": args.keep}, out)
    log(f"rank{args.rank} done: {n_tok:,} positions in "
        f"{time.time() - t0:.0f}s "
        f"({n_tok / (time.time() - t0) / 1000:.1f}k pos/s) -> {out}")


def stage_merge(args):
    run_dir = args.artifact_root / args.tag
    shards = [torch.load(run_dir / f"sweep_{args.banks_tag}_r{r}.pt",
                         weights_only=True, map_location="cpu")
              for r in range(args.world)]
    log(f"merging {len(shards)} shard(s), "
        f"{sum(s['n_tok'] for s in shards):,} positions")
    score = torch.cat([s["score"] for s in shards], 1)
    doc = torch.cat([s["doc"] for s in shards], 1)
    mark = torch.cat([s["mark"] for s in shards], 1)
    window = torch.cat([s["window"] for s in shards], 1)
    keep = min(args.keep, score.shape[1])
    score, order = score.topk(keep, dim=1)
    doc = doc.gather(1, order)
    mark = mark.gather(1, order)
    window = window.gather(1, order[..., None].expand(-1, -1, WINDOW))
    fire = sum(s["fire"] for s in shards)
    usage = sum(s["usage"] for s in shards)
    n_tok = sum(s["n_tok"] for s in shards)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    C = score.shape[0]
    out = {}
    for c in range(C):
        examples, seen = [], {}
        s_c = score[c].tolist()
        d_c = doc[c].tolist()
        m_c = mark[c].tolist()
        for j, v in enumerate(s_c):            # already sorted descending
            if v < args.min_posterior or len(examples) >= args.topk:
                break
            key = d_c[j]
            if seen.get(key, 0) >= args.max_per_doc:
                continue
            seen[key] = seen.get(key, 0) + 1
            ids = window[c, j].tolist()
            k = m_c[j]
            examples.append({
                "share": round(v, 4),
                "text": (f"{tokenizer.decode(ids[:k])}"
                         f"«{tokenizer.decode([ids[k]])}»"
                         f"{tokenizer.decode(ids[k + 1:])}")})
        out[str(c)] = {"examples": examples,
                       "fire_rate": (fire[c] / n_tok).item(),
                       "mean_share": (usage[c] / n_tok).item()}
        if c % 512 == 0:
            log(f"decoded {c}/{C}")
    path = run_dir / (args.out or f"evidence_{args.banks_tag}.json")
    path.write_text(json.dumps(out))
    live = sum(1 for v in out.values() if v["examples"])
    full = sum(1 for v in out.values() if len(v["examples"]) >= args.topk)
    log(f"merge done: {live}/{C} components with examples, {full} at the full "
        f"{args.topk}, {n_tok:,} positions -> {path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    sw = sub.add_parser("sweep")
    common_args(sw)
    sw.add_argument("--rank", type=int, default=0)
    sw.add_argument("--world", type=int, default=1)
    sw.add_argument("--device", default="cuda")
    sw.add_argument("--batches", type=int, default=64)
    sw.add_argument("--batch_seqs", type=int, default=16)
    sw.add_argument("--seq_len", type=int, default=512)
    sw.add_argument("--pos_per_seq", type=int, default=506)
    sw.add_argument("--rank_temperature", type=float, default=0.05)
    sw.add_argument("--min_posterior", type=float, default=0.02)
    sw.add_argument("--data_path", type=Path, default=geo1b.BIN_PATH)
    sw.add_argument("--synthetic_data", action="store_true")
    sw.add_argument("--data_order", default="sequential")

    mg = sub.add_parser("merge")
    common_args(mg)
    mg.add_argument("--world", type=int, default=1)
    mg.add_argument("--max_per_doc", type=int, default=1)
    mg.add_argument("--min_posterior", type=float, default=0.02)
    mg.add_argument("--out", default=None)

    args = parser.parse_args()
    {"sweep": stage_sweep, "merge": stage_merge}[args.stage](args)


if __name__ == "__main__":
    main()
