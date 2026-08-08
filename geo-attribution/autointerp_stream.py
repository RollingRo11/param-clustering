"""Fast auto-interp evidence via the streaming fingerprint posterior.

The exact-attribution evidence stage (autointerp67) einsums over every
component slice (~15 min/batch at C=4096). The decomposition's own frozen
assignment gives the same "which component fires here" signal at collection
speed: fingerprint -> centered/projected/normalized -> cluster posterior.
Top-activating context windows per component are gathered from hundreds of
thousands of positions in minutes, in autointerp67's evidence schema, so the
existing API label stage (or manual labeling) consumes it unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import (make_loader, pass_features, sampled_batch,
                               setup_model)
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch_seqs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pos_per_seq", type=int, default=256)
    parser.add_argument("--rank_temperature", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--max_per_doc", type=int, default=1,
                        help="cap on examples from one source document; "
                             "1 forces every example of a component to come "
                             "from a different document, so apparent "
                             "coherence cannot be document memorization")
    parser.add_argument("--min_posterior", type=float, default=0.02)
    parser.add_argument("--window_before", type=int, default=40)
    parser.add_argument("--window_after", type=int, default=8)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--out", default=None,
                        help="output filename (default evidence_<banks_tag>.json)")
    parser.add_argument("--data_path", type=Path, default=geo1b.BIN_PATH)
    parser.add_argument("--synthetic_data", action="store_true")
    parser.add_argument("--data_order", default="sequential")
    args = parser.parse_args()
    run_dir = args.artifact_root / args.tag
    device = "cuda"
    torch.manual_seed(args.seed)

    bank_meta = {}
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    for key in ("format", "C", "sensor", "gim_tau", "scalar"):
        if key in bank:
            bank_meta[key] = bank[key]
    del bank
    cfg = ranking_args(bank_meta)
    cap = setup_model(cfg, device)
    spec, scales, dim = load_spec(run_dir, device)
    model = load_stream_model(run_dir / "stream_model.pt", device)
    C = int(model["config"]["C"])
    loader = make_loader(args, cap, 0, 1)
    generator = torch.Generator().manual_seed(args.seed)

    bos_id = cap.target.hf.config.bos_token_id

    def prune(entries, limit):
        """Top entries by posterior with at most max_per_doc per document."""
        seen: dict[int, int] = {}
        kept = []
        for entry in sorted(entries, key=lambda x: -x[0]):
            key = entry[3]
            if seen.get(key, 0) >= args.max_per_doc:
                continue
            seen[key] = seen.get(key, 0) + 1
            kept.append(entry)
            if len(kept) >= limit:
                break
        return kept

    K_CAND = 4
    cand: list[list] = [[] for _ in range(C)]
    fire = torch.zeros(C, device=device)
    usage = torch.zeros(C, device=device)
    n_tok = 0
    t0 = time.time()
    for b in range(args.batches):
        idx, pos, bi = sampled_batch(loader, generator, device,
                                     args.pos_per_seq)
        phi, _ = pass_features(cfg, cap, idx, pos, bi, spec, scales, dim)
        x = phi.clamp(-6e4, 6e4).half().float()
        y = F.normalize((x - model["mean"]) @ model["projector"], dim=1)
        sims = y @ model["centroids"].t()
        posterior = torch.softmax(sims / args.rank_temperature, dim=1)
        assigned = sims.argmax(1)
        fire += torch.bincount(assigned, minlength=C).float()
        usage += posterior.sum(0)
        n_tok += posterior.shape[0]
        vals, rows = posterior.t().topk(K_CAND, dim=1)     # [C, K_CAND]
        vals, rows = vals.cpu(), rows.cpu()
        pos_flat = pos.reshape(-1).cpu()
        bi_flat = bi.reshape(-1).cpu()
        idx_cpu = idx.cpu()
        # Document key per (row, position): content hash of the 24 tokens at
        # the row's last BOS before the position. Examples sharing a key came
        # from the same source document.
        doc_keys = {}
        for row_i in range(idx_cpu.shape[0]):
            starts = (idx_cpu[row_i] == bos_id).nonzero().flatten().tolist()
            doc_keys[row_i] = (starts, [
                hash(tuple(idx_cpu[row_i, s:s + 24].tolist())) for s in starts]
                + [hash(tuple(idx_cpu[row_i, :24].tolist()))])

        def doc_key(row_i, position):
            starts, hashes = doc_keys[row_i]
            index = -1
            for j, s in enumerate(starts):
                if s <= position:
                    index = j
            return hashes[index]

        keep = (vals[:, 0] >= args.min_posterior).nonzero().flatten()
        for c in keep.tolist():
            for k in range(K_CAND):
                v = vals[c, k].item()
                if v < args.min_posterior:
                    break
                r = int(rows[c, k])
                bb, tt = int(bi_flat[r]), int(pos_flat[r])
                lo = max(0, tt - args.window_before)
                hi = min(idx_cpu.shape[1], tt + args.window_after)
                cand[c].append((v, idx_cpu[bb, lo:hi].tolist(), tt - lo,
                                doc_key(bb, tt)))
        if b % 8 == 0:
            for c in range(C):
                if len(cand[c]) > 8 * args.topk:
                    cand[c] = prune(cand[c], 4 * args.topk)
            log(f"evidence batch {b}/{args.batches} "
                f"({time.time() - t0:.0f}s, {n_tok:,} positions)")
    for c in range(C):
        cand[c] = prune(cand[c], args.topk)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    out = {}
    for c in range(C):
        examples = []
        for v, ids, mark, _ in cand[c]:
            pre = tokenizer.decode(ids[:mark])
            cur = tokenizer.decode([ids[mark]])
            post = tokenizer.decode(ids[mark + 1:])
            examples.append({"share": round(v, 4),
                             "text": f"{pre}«{cur}»{post}"})
        out[str(c)] = {"examples": examples,
                       "fire_rate": (fire[c] / n_tok).item(),
                       "mean_share": (usage[c] / n_tok).item()}
    output = run_dir / (args.out or f"evidence_{args.banks_tag}.json")
    output.write_text(json.dumps(out))
    live = sum(1 for c in out.values() if c["examples"])
    log(f"evidence done: {live}/{C} components with examples, "
        f"{n_tok:,} positions -> {output}")


if __name__ == "__main__":
    main()
