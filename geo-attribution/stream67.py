"""1B-token decomposition of VPD's 67M Pile target at large C, via reservoirs.

At C=8192 the in-memory path in sensor_study67 is impossible: the bank build
needs every pilot position resident, and with IG K=5 that is five copies of
(P, G). This uses the same trick the production 1B Llama pipeline uses --
stratified Algorithm-R reservoir sampling per cluster (streaming_decomposition.
reservoir_updates, imported unchanged) -- so memory is O(C * quota) rows rather
than O(tokens).

Three phases:
  fit     stratified pilot -> IG K=5 features -> PCA -> spherical k-means,
          saving centroids, the PCA basis and the coordinate spec
  stream  stream N tokens; per batch run the sensor, build features, project,
          assign to the nearest centroid, and let Algorithm-R decide which rows
          survive into the per-cluster reservoir
  bank    build the softpart bank from the reservoir, then evaluate per-input
          sufficiency exactly as sufficiency67 does

Data is stratified by meta.pile_set_name throughout (see pile_data), with
batched tokenisation because 1B tokens document-by-document is far too slow.

    python3.12 stream67.py fit    --C 8192 --sensor ig5
    python3.12 stream67.py stream --C 8192 --sensor ig5 --target_tokens 1_000_000_000
    python3.12 stream67.py bank   --C 8192 --sensor ig5
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS, load67, capture, build_spec, kmeans
from streaming_decomposition import reservoir_updates

RUN = Path(os.environ.get("STREAM67_RUN", "/dev/shm/geo67_stream"))

# cluster port: target ckpt + pile cache live under COFAC_DATA, not /dev/shm
if os.environ.get("COFAC_DATA"):
    import pile_data
    S67.CKPT = Path(os.environ["COFAC_DATA"]) / "target" / "model_step_99999.pt"
    pile_data.CACHE = Path(os.environ["COFAC_DATA"]) / "cofac67" / "piledata"

# TF32 matmul: measured 33k -> 80k tok/s for the IG K=5 sensor. Its 10-bit
# mantissa is the same precision the reservoir already stores (P, G) at.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def log(m):
    print(f"[stream67] {m}", flush=True)


# ------------------------------------------------------------------ data ----

def pile_stream(tok, seq, batch_blocks, text_batch=256, seed=0):
    """Yield [batch_blocks, seq] int64 tensors, round-robin over pile subsets.

    Round-robin keeps every prefix of the stream diverse, which matters because
    a run may be stopped early. Tokenisation is batched -- encoding 1B tokens
    one document at a time is the bottleneck otherwise.
    """
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    pend = collections.defaultdict(list)      # subset -> raw texts
    toks = collections.defaultdict(list)      # subset -> pending token ids
    blocks, labels = [], []
    for ex in ds:
        pend[ex["meta"]["pile_set_name"]].append(ex["text"])
        ready = [k for k, v in pend.items() if len(v) >= text_batch]
        if not ready:
            continue
        for k in sorted(ready):
            enc = tok(pend[k][:text_batch])["input_ids"]
            del pend[k][:text_batch]
            b = toks[k]
            for ids in enc:
                b.extend(ids)
            while len(b) >= seq:
                blocks.append(b[:seq])
                labels.append(k)
                del b[:seq]
        while len(blocks) >= batch_blocks:
            yield (torch.tensor(blocks[:batch_blocks], dtype=torch.long),
                   labels[:batch_blocks])
            del blocks[:batch_blocks]
            del labels[:batch_blocks]



TOKBIN = RUN / "pile_1b_uint16.bin"


def phase_prep(args):
    """Pre-tokenise a stratified corpus to a uint16 memmap.

    Tokenising inside the training loop caps throughput at ~25k tok/s, which is
    11 hours for 1B tokens. The fast tokenizer batched over many documents is
    ~100x that, so pay it once. vocab_size 50277 < 65536, so uint16 is exact.
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    RUN.mkdir(parents=True, exist_ok=True)
    out = TOKBIN
    if out.exists() and out.stat().st_size >= args.target_tokens * 2:
        log(f"{out.name} already complete ({out.stat().st_size/2**30:.1f} GiB)")
        return
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    pend = collections.defaultdict(list)
    written, t0 = 0, time.perf_counter()
    partial = out.with_suffix(".partial")
    with open(partial, "wb") as f:
        for ex in ds:
            pend[ex["meta"]["pile_set_name"]].append(ex["text"])
            ready = [k for k, v in pend.items() if len(v) >= args.text_batch]
            if not ready:
                continue
            texts = []
            for k in sorted(ready):          # round-robin: prefixes stay diverse
                texts.extend(pend[k][:args.text_batch])
                del pend[k][:args.text_batch]
            enc = tok(texts)["input_ids"]
            flat = np.fromiter((t for ids in enc for t in ids), dtype=np.uint16)
            take = min(flat.size, args.target_tokens - written)
            flat[:take].tofile(f)
            written += take
            if written % 50_000_000 < take:
                r = written / max(time.perf_counter() - t0, 1e-9)
                log(f"prep {written/1e6:.0f}M tokens  {r/1e6:.2f}M tok/s  "
                    f"eta {(args.target_tokens-written)/max(r,1)/60:.0f} min")
            if written >= args.target_tokens:
                break
    partial.replace(out)
    log(f"wrote {out} ({out.stat().st_size/2**30:.2f} GiB, {written/1e6:.0f}M tokens)")


def binfile_stream(seq, batch_blocks, target_tokens, shard=0, nshard=1):
    """Yield [batch_blocks, seq] int64 batches from the pre-tokenised memmap.

    With nshard > 1 each shard walks a disjoint contiguous range, so shards can
    run on separate GPUs and their reservoirs merge (each holds `quota` slots
    sampled uniformly from its own range -- the same split the production path
    makes with local_reservoir_quota).
    """
    arr = np.memmap(TOKBIN, dtype=np.uint16, mode="r")
    per = batch_blocks * seq
    total = min(len(arr), target_tokens)
    nb = total // per
    lo, hi = nb * shard // nshard, nb * (shard + 1) // nshard
    arr = arr[lo * per:hi * per]
    n = hi - lo
    for i in range(n):
        chunk = np.asarray(arr[i * per:(i + 1) * per], dtype=np.int64)
        yield torch.from_numpy(chunk).view(batch_blocks, seq), None



def reward_at(lg, t, ids_row, scalar="gold"):
    """Per-token scalar the ranking attribution differentiates.

    Must match the sensor that built the bank, or the components are found
    with one notion of "what matters" and ranked with another.
    """
    if scalar == "gold":
        return lg[0, t, ids_row[t + 1]].float()
    if scalar == "pred":
        return lg[0, t].float().max()
    if scalar == "ent":
        q = F.log_softmax(lg[0, t].float(), -1)
        return -(q.exp() * q).sum()
    raise ValueError(f"unknown scalar {scalar!r}")


# -------------------------------------------------------------- sensor -----

def sensor_pg(model, idx, cfg, W0, Wt, sel):
    """Run the sensor's IG path; return per-step (P, G) at positions `sel`."""
    K, path = cfg["K"], cfg.get("ig_path", "weights")
    Ps, Gs = [], []
    for step in range(K):
        a = (step + 1) / K
        with torch.no_grad():
            for p in MODULES:
                Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
        emb = None
        if path == "inputs":
            with torch.no_grad():
                xc, xf = model.wte(idx), model.wte(idx.roll(1, 0))
            emb = xf + a * (xc - xf)
        P, G = capture(model, idx, embed=emb,
                       scalar=cfg.get("scalar", "gold"))
        Ps.append({p: P[p][:, sel].reshape(-1, P[p].shape[-1]).half()
                   for p in MODULES})
        Gs.append({p: G[p][:, sel].reshape(-1, G[p].shape[-1]).half()
                   for p in MODULES})
        del P, G
    with torch.no_grad():
        for p in MODULES:
            Wt[p].copy_(W0[p])
    return Ps, Gs


def features(Ps, Gs, spec, scales, K, dev):
    N = Ps[0][MODULES[0]].shape[0]
    outs = []
    for p in MODULES:
        r, c = spec[p]
        acc = torch.zeros(N, r.numel(), device=dev)
        for st in range(K):
            acc += Gs[st][p].float()[:, r] * Ps[st][p].float()[:, c]
        outs.append((acc / K) * scales[p][None])
    return torch.cat(outs, 1).clamp(-6e4, 6e4)


# --------------------------------------------------------------- phases ----

def phase_fit(args, dev):
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    cfg = SENSORS[args.sensor]
    K = cfg["K"]
    want_blocks = args.fit_positions // args.pos_per_seq + args.eval_seqs + 8
    ids_cpu, _, pstats = load_pile_blocks(tok, want_blocks, args.seq, seed=0,
                                          tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    fit_ids = IDS[:-args.eval_seqs]
    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}

    gcpu = torch.Generator().manual_seed(args.seed)
    avail = torch.arange(4, args.seq - 2)
    Pa, Ga = [[] for _ in range(K)], [[] for _ in range(K)]
    t0 = time.perf_counter()
    for s in range(0, fit_ids.shape[0], args.batch_blocks):
        b = fit_ids[s:s + args.batch_blocks]
        if b.shape[0] == 0:
            continue
        sel = avail[torch.randperm(avail.numel(),
                                   generator=gcpu)[:args.pos_per_seq]].to(dev)
        Ps, Gs = sensor_pg(model, b, cfg, W0, Wt, sel)
        for st in range(K):
            Pa[st].append({p: v for p, v in Ps[st].items()})
            Ga[st].append({p: v for p, v in Gs[st].items()})
    Ps = [{p: torch.cat([d[p] for d in Pa[st]]) for p in MODULES}
          for st in range(K)]
    Gs = [{p: torch.cat([d[p] for d in Ga[st]]) for p in MODULES}
          for st in range(K)]
    del Pa, Ga
    N = Ps[0][MODULES[0]].shape[0]
    log(f"pilot {N} positions ({time.perf_counter()-t0:.0f}s)")

    p2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Ps]).mean(0)
           for p in MODULES}
    g2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Gs]).mean(0)
           for p in MODULES}
    spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev, p2m, g2m)
    X = features(Ps, Gs, spec, scales, K, dev)
    del Ps, Gs
    torch.cuda.empty_cache()

    mu = X.mean(0)
    Xc = X - mu
    q = min(X.shape[1], N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg, device=dev))[0]
    for _ in range(4):
        Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
    Z = Xc @ Q
    sm = (Z.t() @ Z) / N
    val, vec = torch.linalg.eigh(0.5 * (sm + sm.t()))
    B = (Q @ vec[:, val.argsort(descending=True)[:args.embed_dim]]).contiguous()
    E = Xc @ B
    del Xc, Z, sm, val, vec, Q, X
    cent, _ = kmeans(E, args.C, args.kmeans_iters, args.seed)
    del E
    torch.cuda.empty_cache()
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": cent.cpu(), "basis": B.cpu(), "mean": mu.cpu(),
                "spec": {p: (spec[p][0].cpu(), spec[p][1].cpu())
                         for p in MODULES},
                "scales": {p: scales[p].cpu() for p in MODULES},
                "C": args.C, "sensor": args.sensor, "D": D, "N_pilot": N,
                "pile": pstats}, RUN / f"fit_C{args.C}_{args.sensor}.pt")
    log(f"wrote fit: C={args.C} D={D} pilot N={N}")


def phase_stream(args, dev):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    fit = torch.load(RUN / f"fit_C{args.C}_{args.sensor}{args.fit_tag}.pt",
                     map_location=dev, weights_only=False)
    cfg = SENSORS[args.sensor]
    K = cfg["K"]
    cent, B, mu = fit["centroids"], fit["basis"], fit["mean"]
    spec = {p: (fit["spec"][p][0].to(dev), fit["spec"][p][1].to(dev))
            for p in MODULES}
    scales = {p: fit["scales"][p].to(dev) for p in MODULES}
    centn = F.normalize(cent, dim=1)

    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}

    quota = args.quota
    cap = args.C * quota
    res = [{p: torch.zeros(cap, W0[p].shape[1], dtype=torch.float16, device=dev)
            for p in MODULES} for _ in range(K)]          # P side (d_in)
    resg = [{p: torch.zeros(cap, W0[p].shape[0], dtype=torch.float16, device=dev)
             for p in MODULES} for _ in range(K)]         # G side (d_out)
    gb = sum(v.numel() * 2 for d in res for v in d.values())
    gb += sum(v.numel() * 2 for d in resg for v in d.values())
    log(f"reservoir {cap} rows x {K} steps = {gb/2**30:.1f} GiB")

    seen = torch.zeros(args.C, dtype=torch.int64)
    rgen = torch.Generator().manual_seed(args.seed + 32452843)
    gcpu = torch.Generator().manual_seed(args.seed)
    avail = torch.arange(4, args.seq - 2)
    done, t0, nb = 0, time.perf_counter(), 0
    for idx_cpu, _ in binfile_stream(args.seq, args.batch_blocks,
                                     args.target_tokens, args.shard,
                                     args.nshard):
        idx = idx_cpu.to(dev, non_blocking=True)
        sel = avail[torch.randperm(avail.numel(),
                                   generator=gcpu)[:args.pos_per_seq]].to(dev)
        Ps, Gs = sensor_pg(model, idx, cfg, W0, Wt, sel)
        X = features(Ps, Gs, spec, scales, K, dev)
        E = (X - mu) @ B
        lab = (F.normalize(E, dim=1) @ centn.t()).argmax(1)
        src, dst = reservoir_updates(lab, seen, quota, rgen)
        if src.numel():
            s_d, d_d = src.to(dev), dst.to(dev)
            for st in range(K):
                for p in MODULES:
                    res[st][p][d_d] = Ps[st][p][s_d]
                    resg[st][p][d_d] = Gs[st][p][s_d]
        del Ps, Gs, X, E
        done += idx.numel()
        nb += 1
        if nb % args.log_every == 0:
            r = done / max(time.perf_counter() - t0, 1e-9)
            log(f"{done/1e6:.1f}M tokens  {r/1e3:.1f}k tok/s  "
                f"filled {int((seen.clamp(max=quota)).sum())}/{cap}  "
                f"eta {(args.target_tokens-done)/max(r,1)/60:.0f} min")
        if done >= args.target_tokens:
            break
    torch.save({"res_p": [{p: v.cpu() for p, v in d.items()} for d in res],
                "res_g": [{p: v.cpu() for p, v in d.items()} for d in resg],
                "seen": seen, "quota": quota, "tokens": done},
               RUN / f"res_C{args.C}_{args.sensor}{args.tag}_s{args.shard}"
               f"of{args.nshard}.pt")
    log(f"streamed {done/1e6:.1f}M tokens; wrote reservoir")




# ------------------------------------------------------- unit ownership ----

N_HEAD, N_LAYER = 6, 4
WRITE_SIDE = ("attn.o_proj", "mlp.down_proj")


def is_write_side(p: str) -> bool:
    """True when the module WRITES the residual stream, so its d_in axis is
    what indexes functional units (head dims / MLP neurons)."""
    return p.endswith(WRITE_SIDE)


def unit_groups(args, W0):
    """Map every (module, unit) to a global group id.

    A "unit" is the axis that indexes a functional piece of the model: the
    output row for modules that READ the residual stream (one row of c_fc is
    one neuron's input weights), the input column for modules that WRITE it.
    All entries of a unit share one owner set, so ablating a component removes
    whole neurons instead of fractions of a pre-activation -- gelu(0.3x) is not
    0.3*gelu(x), so a fractionally-owned row does not compose.

    Groups additionally tie units that only make sense ablated together:
      own_mode=neuron_tied  c_fc row o <-> down_proj column o (one MLP neuron)
      qk_joint=head         all q/k rows of one attention head
      qk_joint=dim          q/k rows of one head AND rope pair (d, d+hd/2), so
                            the bilinear product q_d k_d scales coherently
    """
    hd = W0[MODULES[0]].shape[0] // N_HEAD
    gid, named = {}, {}
    nxt = 0

    def fresh(n):
        nonlocal nxt
        g = torch.arange(nxt, nxt + n)
        nxt += n
        return g

    def shared(key, n):
        nonlocal nxt
        if key not in named:
            named[key] = torch.arange(nxt, nxt + n)
            nxt += n
        return named[key]

    for p in MODULES:
        d_out, d_in = W0[p].shape
        n_unit = d_in if is_write_side(p) else d_out
        layer = p.split(".")[1]
        o = torch.arange(n_unit)
        head = o // hd
        if getattr(args, "attn_joint", "off") == "head" and ".attn." in p:
            # the whole head -- q, k, v and o -- as one ablatable unit
            gid[p] = shared(f"attn{layer}", N_HEAD)[head]
        elif args.qk_joint != "off" and p.endswith(("attn.q_proj",
                                                    "attn.k_proj")):
            if args.qk_joint == "head":
                sub, n_sub = head, N_HEAD
            else:
                sub = head * (hd // 2) + (o % hd) % (hd // 2)
                n_sub = N_HEAD * (hd // 2)
            gid[p] = shared(f"qk{layer}", n_sub)[sub]
        elif getattr(args, "ov_joint", "off") != "off" and p.endswith(
                ("attn.v_proj", "attn.o_proj")):
            # OV is linear, so this is not a nonlinearity fix -- it is the same
            # "ablate a circuit, not half of one" argument as QK
            if args.ov_joint == "head":
                sub, n_sub = head, N_HEAD
            else:
                sub, n_sub = o, n_unit          # per head dim, v row <-> o col
            gid[p] = shared(f"ov{layer}", n_sub)[sub]
        elif args.own_mode == "neuron_tied" and p.endswith(("mlp.c_fc",
                                                            "mlp.down_proj")):
            gid[p] = shared(f"mlp{layer}", n_unit)
        else:
            gid[p] = fresh(n_unit)
    return gid, nxt


def unit_bank(args, st, W0, C, K, quota, w_slot, dev):
    """Ownership at unit granularity: pool the per-entry attribution mass onto
    functional units, pick each unit's top-s owners once, broadcast back."""
    gid, n_group = unit_groups(args, W0)
    S = args.soft_s
    agg = torch.zeros(C, n_group, device=dev)
    for p in MODULES:
        d_out, d_in = W0[p].shape
        write = is_write_side(p)
        Mu = torch.zeros(C, d_in if write else d_out, device=dev)
        for c0 in range(0, C, args.bank_chunk):
            cc = min(args.bank_chunk, C - c0)
            for k in range(K):
                Pc = st["res_p"][k][p].view(C, quota, d_in)[c0:c0 + cc] \
                    .to(dev, non_blocking=True).float().abs()
                Gc = st["res_g"][k][p].view(C, quota, d_out)[c0:c0 + cc] \
                    .to(dev, non_blocking=True).float().abs()
                Gc = Gc * w_slot[c0:c0 + cc][:, :, None]
                # sum of the entry masses over the pooled axis
                if write:
                    Mu[c0:c0 + cc] += (Pc * Gc.sum(-1)[:, :, None]).sum(1) / K
                else:
                    Mu[c0:c0 + cc] += (Gc * Pc.sum(-1)[:, :, None]).sum(1) / K
                del Pc, Gc
        # per-module scale so tied modules contribute comparably; a constant
        # per module changes neither the per-unit ranking nor the shares
        agg.index_add_(1, gid[p].to(dev), Mu / Mu.sum().clamp_min(1e-30))
        del Mu
        torch.cuda.empty_cache()
    gvals, gidx = agg.topk(min(S, C), dim=0)             # [S, n_group]
    del agg
    gw = gvals.clamp_min(0)
    tot = gw.sum(0, keepdim=True)
    gw = gw / tot.clamp_min(1e-30)
    gw[0] = torch.where(tot[0] > 0, gw[0], torch.ones_like(gw[0]))
    sidx, swgt = {}, {}
    for p in MODULES:
        d_out, d_in = W0[p].shape
        g = gid[p].to(dev)
        i_u, w_u = gidx[:, g].to(torch.int16), gw[:, g]  # [S, n_unit]
        sl = (slice(None), None, slice(None)) if is_write_side(p) else \
             (slice(None), slice(None), None)
        sidx[p] = i_u[sl].expand(S, d_out, d_in).contiguous()
        swgt[p] = w_u[sl].expand(S, d_out, d_in).contiguous().half()
    log(f"unit bank: own_mode={args.own_mode} qk_joint={args.qk_joint} -> "
        f"{n_group} groups x {S} owners")
    return sidx, swgt, gid, n_group


def pool_acc(acc, gid, n_group, dev):
    """Make the set-cover accumulator constant within each ownership group.

    Refinement reweights every entry independently, which would hand a neuron's
    entries different shares again and undo the whole point of unit ownership.
    Pooling it onto the same groups keeps the refined bank unit-coherent.
    """
    S = next(iter(acc.values())).shape[0]
    agg = torch.zeros(S, n_group, device=dev)
    for p, a in acc.items():
        u = a.sum(1) if is_write_side(p) else a.sum(2)         # [S, n_unit]
        agg.index_add_(1, gid[p].to(dev), u / u.sum().clamp_min(1e-30))
    out = {}
    for p in acc:
        g = agg[:, gid[p].to(dev)]                             # [S, n_unit]
        out[p] = g[:, None, :] if is_write_side(p) else g[:, :, None]
    return out


def phase_bank(args, dev):
    """Softpart bank from the reservoir, then per-input sufficiency."""
    fit = torch.load(RUN / f"fit_C{args.C}_{args.sensor}{args.fit_tag}.pt",
                     map_location="cpu", weights_only=False)
    shards = sorted(RUN.glob(f"res_C{args.C}_{args.sensor}{args.tag}_s*of*.pt"))
    assert shards, f"no reservoir shards for C={args.C} {args.sensor}"
    sts = [torch.load(f, map_location="cpu", weights_only=False)
           for f in shards]
    q1 = sts[0]["quota"]
    quota = q1 * len(sts)
    n_tok = sum(int(x["tokens"]) for x in sts)
    seen = sum(x["seen"] for x in sts)
    Cc = args.C
    st = {"res_p": [], "res_g": []}
    for k in range(SENSORS[args.sensor]["K"]):
        st["res_p"].append({p: torch.cat(
            [x["res_p"][k][p].view(Cc, q1, -1) for x in sts], 1
        ).reshape(Cc * quota, -1) for p in MODULES})
        st["res_g"].append({p: torch.cat(
            [x["res_g"][k][p].view(Cc, q1, -1) for x in sts], 1
        ).reshape(Cc * quota, -1) for p in MODULES})
    valid_tot = sum(x["seen"].clamp(max=q1) for x in sts)
    del sts
    log(f"merged {len(shards)} reservoir shard(s): quota {q1} x {len(shards)}"
        f" = {quota}/cluster")
    K = SENSORS[args.sensor]["K"]
    cfg = SENSORS[args.sensor]
    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}
    C = args.C
    valid = valid_tot.to(dev)                    # filled rows per cluster
    log(f"reservoir: {int(valid.sum())}/{C*quota} slots filled, "
        f"{int((valid == 0).sum())} empty clusters, "
        f"{n_tok/1e6:.0f}M tokens streamed")
    w_slot = (1.0 / valid.clamp_min(1).float())[:, None].expand(C, quota)

    sidx, swgt = {}, {}
    gid = n_group = None
    t0 = time.perf_counter()
    if args.own_mode == "entry":
        for p in MODULES:
            d_out, d_in = W0[p].shape
            S = args.soft_s
            vals = torch.zeros(S, d_out, d_in, device=dev)
            idxs = torch.zeros(S, d_out, d_in, dtype=torch.int16, device=dev)
            for c0 in range(0, C, args.bank_chunk):
                cc = min(args.bank_chunk, C - c0)
                Mm = torch.zeros(cc, d_out, d_in, device=dev)
                for k in range(K):
                    # fixed slots -> [cc, quota, d]; batched matmul beats a
                    # per-cluster loop (C*len(MODULES)*K launches otherwise)
                    Pc = st["res_p"][k][p].view(C, quota, d_in)[c0:c0 + cc] \
                        .to(dev, non_blocking=True).float().abs()
                    Gc = st["res_g"][k][p].view(C, quota, d_out)[c0:c0 + cc] \
                        .to(dev, non_blocking=True).float().abs()
                    Gc = Gc * w_slot[c0:c0 + cc][:, :, None]
                    Mm += torch.bmm(Gc.transpose(1, 2), Pc)
                allv = torch.cat([vals, Mm / K])
                alli = torch.cat([idxs, torch.arange(
                    c0, c0 + cc, device=dev, dtype=torch.int16
                )[:, None, None].expand(-1, d_out, d_in)])
                top, si = allv.topk(S, dim=0)
                vals, idxs = top, alli.gather(0, si)
                del Mm, allv, alli
            w = vals.clamp_min(0)
            tot = w.sum(0, keepdim=True)
            w = w / tot.clamp_min(1e-30)
            w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
            sidx[p], swgt[p] = idxs, w.half()
            del vals, idxs, w
            torch.cuda.empty_cache()
    else:
        sidx, swgt, gid, n_group = unit_bank(args, st, W0, C, K, quota,
                                             w_slot, dev)
    log(f"bank built ({time.perf_counter()-t0:.0f}s)")
    if args.save_bank:
        bpath = RUN / f"bank_C{C}_{args.sensor}{args.bank_tag}.pt"
        torch.save({"sidx": {p: sidx[p].cpu() for p in MODULES},
                    "swgt": {p: swgt[p].cpu() for p in MODULES},
                    "C": C, "sensor": args.sensor, "tokens": n_tok,
                    "own_mode": args.own_mode, "qk_joint": args.qk_joint,
                    "ov_joint": args.ov_joint,
                    "attn_joint": args.attn_joint},
                   bpath)
        log(f"saved bank to {bpath}")
    del st
    torch.cuda.empty_cache()

    # ---- refinement + per-input sufficiency, both banks ----
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_cpu, _, _ = load_pile_blocks(tok, 168, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    fit_rows, eval_ids = IDS[:-args.eval_seqs], IDS[-args.eval_seqs:]
    path = cfg.get("ig_path", "weights")

    @torch.no_grad()
    def restore():
        for p in MODULES:
            Wt[p].copy_(W0[p])

    @torch.no_grad()
    def tok_out(b, t):
        lg = model(eval_ids[b:b + 1])[0, t].float()
        return lg, float(F.cross_entropy(lg[None], eval_ids[b, t + 1][None]))

    def token_attr(ids_row, t):
        A = {p: torch.zeros_like(W0[p]) for p in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            pre, post, hs = {}, {}, []
            for p in MODULES:
                mod = model.get_submodule(p)

                def hook(mm, inp, out, _p=p):
                    pre[_p] = inp[0]
                    out.retain_grad()
                    post[_p] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(ids_row[None])
            for h in hs:
                h.remove()
            reward = reward_at(lg, t, ids_row,
                               cfg.get('scalar', 'gold'))
            grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
            for p, g in zip(MODULES, grads):
                A[p] += (g[0].float().t() @ pre[p][0].float()) / K
            del pre, post, grads, lg
        restore()
        return {p: A[p] * W0[p] for p in MODULES}

    def comp_scores(AW, sw):
        v = torch.zeros(C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for p in MODULES:
                v += torch.bincount(sidx[p].reshape(-1).int(),
                                    weights=(sw[p].float() * AW[p][None]
                                             ).reshape(-1).double(),
                                    minlength=C)
        return v

    # set-cover refinement on fit rows only
    swgt_r = None
    if args.refine_iters > 0:
        t1 = time.perf_counter()
        gref = torch.Generator().manual_seed(args.seed + 555)
        ref = [(int(torch.randint(0, fit_rows.shape[0], (1,), generator=gref)),
                int(torch.randint(64, args.seq - 2, (1,), generator=gref)))
               for _ in range(args.refine_tokens)]
        AWs = [{m: v.half() for m, v in token_attr(fit_rows[b], t).items()}
               for b, t in ref]
        swgt_r = {m: swgt[m].clone() for m in MODULES}
        for it in range(args.refine_iters):
            acc = {m: torch.zeros_like(swgt[m], dtype=torch.float32)
                   for m in MODULES}
            for AW in AWs:
                sc = comp_scores(AW, swgt_r)
                order = torch.argsort(sc, descending=True)
                rank = torch.empty(C, device=dev, dtype=torch.float32)
                rank[order] = torch.arange(C, device=dev, dtype=torch.float32)
                gain = 1.0 / (rank + args.rank_r0)
                with torch.no_grad():
                    for m in MODULES:
                        acc[m] += AW[m].float().abs()[None] * gain[sidx[m].int()]
            if gid is not None:
                acc = pool_acc(acc, gid, n_group, dev)
            with torch.no_grad():
                for m in MODULES:
                    w = swgt[m].float() * acc[m]
                    tot = w.sum(0, keepdim=True)
                    swgt_r[m] = torch.where(tot <= 0, swgt[m].float(),
                                            w / tot.clamp_min(1e-30)).half()
        del AWs
        log(f"set-cover refined ({time.perf_counter()-t1:.0f}s)")

    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]
    KEEP = [k for k in [0, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768,
                        1024, 1536, 2048, 3072, 4096, 6144] if k < C] + [C]
    restore()
    base_lg, base_ces = {}, []
    for b, t in samp:
        lg, ce = tok_out(b, t)
        base_lg[(b, t)] = F.log_softmax(lg, -1)
        base_ces.append(ce)
    base = float(np.mean(base_ces))
    log(f"unablated CE on {len(samp)} target tokens: {base:.4f}")

    results = {}
    arms = [("original", swgt)] + ([("refined", swgt_r)] if swgt_r else [])
    arms += [("random", swgt_r if swgt_r else swgt)]   # random rank, same bank
    AW_cache = [token_attr(eval_ids[b], t) for b, t in samp]
    for name, sw in arms:
        curves = np.zeros((len(samp), len(KEEP)))
        curves_kl = np.zeros((len(samp), len(KEEP)))
        drop_ce = np.zeros((len(samp), len(KEEP)))
        drop_kl = np.zeros((len(samp), len(KEEP)))
        for j, (b, t) in enumerate(samp):
            if name == "random":
                grj = torch.Generator().manual_seed(777 + j)
                order = torch.randperm(C, generator=grj).to(dev)
            else:
                sc = comp_scores(AW_cache[j], sw)
                order = torch.argsort(sc, descending=True)
            rank = torch.empty(C, dtype=torch.int32, device=dev)
            rank[order] = torch.arange(C, dtype=torch.int32, device=dev)
            with torch.no_grad():
                R = {m: rank[sidx[m].int()] for m in MODULES}
                for ki, kk in enumerate(KEEP):
                    for m in MODULES:
                        keep = (sw[m].float() * (R[m] < kk)
                                ).sum(0, dtype=torch.float32)
                        Wt[m].copy_(W0[m] * keep)
                    lg, ce = tok_out(b, t)
                    curves[j, ki] = ce
                    curves_kl[j, ki] = float(F.kl_div(
                        F.log_softmax(lg, -1), base_lg[(b, t)],
                        log_target=True, reduction="sum"))
                    if args.drop_arm:
                        # exact complement: shares sum to 1, so keep + drop = W
                        for m in MODULES:
                            dm = (sw[m].float() * (R[m] >= kk)
                                  ).sum(0, dtype=torch.float32)
                            Wt[m].copy_(W0[m] * dm)
                        lgd, ced = tok_out(b, t)
                        drop_ce[j, ki] = ced
                        drop_kl[j, ki] = float(F.kl_div(
                            F.log_softmax(lgd, -1), base_lg[(b, t)],
                            log_target=True, reduction="sum"))
                del R
            restore()
        mu = curves.mean(0)
        mkl = curves_kl.mean(0)
        thr = next((k for k, v in zip(KEEP, mu) if v - base <= 0.25), C)
        results[name] = {"ce": [round(float(v), 5) for v in mu],
                         "kl": [round(float(v), 5) for v in mkl],
                         "k_within_0.25": thr,
                         "roundtrip_err_at_C": round(float(mu[-1] - base), 6)}
        if args.drop_arm:
            # necessity, not sufficiency: remove the top-k and keep the rest
            results[name]["ce_drop"] = [round(float(v), 5)
                                        for v in drop_ce.mean(0)]
            results[name]["kl_drop"] = [round(float(v), 5)
                                        for v in drop_kl.mean(0)]
        log(f"{name:<9} k needed {thr:>5}/{C}  "
            f"CE@256 {mu[KEEP.index(256)]:.2f}  "
            f"roundtrip {results[name]['roundtrip_err_at_C']:+.1e}")
    del AW_cache

    out = {"format": "stream67_v2", "C": C, "sensor": args.sensor,
           "tag": args.tag, "tokens_streamed": n_tok, "quota": quota,
           "n_samples": len(samp), "base_token_ce": round(base, 5),
           "keep_grid": KEEP, "empty_clusters": int((valid == 0).sum()),
           "slots_filled": int(valid.sum()), "slots_total": C * quota,
           "refine_tokens": args.refine_tokens if swgt_r else 0,
           "own_mode": args.own_mode, "qk_joint": args.qk_joint,
           "ov_joint": args.ov_joint, "attn_joint": args.attn_joint,
           "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}")



def phase_interp(args, dev):
    """Top-firing contexts per component, for monosemanticity labelling.

    "Fires" here is the same quantity the eval ranks by: the attribution of the
    component's OWNED slice at that token, z_c = sum_e share_c(e) A(e) W(e).
    That is what changes between ownership modes -- the input clustering is
    identical across them, so a cluster-assignment view could not tell them
    apart.  Scores are normalised per token before ranking, or a few
    high-attribution tokens win every component's list.

    (token_attr / comp_scores are duplicated from phase_bank on purpose: that
    function is mid-experiment and not worth refactoring under it.)
    """
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    cfg = SENSORS[args.sensor]
    K = cfg["K"]
    path = cfg.get("ig_path", "weights")
    bpath = RUN / f"bank_C{args.C}_{args.sensor}{args.bank_tag}.pt"
    bk = torch.load(bpath, map_location="cpu", weights_only=False)
    C = int(bk["C"])
    sidx = {p: bk["sidx"][p].to(dev) for p in MODULES}
    swgt = {p: bk["swgt"][p].to(dev) for p in MODULES}
    log(f"loaded {bpath.name}: C={C} own_mode={bk.get('own_mode','entry')} "
        f"qk_joint={bk.get('qk_joint','off')}")

    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    # a separate, larger corpus draw than the fit/eval set: labelling wants
    # many documents, and a 32-sequence pool makes any component look coherent
    ids_cpu, _, _ = load_pile_blocks(tok, args.interp_blocks, args.seq,
                                     seed=args.interp_seed,
                                     tokenizer_name=S67.TOKENIZER)
    eval_ids = ids_cpu.to(dev)

    @torch.no_grad()
    def restore():
        for p in MODULES:
            Wt[p].copy_(W0[p])

    def token_attr(ids_row, t):
        A = {p: torch.zeros_like(W0[p]) for p in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            pre, post, hs = {}, {}, []
            for p in MODULES:
                mod = model.get_submodule(p)

                def hook(mm, inp, out, _p=p):
                    pre[_p] = inp[0]
                    out.retain_grad()
                    post[_p] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(ids_row[None])
            for h in hs:
                h.remove()
            reward = reward_at(lg, t, ids_row,
                               cfg.get('scalar', 'gold'))
            grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
            for p, g in zip(MODULES, grads):
                A[p] += (g[0].float().t() @ pre[p][0].float()) / K
            del pre, post, grads, lg
        restore()
        return {p: A[p] * W0[p] for p in MODULES}

    def comp_scores(AW):
        v = torch.zeros(C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for p in MODULES:
                v += torch.bincount(sidx[p].reshape(-1).int(),
                                    weights=(swgt[p].float() * AW[p][None]
                                             ).reshape(-1).double(),
                                    minlength=C)
        return v

    g = torch.Generator().manual_seed(4242)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=g)),
             int(torch.randint(32, args.seq - 2, (1,), generator=g)))
            for _ in range(args.interp_tokens)]
    Z = torch.zeros(len(samp), C, device=dev)
    t0 = time.perf_counter()
    for j, (b, t) in enumerate(samp):
        z = comp_scores(token_attr(eval_ids[b], t)).float()
        Z[j] = z / z.abs().max().clamp_min(1e-30)     # per-token normalisation
        if (j + 1) % 200 == 0:
            r = (j + 1) / (time.perf_counter() - t0)
            log(f"interp {j+1}/{len(samp)} tokens ({r:.1f}/s, "
                f"eta {(len(samp)-j-1)/r/60:.0f} min)")
    log(f"scored {len(samp)} tokens ({time.perf_counter()-t0:.0f}s)")

    N = args.interp_ctx
    peak, _ = Z.max(0)
    hits = (Z > 0.25 * peak[None, :].clamp_min(1e-9)).sum(0)
    live = (peak > 0).nonzero().flatten()
    order = peak[live].argsort(descending=True)
    strong = live[order[:args.interp_comps // 2]].tolist()
    rest = [int(c) for c in live[order[args.interp_comps // 2:]].tolist()]
    gsel = torch.Generator().manual_seed(7)
    pick = torch.randperm(len(rest), generator=gsel)[:args.interp_comps
                                                     - len(strong)].tolist()
    chosen = strong + [rest[i] for i in pick]

    out = {"bank": bpath.name, "C": C, "own_mode": bk.get("own_mode", "entry"),
           "qk_joint": bk.get("qk_joint", "off"), "n_tokens": len(samp),
           "n_live": int(live.numel()), "components": {}}
    for c in chosen:
        vals, idx = Z[:, c].topk(min(N, Z.shape[0]))
        ctxs = []
        for v, j in zip(vals.tolist(), idx.tolist()):
            b, t = samp[j]
            row = eval_ids[b]
            lo = max(0, t - 24)
            ctxs.append({
                "score": round(float(v), 4),
                "before": tok.decode(row[lo:t].tolist()),
                "token": tok.decode(row[t:t + 1].tolist()),
                "next": tok.decode(row[t + 1:t + 2].tolist()),
                "after": tok.decode(row[t + 1:t + 5].tolist())})
        out["components"][str(c)] = {
            "peak": round(float(peak[c]), 4), "hits": int(hits[c]),
            "in_top_half": bool(c in strong), "contexts": ctxs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out} ({len(chosen)} components, {int(live.numel())} live)")


def phase_ablate(args, dev):
    """Is a single component individually ablatable?

    The keep/drop curves say whether the top-k components as a SET carry the
    model. This asks the interpretability question instead: take one component,
    delete the weight mass it owns (permanent surgery, plain forward), and see
    whether the damage lands on the tokens that component fires on rather than
    on text in general. A crisp mechanism should show dCE(its own contexts) >>
    dCE(control text), and beat deleting a random component of the same mass.
    """
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    cfg = SENSORS[args.sensor]
    K = cfg["K"]
    path = cfg.get("ig_path", "weights")
    bpath = RUN / f"bank_C{args.C}_{args.sensor}{args.bank_tag}.pt"
    bk = torch.load(bpath, map_location="cpu", weights_only=False)
    C = int(bk["C"])
    sidx = {p: bk["sidx"][p].to(dev) for p in MODULES}
    swgt = {p: bk["swgt"][p].to(dev) for p in MODULES}
    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_cpu, _, _ = load_pile_blocks(tok, args.interp_blocks, args.seq,
                                     seed=args.interp_seed,
                                     tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    log(f"loaded {bpath.name}: own_mode={bk.get('own_mode','entry')} "
        f"qk_joint={bk.get('qk_joint','off')} C={C}")

    @torch.no_grad()
    def restore():
        for p in MODULES:
            Wt[p].copy_(W0[p])

    @torch.no_grad()
    def share_of(c):
        return {p: (swgt[p].float() * (sidx[p] == c)).sum(0) for p in MODULES}

    @torch.no_grad()
    def owned_mass(sh):
        return float(sum((sh[p] * W0[p].pow(2)).sum() for p in MODULES))

    @torch.no_grad()
    def ce_on(pairs):
        tot = 0.0
        for b, t in pairs:
            lg = model(IDS[b:b + 1])[0, t].float()
            tot += float(F.cross_entropy(lg[None], IDS[b, t + 1][None]))
        return tot / max(len(pairs), 1)

    @torch.no_grad()
    def ablate(sh, alpha=0.0):
        for p in MODULES:
            Wt[p].copy_(W0[p] * (1.0 - (1.0 - alpha) * sh[p]))

    def token_attr(ids_row, t):
        A = {p: torch.zeros_like(W0[p]) for p in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            pre, post, hs = {}, {}, []
            for p in MODULES:
                mod = model.get_submodule(p)

                def hook(mm, inp, out, _p=p):
                    pre[_p] = inp[0]
                    out.retain_grad()
                    post[_p] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(ids_row[None])
            for h in hs:
                h.remove()
            reward = reward_at(lg, t, ids_row,
                               cfg.get('scalar', 'gold'))
            grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
            for p, g in zip(MODULES, grads):
                A[p] += (g[0].float().t() @ pre[p][0].float()) / K
            del pre, post, grads, lg
        restore()
        return {p: A[p] * W0[p] for p in MODULES}

    def comp_scores(AW):
        v = torch.zeros(C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for p in MODULES:
                v += torch.bincount(sidx[p].reshape(-1).int(),
                                    weights=(swgt[p].float() * AW[p][None]
                                             ).reshape(-1).double(),
                                    minlength=C)
        return v

    g = torch.Generator().manual_seed(4242)
    samp = [(int(torch.randint(0, IDS.shape[0], (1,), generator=g)),
             int(torch.randint(32, args.seq - 2, (1,), generator=g)))
            for _ in range(args.interp_tokens)]
    Z = torch.zeros(len(samp), C, device=dev)
    t0 = time.perf_counter()
    for j, (b, t) in enumerate(samp):
        z = comp_scores(token_attr(IDS[b], t)).float()
        Z[j] = z / z.abs().max().clamp_min(1e-30)
        if (j + 1) % 250 == 0:
            log(f"ablate scoring {j+1}/{len(samp)} "
                f"({(j+1)/(time.perf_counter()-t0):.1f}/s)")
    restore()

    peak, _ = Z.max(0)
    hits = (Z > 0.25 * peak[None, :].clamp_min(1e-9)).sum(0)
    if args.ablate_comps:
        comps = [int(x) for x in args.ablate_comps.split(",")]
    else:
        ok = ((hits >= args.ablate_min_hits) & (peak > 0)).nonzero().flatten()
        comps = ok[peak[ok].argsort(descending=True)][:args.ablate_n].tolist()
    log(f"ablating {len(comps)} components: {comps}")

    W_tot = float(sum(W0[p].pow(2).sum() for p in MODULES))
    gctl = torch.Generator().manual_seed(99)
    out = {"bank": bpath.name, "C": C, "own_mode": bk.get("own_mode", "entry"),
           "qk_joint": bk.get("qk_joint", "off"), "alpha": args.ablate_alpha,
           "n_scored": len(samp), "components": []}
    for c in comps:
        tgt_idx = Z[:, c].topk(min(args.ablate_target, Z.shape[0])).indices
        tgt = [samp[int(i)] for i in tgt_idx]
        pool = [j for j in range(len(samp)) if j not in set(tgt_idx.tolist())]
        ctl_idx = torch.randperm(len(pool), generator=gctl)[:args.ablate_ctl]
        ctl = [samp[pool[int(i)]] for i in ctl_idx]
        sh = share_of(c)
        m_c = owned_mass(sh) / W_tot
        base_t, base_c = ce_on(tgt), ce_on(ctl)
        ablate(sh, args.ablate_alpha)
        d_t, d_c = ce_on(tgt) - base_t, ce_on(ctl) - base_c
        restore()
        # null: a different component of the closest owned mass
        cand = torch.randperm(C, generator=gctl)[:args.ablate_null_pool].tolist()
        cand = [x for x in cand if x != c]
        best, best_gap = None, None
        for x in cand:
            gap = abs(owned_mass(share_of(x)) / W_tot - m_c)
            if best_gap is None or gap < best_gap:
                best, best_gap = x, gap
        shn = share_of(best)
        ablate(shn, args.ablate_alpha)
        n_t, n_c = ce_on(tgt) - base_t, ce_on(ctl) - base_c
        restore()
        rec = {"c": int(c), "peak": round(float(peak[c]), 4),
               "hits": int(hits[c]), "owned_mass_frac": round(m_c, 6),
               "base_ce_target": round(base_t, 4),
               "base_ce_control": round(base_c, 4),
               "d_target": round(d_t, 4), "d_control": round(d_c, 4),
               "null_c": int(best), "null_d_target": round(n_t, 4),
               "null_d_control": round(n_c, 4),
               "selectivity": round(d_t / max(abs(d_c), 1e-6), 2)}
        out["components"].append(rec)
        log(f"c{c:<5} dCE target {d_t:+7.3f}  control {d_c:+7.3f}  "
            f"| null c{best}: {n_t:+7.3f} / {n_c:+7.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}")


def phase_fitstream(args, dev):
    """Streaming fit: exact covariance PCA + minibatch spherical k-means.

    Replaces the in-memory pilot fit. Phase A0 accumulates p^2/g^2 second
    moments (for the spec) over the first slice; A1 accumulates the exact
    feature covariance [D, D] and mean; B runs Sculley minibatch k-means over
    everything after. Saves the same fit artifact phase_fit writes, so
    stream/bank run unchanged.
    """
    cfg = SENSORS[args.sensor]
    K = cfg["K"]
    model = load67(dev, cfg["mode"])
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}
    gcpu = torch.Generator().manual_seed(args.seed)
    avail = torch.arange(4, args.seq - 2)

    n_stats, n_cov = args.fs_stats_batches, args.fs_cov_batches
    p2s = {p: 0.0 for p in MODULES}
    g2s = {p: 0.0 for p in MODULES}
    cov = mu = None
    n_mu = 0
    cent = None
    counts = None
    spec = scales = None
    t0 = time.perf_counter()
    for bi, (idx_cpu, _) in enumerate(binfile_stream(
            args.seq, args.batch_blocks, args.fs_tokens)):
        idx = idx_cpu.to(dev, non_blocking=True)
        sel = avail[torch.randperm(avail.numel(),
                                   generator=gcpu)[:args.pos_per_seq]].to(dev)
        Ps, Gs = sensor_pg(model, idx, cfg, W0, Wt, sel)
        if bi < n_stats:                                   # A0: spec moments
            for p in MODULES:
                p2s[p] = p2s[p] + torch.stack(
                    [Ps[st][p].float().pow(2).mean(0) for st in range(K)]
                ).mean(0)
                g2s[p] = g2s[p] + torch.stack(
                    [Gs[st][p].float().pow(2).mean(0) for st in range(K)]
                ).mean(0)
            del Ps, Gs
            continue
        if spec is None:
            p2m = {p: v / n_stats for p, v in p2s.items()}
            g2m = {p: v / n_stats for p, v in g2s.items()}
            spec, scales, D = build_spec(model, args.feat_dim, args.seed,
                                         dev, p2m, g2m)
            cov = torch.zeros(D, D, device=dev)
            mu = torch.zeros(D, device=dev)
            log(f"spec built D={D} ({time.perf_counter()-t0:.0f}s)")
        X = features(Ps, Gs, spec, scales, K, dev)
        del Ps, Gs
        if bi < n_stats + n_cov:                           # A1: covariance
            cov.addmm_(X.t(), X)         # in place: no second [D, D] temporary
            mu += X.sum(0)
            n_mu += X.shape[0]
            del X
            continue
        if cent is None:                                   # eigenbasis once
            mu = mu / n_mu
            cov.div_(n_mu).addr_(mu, mu, alpha=-1.0)   # in place: no [D, D] temporaries
            gg = torch.Generator(device=dev).manual_seed(args.seed)
            Q = torch.linalg.qr(torch.randn(cov.shape[0], args.embed_dim + 64,
                                            generator=gg, device=dev))[0]
            for _ in range(6):
                Q = torch.linalg.qr(cov @ Q)[0]
            w = torch.linalg.eigh(Q.t() @ cov @ Q)
            B = (Q @ w.eigenvectors[:, w.eigenvalues.argsort(
                descending=True)[:args.embed_dim]]).contiguous()
            del cov, Q
            gk = torch.Generator(device=dev).manual_seed(args.seed + 1)
            cent = None
            basis = B
            log(f"PCA basis frozen over {n_mu} positions "
                f"({time.perf_counter()-t0:.0f}s)")
        E = F.normalize((X - mu[None]) @ basis, dim=1)
        del X
        if cent is None:
            cent = E[torch.randperm(E.shape[0], generator=gk,
                                    device=dev)[:args.C]].clone()
            if cent.shape[0] < args.C:
                reps = args.C // cent.shape[0] + 1
                cent = cent.repeat(reps, 1)[:args.C]
            counts = torch.zeros(args.C, device=dev)
        lab = (E @ cent.t()).argmax(1)                     # B: minibatch
        sums = torch.zeros_like(cent).index_add_(0, lab, E)
        n_b = torch.bincount(lab, minlength=args.C).float()
        counts += n_b
        eta = (n_b / counts.clamp_min(1.0)).unsqueeze(1)
        m_b = sums / n_b.clamp_min(1.0).unsqueeze(1)
        cent = F.normalize(cent * (1 - eta) + m_b * eta, dim=1)
        dead = (counts == 0).nonzero().flatten()
        if dead.numel():
            # the init draws C centroids from one 512-point batch (duplicates
            # beyond 512), so at large C most centroids never win a point;
            # reseed dead ones with the batch's worst-fit points
            worst = (E * cent[lab]).sum(1).argsort()[:min(dead.numel(), E.shape[0])]
            cent[dead[:worst.numel()]] = E[worst]
            counts[dead[:worst.numel()]] = 1.0
    n_dead = int((counts == 0).sum())
    log(f"minibatch k-means done: {int(counts.sum())} positions, "
        f"{n_dead} dead clusters ({time.perf_counter()-t0:.0f}s)")
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": cent.cpu(), "basis": basis.cpu(),
                "mean": mu.cpu(),
                "spec": {p: (spec[p][0].cpu(), spec[p][1].cpu())
                         for p in MODULES},
                "scales": {p: scales[p].cpu() for p in MODULES},
                "C": args.C, "sensor": args.sensor, "D": basis.shape[0],
                "N_pilot": int(counts.sum()), "pile": {"fitstream": True}},
               RUN / f"fit_C{args.C}_{args.sensor}{args.tag}.pt")
    log(f"wrote streaming fit (tag {args.tag!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prep", "fit", "fitstream", "stream",
                                     "bank", "interp", "ablate"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--sensor", default="ig5")
    ap.add_argument("--fit_positions", type=int, default=65536)
    ap.add_argument("--pos_per_seq", type=int, default=64)
    ap.add_argument("--batch_blocks", type=int, default=8)
    ap.add_argument("--feat_dim", type=int, default=65536)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--quota", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--target_tokens", type=int, default=1_000_000_000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--text_batch", type=int, default=1024)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--bank_chunk", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--out", default="out/stream67.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--save_bank", action="store_true")
    ap.add_argument("--own_mode", choices=["entry", "neuron", "neuron_tied"],
                    default="entry",
                    help="ownership granularity: per weight entry (default), "
                         "per functional unit (row/column = MLP neuron or head "
                         "dim), or per unit with the MLP neuron tied across "
                         "c_fc and down_proj")
    ap.add_argument("--qk_joint", choices=["off", "head", "dim"], default="off",
                    help="tie q_proj/k_proj ownership so the bilinear QK "
                         "product is scaled coherently: by head, or by head "
                         "and rope pair")
    ap.add_argument("--ov_joint", choices=["off", "dim", "head"], default="off",
                    help="tie v_proj/o_proj ownership per head dim or per head "
                         "(the OV circuit, the linear analogue of --qk_joint)")
    ap.add_argument("--attn_joint", choices=["off", "head"], default="off",
                    help="tie all four attention matrices of a head into one "
                         "unit; supersedes --qk_joint/--ov_joint")
    ap.add_argument("--bank_tag", default="",
                    help="suffix for the saved bank file (output naming only; "
                         "--tag selects the reservoir)")
    ap.add_argument("--drop_arm", action="store_true",
                    help="also measure the complement curve (remove the top-k, "
                         "keep the rest) -- necessity rather than sufficiency")
    ap.add_argument("--interp_tokens", type=int, default=1536,
                    help="interp: held-out token positions to score")
    ap.add_argument("--interp_comps", type=int, default=64,
                    help="interp: components to report (half strongest, half "
                         "sampled from the rest)")
    ap.add_argument("--interp_ctx", type=int, default=8,
                    help="interp: top contexts per reported component")
    ap.add_argument("--interp_blocks", type=int, default=512,
                    help="interp: corpus blocks to draw contexts from")
    ap.add_argument("--ablate_comps", default="",
                    help="ablate: comma-separated component ids (default: the "
                         "--ablate_n highest-peak components with enough hits)")
    ap.add_argument("--ablate_n", type=int, default=12)
    ap.add_argument("--ablate_min_hits", type=int, default=8)
    ap.add_argument("--ablate_target", type=int, default=24,
                    help="ablate: top-firing tokens used as the target set")
    ap.add_argument("--ablate_ctl", type=int, default=64,
                    help="ablate: random control tokens")
    ap.add_argument("--ablate_alpha", type=float, default=0.0,
                    help="ablate: 0 = delete owned mass, -1 = invert it")
    ap.add_argument("--ablate_null_pool", type=int, default=48,
                    help="ablate: candidates searched for a mass-matched null")
    ap.add_argument("--interp_seed", type=int, default=999,
                    help="interp: corpus seed (999/555 are prestaged; seed 0 "
                         "is the fit/eval draw, so keep off it)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fit_tag", default="")
    ap.add_argument("--refine_iters", type=int, default=1)
    ap.add_argument("--refine_tokens", type=int, default=256)
    ap.add_argument("--rank_r0", type=float, default=16.0)
    ap.add_argument("--fs_tokens", type=int, default=80_000_000)
    ap.add_argument("--fs_stats_batches", type=int, default=60)
    ap.add_argument("--fs_cov_batches", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    if args.phase == "fitstream":
        phase_fitstream(args, dev)
    elif args.phase == "prep":
        phase_prep(args)
    elif args.phase == "fit":
        phase_fit(args, dev)
    elif args.phase == "stream":
        phase_stream(args, dev)
    elif args.phase == "bank":
        phase_bank(args, dev)
    elif args.phase == "interp":
        phase_interp(args, dev)
    elif args.phase == "ablate":
        phase_ablate(args, dev)


if __name__ == "__main__":
    main()
