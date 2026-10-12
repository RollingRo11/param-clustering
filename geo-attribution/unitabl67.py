"""Ownership from MEASURED unit ablations on the C=8192 geo decomposition.

The geo bank splits weight mass among clusters in proportion to first-order
attribution, a linear stand-in for "what happens when this weight is removed".
This module measures the real thing at the granularity where ablation is
well defined -- functional units, each an additive term of its layer's output:

    mlp   one MLP neuron: c_fc row n  +  down_proj column n
    head  one attention head: q/k/v rows of the head + o_proj columns
    qk    a head's q/k rows only (its attention pattern)
    ov    one head dimension: v_proj row d + o_proj column d

Groupings partition every decomposed matrix exactly:
    head = mlp + head            (12,312 units)
    fine = mlp + qk + ov         (15,384 units)

Phases (training-free; forward passes only, except the ig5 diagnostics):
    label    label every position of a fresh stratified corpus draw with its
             stream cluster, exactly as phase_stream assigns them
    effects  E[u, x] = KL(full || unit u zeroed) and dlogp(gold) at every token
             x of that corpus; one forward pass per unit; sharded, resumable
    own      pool E to clusters, centre per unit, binary OVERLAPPING membership
             by t-statistic; units with no differential owner form a shared
             pool, split by clustering their centred profiles (no size cap)
    eval     union-keep / drop curves on the 48 phase_bank eval tokens (same
             selection), canonical (attribution) and oracle (component
             ablation) rankings, per-component selectivity; also the existing
             tiedqk bank re-read under union semantics
    diag     per-token unit attribution vs measured effect

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k  python -u unitabl67.py <phase> ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stream67 as S                      # sets CKPT / pile cache, TF32
from stream67 import MODULES, SENSORS, load67, sensor_pg, features, reward_at
import sensor_study67 as S67

D_MODEL, N_HEAD, D_MLP, N_LAYER = 768, 6, 3072, 4
TYPES = ("mlp", "head", "qk", "ov")
GROUPINGS = {"head": ("mlp", "head"), "fine": ("mlp", "qk", "ov")}
KEEP_GRID = [0, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536,
             2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576]


def log(m):
    print(f"[unitabl] {m}", flush=True)


def out_dir(args) -> Path:
    d = Path(args.out_dir) if args.out_dir else S.RUN.parent / "unitabl"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ units ---

def build_units(d_model=None, n_head=None, d_mlp=None, n_layer=None):
    """(type, layer, index, slices) per unit; slice = (module, axis, lo, hi),
    axis 0 = rows of a read-side matrix, axis 1 = columns of a write-side one."""
    d_model = D_MODEL if d_model is None else d_model
    n_head = N_HEAD if n_head is None else n_head
    d_mlp = D_MLP if d_mlp is None else d_mlp
    n_layer = N_LAYER if n_layer is None else n_layer
    hd = d_model // n_head
    U = []
    for l in range(n_layer):
        A, M = f"h.{l}.attn.", f"h.{l}.mlp."
        for n in range(d_mlp):
            U.append(("mlp", l, n, ((M + "c_fc", 0, n, n + 1),
                                    (M + "down_proj", 1, n, n + 1))))
        for h in range(n_head):
            lo, hi = h * hd, (h + 1) * hd
            U.append(("head", l, h, ((A + "q_proj", 0, lo, hi),
                                     (A + "k_proj", 0, lo, hi),
                                     (A + "v_proj", 0, lo, hi),
                                     (A + "o_proj", 1, lo, hi))))
        for h in range(n_head):
            lo, hi = h * hd, (h + 1) * hd
            U.append(("qk", l, h, ((A + "q_proj", 0, lo, hi),
                                   (A + "k_proj", 0, lo, hi))))
        for d in range(d_model):
            U.append(("ov", l, d, ((A + "v_proj", 0, d, d + 1),
                                   (A + "o_proj", 1, d, d + 1))))
    return U


def unit_ids_of(units, grouping):
    types = GROUPINGS[grouping]
    return torch.tensor([g for g, u in enumerate(units) if u[0] in types])


def type_mask(units, t):
    return torch.tensor([u[0] == t for u in units])


def type_lookup(units, types, W0, dev, strict):
    """Per module: unit id owning each row (read side) / column (write side),
    restricted to `types`; -1 where none of those types touches the entry."""
    look = {}
    for p in MODULES:
        n = W0[p].shape[1] if S.is_write_side(p) else W0[p].shape[0]
        look[p] = torch.full((n,), -1, dtype=torch.long)
    for g, (t, l, i, sl) in enumerate(units):
        if t not in types:
            continue
        for p, axis, lo, hi in sl:
            assert (look[p][lo:hi] < 0).all(), "overlapping units in grouping"
            look[p][lo:hi] = g
    if strict:
        for p in MODULES:
            assert (look[p] >= 0).all(), (types, p)
    return {p: v.to(dev) for p, v in look.items()}


def unit_lookup(units, grouping, W0, dev):
    return type_lookup(units, GROUPINGS[grouping], W0, dev, strict=True)


def unit_mass(units, W0):
    m = torch.zeros(len(units), dtype=torch.float64)
    for g, (t, l, i, sl) in enumerate(units):
        for p, axis, lo, hi in sl:
            blk = W0[p][lo:hi] if axis == 0 else W0[p][:, lo:hi]
            m[g] += float(blk.double().pow(2).sum())
    return m


class Net:
    def __init__(self, dev, mode="plain", model=None):
        self.model = model if model is not None else load67(dev, mode)
        self.W0 = {p: self.model.get_submodule(p).weight.detach().clone()
                   for p in MODULES}
        self.Wt = {p: self.model.get_submodule(p).weight for p in MODULES}
        self.dev = dev

    def half(self):
        """fp16 weights and activations (V100 has no TF32); W0/Wt rebuilt."""
        self.model.half()
        self.W0 = {p: self.model.get_submodule(p).weight.detach().clone()
                   for p in MODULES}
        self.Wt = {p: self.model.get_submodule(p).weight for p in MODULES}
        return self

    @torch.no_grad()
    def restore(self):
        for p in MODULES:
            self.Wt[p].copy_(self.W0[p])

    @torch.no_grad()
    def zero_unit(self, unit):
        for p, axis, lo, hi in unit[3]:
            if axis == 0:
                self.Wt[p][lo:hi].zero_()
            else:
                self.Wt[p][:, lo:hi].zero_()

    @torch.no_grad()
    def restore_unit(self, unit):
        for p, axis, lo, hi in unit[3]:
            if axis == 0:
                self.Wt[p][lo:hi].copy_(self.W0[p][lo:hi])
            else:
                self.Wt[p][:, lo:hi].copy_(self.W0[p][:, lo:hi])

    @torch.no_grad()
    def apply_unit_mask(self, keep, look):
        """keep: bool [n_units_total].  The grouping's lookup decides which
        rows / columns each unit owns; units of other types are ignored."""
        for p in MODULES:
            km = keep[look[p]].to(self.W0[p].dtype)
            if S.is_write_side(p):
                self.Wt[p].copy_(self.W0[p] * km[None, :])
            else:
                self.Wt[p].copy_(self.W0[p] * km[:, None])

    @torch.no_grad()
    def apply_entry_mask(self, masks):
        for p in MODULES:
            self.Wt[p].copy_(self.W0[p] * masks[p].to(self.W0[p].dtype))


def token_attr(net, cfg, ids_row, t):
    """IG-K attribution of the gold logit at position t to every weight
    entry (A * W), identical to stream67.phase_bank's token_attr."""
    K, path = cfg["K"], cfg.get("ig_path", "weights")
    A = {p: torch.zeros_like(net.W0[p]) for p in MODULES}
    for step in range(K):
        a = (step + 1) / K
        with torch.no_grad():
            for p in MODULES:
                net.Wt[p].copy_(net.W0[p] * (a if path == "weights" else 1.0))
        pre, post, hs = {}, {}, []
        for p in MODULES:
            mod = net.model.get_submodule(p)

            def hook(mm, inp, out, _p=p):
                pre[_p] = inp[0]
                out.retain_grad()
                post[_p] = out
            hs.append(mod.register_forward_hook(hook))
        lg = net.model(ids_row[None])
        for h in hs:
            h.remove()
        reward = reward_at(lg, t, ids_row, cfg.get("scalar", "gold"))
        grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
        with torch.no_grad():
            for p, g in zip(MODULES, grads):
                A[p] += (g[0].float().t() @ pre[p][0].detach().float()) / K
        del pre, post, grads, lg
    net.restore()
    with torch.no_grad():
        return {p: A[p] * net.W0[p] for p in MODULES}


def unit_attr(AW, units, looks_by_type):
    """Signed attribution summed over each unit's entries, for every unit of
    every type (a q row contributes to both its head unit and its qk unit)."""
    a = torch.zeros(len(units), dtype=torch.float64,
                    device=AW[MODULES[0]].device)
    for p in MODULES:
        s = AW[p].double().sum(0 if S.is_write_side(p) else 1)
        for lk in looks_by_type.values():
            idx = lk[p]
            ok = idx >= 0
            if ok.any():
                a.index_add_(0, idx[ok], s[ok])
    return a


# ----------------------------------------------------------------- tokens ---

def load_eval_tokens(args, dev):
    """The 48 (sequence, position) pairs phase_bank evaluates on."""
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_cpu, _, _ = load_pile_blocks(tok, 168, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    eval_ids = ids_cpu[-args.eval_seqs:].to(dev)
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]
    return tok, eval_ids, samp


def lab_path(args, with_clu=True):
    """lab{lab_tag}.pt holds the corpus ids (and labels under the default
    clustering); lab{lab_tag}{clu_tag}.pt the same corpus labelled by another
    clustering.  The effect matrix depends on the corpus only."""
    return out_dir(args) / f"lab{args.lab_tag}{args.clu_tag if with_clu else ''}.pt"


# ------------------------------------------------------------------ label ---

def label_batch(net, cfg, fit, idx, sel, dev):
    """Cluster label of every selected position of idx, phase_stream's way."""
    K = cfg["K"]
    spec = {p: (fit["spec"][p][0].to(dev), fit["spec"][p][1].to(dev))
            for p in MODULES}
    scales = {p: fit["scales"][p].to(dev) for p in MODULES}
    centn = F.normalize(fit["centroids"].to(dev), dim=1)
    Ps, Gs = sensor_pg(net.model, idx, cfg, net.W0, net.Wt, sel)
    X = features(Ps, Gs, spec, scales, K, dev)
    Z = (X - fit["mean"].to(dev)) @ fit["basis"].to(dev)
    lab = (F.normalize(Z, dim=1) @ centn.t()).argmax(1)
    del Ps, Gs, X, Z
    return lab.view(idx.shape[0], -1)


def phase_label(args, dev):
    fit = torch.load(S.RUN / f"fit_C{args.C}_{args.sensor}{args.fit_tag}.pt",
                     map_location="cpu", weights_only=False)
    cfg = SENSORS[args.sensor]
    net = Net(dev, cfg["mode"])
    base = lab_path(args, with_clu=False)
    if args.clu_tag and base.exists():
        # same corpus, another clustering: keep the ids, redo the labels
        ids_cpu = torch.load(base, map_location="cpu", weights_only=False)["ids"].long()
        log(f"re-labelling the corpus of {base.name} with fit tag {args.fit_tag!r}")
    elif args.lab_source == "bin":
        # the bin is round-robin over pile subsets in batches of ~1k documents,
        # so a contiguous window is a few subsets; stride across it instead
        arr = np.memmap(S.TOKBIN, dtype=np.uint16, mode="r")
        stride = args.lab_stride or ((len(arr) - args.lab_offset - args.seq) // args.n_seq)
        starts = [args.lab_offset + i * stride for i in range(args.n_seq)]
        assert starts[-1] + args.seq <= len(arr), "corpus runs past the bin"
        ids_cpu = torch.from_numpy(np.stack([
            np.asarray(arr[o:o + args.seq], dtype=np.int64) for o in starts]))
        log(f"corpus: {args.n_seq} x {args.seq} tokens from the bin, offset "
            f"{args.lab_offset}, stride {stride} (bin has {len(arr)} tokens)")
    else:
        from transformers import AutoTokenizer
        from pile_data import load_pile_blocks
        tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
        ids_cpu, _, _ = load_pile_blocks(tok, args.n_seq, args.seq,
                                         seed=args.lab_seed,
                                         tokenizer_name=S67.TOKENIZER)
    N, T = ids_cpu.shape
    sel = torch.arange(4, T - 2).to(dev)     # phase_stream's `avail`
    labels = torch.full((N, T), -1, dtype=torch.int16)
    t0 = time.perf_counter()
    for b0 in range(0, N, args.lab_batch):
        idx = ids_cpu[b0:b0 + args.lab_batch].to(dev)
        lab = label_batch(net, cfg, fit, idx, sel, dev)
        labels[b0:b0 + idx.shape[0], 4:T - 2] = lab.to(torch.int16).cpu()
        if (b0 // args.lab_batch) % 8 == 0:
            log(f"label {b0 + idx.shape[0]}/{N} seqs "
                f"({time.perf_counter() - t0:.0f}s)")
    units = build_units()
    mass = unit_mass(units, {p: net.W0[p].cpu() for p in MODULES})
    valid = labels[labels >= 0].long()
    counts = torch.bincount(valid, minlength=args.C)
    d = {"ids": ids_cpu.to(torch.int32), "labels": labels,
         "seed": args.lab_seed, "C": args.C, "sensor": args.sensor,
         "fit_tag": args.fit_tag, "lab_source": args.lab_source,
         "counts": counts, "unit_mass": mass,
         "unit_type": [u[0] for u in units],
         "unit_layer": [u[1] for u in units],
         "unit_index": [u[2] for u in units]}
    torch.save(d, lab_path(args))
    cov = {n: int((counts >= n).sum()) for n in (1, 4, 8, 16, 32)}
    summ = {"n_seq": N, "seq": T, "n_labeled": int(valid.numel()),
            "clusters_with_at_least": cov,
            "count_quantiles": {q: float(np.quantile(counts.numpy(), q))
                                for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
            "largest_cluster": int(counts.max()),
            "n_units": len(units),
            "units_by_type": {t: int(type_mask(units, t).sum()) for t in TYPES},
            "mass_by_type": {t: float(mass[type_mask(units, t)].sum())
                             for t in TYPES}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summ, indent=1))
    log(f"labelled {valid.numel()} positions of {N}x{T}; clusters with >=4 "
        f"tokens: {cov[4]}/{args.C}; wrote {lab_path(args).name}, {args.out}")


# ---------------------------------------------------------------- effects ---

def shard_range(n_all, shard, nshard):
    return n_all * shard // nshard, n_all * (shard + 1) // nshard


def eff_paths(args, kind, shard, q=0):
    qs = f"_q{q}of{args.nq}" if args.nq > 1 else ""
    return out_dir(args) / f"E{args.lab_tag}_{kind}_s{shard}of{args.nshard}{qs}.f16"


def done_path(args, shard, q=0):
    qs = f"_q{q}of{args.nq}" if args.nq > 1 else ""
    return out_dir(args) / f"E{args.lab_tag}_done_s{shard}of{args.nshard}{qs}.npy"


def resolve_device_defaults(args, dev):
    """--precision auto: fp32 (TF32 matmuls) on Ampere+, fp16 on older GPUs
    (no TF32 there, fp32 is 8x slower).  --chunk 0: by GPU memory."""
    if str(dev).startswith("cuda"):
        cap = torch.cuda.get_device_capability(dev)[0]
        mem = torch.cuda.get_device_properties(dev).total_memory / 2**30
        name = torch.cuda.get_device_name(dev)
    else:
        cap, mem, name = 8, 64.0, "cpu"
    if args.precision == "auto":
        args.precision = "fp32" if cap >= 8 else "fp16"
    if args.chunk <= 0:
        args.chunk = 16 if mem >= 30 else (8 if mem >= 20 else 4)
    log(f"device {name}: {mem:.0f} GiB, capability {cap} -> precision "
        f"{args.precision}, chunk {args.chunk}")


def phase_effects(args, dev):
    """One (unit shard, corpus slice) block of E.  --q/--nq slice the corpus
    by sequence so a 32 GB GPU can hold the base distribution; --precision
    fp16 runs the model in half (base and masked passes alike) and logs the
    KL between the fp32 and fp16 models as the measurement's noise floor."""
    lab = torch.load(lab_path(args, with_clu=False), map_location="cpu",
                     weights_only=False)
    N_all = lab["ids"].shape[0]
    q_lo, q_hi = shard_range(N_all, args.q, args.nq)
    ids = lab["ids"][q_lo:q_hi].long().to(dev)
    N, T = ids.shape
    units = build_units()
    lo, hi = shard_range(len(units), args.shard, args.nshard)
    net = Net(dev)
    V = net.model.cfg["vocab_size"]
    t0 = time.perf_counter()
    resolve_device_defaults(args, dev)
    c_chk = min(args.chunk, N)
    with torch.no_grad():
        ref32 = F.log_softmax(net.model(ids[:c_chk]).float(), -1)
    if args.precision == "fp16":
        net.half()
    with torch.no_grad():
        ref = F.log_softmax(net.model(ids[:c_chk]).float(), -1)
        floor = (ref32.exp() * (ref32 - ref)).sum(-1)
    log(f"precision {args.precision}: KL(fp32 model || run model) on {c_chk} "
        f"seqs: mean {float(floor.mean()):.2e}, max {float(floor.max()):.2e} "
        f"(the noise floor of every effect below)")
    del ref32, ref, floor
    base_lp = torch.empty(N, T, V, device=dev, dtype=torch.float32)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            base_lp[c0:c0 + args.chunk] = F.log_softmax(
                net.model(ids[c0:c0 + args.chunk]).float(), -1)
    gold = ids[:, 1:]
    base_gold = base_lp[:, :-1].gather(-1, gold[..., None])[..., 0]
    mem = (f", {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB"
           if str(dev).startswith("cuda") else "")
    log(f"base log-probs for seqs [{q_lo}, {q_hi}) x {T} tokens "
        f"({time.perf_counter() - t0:.0f}s{mem})")

    fk = eff_paths(args, "kl", args.shard, args.q)
    fd = eff_paths(args, "dl", args.shard, args.q)
    fdone = done_path(args, args.shard, args.q)
    shape = (hi - lo, N * T)
    resume = fk.exists() and fd.exists() and fdone.exists()
    Ek = np.memmap(fk, np.float16, "r+" if resume else "w+", shape=shape)
    Ed = np.memmap(fd, np.float16, "r+" if resume else "w+", shape=shape)
    done = np.load(fdone) if resume else np.zeros(hi - lo, dtype=bool)
    log(f"shard {args.shard}/{args.nshard} slice {args.q}/{args.nq}: units "
        f"[{lo}, {hi}), {int(done.sum())} already done")

    t0, n_new = time.perf_counter(), 0
    kl_buf = torch.empty(N, T, device=dev)
    dl_buf = torch.zeros(N, T, device=dev)
    for j in range(hi - lo):
        if done[j]:
            continue
        u = units[lo + j]
        net.zero_unit(u)
        with torch.no_grad():
            for c0 in range(0, N, args.chunk):
                c1 = min(N, c0 + args.chunk)
                lp = F.log_softmax(net.model(ids[c0:c1]).float(), -1)
                b = base_lp[c0:c1]
                kl_buf[c0:c1] = (b.exp() * (b - lp)).sum(-1)
                dl_buf[c0:c1, :-1] = (lp[:, :-1].gather(
                    -1, gold[c0:c1, :, None])[..., 0] - base_gold[c0:c1])
                del lp
        net.restore_unit(u)
        Ek[j] = kl_buf.flatten().half().cpu().numpy()
        Ed[j] = dl_buf.flatten().half().cpu().numpy()
        done[j] = True
        n_new += 1
        if n_new % args.save_every == 0 or j == hi - lo - 1:
            Ek.flush()
            Ed.flush()
            np.save(fdone, done)
            r = n_new / (time.perf_counter() - t0)
            left = int((~done).sum())
            log(f"unit {lo + j} ({u[0]} L{u[1]} #{u[2]}): mean KL "
                f"{float(kl_buf.mean()):.5f}  {r:.2f} units/s  "
                f"eta {left / max(r, 1e-9) / 60:.0f} min")
    net.restore()
    log(f"shard {args.shard} slice {args.q} done ({time.perf_counter() - t0:.0f}s)")


def load_effects(args, kind):
    """(lab dict, list of (lo, hi, [memmap per corpus slice])) over shards."""
    lab = torch.load(lab_path(args), map_location="cpu", weights_only=False)
    N_all, T = lab["ids"].shape
    n_all = len(build_units())
    parts = []
    for s in range(args.nshard):
        lo, hi = shard_range(n_all, s, args.nshard)
        mms = []
        for q in range(args.nq):
            q_lo, q_hi = shard_range(N_all, q, args.nq)
            done = np.load(done_path(args, s, q))
            assert done.all(), f"shard {s} slice {q}: {int((~done).sum())} units missing"
            mms.append(np.memmap(eff_paths(args, kind, s, q), np.float16, "r",
                                 shape=(hi - lo, (q_hi - q_lo) * T)))
        parts.append((lo, hi, mms))
    return lab, parts


def read_rows(parts, g0, g1):
    """rows [g0, g1) of E as float16 [g1-g0, N*T], token order = sequence-major"""
    out = []
    for lo, hi, mms in parts:
        a, b = max(g0, lo), min(g1, hi)
        if a < b:
            out.append(np.concatenate([np.asarray(mm[a - lo:b - lo]) for mm in mms], 1))
    return np.concatenate(out, 0)


# -------------------------------------------------------------------- own ---

def split_pool(P, pool_ids, seed):
    """Split the shared pool by clustering its units' centred profiles.
    P: [n_cov, n_pool] t-statistics.  k = participation ratio of the gram of
    the pool-centred profiles -- data-driven, no size cap."""
    X = P.t().float()
    if X.shape[0] < 4:
        return [pool_ids], 1.0
    Xc = X - X.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Xc).pow(2)
    pr = float(lam.sum().pow(2) / lam.pow(2).sum().clamp_min(1e-30))
    k = max(2, min(int(round(pr)), X.shape[0] // 2))
    _, lab = S67.kmeans(Xc, k, 50, seed, spherical=True)
    comps = [pool_ids[lab == j] for j in range(k)]
    comps = [c for c in comps if c.numel() > 0]
    log(f"pool split: {X.shape[0]} units, participation ratio {pr:.1f} -> "
        f"{len(comps)} pool components, sizes "
        f"{sorted([int(c.numel()) for c in comps], reverse=True)[:12]}")
    return comps, pr


def qdict(x, qs=(0.1, 0.5, 0.9, 0.99)):
    x = np.asarray(x, dtype=np.float64)
    return {str(q): float(np.quantile(x, q)) for q in qs} if x.size else {}


def pool_effects(args):
    """Cluster-by-unit mean effect M, plus per-unit corpus mean / std.
    Saved to pooled.pt so ownership rules can be iterated without re-reading
    the 8 GB effect matrix; --from_pooled reuses it."""
    pooled_f = out_dir(args) / f"pooled{args.lab_tag}{args.clu_tag}.pt"
    lab = torch.load(lab_path(args), map_location="cpu", weights_only=False)
    if args.from_pooled and pooled_f.exists():
        P = torch.load(pooled_f, map_location="cpu", weights_only=False)
        log(f"loaded {pooled_f.name}")
        return lab, P["M"], P["mean_all"], P["std_all"], P["counts"]
    _, parts = load_effects(args, "kl")
    C = int(lab["C"])
    labels = lab["labels"].flatten().long()
    valid = labels >= 0
    lv = labels[valid]
    counts = torch.bincount(lv, minlength=C)
    n_all = len(build_units())
    t0 = time.perf_counter()
    M = torch.zeros(C, n_all)
    mean_all = torch.zeros(n_all)
    sq_all = torch.zeros(n_all)
    for g0 in range(0, n_all, args.own_chunk):
        g1 = min(n_all, g0 + args.own_chunk)
        X = torch.from_numpy(read_rows(parts, g0, g1)).float()[:, valid]
        M[:, g0:g1] = torch.zeros(C, g1 - g0).index_add_(0, lv, X.t())
        mean_all[g0:g1] = X.mean(1)
        sq_all[g0:g1] = X.pow(2).mean(1)
        del X
    M /= counts.clamp_min(1)[:, None].float()
    std_all = (sq_all - mean_all.pow(2)).clamp_min(0).sqrt()
    torch.save({"M": M, "mean_all": mean_all, "std_all": std_all,
                "counts": counts, "C": C}, pooled_f)
    log(f"pooled E ({time.perf_counter() - t0:.0f}s) -> {pooled_f.name}")
    return lab, M, mean_all, std_all, counts


def share_sets(W, frac):
    """Per unit (column of W >= 0), the smallest set of rows carrying `frac`
    of the column's total: bool mask [n_rows, n_units] and set sizes."""
    tot = W.sum(0).clamp_min(1e-30)
    vals, order = W.sort(0, descending=True)
    cum = vals.cumsum(0) / tot[None]
    # rows up to and including the one that crosses frac
    inset_sorted = torch.cat([torch.ones(1, W.shape[1], dtype=torch.bool),
                              (cum[:-1] < frac)], 0)
    memb = torch.zeros_like(W, dtype=torch.bool).scatter_(0, order, inset_sorted)
    return memb, inset_sorted.sum(0)


def phase_own(args):
    lab, M, mean_all, std_all, counts = pool_effects(args)
    N, T = lab["ids"].shape
    C = int(lab["C"])
    units = build_units()
    n_all = len(units)
    mass = lab["unit_mass"].float()
    covered = counts >= args.n_min
    cov_ids = covered.nonzero().flatten()
    n_c = counts[covered].float()
    log(f"{int(covered.sum())} covered clusters; unit mean-KL quantiles "
        + ", ".join(f"{q}:{v:.2e}" for q, v in qdict(mean_all).items()))

    summary = {"format": "unitabl_own_v1", "n_seq": N, "seq": T, "C": C,
               "n_min": args.n_min, "covered_clusters": int(covered.sum()),
               "pool_floor": args.pool_floor, "floor": args.floor,
               "taus": args.taus, "topS": args.topS, "groupings": {}}
    maxz = {}
    for grouping in GROUPINGS:
        sub = unit_ids_of(units, grouping)
        Mc = M[covered][:, sub]
        mu, sd = mean_all[sub], std_all[sub]
        D = Mc - mu[None]                                # effect above the unit's corpus mean
        Z = D / (sd[None] / n_c.sqrt()[:, None]).clamp_min(1e-12)   # t-statistic
        R = Mc / mu[None].clamp_min(1e-12)               # ratio to the unit's corpus mean
        maxz[grouping] = Z.max(0).values
        variants, shared = {}, {}
        for tau in args.taus:
            variants[f"t{tau:g}"] = (Z > tau) & (D > args.floor)
        if args.topS > 0:
            topz, topi = Z.topk(min(args.topS, Z.shape[0]), dim=0)
            mtop = torch.zeros_like(Z, dtype=torch.bool)
            mtop.scatter_(0, topi, topz > 2.0)
            variants[f"top{args.topS}"] = mtop & (D > args.floor)
        for r in args.ratios:
            # the unit matters r times more here than on an average token, and
            # the excess is not noise
            variants[f"r{r:g}"] = (R > r) & (Z > 2.0) & (D > args.floor)
        Wm = Mc * n_c[:, None]                           # effect mass per (cluster, unit)
        n_cov = int(covered.sum())
        conc = {}
        for f in args.shares:
            memb, nset = share_sets(Wm, f)
            name = f"s{int(round(100 * f))}"
            # a unit whose top-f effect set spans more than pool_frac of the
            # clusters is shared machinery, not owned by anyone
            shared[name] = nset > args.pool_frac * n_cov
            variants[name] = memb & ~shared[name][None]
            conc[name] = nset
        n50 = conc.get("s50", share_sets(Wm, 0.5)[1])
        tot_mass = float(mass[sub].sum())
        sub_types = [units[g][0] for g in sub.tolist()]
        gsum = {"n_units": int(sub.numel()), "variants": {},
                "unit_stats": {t: {
                    "n": sub_types.count(t),
                    "mean_kl_quantiles": qdict(mu[torch.tensor([s == t for s in sub_types])]),
                    "max_z_quantiles": qdict(maxz[grouping][torch.tensor([s == t for s in sub_types])]),
                    "max_ratio_quantiles": qdict(R.max(0).values[torch.tensor([s == t for s in sub_types])]),
                    "n50_quantiles": qdict(n50[torch.tensor([s == t for s in sub_types])].float()),
                    "clusters_over_3x_quantiles": qdict((R > 3).sum(0)[torch.tensor([s == t for s in sub_types])].float())}
                    for t in GROUPINGS[grouping]}}
        log(f"{grouping}: n50 (clusters carrying half a unit's effect) quantiles "
            + ", ".join(f"{q}:{v:.0f}" for q, v in qdict(n50.float()).items())
            + "; clusters with >3x the unit's mean: "
            + ", ".join(f"{q}:{v:.0f}" for q, v in qdict((R > 3).sum(0).float()).items()))
        for name, memb in variants.items():
            has = memb.any(0)
            members = [[] for _ in range(C)]
            for ci, ui in memb.nonzero().tolist():
                members[int(cov_ids[ci])].append(int(sub[ui]))
            members = [torch.tensor(m, dtype=torch.long) for m in members]
            unowned = ~has
            pool_m = unowned & (mu > args.pool_floor)
            inert_m = unowned & ~(mu > args.pool_floor)
            pool_ids, inert_ids = sub[pool_m], sub[inert_m]
            pool_comps, pr = split_pool(Z[:, pool_m], pool_ids, args.seed)
            sizes = torch.tensor([m.numel() for m in members])
            per_unit = memb.sum(0)
            own = {"grouping": grouping, "variant": name, "C": C,
                   "members": members, "pool": pool_comps, "inert": inert_ids,
                   "unit_ids": sub, "unit_mean_kl": mu, "unit_std_kl": sd,
                   "unit_max_z": maxz[grouping], "unit_n50": n50,
                   "unit_n_memberships": per_unit,
                   "covered": cov_ids, "counts": counts}
            torch.save(own, out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{grouping}_{name}.pt")
            vs = {"live_components": int((sizes > 0).sum()),
                  "memberships": int(memb.sum()),
                  "units_owned": int(has.sum()),
                  "units_pool": int(pool_m.sum()),
                  "units_inert": int(inert_m.sum()),
                  "mass_owned": float(mass[sub[has]].sum() / tot_mass),
                  "mass_pool": float(mass[pool_ids].sum() / tot_mass),
                  "mass_inert": float(mass[inert_ids].sum() / tot_mass),
                  "pool_components": len(pool_comps), "pool_pr": pr,
                  "pool_sizes": sorted([int(c.numel()) for c in pool_comps],
                                       reverse=True),
                  "size_quantiles": qdict(sizes[sizes > 0]),
                  "largest_component": int(sizes.max()),
                  "memberships_per_unit_quantiles": qdict(per_unit[has], (0.5, 0.9, 0.99)),
                  "max_memberships_per_unit": int(per_unit.max()),
                  "owned_by_type": {t: sum(1 for g in sub[has].tolist()
                                           if units[g][0] == t) for t in TYPES},
                  "pool_by_type": {t: sum(1 for g in pool_ids.tolist()
                                          if units[g][0] == t) for t in TYPES},
                  "pool_top_clusters": [
                      [int(cov_ids[i]) for i in
                       Z[:, torch.isin(sub, c)].mean(1).topk(
                           min(5, int(covered.sum()))).indices.tolist()]
                      for c in pool_comps],
                  "sizes": sizes.tolist()}
            gsum["variants"][name] = vs
            log(f"{grouping}/{name}: live comps {vs['live_components']}, "
                f"memberships {vs['memberships']}, owned {vs['units_owned']} "
                f"pool {vs['units_pool']} inert {vs['units_inert']} units; "
                f"mass owned/pool/inert {vs['mass_owned']:.2f}/"
                f"{vs['mass_pool']:.2f}/{vs['mass_inert']:.2f}; "
                f"largest comp {vs['largest_component']}")
        summary["groupings"][grouping] = gsum
    zh = torch.zeros(n_all).index_copy_(0, unit_ids_of(units, "head"), maxz["head"])
    zf = torch.zeros(n_all).index_copy_(0, unit_ids_of(units, "fine"), maxz["fine"])
    n50_all = torch.zeros(n_all)
    for grouping in GROUPINGS:
        sub = unit_ids_of(units, grouping)
        Wm = M[covered][:, sub] * n_c[:, None]
        n50_all[sub] = share_sets(Wm, 0.5)[1].float()
    summary["arrays"] = {
        "unit_type": [u[0] for u in units],
        "unit_layer": [u[1] for u in units],
        "unit_n50": [int(v) for v in n50_all.tolist()],
        "unit_mean_kl": [round(float(v), 7) for v in mean_all.tolist()],
        "unit_max_z_head": [round(float(v), 3) for v in zh.tolist()],
        "unit_max_z_fine": [round(float(v), 3) for v in zf.tolist()],
        "unit_mass": [round(float(v), 4) for v in mass.tolist()],
        "cluster_counts": counts.tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    log(f"wrote {args.out}")


# ------------------------------------------------------------------- eval ---

def kl_of(lp_masked, lp_base):
    return float(F.kl_div(lp_masked, lp_base, log_target=True, reduction="sum"))


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def score_summary(score_attr, score_orc, samp):
    """How well the canonical (attribution) ranking agrees with the oracle."""
    out = {"spearman": [], "top32_overlap": [], "top8_overlap": []}
    for j in range(len(samp)):
        a, o = score_attr[:, j], score_orc[:, j]
        out["spearman"].append(spearman(a, o))
        for k, key in ((32, "top32_overlap"), (8, "top8_overlap")):
            k = min(k, a.numel())
            out[key].append(float(torch.isin(a.topk(k).indices,
                                             o.topk(k).indices).float().mean()))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def run_curves(args, net, samp, kl_token, rankings, modes, n_comp, res, dev,
               entry_level, comp_masks=None, unit_cm=None, n_diff=None):
    """Keep top-k (union of the kept components' units at full strength) and
    drop top-k (its exact complement), per token, ranking, and mode."""
    for mode_name, always in modes:
        n_rank = n_comp if always is None else n_diff
        if getattr(args, "keep_every", 0) > 0:
            # dense grid: every keep_every-th k (1 = every single component)
            KEEP = list(range(0, n_rank, args.keep_every)) + [n_rank]
        else:
            KEEP = [k for k in KEEP_GRID if k < n_rank] + [n_rank]
        rank_names = list(rankings) + ([] if getattr(args, "no_random", False) else ["random"])
        for rank_name in rank_names:
            keep_kl = np.zeros((len(samp), len(KEEP)))
            drop_kl = np.zeros((len(samp), len(KEEP)))
            kept_units = np.zeros((len(samp), len(KEEP)))
            kept_mass = np.zeros((len(samp), len(KEEP)))
            t0 = time.perf_counter()
            for j in range(len(samp)):
                if rank_name == "random":
                    g = torch.Generator().manual_seed(777 + j)
                    order = torch.randperm(n_rank, generator=g).to(dev)
                else:
                    order = rankings[rank_name][:n_rank, j].argsort(descending=True)
                if entry_level:
                    kb = torch.zeros(n_comp, dtype=torch.bool, device=dev)
                else:
                    CM, look, sub, mass = unit_cm
                    sel = torch.zeros(CM.shape[1], dtype=torch.bool, device=dev)
                prev = 0
                for ki, kk in enumerate(KEEP):
                    if kk > prev:
                        if entry_level:
                            kb[order[prev:kk]] = True
                        else:
                            sel |= CM[order[prev:kk]].any(0)
                    prev = kk
                    if not entry_level:
                        # keep = the ranked components' units (+ the always-on
                        # sets in poolon mode); drop = exactly those units
                        # removed, everything else -- pool included -- intact
                        keep = sel if always is None else (sel | always)
                    with torch.no_grad():
                        if entry_level:
                            masks = comp_masks(kb)
                            net.apply_entry_mask(masks)
                            kept_units[j, ki] = float(sum(m.sum() for m in masks.values()))
                            kept_mass[j, ki] = float(sum(
                                (net.W0[p].pow(2) * masks[p]).sum() for p in MODULES))
                            keep_kl[j, ki] = kl_token(j)
                            net.apply_entry_mask({p: ~masks[p] for p in MODULES})
                            drop_kl[j, ki] = kl_token(j)
                        else:
                            net.apply_unit_mask(keep, look)
                            ks = keep[sub]
                            kept_units[j, ki] = float(ks.sum())
                            kept_mass[j, ki] = float(mass[sub][ks].sum())
                            keep_kl[j, ki] = kl_token(j)
                            net.apply_unit_mask(~sel, look)
                            drop_kl[j, ki] = kl_token(j)
                net.restore()
            unit_total = kept_units[:, -1].mean()
            mass_total = kept_mass[:, -1].mean()
            name = f"{mode_name}/{rank_name}"
            res["results"][name] = {
                "keep_grid": KEEP,
                "kl": [round(float(v), 5) for v in keep_kl.mean(0)],
                "kl_drop": [round(float(v), 5) for v in drop_kl.mean(0)],
                "kl_median": [round(float(v), 5) for v in np.median(keep_kl, 0)],
                "kl_drop_median": [round(float(v), 5) for v in np.median(drop_kl, 0)],
                "units_kept_frac": [round(float(v), 5) for v in kept_units.mean(0) / unit_total],
                "mass_kept_frac": [round(float(v), 5) for v in kept_mass.mean(0) / mass_total],
                "k_at_kl_0.1": [int(next((k for k, v in zip(KEEP, keep_kl[j]) if v < 0.1), n_rank))
                                for j in range(len(samp))],
                "k_at_kl_1": [int(next((k for k, v in zip(KEEP, keep_kl[j]) if v < 1.0), n_rank))
                              for j in range(len(samp))]}
            mk, md = keep_kl.mean(0), drop_kl.mean(0)
            show = [k for k in (8, 64, 256, 2048) if k in KEEP]
            log(f"{name:<16} keep " + " ".join(f"@{k} {mk[KEEP.index(k)]:.3f}" for k in show)
                + " | drop " + " ".join(f"@{k} {md[KEEP.index(k)]:.3f}" for k in show[:2])
                + f" | roundtrip {mk[-1]:.1e}/{md[0]:.1e} ({time.perf_counter() - t0:.0f}s)")


def selectivity(args, net, look, comp_units, comp_kind, own, units, res, dev):
    """Ablate whole components on the labelled corpus: KL on the component's
    own cluster vs everywhere else, against a size-matched random unit set.
    --sel_lab_tag evaluates on another corpus labelled by the same clustering
    (so ownership fit on one corpus can be judged on the other)."""
    sel_tag = args.lab_tag if args.sel_lab_tag is None else args.sel_lab_tag
    lab = torch.load(out_dir(args) / f"lab{sel_tag}{args.clu_tag}.pt", map_location="cpu", weights_only=False)
    res["selectivity_corpus"] = f"lab{sel_tag}{args.clu_tag}.pt"
    ids = lab["ids"].long()[:args.sel_seqs].to(dev)
    labels = lab["labels"][:args.sel_seqs].to(dev)
    N, T = ids.shape
    net.restore()
    base_lp = torch.empty(N, T, net.model.cfg["vocab_size"], device=dev)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            base_lp[c0:c0 + args.chunk] = F.log_softmax(
                net.model(ids[c0:c0 + args.chunk]).float(), -1)

    @torch.no_grad()
    def kl_map():
        out = torch.empty(N, T, device=dev)
        for c0 in range(0, N, args.chunk):
            lp = F.log_softmax(net.model(ids[c0:c0 + args.chunk]).float(), -1)
            b = base_lp[c0:c0 + args.chunk]
            out[c0:c0 + args.chunk] = (b.exp() * (b - lp)).sum(-1)
            del lp
        return out

    C = int(own["C"])
    sizes = torch.tensor([m.numel() for m in comp_units])
    diff_ok = (sizes[:C] >= args.sel_min_units).nonzero().flatten()
    g = torch.Generator().manual_seed(args.seed + 99)
    by_size = diff_ok[sizes[diff_ok].argsort(descending=True)][:args.sel_comps // 2].tolist()
    rest = [c for c in diff_ok.tolist() if c not in set(by_size)]
    rnd = [rest[i] for i in torch.randperm(len(rest), generator=g)[
        :args.sel_comps - len(by_size)].tolist()]
    chosen = by_size + rnd + list(range(C, len(comp_units) - 1))   # + pool comps
    owned_all = torch.cat([m for m in comp_units[:C] if m.numel()]).unique()
    valid = labels >= 0
    rows = []
    t0 = time.perf_counter()
    for c in chosen:
        m = comp_units[c]
        keep = torch.ones(len(units), dtype=torch.bool, device=dev)
        keep[m] = False
        net.apply_unit_mask(keep, look)
        K = kl_map()
        own_m = (labels == c) if c < C else valid
        other = valid & ~own_m
        r = {"comp": int(c), "kind": comp_kind[c], "n_units": int(m.numel()),
             "own_tokens": int(own_m.sum()),
             "kl_own": float(K[own_m].mean()) if own_m.any() else None,
             "kl_other": float(K[other].mean()) if other.any() else None,
             "kl_all": float(K[valid].mean())}
        nm = owned_all[torch.randperm(owned_all.numel(), generator=g)[:m.numel()].to(dev)]
        keep = torch.ones(len(units), dtype=torch.bool, device=dev)
        keep[nm] = False
        net.apply_unit_mask(keep, look)
        Kn = kl_map()
        r["null_kl_own"] = float(Kn[own_m].mean()) if own_m.any() else None
        r["null_kl_other"] = float(Kn[other].mean()) if other.any() else None
        rows.append(r)
    net.restore()
    res["selectivity"] = {"n_seqs": N, "sel_min_units": args.sel_min_units,
                          "rows": rows}
    dif = [r for r in rows if r["kind"] == "diff" and r["kl_own"] is not None]
    if dif:
        log(f"selectivity on {len(dif)} diff comps: own "
            f"{np.mean([r['kl_own'] for r in dif]):.4f} other "
            f"{np.mean([r['kl_other'] for r in dif]):.4f} null-own "
            f"{np.mean([r['null_kl_own'] for r in dif]):.4f} "
            f"({time.perf_counter() - t0:.0f}s)")


def selectivity_bank(args, net, sidx, owned, ent_count, C, res, dev):
    """The bank's components index the same clusters as the labels, so the
    same own-cluster / other-token test applies: delete every entry the
    component owns (union semantics).  Null = the other component nearest in
    entry count, deleted and measured on the same own tokens."""
    lab = torch.load(lab_path(args), map_location="cpu", weights_only=False)
    ids = lab["ids"].long()[:args.sel_seqs].to(dev)
    labels = lab["labels"][:args.sel_seqs].to(dev)
    N, T = ids.shape
    net.restore()
    base_lp = torch.empty(N, T, net.model.cfg["vocab_size"], device=dev)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            base_lp[c0:c0 + args.chunk] = F.log_softmax(
                net.model(ids[c0:c0 + args.chunk]).float(), -1)

    @torch.no_grad()
    def kl_map():
        out = torch.empty(N, T, device=dev)
        for c0 in range(0, N, args.chunk):
            lp = F.log_softmax(net.model(ids[c0:c0 + args.chunk]).float(), -1)
            b = base_lp[c0:c0 + args.chunk]
            out[c0:c0 + args.chunk] = (b.exp() * (b - lp)).sum(-1)
            del lp
        return out

    def ablate(c):
        net.apply_entry_mask({p: ~((sidx[p] == c) & owned[p]).any(0) for p in MODULES})

    cnt = ent_count.cpu()
    ok = (cnt >= 1).nonzero().flatten()
    g = torch.Generator().manual_seed(args.seed + 99)
    by_size = ok[cnt[ok].argsort(descending=True)][:args.sel_comps // 2].tolist()
    rest = [c for c in ok.tolist() if c not in set(by_size)]
    rnd = [rest[i] for i in torch.randperm(len(rest), generator=g)[:args.sel_comps - len(by_size)].tolist()]
    valid = labels >= 0
    rows = []
    t0 = time.perf_counter()
    for c in by_size + rnd:
        ablate(c)
        K = kl_map()
        own_m = labels == c
        other = valid & ~own_m
        r = {"comp": int(c), "kind": "diff", "n_units": int(cnt[c]),
             "own_tokens": int(own_m.sum()),
             "kl_own": float(K[own_m].mean()) if own_m.any() else None,
             "kl_other": float(K[other].mean()) if other.any() else None,
             "kl_all": float(K[valid].mean())}
        others = ok[ok != c]
        nc = int(others[(cnt[others] - cnt[c]).abs().argmin()])
        ablate(nc)
        Kn = kl_map()
        r["null_comp"] = nc
        r["null_kl_own"] = float(Kn[own_m].mean()) if own_m.any() else None
        r["null_kl_other"] = float(Kn[other].mean()) if other.any() else None
        rows.append(r)
    net.restore()
    res["selectivity"] = {"n_seqs": N, "null": "entry-count-matched other component",
                          "rows": rows}
    dif = [r for r in rows if r["kl_own"] is not None and r["kl_other"]]
    if dif:
        log(f"bank selectivity on {len(dif)} comps: own "
            f"{np.mean([r['kl_own'] for r in dif]):.4f} other "
            f"{np.mean([r['kl_other'] for r in dif]):.4f} null-own "
            f"{np.mean([r['null_kl_own'] for r in dif]):.4f} "
            f"({time.perf_counter() - t0:.0f}s)")


def selectivity_hybrid(args, net, comp_entry_mask, comp_units, C, res, dev):
    """Same own-cluster / other-token test as selectivity(), with the hybrid
    entry masks; null = the same component's units kept whole minus the slice
    is not a null, so the null is a size-matched random set of the component's
    own units' entries... kept simple: random other component of similar size."""
    lab = torch.load(lab_path(args), map_location="cpu", weights_only=False)
    ids = lab["ids"].long()[:args.sel_seqs].to(dev)
    labels = lab["labels"][:args.sel_seqs].to(dev)
    N, T = ids.shape
    net.restore()
    base_lp = torch.empty(N, T, net.model.cfg["vocab_size"], device=dev)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            base_lp[c0:c0 + args.chunk] = F.log_softmax(
                net.model(ids[c0:c0 + args.chunk]).float(), -1)

    @torch.no_grad()
    def kl_map():
        out = torch.empty(N, T, device=dev)
        for c0 in range(0, N, args.chunk):
            lp = F.log_softmax(net.model(ids[c0:c0 + args.chunk]).float(), -1)
            b = base_lp[c0:c0 + args.chunk]
            out[c0:c0 + args.chunk] = (b.exp() * (b - lp)).sum(-1)
            del lp
        return out

    sizes = torch.tensor([m.numel() for m in comp_units[:C]])
    ok = (sizes >= args.sel_min_units).nonzero().flatten()
    g = torch.Generator().manual_seed(args.seed + 99)
    by_size = ok[sizes[ok].argsort(descending=True)][:args.sel_comps // 2].tolist()
    rest = [c for c in ok.tolist() if c not in set(by_size)]
    rnd = [rest[i] for i in torch.randperm(len(rest), generator=g)[:args.sel_comps - len(by_size)].tolist()]
    valid = labels >= 0
    rows = []
    t0 = time.perf_counter()
    for c in by_size + rnd:
        m = comp_entry_mask(c)
        net.apply_entry_mask({p: ~m[p] for p in MODULES})
        K = kl_map()
        own_m = labels == c
        other = valid & ~own_m
        r = {"comp": int(c), "kind": "diff", "n_units": int(sizes[c]),
             "n_entries": int(sum(int(v.sum()) for v in m.values())),
             "own_tokens": int(own_m.sum()),
             "kl_own": float(K[own_m].mean()) if own_m.any() else None,
             "kl_other": float(K[other].mean()) if other.any() else None,
             "kl_all": float(K[valid].mean())}
        others = ok[ok != c]
        nc = int(others[(sizes[others] - sizes[c]).abs().argmin()])
        mn = comp_entry_mask(nc)
        net.apply_entry_mask({p: ~mn[p] for p in MODULES})
        Kn = kl_map()
        r["null_comp"] = nc
        r["null_kl_own"] = float(Kn[own_m].mean()) if own_m.any() else None
        r["null_kl_other"] = float(Kn[other].mean()) if other.any() else None
        rows.append(r)
    net.restore()
    res["selectivity"] = {"n_seqs": N, "null": "unit-count-matched other component", "rows": rows}
    dif = [r for r in rows if r["kl_own"] is not None and r["kl_other"]]
    if dif:
        log(f"hybrid selectivity on {len(dif)} comps: own {np.mean([r['kl_own'] for r in dif]):.4f} "
            f"other {np.mean([r['kl_other'] for r in dif]):.4f} null-own "
            f"{np.mean([r['null_kl_own'] for r in dif]):.4f} ({time.perf_counter() - t0:.0f}s)")


def phase_eval(args, dev):
    cfg = SENSORS[args.sensor]
    net = Net(dev, cfg["mode"])
    resolve_device_defaults(args, dev)
    units = build_units()
    n_all = len(units)
    mass = unit_mass(units, {p: net.W0[p].cpu() for p in MODULES}).to(dev)
    tok, eval_ids, samp = load_eval_tokens(args, dev)
    B_eval, T = eval_ids.shape
    net.restore()
    base_lp = {}
    with torch.no_grad():
        for b, t in samp:
            base_lp[(b, t)] = F.log_softmax(
                net.model(eval_ids[b:b + 1])[0, t].float(), -1)
    base_ce = float(np.mean([-float(base_lp[(b, t)][eval_ids[b, t + 1]])
                             for b, t in samp]))
    log(f"{len(samp)} eval tokens, base CE {base_ce:.4f}")
    looks_by_type = {t: type_lookup(units, (t,), net.W0, dev, strict=False)
                     for t in TYPES}
    bk = sidx = swgt = owned = None
    if args.variant == "bank" or args.variant.startswith("hybrid:"):
        bk = torch.load(S.RUN / f"bank_C{args.C}_{args.sensor}{args.bank_tag}.pt",
                        map_location="cpu", weights_only=False)
        sidx = {p: bk["sidx"][p].to(dev).long() for p in MODULES}
        swgt = {p: bk["swgt"][p].to(dev) for p in MODULES}
        # owner slots are sorted by share; --bank_topS keeps only the S largest
        # (S=1: every entry belongs to its single strongest cluster, a partition)
        owned = {p: (swgt[p] > 0) & (torch.arange(swgt[p].shape[0], device=dev)[:, None, None]
                                     < args.bank_topS) for p in MODULES}
    # per token: the full-weight attribution is 113 MB, so it is reduced to
    # per-unit sums (and the bank's share-weighted scores) and freed at once
    a_cache = torch.zeros(len(samp), n_all, dtype=torch.float64, device=dev)
    score_attr_bank = (torch.zeros(int(bk["C"]), len(samp), device=dev,
                                   dtype=torch.float64) if bk and args.variant == "bank" else None)
    for j, (b, t) in enumerate(samp):
        AW = token_attr(net, cfg, eval_ids[b], t)
        a_cache[j] = unit_attr(AW, units, looks_by_type)
        if score_attr_bank is not None:
            with torch.no_grad():
                for p in MODULES:
                    score_attr_bank[:, j] += torch.bincount(
                        sidx[p].reshape(-1), minlength=int(bk["C"]),
                        weights=(swgt[p].float() * AW[p][None]).reshape(-1).double())
        del AW
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    if args.precision == "fp16":
        # attribution needed fp32 gradients; everything after is forward-only
        net.half()
        with torch.no_grad():
            fl = []
            for b, t in samp:
                lp = F.log_softmax(net.model(eval_ids[b:b + 1])[0, t].float(), -1)
                fl.append(kl_of(lp, base_lp[(b, t)]))
                base_lp[(b, t)] = lp
        log(f"fp16 model: KL to the fp32 model on the eval tokens mean "
            f"{np.mean(fl):.2e}, max {np.max(fl):.2e}; base re-taken in fp16")

    @torch.no_grad()
    def kl_all_eval():
        """KL at the sample positions under the current weights."""
        out = torch.zeros(len(samp), device=dev)
        for c0 in range(0, B_eval, args.chunk):
            lp = F.log_softmax(net.model(eval_ids[c0:c0 + args.chunk]).float(), -1)
            for j, (b, t) in enumerate(samp):
                if c0 <= b < c0 + args.chunk:
                    out[j] = kl_of(lp[b - c0, t], base_lp[(b, t)])
            del lp
        return out

    @torch.no_grad()
    def kl_token(j):
        b, t = samp[j]
        lp = F.log_softmax(net.model(eval_ids[b:b + 1])[0, t].float(), -1)
        return kl_of(lp, base_lp[(b, t)])

    res = {"format": "unitabl_eval_v1", "variant": args.variant,
           "n_samples": len(samp), "base_token_ce": round(base_ce, 5),
           "eval": "stream67 phase_bank selection", "results": {}}

    if args.variant.startswith("hybrid:"):
        # measured ownership says which units a cluster owns (own file); the
        # per-entry attribution bank says which of those units' weights are the
        # cluster's slice.  A unit whose slice for that cluster is empty (the
        # cluster is not among the entry's top owners anywhere) stays whole.
        own = torch.load(out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{args.variant[7:]}.pt",
                         map_location="cpu", weights_only=False)
        grouping = own["grouping"]
        C = int(own["C"])
        bk = torch.load(S.RUN / f"bank_C{args.C}_{args.sensor}{args.bank_tag}.pt",
                        map_location="cpu", weights_only=False)
        sidx = {p: bk["sidx"][p].to(dev).long() for p in MODULES}
        swgt = {p: bk["swgt"][p].to(dev) for p in MODULES}
        owned = {p: (swgt[p] > 0) & (torch.arange(swgt[p].shape[0], device=dev)[:, None, None]
                                     < args.bank_topS) for p in MODULES}
        comp_units = ([m.to(dev) for m in own["members"]]
                      + [c.to(dev) for c in own["pool"]] + [own["inert"].to(dev)])
        n_comp = len(comp_units)
        # Every weight must belong to some component (faithfulness).  For each
        # unit: its owning components (measured) each take the weights the
        # attribution bank assigns to that cluster; the weights of the unit
        # that none of its owners claim go to the owner with the largest
        # slice; a unit whose owners claim nothing stays whole with that
        # owner.  Pool / inert components (index >= C) always take the whole
        # unit.
        t0 = time.perf_counter()
        unit_owners = {}
        for c in range(n_comp):
            for g in comp_units[c].tolist():
                unit_owners.setdefault(g, []).append(c)

        def slice_of(c, g):
            """bool masks (per slice of unit g) of the entries the bank gives cluster c"""
            out = []
            for p, axis, lo, hi in units[g][3]:
                if c >= C:
                    out.append(None)
                elif axis == 0:
                    out.append(((sidx[p][:, lo:hi, :] == c) & owned[p][:, lo:hi, :]).any(0))
                else:
                    out.append(((sidx[p][:, :, lo:hi] == c) & owned[p][:, :, lo:hi]).any(0))
            return out

        extra = {}          # (c, g) -> list of residual masks (per slice) that c also takes
        stats = {"units": 0, "sliced": 0, "entries_slice": 0, "entries_residual": 0, "entries_whole": 0}
        for g, owners in unit_owners.items():
            sl = {c: slice_of(c, g) for c in owners}
            sizes = {c: sum(int(m.sum()) for m in sl[c] if m is not None) for c in owners}
            stats["units"] += len(owners)
            whole_n = sum((hi - lo) * (net.W0[p].shape[1] if axis == 0 else net.W0[p].shape[0])
                          for p, axis, lo, hi in units[g][3])
            if any(c >= C for c in owners) or max(sizes.values()) == 0:
                # a pool/inert owner, or no slice at all: the whole unit goes to every owner that is
                # pool/inert, else to the first owner; nothing sliced
                primary = next((c for c in owners if c >= C), owners[0])
                extra[(primary, g)] = "whole"
                stats["entries_whole"] += whole_n
                continue
            primary = max(owners, key=lambda c: sizes[c])
            covered = [torch.zeros_like(m) for m in sl[primary]]
            for c in owners:
                for i, m in enumerate(sl[c]):
                    covered[i] |= m
            residual = [~m for m in covered]
            extra[(primary, g)] = residual
            stats["sliced"] += sum(1 for c in owners if sizes[c] > 0)
            stats["entries_slice"] += sum(sizes.values())
            stats["entries_residual"] += sum(int(m.sum()) for m in residual)

        # sparse index lists (CSR over the concatenated weights) per component:
        # building masks per evaluation from 28M-entry dense tensors is too slow
        off, n_flat = {}, 0
        for p in MODULES:
            off[p] = n_flat
            n_flat += net.W0[p].numel()
        grids = {}
        for p in MODULES:
            d_out, d_in = net.W0[p].shape
            grids[p] = torch.arange(d_out * d_in, device=dev).view(d_out, d_in) + off[p]

        def comp_indices(c):
            parts = []
            for g in comp_units[c].tolist():
                ex = extra.get((c, g))
                sl = slice_of(c, g)
                for i, (p, axis, lo, hi) in enumerate(units[g][3]):
                    blk = grids[p][lo:hi, :] if axis == 0 else grids[p][:, lo:hi]
                    if ex == "whole":
                        parts.append(blk.reshape(-1))
                        continue
                    m = sl[i] if sl[i] is not None else torch.zeros_like(blk, dtype=torch.bool)
                    if ex is not None:
                        m = m | ex[i]
                    parts.append(blk[m])
            return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long, device=dev)

        t1 = time.perf_counter()
        idx_list = [comp_indices(c) for c in range(n_comp)]
        ptr = torch.tensor([0] + [int(v.numel()) for v in idx_list]).cumsum(0)
        idx_all = torch.cat(idx_list).to(torch.int32)
        del idx_list
        owner_of_entry = torch.repeat_interleave(torch.arange(n_comp, device=dev),
                                                 (ptr[1:] - ptr[:-1]).to(dev))
        log(f"hybrid: {int(ptr[-1])} weight memberships over {n_flat} weights "
            f"({int(ptr[-1]) / n_flat:.2f}x) in {time.perf_counter() - t1:.0f}s")

        def flat_to_masks(flat):
            return {p: flat[off[p]:off[p] + net.W0[p].numel()].view(net.W0[p].shape) for p in MODULES}

        def comp_entry_mask(c):
            flat = torch.zeros(n_flat, dtype=torch.bool, device=dev)
            flat[idx_all[ptr[c]:ptr[c + 1]].long()] = True
            return flat_to_masks(flat)

        ent_count = (ptr[1:] - ptr[:-1]).float().to(dev)
        tot = stats["entries_slice"] + stats["entries_residual"] + stats["entries_whole"]
        log(f"hybrid: {stats['units']} unit memberships, {stats['sliced']} sliced; of the kept "
            f"weights {stats['entries_slice'] / max(1, tot):.0%} are cluster slices, "
            f"{stats['entries_residual'] / max(1, tot):.0%} residual (to the largest owner), "
            f"{stats['entries_whole'] / max(1, tot):.0%} whole units ({time.perf_counter() - t0:.0f}s)")
        res.update({"grouping": grouping, "C": C, "n_components": n_comp, "hybrid": True,
                    "bank_tag": args.bank_tag, "bank_topS": args.bank_topS, "slice_stats": stats,
                    "live_diff_components": int(sum(1 for m in own["members"] if m.numel() > 0))})

        def comp_masks(kept):
            """entry mask over the union of kept components' weights"""
            flat = torch.zeros(n_flat, dtype=torch.bool, device=dev)
            flat[idx_all[kept[owner_of_entry]].long()] = True
            return flat_to_masks(flat)

        looks = unit_lookup(units, grouping, net.W0, dev)
        CM = torch.zeros(n_comp, n_all, dtype=torch.bool, device=dev)
        flat_c, flat_u = [], []
        for c, m in enumerate(comp_units):
            if m.numel():
                CM[c, m] = True
                flat_c.append(torch.full((m.numel(),), c, device=dev))
                flat_u.append(m)
        flat_c, flat_u = torch.cat(flat_c), torch.cat(flat_u)
        score_attr = torch.zeros(n_comp, len(samp), device=dev, dtype=torch.float64)
        for j in range(len(samp)):
            score_attr[:, j].index_add_(0, flat_c, a_cache[j][flat_u])
        t0 = time.perf_counter()
        score_orc = torch.zeros(n_comp, len(samp), device=dev)
        for c in range(n_comp):
            if comp_units[c].numel() == 0:
                continue
            m = comp_entry_mask(c)
            net.apply_entry_mask({p: ~m[p] for p in MODULES})
            score_orc[c] = kl_all_eval()
            if (c + 1) % 1024 == 0:
                log(f"oracle {c + 1}/{n_comp} ({time.perf_counter() - t0:.0f}s)")
        net.restore()
        run_curves(args, net, samp, kl_token,
                   {"attr": score_attr.float(), "oracle": score_orc},
                   [("all", None)], n_comp, res, dev, entry_level=True,
                   comp_masks=comp_masks)
        res["component_entries_quantiles"] = qdict(ent_count[ent_count > 0].cpu().numpy())
        res["score_summary"] = score_summary(score_attr.float(), score_orc, samp)
        if args.sel_comps > 0:
            selectivity_hybrid(args, net, comp_entry_mask, comp_units, C, res, dev)
    elif args.variant == "bank":
        C = int(bk["C"])
        res.update({"bank": bk.get("own_mode"), "qk_joint": bk.get("qk_joint"),
                    "C": C, "bank_topS": args.bank_topS, "bank_tag": args.bank_tag})
        ent_count = torch.zeros(C, device=dev)
        for p in MODULES:
            for s in range(sidx[p].shape[0]):
                ok = owned[p][s]
                ent_count.index_add_(0, sidx[p][s][ok],
                                     torch.ones(int(ok.sum()), device=dev))

        def comp_masks(kept):
            """entry kept iff one of its owners is a kept component"""
            return {p: (kept[sidx[p]] & owned[p]).any(0) for p in MODULES}

        score_attr = score_attr_bank            # the bank's own canonical score
        t0 = time.perf_counter()
        score_orc = torch.zeros(C, len(samp), device=dev)
        for c in range(C):
            # ablating component c removes every entry it owns, co-owned or
            # not -- the drop arm's semantics (keeping "all but c" would leave
            # co-owned entries in place and measure exclusive ownership only)
            net.apply_entry_mask({p: ~((sidx[p] == c) & owned[p]).any(0)
                                  for p in MODULES})
            score_orc[c] = kl_all_eval()
            if (c + 1) % 1024 == 0:
                log(f"oracle {c + 1}/{C} ({time.perf_counter() - t0:.0f}s)")
        net.restore()
        run_curves(args, net, samp, kl_token,
                   {"attr": score_attr.float(), "oracle": score_orc},
                   [("all", None)], C, res, dev, entry_level=True,
                   comp_masks=comp_masks)
        res["component_entries_quantiles"] = qdict(ent_count.cpu().numpy())
        res["score_summary"] = score_summary(score_attr.float(), score_orc, samp)
        if args.sel_comps > 0:
            selectivity_bank(args, net, sidx, owned, ent_count, C, res, dev)
    else:
        own = torch.load(out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{args.variant}.pt",
                         map_location="cpu", weights_only=False)
        grouping = own["grouping"]
        C = int(own["C"])
        look = unit_lookup(units, grouping, net.W0, dev)
        sub = own["unit_ids"].to(dev)
        comp_units = ([m.to(dev) for m in own["members"]]
                      + [c.to(dev) for c in own["pool"]] + [own["inert"].to(dev)])
        comp_kind = ["diff"] * C + ["pool"] * len(own["pool"]) + ["inert"]
        n_comp = len(comp_units)
        res.update({"grouping": grouping, "C": C, "n_components": n_comp,
                    "n_pool_components": len(own["pool"]),
                    "live_diff_components": int(sum(1 for m in own["members"]
                                                    if m.numel() > 0)),
                    "pool_units": int(sum(c.numel() for c in own["pool"])),
                    "inert_units": int(own["inert"].numel()),
                    "n_units": int(sub.numel())})
        CM = torch.zeros(n_comp, n_all, dtype=torch.bool, device=dev)
        flat_c, flat_u = [], []
        for c, m in enumerate(comp_units):
            if m.numel():
                CM[c, m] = True
                flat_c.append(torch.full((m.numel(),), c, device=dev))
                flat_u.append(m)
        flat_c, flat_u = torch.cat(flat_c), torch.cat(flat_u)
        score_attr = torch.zeros(n_comp, len(samp), device=dev, dtype=torch.float64)
        for j in range(len(samp)):
            score_attr[:, j].index_add_(0, flat_c, a_cache[j][flat_u])
        t0 = time.perf_counter()
        score_orc = torch.zeros(n_comp, len(samp), device=dev)
        for c in range(n_comp):
            if comp_units[c].numel() == 0:
                continue
            keep = torch.ones(n_all, dtype=torch.bool, device=dev)
            keep[comp_units[c]] = False
            net.apply_unit_mask(keep, look)
            score_orc[c] = kl_all_eval()
            if (c + 1) % 1024 == 0:
                log(f"oracle {c + 1}/{n_comp} ({time.perf_counter() - t0:.0f}s)")
        net.restore()
        always = torch.zeros(n_all, dtype=torch.bool, device=dev)
        for c in range(C, n_comp):
            always[comp_units[c]] = True
        # the poolon arm (pool + inert always on) only when there are pool components
        modes = [("all", None)] + ([("poolon", always)] if len(own["pool"]) > 0 else [])
        run_curves(args, net, samp, kl_token,
                   {"attr": score_attr.float(), "oracle": score_orc},
                   modes, n_comp, res, dev,
                   entry_level=False, unit_cm=(CM, look, sub, mass), n_diff=C)
        res["score_summary"] = score_summary(score_attr.float(), score_orc, samp)
        res["pool_scores"] = {
            "oracle_mean_kl": [round(float(score_orc[c].mean()), 5)
                               for c in range(C, n_comp)],
            "sizes": [int(comp_units[c].numel()) for c in range(C, n_comp)],
            "kind": comp_kind[C:]}
        res["component_sizes"] = [int(m.numel()) for m in comp_units[:C]]
        if args.sel_comps > 0 and "group_label" not in own:
            # selectivity needs components indexed by token cluster; unit
            # groups (column clustering) are not, so it is skipped for them
            selectivity(args, net, look, comp_units, comp_kind, own, units, res, dev)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {args.out}")


# ------------------------------------------------------------------- diag ---

def phase_diag(args, dev):
    """Per token: first-order unit attribution vs the measured effect."""
    lab, parts_kl = load_effects(args, "kl")
    _, parts_dl = load_effects(args, "dl")
    ids_cpu = lab["ids"].long()
    labels = lab["labels"]
    N, T = ids_cpu.shape
    units = build_units()
    n_all = len(units)
    g = torch.Generator().manual_seed(31)
    cand = ((labels >= 0) & (torch.arange(T)[None] >= 32)).nonzero()
    pick = cand[torch.randperm(cand.shape[0], generator=g)[:args.diag_tokens]]
    flat = (pick[:, 0] * T + pick[:, 1]).numpy()
    Ek = np.zeros((n_all, len(flat)), np.float32)
    Ed = np.zeros((n_all, len(flat)), np.float32)
    for g0 in range(0, n_all, args.own_chunk):
        g1 = min(n_all, g0 + args.own_chunk)
        Ek[g0:g1] = read_rows(parts_kl, g0, g1)[:, flat]
        Ed[g0:g1] = read_rows(parts_dl, g0, g1)[:, flat]
    log(f"read measured effects for {len(flat)} tokens")
    cfg = SENSORS[args.sensor]
    net = Net(dev, cfg["mode"])
    looks_by_type = {t: type_lookup(units, (t,), net.W0, dev, strict=False)
                     for t in TYPES}
    types = np.array([u[0] for u in units])
    Ek_t, Ed_t = torch.from_numpy(Ek), torch.from_numpy(Ed)
    per_tok = {"spearman_abs_kl": [], "pearson_neg_dl": [], "top32_overlap": [],
               "top8_overlap": [], "kl_frac_in_attr_top32": []}
    per_type = {t: {"spearman_abs_kl": [], "top32_overlap": []} for t in TYPES}
    scatter = []
    t0 = time.perf_counter()
    for j, (b, t) in enumerate(pick.tolist()):
        AW = token_attr(net, cfg, ids_cpu[b].to(dev), t)
        a = unit_attr(AW, units, looks_by_type).float().cpu()
        kl, dl = Ek_t[:, j], Ed_t[:, j]
        per_tok["spearman_abs_kl"].append(spearman(a.abs(), kl))
        x, y = -a, dl
        per_tok["pearson_neg_dl"].append(float(
            ((x - x.mean()) * (y - y.mean())).mean()
            / (x.std() * y.std()).clamp_min(1e-12)))
        ta, tk = a.abs().topk(32).indices, kl.topk(32).indices
        per_tok["top32_overlap"].append(float(torch.isin(ta, tk).float().mean()))
        per_tok["top8_overlap"].append(float(torch.isin(
            a.abs().topk(8).indices, kl.topk(8).indices).float().mean()))
        per_tok["kl_frac_in_attr_top32"].append(
            float(kl[ta].sum() / kl.sum().clamp_min(1e-12)))
        for ty in TYPES:
            sel = torch.from_numpy(types == ty)
            per_type[ty]["spearman_abs_kl"].append(spearman(a[sel].abs(), kl[sel]))
            kk = min(32, int(sel.sum()))
            per_type[ty]["top32_overlap"].append(float(torch.isin(
                a[sel].abs().topk(kk).indices, kl[sel].topk(kk).indices).float().mean()))
        if j < 64:
            idx = torch.randperm(n_all, generator=g)[:256]
            scatter += [(ty, round(float(a[i]), 6), round(float(dl[i]), 6),
                         round(float(kl[i]), 6))
                        for i, ty in zip(idx.tolist(), types[idx.numpy()].tolist())]
        if (j + 1) % 128 == 0:
            log(f"diag {j + 1}/{len(pick)} ({time.perf_counter() - t0:.0f}s)")
    out = {"format": "unitabl_diag_v1", "n_tokens": len(pick),
           "per_token": {k: [round(float(x), 4) for x in v]
                         for k, v in per_tok.items()},
           "per_token_mean": {k: round(float(np.mean(v)), 4)
                              for k, v in per_tok.items()},
           "per_token_median": {k: round(float(np.median(v)), 4)
                                for k, v in per_tok.items()},
           "per_type": {ty: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
                        for ty, d in per_type.items()},
           "unit_mean_kl_by_type": {ty: round(float(Ek[types == ty].mean()), 6)
                                    for ty in TYPES},
           "scatter": scatter}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}: " + ", ".join(f"{k} {v}" for k, v in
                                          out["per_token_mean"].items()))


# ------------------------------------------------------------------- main ---

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["label", "effects", "own", "eval", "diag"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--sensor", default="ig5")
    ap.add_argument("--fit_tag", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--out", default="out/unitabl.json")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    # label
    ap.add_argument("--lab_seed", type=int, default=999)
    ap.add_argument("--lab_tag", default="", help="corpus tag (effect matrix)")
    ap.add_argument("--clu_tag", default="",
                    help="clustering tag for labels / ownership on that corpus")
    ap.add_argument("--lab_source", choices=["pile", "bin"], default="pile",
                    help="corpus from the stratified pile draw or the token bin")
    ap.add_argument("--lab_offset", type=int, default=80_000_000,
                    help="bin: first token of the corpus (default: past the "
                         "streaming fit's 80M)")
    ap.add_argument("--lab_stride", type=int, default=0,
                    help="bin: tokens between sequence starts (0 = spread "
                         "evenly over the rest of the bin)")
    ap.add_argument("--n_seq", type=int, default=512)
    ap.add_argument("--lab_batch", type=int, default=16)
    # effects
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--q", type=int, default=0, help="corpus slice index")
    ap.add_argument("--nq", type=int, default=1, help="corpus slices (by sequence)")
    ap.add_argument("--precision", choices=["fp32", "fp16", "auto"], default="fp32")
    # own
    ap.add_argument("--own_chunk", type=int, default=512)
    ap.add_argument("--n_min", type=int, default=4,
                    help="tokens a cluster needs before it can own anything")
    ap.add_argument("--taus", type=float, nargs="*", default=[4.0, 6.0])
    ap.add_argument("--topS", type=int, default=8, help="0 disables")
    ap.add_argument("--ratios", type=float, nargs="*", default=[3.0, 10.0],
                    help="ratio rule: cluster mean > r x the unit's corpus mean")
    ap.add_argument("--shares", type=float, nargs="*", default=[0.5, 0.25],
                    help="share rule: smallest cluster set carrying this "
                         "fraction of the unit's effect mass")
    ap.add_argument("--pool_frac", type=float, default=0.1,
                    help="share rule: units whose set spans more than this "
                         "fraction of covered clusters are shared (pool)")
    ap.add_argument("--from_pooled", action="store_true",
                    help="reuse pooled.pt instead of re-reading E")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="minimum centred effect (nats) for a membership")
    ap.add_argument("--pool_floor", type=float, default=1e-4,
                    help="unowned units with mean KL above this are shared "
                         "(pool); below, inert")
    # eval
    ap.add_argument("--variant", default="head_t4",
                    help="<grouping>_<variant> from the own phase, or 'bank'")
    ap.add_argument("--bank_tag", default="_tiedqk")
    ap.add_argument("--bank_topS", type=int, default=8,
                    help="bank variant: owners per entry kept (1 = partition)")
    ap.add_argument("--keep_every", type=int, default=0,
                    help="eval: dense keep/drop grid, every keep_every-th k "
                         "(1 = every single component); 0 = the coarse KEEP_GRID")
    ap.add_argument("--no_random", action="store_true",
                    help="eval: skip the random-ranking arm")
    ap.add_argument("--sel_comps", type=int, default=192)
    ap.add_argument("--sel_min_units", type=int, default=2)
    ap.add_argument("--sel_seqs", type=int, default=256)
    ap.add_argument("--sel_lab_tag", default=None,
                    help="corpus tag for the selectivity test (default: the ownership corpus)")
    # diag
    ap.add_argument("--diag_tokens", type=int, default=768)
    return ap.parse_args(argv)


def main():
    args = parse()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    {"label": phase_label, "effects": phase_effects,
     "own": lambda a, d: phase_own(a), "eval": phase_eval,
     "diag": phase_diag}[args.phase](args, dev)


if __name__ == "__main__":
    main()
