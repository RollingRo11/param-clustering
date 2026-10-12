"""Sub-unit atoms for zcofac67: split each unit's read row along the input
directions its own firing tokens separate into.

A neuron reads the residual stream through one row w of c_fc.  If the
positions where it fires fall into a few groups with different inputs, the
row is split into parts w = sum_g w_g, one per group, each chosen to be
SILENT on the other groups' inputs and on random quiet positions while the
parts still sum to w exactly:

    min  sum_g || X_{not g} w_g ||^2     s.t.  sum_g w_g = w

(X_{not g}: inputs of the other groups plus a quiet sample; closed form).
A split is kept only if every part's leakage energy, silent set over own
group, is below --leak_max; otherwise the unit stays one atom.  The
down_proj column is a separate write atom (used whenever the neuron fires,
whichever part fired it).  OV dims are treated the same way on v_proj rows
with the o_proj column as the write atom.  QK heads stay whole.

Atoms: mlp read parts, mlp write columns, qk heads, ov read parts, ov write
columns; they sum to the weights exactly.  Attribution of a read part at a
position is grad(pre-activation) x (w_g . x); of a write atom grad x
activation at that coordinate; QK as in zcofac67.

Phases
  split    stream inputs, reservoir-sample firing positions per unit,
           cluster, solve, write split{tag}.pt
  stream   fused GIM sensor + online fit over the token bin (zcofac67's
           algorithm, split atoms), writes fit files + own{tag}_split_z{C}.pt
  eval     keep / drop curves on unitabl67's 48 tokens with attribution,
           oracle and random rankings; masks rebuild rows from kept parts
  report   components with atoms and top events

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k \
    python -u zsplit67.py split  --tag _100m --lab_source bin --n_seq 2048
    python -u zsplit67.py stream --tag _100m --lab_source bin --n_seq 208800 --C 2048 4096 8192
    python -u zsplit67.py eval   --tag _100m --C 2048
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stream67 as S                      # sets CKPT / pile cache, TF32
import sensor_study67 as S67
from sensor_study67 import MODULES
import unitabl67 as U
import zcofac67 as Zc
from zcofac67 import (Corpus, make_net, reward_all_positions, reward_at_positions,
                      build_D, sparse_code, Accum, TopEvents, ownership_step,
                      rms_stats, kmeans_init, META_KEYS, TOP_EVENTS,
                      N_LAYER, N_HEAD, D_MODEL, D_MLP)

HD = D_MODEL // N_HEAD
KINDS = ("mr", "mw", "qk", "or", "ow")   # mlp read part, mlp write, qk head, ov read part, ov write


def log(m):
    print(f"[zsplit] {m}", flush=True)


def zdir(args) -> Path:
    d = Path(args.z_dir) if args.z_dir else S.RUN.parent / "zcofac"
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_path(args) -> Path:
    return zdir(args) / f"split{args.tag}.pt"


def fit_path(args, C) -> Path:
    return zdir(args) / f"fitsplit{args.tag}_{args.sensor}_{args.scalar}_C{C}.pt"


def own_path(args, C) -> Path:
    return U.out_dir(args) / f"own{args.tag}_split_z{C}.pt"


# ------------------------------------------------------------------ split ---

def reservoir_add(store, count, X, pairs_u, pairs_p, M, gen):
    """Algorithm-R over (unit, position) firing pairs, vectorised.
    store [U, M, d] fp16, count [U] int64 (firings seen), X [n, d] inputs."""
    if pairs_u.numel() == 0:
        return
    order = pairs_u.argsort()
    u, p = pairs_u[order], pairs_p[order]
    per = torch.bincount(u, minlength=store.shape[0])
    starts = per.cumsum(0) - per
    rank = torch.arange(u.numel(), device=u.device) - starts[u]
    gidx = count[u] + rank                                  # index among the unit's firings
    r = torch.rand(u.numel(), generator=gen, device=u.device)
    accept = (gidx < M) | (r < M / gidx.clamp_min(1).float())
    slot = torch.where(gidx < M, gidx,
                       (torch.rand(u.numel(), generator=gen, device=u.device) * M).long())
    u, p, slot = u[accept], p[accept], slot[accept]
    store[u, slot] = X[p].half()
    count += per


def batched_kmeans(Xw, valid, G, iters, gen):
    """Xw [U, M, d] whitened inputs, valid [U, M] bool; returns labels [U, M]
    (garbage for invalid rows) and centroids [U, G, d]."""
    Ub, M, d = Xw.shape
    # init: G random valid points per unit
    r = torch.rand(Ub, M, generator=gen, device=Xw.device)
    r[~valid] = 2.0
    init = r.argsort(1)[:, :G]                                  # [U, G]
    cent = torch.gather(Xw, 1, init[..., None].expand(-1, -1, d)).clone()
    for _ in range(iters):
        d2 = (Xw[:, :, None, :] - cent[:, None, :, :]).pow(2).sum(-1)   # [U, M, G]
        lab = d2.argmin(-1)
        onehot = F.one_hot(lab, G).float() * valid[..., None].float()   # [U, M, G]
        sums = torch.einsum("umg,umd->ugd", onehot, Xw)
        cnt = onehot.sum(1)                                            # [U, G]
        new = sums / cnt.clamp_min(1)[..., None]
        cent = torch.where(cnt[..., None] > 0, new, cent)
    d2 = (Xw[:, :, None, :] - cent[:, None, :, :]).pow(2).sum(-1)
    return d2.argmin(-1), cent


def solve_parts(w, A, B):
    """w [U, d] rows; A [U, G, d, d] silent scatters per part; B [U, G, d, d]
    own-group scatters.  Parts w_g minimising sum_g w_g^T A_g w_g with
    sum_g w_g = w (part 0 eliminated).  Returns parts [U, G, d], leak [U, G]."""
    Ub, G, d, _ = A.shape
    eye = torch.eye(d, device=w.device)
    ridge = 1e-4 * A.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-8
    A = A + ridge * eye
    if G == 1:
        parts = w[:, None, :]
    else:
        # unknowns v_1..v_{G-1}; block system K v = rhs with
        # K_kk = A_k + A_0, K_kj = A_0 (k != j), rhs_k = A_0 w
        n = (G - 1) * d
        K = A[:, 0:1, :, :].expand(-1, G - 1, -1, -1)                 # A_0 blocks
        K = K[:, :, None, :, :].expand(-1, -1, G - 1, -1, -1).clone()  # [U, G-1, G-1, d, d]
        for k in range(G - 1):
            K[:, k, k] += A[:, k + 1]
        K = K.permute(0, 1, 3, 2, 4).reshape(Ub, n, n)
        rhs = torch.einsum("uij,uj->ui", A[:, 0], w)[:, None, :].expand(-1, G - 1, -1).reshape(Ub, n)
        v = torch.linalg.solve(K, rhs).view(Ub, G - 1, d)
        parts = torch.cat([(w - v.sum(1))[:, None, :], v], dim=1)
    silent = torch.einsum("ugd,ugde,uge->ug", parts, A, parts)
    own = torch.einsum("ugd,ugde,uge->ug", parts, B, parts)
    return parts, silent / own.clamp_min(1e-12)


def split_units(store, count, thr_ok, quiet, w_rows, Gmax, args, gen, dev):
    """store [U, M, d] fp16 firing inputs, count [U] firings seen, quiet
    [Nq, d] fp16 quiet inputs of the layer, w_rows [U, d].  Returns parts
    [U, Gmax, d] (zero padded), G [U], leak [U, Gmax]."""
    Ub, M, d = store.shape
    Xq = quiet.float()
    mu = Xq.mean(0)
    cov = (Xq - mu).t() @ (Xq - mu) / Xq.shape[0]
    ev, evec = torch.linalg.eigh(cov + 1e-6 * torch.eye(d, device=dev))
    Wh = evec @ torch.diag(ev.clamp_min(1e-8).rsqrt()) @ evec.t()   # whitening
    Aq = Xq.t() @ Xq                                              # quiet scatter
    parts_out = torch.zeros(Ub, Gmax, d, device=dev)
    G_out = torch.ones(Ub, dtype=torch.long, device=dev)
    leak_out = torch.zeros(Ub, Gmax, device=dev)
    leak2_out = torch.full((Ub,), float("nan"), device=dev)     # G=2 leakage, every unit
    parts_out[:, 0] = w_rows
    nq = Xq.shape[0]
    for u0 in range(0, Ub, args.unit_batch):
        u1 = min(Ub, u0 + args.unit_batch)
        X = store[u0:u1].float()                                   # [u, M, d]
        m = count[u0:u1].clamp(max=M)
        valid = torch.arange(M, device=dev)[None] < m[:, None]
        valid &= thr_ok[u0:u1, None]
        w = w_rows[u0:u1]
        best_parts = w[:, None, :].expand(-1, Gmax, -1).clone() * 0
        best_parts[:, 0] = w
        best_G = torch.ones(u1 - u0, dtype=torch.long, device=dev)
        best_leak = torch.zeros(u1 - u0, Gmax, device=dev)
        Xw = (X - mu) @ Wh
        for G in range(2, Gmax + 1):
            lab, _ = batched_kmeans(Xw, valid, G, args.kmeans_iters, gen)
            onehot = F.one_hot(lab, G).float() * valid[..., None].float()   # [u, M, G]
            sizes = onehot.sum(1)                                           # [u, G]
            # own-group scatter B_g and other-groups scatter; quiet weighted to the firing count
            Bg = torch.einsum("umg,umd,ume->ugde", onehot, X, X)            # [u, G, d, d]
            tot = Bg.sum(1, keepdim=True)
            qw = (args.quiet_w * m.float() / nq)[:, None, None, None]
            Ag = tot - Bg + qw * Aq[None, None]
            parts, leak = solve_parts(w, Ag, Bg)
            if G == 2:
                leak2_out[u0:u1] = torch.where(m >= 2 * args.min_group, leak.max(1).values,
                                               torch.full_like(leak[:, 0], float("nan")))
            ok = (leak.max(1).values < args.leak_max) & (sizes.min(1).values >= args.min_group) \
                & (m >= 2 * args.min_group)
            upd = ok & (best_G < G)
            best_parts[upd] = 0
            best_parts[upd, :G] = parts[upd]
            best_G[upd] = G
            best_leak[upd] = 0
            best_leak[upd, :G] = leak[upd]
        parts_out[u0:u1] = best_parts
        G_out[u0:u1] = best_G
        leak_out[u0:u1] = best_leak
    return parts_out, G_out, leak_out, leak2_out


def phase_split(args, dev):
    corpus = Corpus(args)
    T, lo = corpus.T, args.min_pos
    N_seq = len(corpus)
    SB = args.seq_per_batch
    net = make_net(args, dev, "plain")
    model = net.model
    M, Gmax = args.reservoir, args.Gmax
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    acts = {}

    def mk(name, take_input):
        def hook(mm, inp, out):
            acts[name] = (inp[0] if take_input else out).detach()
        return hook
    hs = []
    for l in range(N_LAYER):
        hs.append(model.get_submodule(f"h.{l}.mlp.c_fc").register_forward_hook(mk(f"xm{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.mlp.c_fc").register_forward_hook(mk(f"h{l}", False)))
        hs.append(model.get_submodule(f"h.{l}.attn.v_proj").register_forward_hook(mk(f"xa{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.v_proj").register_forward_hook(mk(f"v{l}", False)))
    # pilot: firing thresholds = --fire_q quantile of the activation magnitude
    n_pilot = min(args.pilot_seqs, N_seq)
    pil = {f"a{l}": [] for l in range(N_LAYER)} | {f"v{l}": [] for l in range(N_LAYER)}
    with torch.no_grad():
        for b0 in range(0, n_pilot, SB):
            idx = corpus.rows(torch.arange(b0, min(n_pilot, b0 + SB))).to(dev)
            model(idx)
            for l in range(N_LAYER):
                pil[f"a{l}"].append(F.gelu(acts[f"h{l}"][:, lo:-1], approximate="tanh").abs().reshape(-1, D_MLP))
                pil[f"v{l}"].append(acts[f"v{l}"][:, lo:-1].abs().reshape(-1, D_MODEL))
    thr = {}
    for k, v in pil.items():
        X = torch.cat(v)
        thr[k] = torch.quantile(X[:min(X.shape[0], 20000)], args.fire_q, dim=0)
    del pil
    log(f"split: {N_seq} seqs, thresholds from {n_pilot} pilot seqs at q={args.fire_q}; "
        f"reservoir {M}/unit, Gmax {Gmax}")
    store = {f"m{l}": torch.zeros(D_MLP, M, D_MODEL, dtype=torch.float16, device=dev) for l in range(N_LAYER)}
    store |= {f"o{l}": torch.zeros(D_MODEL, M, D_MODEL, dtype=torch.float16, device=dev) for l in range(N_LAYER)}
    count = {k: torch.zeros(v.shape[0], dtype=torch.long, device=dev) for k, v in store.items()}
    quiet = {f"m{l}": torch.zeros(args.quiet_n, D_MODEL, dtype=torch.float16, device=dev) for l in range(N_LAYER)}
    quiet |= {f"o{l}": torch.zeros(args.quiet_n, D_MODEL, dtype=torch.float16, device=dev) for l in range(N_LAYER)}
    qcount = {k: torch.zeros(1, dtype=torch.long, device=dev) for k in quiet}
    t0 = time.perf_counter()
    with torch.no_grad():
        for b0 in range(0, N_seq, SB):
            idx = corpus.rows(torch.arange(b0, min(N_seq, b0 + SB))).to(dev)
            model(idx)
            for l in range(N_LAYER):
                xm = acts[f"xm{l}"][:, lo:-1].reshape(-1, D_MODEL)
                a = F.gelu(acts[f"h{l}"][:, lo:-1], approximate="tanh").abs().reshape(-1, D_MLP)
                xa = acts[f"xa{l}"][:, lo:-1].reshape(-1, D_MODEL)
                v = acts[f"v{l}"][:, lo:-1].abs().reshape(-1, D_MODEL)
                for key, act, th, X in ((f"m{l}", a, thr[f"a{l}"], xm), (f"o{l}", v, thr[f"v{l}"], xa)):
                    fire = act > th[None]                                  # [n, U]
                    p, u = fire.nonzero(as_tuple=True)
                    reservoir_add(store[key], count[key], X, u, p, M, gen)
                    # quiet sample: every position is a candidate for the one-unit reservoir
                    allp = torch.arange(X.shape[0], device=dev)
                    reservoir_add(quiet[key][None], qcount[key], X, torch.zeros_like(allp), allp,
                                  args.quiet_n, gen)
            if (b0 // SB) % 32 == 0:
                log(f"seq {b0}/{N_seq} ({time.perf_counter() - t0:.0f}s)")
    for h in hs:
        h.remove()
    out = {"format": "zsplit67_v1", "Gmax": Gmax, "reservoir": M, "fire_q": args.fire_q,
           "leak_max": args.leak_max, "min_group": args.min_group, "quiet_w": args.quiet_w,
           "n_seq": N_seq, "lab_source": args.lab_source, "bin_offset": args.bin_offset,
           "layers": []}
    for l in range(N_LAYER):
        lay = {}
        for kind, key, mod, thk in (("mlp", f"m{l}", f"h.{l}.mlp.c_fc", f"a{l}"),
                                    ("ov", f"o{l}", f"h.{l}.attn.v_proj", f"v{l}")):
            w_rows = model.get_submodule(mod).weight.detach().float()
            enough = count[key] >= 2 * args.min_group
            parts, G, leak, leak2 = split_units(store[key], count[key], enough, quiet[key],
                                                w_rows, Gmax, args, gen, dev)
            err = (parts.sum(1) - w_rows).abs().max().item()
            lay[f"P_{kind}"] = parts.cpu()
            lay[f"G_{kind}"] = G.cpu()
            lay[f"leak_{kind}"] = leak.cpu()
            lay[f"leak2_{kind}"] = leak2.cpu()
            l2 = leak2[~torch.isnan(leak2)]
            if l2.numel():
                log(f"layer {l} {kind}: G=2 leakage quantiles "
                    f"{ {q: round(float(torch.quantile(l2, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} }, "
                    f"frac < 0.3: {float((l2 < 0.3).float().mean()):.3f}, "
                    f"< 0.5: {float((l2 < 0.5).float().mean()):.3f}, < 1: {float((l2 < 1).float().mean()):.3f}")
            lay[f"thr_{kind}"] = thr[thk].cpu()
            lay[f"fired_{kind}"] = count[key].cpu()
            hist = torch.bincount(G, minlength=Gmax + 1).tolist()
            log(f"layer {l} {kind}: G histogram {hist[1:]}, sum-to-row max err {err:.2e}, "
                f"units with < {2 * args.min_group} firings: {int((~enough).sum())} "
                f"({time.perf_counter() - t0:.0f}s)")
        out["layers"].append(lay)
    torch.save(out, split_path(args))
    log(f"wrote {split_path(args)}")


# ------------------------------------------------------------------ atoms ---

class SplitAtoms:
    """Atom table over a split file plus the dense->active index for
    attribution and the per-layer structures for masking."""

    def __init__(self, split, dev):
        self.Gmax = int(split["Gmax"])
        self.dev = dev
        self.P_mlp = [lay["P_mlp"].to(dev) for lay in split["layers"]]     # [3072, Gmax, 768]
        self.P_ov = [lay["P_ov"].to(dev) for lay in split["layers"]]       # [768, Gmax, 768]
        self.G_mlp = [lay["G_mlp"].to(dev) for lay in split["layers"]]
        self.G_ov = [lay["G_ov"].to(dev) for lay in split["layers"]]
        kinds, layers, units, parts = [], [], [], []
        dense_idx = []                       # index into the per-layer dense attribution layout
        self.dense_per_layer = D_MLP * self.Gmax + D_MLP + N_HEAD + D_MODEL * self.Gmax + D_MODEL
        for l in range(N_LAYER):
            base = l * self.dense_per_layer
            Gm, Go = self.G_mlp[l].cpu(), self.G_ov[l].cpu()
            for n in range(D_MLP):
                for g in range(int(Gm[n])):
                    kinds.append(0); layers.append(l); units.append(n); parts.append(g)
                    dense_idx.append(base + n * self.Gmax + g)
            for n in range(D_MLP):
                kinds.append(1); layers.append(l); units.append(n); parts.append(0)
                dense_idx.append(base + D_MLP * self.Gmax + n)
            for h in range(N_HEAD):
                kinds.append(2); layers.append(l); units.append(h); parts.append(0)
                dense_idx.append(base + D_MLP * self.Gmax + D_MLP + h)
            for d in range(D_MODEL):
                for g in range(int(Go[d])):
                    kinds.append(3); layers.append(l); units.append(d); parts.append(g)
                    dense_idx.append(base + D_MLP * self.Gmax + D_MLP + N_HEAD + d * self.Gmax + g)
            for d in range(D_MODEL):
                kinds.append(4); layers.append(l); units.append(d); parts.append(0)
                dense_idx.append(base + D_MLP * self.Gmax + D_MLP + N_HEAD + D_MODEL * self.Gmax + d)
        self.kind = torch.tensor(kinds, device=dev)
        self.layer = torch.tensor(layers, device=dev)
        self.unit = torch.tensor(units, device=dev)
        self.part = torch.tensor(parts, device=dev)
        self.dense_idx = torch.tensor(dense_idx, device=dev)
        self.J = len(kinds)

    def describe(self, q):
        k, l, u, g = int(self.kind[q]), int(self.layer[q]), int(self.unit[q]), int(self.part[q])
        return f"L{l}:{KINDS[k]}{u}" + (f".{g}" if k in (0, 3) else "")

    def mass(self, net):
        """Squared weight mass per atom."""
        m = torch.zeros(self.J, device=self.dev)
        for l in range(N_LAYER):
            sel = self.layer == l
            for k, val in ((0, self.P_mlp[l].pow(2).sum(-1)),            # [3072, Gmax]
                           (3, self.P_ov[l].pow(2).sum(-1))):
                s = sel & (self.kind == k)
                m[s] = val[self.unit[s], self.part[s]]
            m[sel & (self.kind == 1)] = net.W0[f"h.{l}.mlp.down_proj"].pow(2).sum(0)[self.unit[sel & (self.kind == 1)]]
            m[sel & (self.kind == 4)] = net.W0[f"h.{l}.attn.o_proj"].pow(2).sum(0)[self.unit[sel & (self.kind == 4)]]
            qk = sel & (self.kind == 2)
            hm = (net.W0[f"h.{l}.attn.q_proj"].pow(2).sum(1) + net.W0[f"h.{l}.attn.k_proj"].pow(2).sum(1)
                  ).view(N_HEAD, HD).sum(1)
            m[qk] = hm[self.unit[qk]]
        return m

    @torch.no_grad()
    def apply_keep(self, net, keep):
        """keep: bool [J] -> module weights rebuilt from the kept atoms."""
        for l in range(N_LAYER):
            sel = self.layer == l
            km = torch.zeros(D_MLP, self.Gmax, device=self.dev)
            s = sel & (self.kind == 0)
            km[self.unit[s], self.part[s]] = keep[s].float()
            net.Wt[f"h.{l}.mlp.c_fc"].copy_(torch.einsum("ng,ngd->nd", km, self.P_mlp[l]))
            s = sel & (self.kind == 1)
            kw = torch.zeros(D_MLP, device=self.dev)
            kw[self.unit[s]] = keep[s].float()
            net.Wt[f"h.{l}.mlp.down_proj"].copy_(net.W0[f"h.{l}.mlp.down_proj"] * kw[None, :])
            s = sel & (self.kind == 2)
            kq = torch.zeros(N_HEAD, device=self.dev)
            kq[self.unit[s]] = keep[s].float()
            rowm = kq.repeat_interleave(HD)
            net.Wt[f"h.{l}.attn.q_proj"].copy_(net.W0[f"h.{l}.attn.q_proj"] * rowm[:, None])
            net.Wt[f"h.{l}.attn.k_proj"].copy_(net.W0[f"h.{l}.attn.k_proj"] * rowm[:, None])
            ko = torch.zeros(D_MODEL, self.Gmax, device=self.dev)
            s = sel & (self.kind == 3)
            ko[self.unit[s], self.part[s]] = keep[s].float()
            net.Wt[f"h.{l}.attn.v_proj"].copy_(torch.einsum("ng,ngd->nd", ko, self.P_ov[l]))
            s = sel & (self.kind == 4)
            kw = torch.zeros(D_MODEL, device=self.dev)
            kw[self.unit[s]] = keep[s].float()
            net.Wt[f"h.{l}.attn.o_proj"].copy_(net.W0[f"h.{l}.attn.o_proj"] * kw[None, :])


def atom_attr_split(net, atoms, idx, reward_fn):
    """Per-position attribution A [B, T, J] over split atoms (first order, GIM)."""
    model = net.model
    acts, hs = {}, []

    def mk(name, take_input):
        def hook(mm, inp, out):
            acts[name] = inp[0] if take_input else out
        return hook
    for l in range(N_LAYER):
        hs.append(model.get_submodule(f"h.{l}.mlp.c_fc").register_forward_hook(mk(f"xm{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.mlp.c_fc").register_forward_hook(mk(f"h{l}", False)))
        hs.append(model.get_submodule(f"h.{l}.mlp.down_proj").register_forward_hook(mk(f"a{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.v_proj").register_forward_hook(mk(f"xa{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.v_proj").register_forward_hook(mk(f"v{l}", False)))
        hs.append(model.get_submodule(f"h.{l}.attn.o_proj").register_forward_hook(mk(f"o{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.q_proj").register_forward_hook(mk(f"q{l}", False)))
        hs.append(model.get_submodule(f"h.{l}.attn.k_proj").register_forward_hook(mk(f"k{l}", False)))
    lg = model(idx)
    for h in hs:
        h.remove()
    reward = reward_fn(lg)
    names = [f"{k}{l}" for l in range(N_LAYER) for k in ("h", "a", "v", "o", "q", "k")]
    grads = torch.autograd.grad(reward, [acts[n] for n in names])
    B, T = idx.shape
    Gm = atoms.Gmax
    with torch.no_grad():
        dense = []
        for l in range(N_LAYER):
            gh, ga, gv, go, gq, gk = [g.float() for g in grads[6 * l:6 * l + 6]]
            xm, xa = acts[f"xm{l}"].float(), acts[f"xa{l}"].float()
            # read parts: (x . w_g) * grad at the pre-activation
            pm = torch.einsum("btd,ngd->btng", xm, atoms.P_mlp[l]) * gh[..., None]   # [B,T,3072,Gm]
            po = torch.einsum("btd,ngd->btng", xa, atoms.P_ov[l]) * gv[..., None]    # [B,T,768,Gm]
            mw = ga * acts[f"a{l}"].float()
            ow = go * acts[f"o{l}"].float()
            qk = ((gq * acts[f"q{l}"].float()).view(B, T, N_HEAD, HD).sum(-1)
                  + (gk * acts[f"k{l}"].float()).view(B, T, N_HEAD, HD).sum(-1))
            dense += [pm.reshape(B, T, -1), mw, qk, po.reshape(B, T, -1), ow]
        A = torch.cat(dense, dim=-1)[..., atoms.dense_idx]
    del grads, acts, lg
    return A


# ----------------------------------------------------------------- stream ---

def finish_C(args, C, own, p, acc, top, rms, live, dead, atoms, hist, N, src_info, t0):
    J, Jl = rms.numel(), live.numel()
    sizes = torch.bincount(own, minlength=C)
    usage_freq = acc.nz / max(acc.n, 1)
    own_full = torch.full((J,), -1, dtype=torch.long)
    own_full[live.cpu()] = own.cpu()
    p_full = torch.zeros(J)
    p_full[live.cpu()] = p.cpu()
    members = [torch.nonzero(own_full == c).flatten() for c in range(C)]
    inert = torch.nonzero(dead.cpu()).flatten()
    kind = atoms.kind.cpu().numpy()
    layer = atoms.layer.cpu().numpy()
    mixed = sum(1 for c in range(C) if sizes[c] > 1 and len(set(kind[(own_full == c).numpy()])) > 1)
    multilayer = sum(1 for c in range(C) if sizes[c] > 1 and len(set(layer[(own_full == c).numpy()])) > 1)
    aq = acc.act_quantiles()
    fit = {"format": "zsplit67_fit_v1", "C": C, "N": N, "J": J, "Jl": Jl,
           "sensor": args.sensor, "scalar": args.scalar, "tag": args.tag, **src_info,
           "own": own_full, "p": p_full, "live": live.cpu(), "dead": dead.cpu(),
           "rms": rms.cpu(), "sizes": sizes.cpu(), "usage_freq": usage_freq.cpu(),
           "usage_pos_frac": (acc.npos / acc.nz.clamp_min(1)).cpu(),
           "mean_abs_coef": (acc.asum / acc.nz.clamp_min(1)).cpu(),
           "active_hist": acc.act_hist.cpu(), "top_events": top.state(),
           "atom_kind": atoms.kind.cpu(), "atom_layer": atoms.layer.cpu(),
           "atom_unit": atoms.unit.cpu(), "atom_part": atoms.part.cpu(),
           "hist": hist, "seed": args.seed}
    torch.save(fit, fit_path(args, C))
    torch.save({"format": "zsplit67_own_v1", "variant": f"split_z{C}", "C": C,
                "members": members, "inert": inert, "own": own_full, "coef": p_full,
                "split_file": str(split_path(args)), "tag": args.tag}, own_path(args, C))
    summ = {"C": C, "N": N, "J": J, "dead": int(dead.sum()),
            "explained_code": round(acc.expl / max(acc.n, 1), 5), "iters": len(hist),
            "active_per_event": {"median": aq[0.5], "p90": aq[0.9]},
            "sizes": U.qdict(sizes[sizes > 0].cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "largest": int(sizes.max()), "empty": int((sizes == 0).sum()),
            "singletons": int((sizes == 1).sum()),
            "usage_freq": U.qdict(usage_freq.cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "components_mixing_kinds": mixed, "components_spanning_layers": multilayer,
            "seconds": round(time.perf_counter() - t0, 1)}
    log(f"C={C} done: sizes median {int(sizes.median())} max {int(sizes.max())} "
        f"singletons {summ['singletons']}, usage freq median {usage_freq.median():.4f}, "
        f"{mixed} mix kinds, {multilayer} span layers -> {own_path(args, C).name}")
    outj = Path(args.out).with_name(Path(args.out).stem + f"_C{C}.json")
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps({**summ, "hist": hist}, indent=1))


def phase_stream(args, dev):
    cfg = S67.SENSORS[args.sensor]
    assert cfg["K"] == 1
    split = torch.load(split_path(args), map_location="cpu", weights_only=False)
    corpus = Corpus(args)
    T, lo = corpus.T, args.min_pos
    n_pos = T - 1 - lo
    N_seq = len(corpus)
    SB = args.seq_per_batch
    n_batches = math.ceil(N_seq / SB)
    net = make_net(args, dev, cfg["mode"])
    atoms = SplitAtoms(split, dev)
    J = atoms.J
    Cs = list(args.C)
    ck = zdir(args) / f"streamsplit{args.tag}_{args.sensor}_{args.scalar}_ckpt.pt"
    log(f"stream: {N_seq} seqs x {n_pos} positions in {n_batches} batches; J={J} split atoms; "
        f"C={Cs}; chunk {args.chunk_events}, decay {args.decay}")
    st = {"b": 0, "phase": "rms", "sq": torch.zeros(J, dtype=torch.float64, device=dev), "n_rms": 0,
          "Y": None, "n_init": 0, "rms": None, "dead": None, "live": None, "scale": None,
          "colsq": None, "own": {}, "p": {}, "acc": {}, "top": {}, "hist": {c: [] for c in Cs},
          "chunk_n": 0, "n_events": 0}
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        log(f"resuming from {ck.name}: batch {st['b']}/{n_batches}, phase {st['phase']}")
        for key in ("sq", "Y", "rms", "dead", "live", "scale", "colsq"):
            if st[key] is not None:
                st[key] = st[key].to(dev)
        for c in Cs:
            if c in st["own"]:
                st["own"][c], st["p"][c] = st["own"][c].to(dev), st["p"][c].to(dev)
                acc = Accum(c, st["scale"].numel(), dev)
                for k, v in st["acc"][c].items():
                    setattr(acc, k, v.to(dev) if torch.is_tensor(v) else v)
                st["acc"][c] = acc
                top = TopEvents(c, TOP_EVENTS, dev)
                top.load(st["top"][c])
                st["top"][c] = top

    def save_ckpt():
        out = {k: v for k, v in st.items() if k not in ("acc", "top")}
        for key in ("sq", "Y", "rms", "dead", "live", "scale", "colsq"):
            if out[key] is not None:
                out[key] = out[key].cpu()
        out["own"] = {c: t.cpu() for c, t in st["own"].items()}
        out["p"] = {c: t.cpu() for c, t in st["p"].items()}
        out["acc"] = {c: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(a).items()}
                      for c, a in st["acc"].items()}
        out["top"] = {c: t.state() for c, t in st["top"].items()}
        tmp = ck.with_suffix(".tmp")
        torch.save(out, tmp)
        tmp.replace(ck)

    t0 = time.perf_counter()
    D = {c: build_D(st["own"][c], st["p"][c], c) for c in Cs if c in st["own"]}
    for b in range(st["b"], n_batches):
        seq_ids = torch.arange(b * SB, min(N_seq, (b + 1) * SB))
        idx = corpus.rows(seq_ids).to(dev)
        rec = {}

        def reward_fn(lg):
            r, tgt, lp = reward_all_positions(lg, idx, lo, args.scalar)
            rec["tgt"], rec["lp"] = tgt, lp
            return r
        A = atom_attr_split(net, atoms, idx, reward_fn)
        Zraw = A[:, lo:-1].reshape(-1, J)
        n = Zraw.shape[0]
        del A
        if st["phase"] == "rms":
            st["sq"] += Zraw.pow(2).sum(0).double()
            st["n_rms"] += n
            if b + 1 >= args.pilot_batches:
                rms, dead, live, scale = rms_stats(st["sq"], st["n_rms"], args)
                st.update(rms=rms, dead=dead, live=live if live.numel() < J else None,
                          scale=scale, phase="init",
                          Y=torch.zeros(live.numel(), args.proj_dim, device=dev),
                          colsq=torch.zeros(live.numel(), device=dev, dtype=torch.float64))
        elif st["phase"] == "init":
            Zb = Zc.standardize(Zraw, st["live"], st["scale"])
            g = torch.Generator(device=dev).manual_seed(args.seed * 7919 + b)
            R = torch.randn(n, args.proj_dim, generator=g, device=dev)
            st["Y"] += Zb.t() @ R
            st["colsq"] += Zb.pow(2).sum(0).double()
            st["n_init"] += n
            if st["n_init"] >= args.pilot_batches * n:
                Y = st["Y"] / math.sqrt(args.proj_dim)
                for c in Cs:
                    st["own"][c], st["p"][c] = kmeans_init(Y, c, args, dev)
                    st["acc"][c] = Accum(c, Y.shape[0], dev)
                    st["top"][c] = TopEvents(c, TOP_EVENTS, dev)
                    D[c] = build_D(st["own"][c], st["p"][c], c)
                st["Y"] = None
                st["phase"] = "fit"
                log(f"init done after {st['n_init']} events: k-means for C={Cs}")
        else:
            Zb = Zc.standardize(Zraw, st["live"], st["scale"])
            st["colsq"] += Zb.pow(2).sum(0).double()
            meta = {"seq": seq_ids.to(dev)[:, None].expand(-1, n_pos).reshape(-1),
                    "pos": torch.arange(lo, T - 1, device=dev)[None].expand(len(seq_ids), -1).reshape(-1),
                    "pred": rec["tgt"].reshape(-1), "gold": idx[:, lo + 1:].reshape(-1),
                    "lp": rec["lp"].detach().reshape(-1)}
            for c in Cs:
                Ub = sparse_code(Zb, D[c], args.k)
                st["acc"][c].add(Ub, Zb)
                st["top"][c].update(Ub, meta)
            st["chunk_n"] += n
            st["n_events"] += n
            if st["chunk_n"] >= args.chunk_events or b + 1 == n_batches:
                colsq = (st["colsq"] / max(st["n_events"], 1) * st["acc"][Cs[0]].n).float().clamp_min(1e-12)
                for c in Cs:
                    acc = st["acc"][c]
                    own, p, own_expl, changed, n_empty = ownership_step(acc, st["own"][c], st["p"][c], colsq)
                    st["own"][c], st["p"][c] = own, p
                    D[c] = build_D(own, p, c)
                    aq = acc.act_quantiles()
                    tot = max(acc.n, 1)
                    st["hist"][c].append({"batch": b + 1, "events_seen": st["n_events"],
                                          "explained_code": round(acc.expl / tot, 5),
                                          "explained_own": round(own_expl / tot, 5),
                                          "changed": round(changed, 5), "active_median": aq[0.5],
                                          "active_p90": aq[0.9], "empty_reseeded": n_empty})
                    log(f"C={c} chunk @ {st['n_events']} events: explained {acc.expl / tot:.4f} "
                        f"(own step {own_expl / tot:.4f}), owners changed {changed:.4f}, active/event "
                        f"median {aq[0.5]} p90 {aq[0.9]}, reseeded {n_empty} ({time.perf_counter() - t0:.0f}s)")
                    acc.decay(args.decay)
                st["chunk_n"] = 0
        del Zraw
        st["b"] = b + 1
        if (b + 1) % 50 == 0:
            log(f"batch {b + 1}/{n_batches}, phase {st['phase']}, {st['n_events']} fit events "
                f"({time.perf_counter() - t0:.0f}s)")
        if (b + 1) % args.ckpt_every == 0 and b + 1 < n_batches:
            save_ckpt()
            if args.stop_after and b + 1 >= args.stop_after:
                log(f"stop_after {args.stop_after} batches: checkpoint written, exiting")
                return
    src_info = {"lab_source": args.lab_source, "bin_offset": args.bin_offset, "n_seq": N_seq,
                "positions_per_seq": n_pos, "stream": True, "chunk_events": args.chunk_events,
                "decay": args.decay, "split_file": str(split_path(args))}
    live = st["live"] if st["live"] is not None else torch.arange(J, device=dev)
    for c in Cs:
        finish_C(args, c, st["own"][c], st["p"][c], st["acc"][c], st["top"][c], st["rms"], live,
                 st["dead"], atoms, st["hist"][c], st["n_events"], src_info, t0)
    if ck.exists():
        ck.unlink()
    log(f"stream done: {st['n_events']} events ({time.perf_counter() - t0:.0f}s)")


# ------------------------------------------------------------------- eval ---

def phase_eval(args, dev):
    """unitabl67's keep / drop protocol (same 48 tokens, grid, rankings) with
    split-atom masking."""
    cfg = S67.SENSORS[args.sensor]
    split = torch.load(split_path(args), map_location="cpu", weights_only=False)
    net = make_net(args, dev, cfg["mode"])
    atoms = SplitAtoms(split, dev)
    J = atoms.J
    mass = atoms.mass(net)
    # faithfulness of the split: all atoms kept must reproduce the weights
    atoms.apply_keep(net, torch.ones(J, dtype=torch.bool, device=dev))
    ferr = max((net.Wt[p] - net.W0[p]).abs().max().item() for p in MODULES)
    net.restore()
    log(f"{J} split atoms; all-kept reconstruction max |err| {ferr:.2e}")
    if args.smoke:
        g = torch.Generator().manual_seed(1)
        eval_ids = torch.randint(0, 50277, (4, args.seq), generator=g).to(dev)
        samp = [(0, 10), (1, 20), (2, 30), (3, 40)]
    else:
        _, eval_ids, samp = U.load_eval_tokens(args, dev)
    B_eval = eval_ids.shape[0]
    base_lp = {}
    with torch.no_grad():
        for b, t in samp:
            base_lp[(b, t)] = F.log_softmax(net.model(eval_ids[b:b + 1])[0, t].float(), -1)
    base_ce = float(np.mean([-float(base_lp[(b, t)][eval_ids[b, t + 1]]) for b, t in samp]))
    log(f"{len(samp)} eval tokens, base CE {base_ce:.4f}")

    @torch.no_grad()
    def kl_token(j):
        b, t = samp[j]
        lp = F.log_softmax(net.model(eval_ids[b:b + 1])[0, t].float(), -1)
        return U.kl_of(lp, base_lp[(b, t)])

    @torch.no_grad()
    def kl_all_eval():
        out = torch.zeros(len(samp), device=dev)
        for c0 in range(0, B_eval, args.chunk):
            lp = F.log_softmax(net.model(eval_ids[c0:c0 + args.chunk]).float(), -1)
            for j, (b, t) in enumerate(samp):
                if c0 <= b < c0 + args.chunk:
                    out[j] = U.kl_of(lp[b - c0, t], base_lp[(b, t)])
            del lp
        return out

    # per-token atom attribution (gold logit at the target, as unitabl67 does)
    a_cache = torch.zeros(len(samp), J, device=dev)
    ar = torch.arange(1, device=dev)
    for j, (b, t) in enumerate(samp):
        idx = eval_ids[b:b + 1]
        posf = torch.tensor([t], device=dev)

        def reward_fn(lg):
            return reward_at_positions(lg, ar, posf, "gold", idx)[0]
        a_cache[j] = atom_attr_split(net, atoms, idx, reward_fn).sum(1)[0]
    res = {"format": "zsplit67_eval_v1", "n_samples": len(samp), "base_token_ce": round(base_ce, 5),
           "J": J, "split_recon_err": ferr, "results": {}}
    for C in args.C:
        own = torch.load(own_path(args, C), map_location="cpu", weights_only=False)
        members = [m.to(dev) for m in own["members"]] + [own["inert"].to(dev)]
        n_comp = len(members)
        CM = torch.zeros(n_comp, J, dtype=torch.bool, device=dev)
        for c, m in enumerate(members):
            if m.numel():
                CM[c, m] = True
        score_attr = (a_cache @ CM.float().t()).t()                  # [n_comp, samp]
        t0 = time.perf_counter()
        score_orc = torch.zeros(n_comp, len(samp), device=dev)
        for c in range(n_comp):
            if members[c].numel() == 0:
                continue
            keep = torch.ones(J, dtype=torch.bool, device=dev)
            keep[members[c]] = False
            atoms.apply_keep(net, keep)
            score_orc[c] = kl_all_eval()
            if (c + 1) % 1024 == 0:
                log(f"oracle {c + 1}/{n_comp} ({time.perf_counter() - t0:.0f}s)")
        net.restore()
        rankings = {"attr": score_attr, "oracle": score_orc}
        KEEP = [k for k in U.KEEP_GRID if k < n_comp] + [n_comp]
        for rank_name in ("attr", "oracle", "random"):
            keep_kl = np.zeros((len(samp), len(KEEP)))
            drop_kl = np.zeros((len(samp), len(KEEP)))
            kept_atoms = np.zeros((len(samp), len(KEEP)))
            kept_mass = np.zeros((len(samp), len(KEEP)))
            t1 = time.perf_counter()
            for j in range(len(samp)):
                if rank_name == "random":
                    g = torch.Generator().manual_seed(777 + j)
                    order = torch.randperm(n_comp, generator=g).to(dev)
                else:
                    order = rankings[rank_name][:, j].argsort(descending=True)
                sel = torch.zeros(J, dtype=torch.bool, device=dev)
                prev = 0
                for ki, kk in enumerate(KEEP):
                    if kk > prev:
                        sel |= CM[order[prev:kk]].any(0)
                    prev = kk
                    with torch.no_grad():
                        atoms.apply_keep(net, sel)
                        kept_atoms[j, ki] = float(sel.sum())
                        kept_mass[j, ki] = float(mass[sel].sum())
                        keep_kl[j, ki] = kl_token(j)
                        atoms.apply_keep(net, ~sel)
                        drop_kl[j, ki] = kl_token(j)
                net.restore()
            name = f"C{C}/{rank_name}"
            res["results"][name] = {
                "keep_grid": KEEP,
                "kl": [round(float(v), 5) for v in keep_kl.mean(0)],
                "kl_drop": [round(float(v), 5) for v in drop_kl.mean(0)],
                "kl_median": [round(float(v), 5) for v in np.median(keep_kl, 0)],
                "kl_drop_median": [round(float(v), 5) for v in np.median(drop_kl, 0)],
                "units_kept_frac": [round(float(v), 5) for v in kept_atoms.mean(0) / J],
                "mass_kept_frac": [round(float(v), 5) for v in kept_mass.mean(0) / float(mass.sum())],
                "k_at_kl_0.1": [int(next((k for k, v in zip(KEEP, keep_kl[j]) if v < 0.1), n_comp))
                                for j in range(len(samp))],
                "k_at_kl_1": [int(next((k for k, v in zip(KEEP, keep_kl[j]) if v < 1.0), n_comp))
                              for j in range(len(samp))]}
            mk = keep_kl.mean(0)
            show = [k for k in (8, 64, 256, 2048) if k in KEEP]
            log(f"{name:<16} keep " + " ".join(f"@{k} {mk[KEEP.index(k)]:.3f}" for k in show)
                + f" | roundtrip {mk[-1]:.1e} ({time.perf_counter() - t1:.0f}s)")
        res[f"C{C}"] = {"n_components": n_comp, "score_summary": U.score_summary(score_attr, score_orc, samp),
                        "component_sizes": [int(m.numel()) for m in members[:-1]]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {args.out}")


# ----------------------------------------------------------------- report ---

def phase_report(args, dev):
    corpus = Corpus(args)
    split = torch.load(split_path(args), map_location="cpu", weights_only=False)
    atoms = SplitAtoms(split, "cpu")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    except Exception as e:                                # noqa: BLE001
        log(f"tokenizer unavailable ({e}); contexts as token ids")
        tok = None

    def ctx(m, j, c, n_before=args.ctx_tokens):
        s, t = int(m["seq"][j, c]), int(m["pos"][j, c])
        row = corpus[s]
        left = row[max(0, t - n_before):t + 1].tolist()
        pred, gold = int(m["pred"][j, c]), int(m["gold"][j, c])
        lp = round(float(m["lp"][j, c]), 3)
        if tok is None:
            return {"left": left, "pred": pred, "gold": gold, "lp": lp}
        return {"left": tok.decode(left), "pred": tok.decode([pred]), "gold": tok.decode([gold]), "lp": lp}

    for C in args.C:
        fit = torch.load(fit_path(args, C), map_location="cpu", weights_only=False)
        own, sizes, freq = fit["own"], fit["sizes"], fit["usage_freq"]
        te = fit["top_events"]
        order = freq.argsort(descending=True)
        g = torch.Generator().manual_seed(args.seed)
        picks = order[:args.n_show].tolist()
        picks += [c for c in torch.randperm(C, generator=g)[:args.n_show].tolist() if c not in picks]
        comps = []
        for c in picks:
            qs = (own == c).nonzero().flatten()
            if qs.numel() == 0:
                continue
            desc = [atoms.describe(int(q)) + ("+" if fit["p"][q] >= 0 else "-") for q in qs.tolist()]
            ev = {}
            for s in ("pos", "neg"):
                v = te["v"][s][:, c]
                ok = (v > 0).nonzero().flatten()[:args.top_events].tolist()
                ev[s] = [{"coef": round(float(v[j]) * (1 if s == "pos" else -1), 4),
                          **ctx(te["meta"][s], j, c)} for j in ok]
            comps.append({"component": int(c), "n_atoms": int(sizes[c]),
                          "usage_freq": round(float(freq[c]), 5),
                          "pos_frac": round(float(fit["usage_pos_frac"][c]), 3),
                          "atoms": desc[:64], "top_positive": ev["pos"], "top_negative": ev["neg"]})
        out = Path(args.out).with_name(Path(args.out).stem + f"_report_C{C}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"C": C, "components": comps}, indent=1))
        log(f"C={C}: report on {len(comps)} components -> {out}")


# ------------------------------------------------------------------- main ---

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["split", "stream", "eval", "report"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="_100m")
    ap.add_argument("--lab_tag", default="_1m", help="pile corpus tag (Corpus) / unitabl paths")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--z_dir", default="")
    ap.add_argument("--out", default="out/zsplit67.json")
    ap.add_argument("--sensor", default="gim")
    ap.add_argument("--scalar", choices=["pred", "gold"], default="pred")
    # corpus
    ap.add_argument("--lab_source", choices=["pile", "bin"], default="bin")
    ap.add_argument("--tokbin", default=str(S.RUN / "pile_1b_uint16.bin"))
    ap.add_argument("--bin_offset", type=int, default=82_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n_seq", type=int, default=0)
    ap.add_argument("--seq_per_batch", type=int, default=8)
    ap.add_argument("--min_pos", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop_after", type=int, default=0)
    # split
    ap.add_argument("--Gmax", type=int, default=3)
    ap.add_argument("--reservoir", type=int, default=256, help="firing inputs kept per unit")
    ap.add_argument("--quiet_n", type=int, default=4096, help="quiet inputs kept per layer")
    ap.add_argument("--quiet_w", type=float, default=1.0,
                    help="weight of the quiet set relative to the unit's own firing set")
    ap.add_argument("--fire_q", type=float, default=0.98, help="firing = |activation| above this quantile")
    ap.add_argument("--pilot_seqs", type=int, default=64)
    ap.add_argument("--leak_max", type=float, default=0.3)
    ap.add_argument("--min_group", type=int, default=24)
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--unit_batch", type=int, default=256)
    # stream / fit
    ap.add_argument("--C", type=int, nargs="+", default=[2048])
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--dead_frac", type=float, default=1e-3)
    ap.add_argument("--floor_frac", type=float, default=1e-2)
    ap.add_argument("--pilot_batches", type=int, default=64)
    ap.add_argument("--chunk_events", type=int, default=1_000_000)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--ckpt_every", type=int, default=500)
    # eval
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=16)
    # report
    ap.add_argument("--n_show", type=int, default=32)
    ap.add_argument("--top_events", type=int, default=8)
    ap.add_argument("--ctx_tokens", type=int, default=24)
    return ap.parse_args(argv)


def main():
    args = parse()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    {"split": phase_split, "stream": phase_stream, "eval": phase_eval,
     "report": phase_report}[args.phase](args, dev)


if __name__ == "__main__":
    main()
