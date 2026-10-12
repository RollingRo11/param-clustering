"""Rank-one direction atoms for zcofac67: partial parameters with a null space.

Every decomposed matrix W [d_out, d_in] is written in a basis of its INPUT
space, W = sum_j (W m_j) a_j^T with A M = I (rows a_j of A, columns m_j of
M), so piece j is a rank-one matrix that is inert on any input orthogonal to
a_j.  Each weight entry is partially owned by every piece, exactly as with
VPD's rank-one subcomponents, but the pieces sum to W with nothing trained
and nothing left over.  Bases come from dirdecomp67's dirbases.pt: ICA or PCA
of each module's input activations (q/k/v of a layer share one space), the
SVD right singular vectors, or the standard basis.

Attribution of piece j at a position: (a_j . x) (g . W m_j), first order,
GIM backward.  Membership: zcofac67's signed streaming fit assigns every
piece to one component.  Ablating a set S of pieces is a weight-space edit,
W' = W - (W M) diag(1_S) A, so the eval needs no hooks.

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k \
    python -u zdir67.py stream --tag _100m --basis_read ica --basis_write std \
        --lab_source bin --n_seq 208800 --C 2048 4096 8192
    python -u zdir67.py eval --tag _100m --C 2048
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
                      rms_stats, kmeans_init, TOP_EVENTS, N_LAYER)

DATA = Path(os.environ.get("COFAC_DATA", "/data"))
DIRBASES = Path(os.environ.get("DIRBASES", str(DATA / "geo67" / "dir" / "dirbases.pt")))
SHORT = {"attn.q_proj": "q", "attn.k_proj": "k", "attn.v_proj": "v", "attn.o_proj": "o",
         "mlp.c_fc": "fc", "mlp.down_proj": "dn"}
WRITE_SIDE = ("attn.o_proj", "mlp.down_proj")


def log(m):
    print(f"[zdir] {m}", flush=True)


def zdir(args) -> Path:
    d = Path(args.z_dir) if args.z_dir else S.RUN.parent / "zcofac"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_stem(args) -> str:
    return f"dir{args.tag}_{args.basis_read}{args.basis_write}_{args.sensor}_{args.scalar}"


def fit_path(args, C) -> Path:
    return zdir(args) / f"fit{run_stem(args)}_C{C}.pt"


def own_path(args, C) -> Path:
    return U.out_dir(args) / f"own{args.tag}_dir{args.basis_read}{args.basis_write}_z{C}.pt"


def space_of(m: str) -> str:
    layer = m.split(".")[1]
    if m.endswith(("attn.q_proj", "attn.k_proj", "attn.v_proj")):
        return f"attn_in{layer}"
    if m.endswith("attn.o_proj"):
        return f"attn_out{layer}"
    if m.endswith("mlp.c_fc"):
        return f"mlp_in{layer}"
    return f"mlp_out{layer}"


class DirAtoms:
    """Per module: A [d_in, d_in] (rows a_j), M (columns m_j), WM = W M."""

    def __init__(self, net, args, dev, blob=None):
        self.dev = dev
        self.A, self.M, self.WM = {}, {}, {}
        self.mod_of, self.j_of, self.layer, self.kind = [], [], [], []
        need = {args.basis_read, args.basis_write} - {"std"}
        if need and blob is None:
            blob = torch.load(DIRBASES, map_location="cpu", weights_only=False)
        for mi, m in enumerate(MODULES):
            W = net.W0[m].float()
            d_in = W.shape[1]
            basis = args.basis_write if m.endswith(WRITE_SIDE) else args.basis_read
            if basis == "std":
                A = M = torch.eye(d_in, device=dev)
            elif basis == "svd":
                A, M = blob["svd"][m]["A"].to(dev).float(), blob["svd"][m]["M"].to(dev).float()
            else:
                sp = blob["spaces"][space_of(m)]
                A, M = sp[f"A_{basis}"].to(dev).float(), sp[f"M_{basis}"].to(dev).float()
            err = float((A @ M - torch.eye(d_in, device=dev)).abs().max())
            assert err < 1e-2, (m, basis, err)
            self.A[m], self.M[m] = A, M
            self.WM[m] = W @ M                                       # [d_out, d_in]
            self.mod_of += [mi] * d_in
            self.j_of += list(range(d_in))
            self.layer += [int(m.split(".")[1])] * d_in
            self.kind += [mi % 6] * d_in
        self.mod_of = torch.tensor(self.mod_of, device=dev)
        self.j_of = torch.tensor(self.j_of, device=dev)
        self.layer = torch.tensor(self.layer, device=dev)
        self.kind = torch.tensor(self.kind, device=dev)
        self.J = self.mod_of.numel()
        self.offset = {}
        o = 0
        for m in MODULES:
            self.offset[m] = o
            o += self.A[m].shape[0]

    def describe(self, q):
        m = MODULES[int(self.mod_of[q])]
        return f"L{m.split('.')[1]}:{SHORT[m.split('.', 2)[2]]}{int(self.j_of[q])}"

    def mass(self, net):
        out = torch.zeros(self.J, device=self.dev)
        for m in MODULES:
            o, d = self.offset[m], self.A[m].shape[0]
            out[o:o + d] = self.WM[m].pow(2).sum(0) * self.A[m].pow(2).sum(1)
        return out

    @torch.no_grad()
    def apply_keep(self, net, keep):
        """W' = W - sum_{j removed} (W m_j) a_j^T, per module."""
        for m in MODULES:
            o, d = self.offset[m], self.A[m].shape[0]
            drop = (~keep[o:o + d]).float()
            if drop.any():
                net.Wt[m].copy_(net.W0[m] - (self.WM[m] * drop[None, :]) @ self.A[m])
            else:
                net.Wt[m].copy_(net.W0[m])

    def attr(self, net, idx, reward_fn):
        """A [B, T, J]: (a_j . x) * (g . W m_j) for every module and direction."""
        model = net.model
        pre, post, hs = {}, {}, []
        for m in MODULES:
            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
                post[_m] = out
            hs.append(model.get_submodule(m).register_forward_hook(hook))
        lg = model(idx)
        for h in hs:
            h.remove()
        reward = reward_fn(lg)
        grads = torch.autograd.grad(reward, [post[m] for m in MODULES])
        with torch.no_grad():
            parts = []
            for m, g in zip(MODULES, grads):
                coord = pre[m].float() @ self.A[m].t()                # [B,T,d_in]
                wt = g.float() @ self.WM[m]                            # [B,T,d_in]
                parts.append(coord * wt)
            A = torch.cat(parts, dim=-1)
        del grads, pre, post, lg
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
    fit = {"format": "zdir67_fit_v1", "C": C, "N": N, "J": J, "Jl": Jl,
           "sensor": args.sensor, "scalar": args.scalar, "tag": args.tag,
           "basis_read": args.basis_read, "basis_write": args.basis_write, **src_info,
           "own": own_full, "p": p_full, "live": live.cpu(), "dead": dead.cpu(),
           "rms": rms.cpu(), "sizes": sizes.cpu(), "usage_freq": usage_freq.cpu(),
           "usage_pos_frac": (acc.npos / acc.nz.clamp_min(1)).cpu(),
           "mean_abs_coef": (acc.asum / acc.nz.clamp_min(1)).cpu(),
           "active_hist": acc.act_hist.cpu(), "top_events": top.state(),
           "atom_mod": atoms.mod_of.cpu(), "atom_j": atoms.j_of.cpu(),
           "atom_layer": atoms.layer.cpu(), "atom_kind": atoms.kind.cpu(),
           "hist": hist, "seed": args.seed}
    torch.save(fit, fit_path(args, C))
    torch.save({"format": "zdir67_own_v1", "variant": f"dir_z{C}", "C": C,
                "members": members, "inert": inert, "own": own_full, "coef": p_full,
                "basis_read": args.basis_read, "basis_write": args.basis_write,
                "tag": args.tag}, own_path(args, C))
    summ = {"C": C, "N": N, "J": J, "dead": int(dead.sum()),
            "explained_code": round(acc.expl / max(acc.n, 1), 5), "iters": len(hist),
            "active_per_event": {"median": aq[0.5], "p90": aq[0.9]},
            "sizes": U.qdict(sizes[sizes > 0].cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "largest": int(sizes.max()), "empty": int((sizes == 0).sum()),
            "singletons": int((sizes == 1).sum()),
            "usage_freq": U.qdict(usage_freq.cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "components_mixing_matrix_types": mixed, "components_spanning_layers": multilayer,
            "seconds": round(time.perf_counter() - t0, 1)}
    log(f"C={C} done: sizes median {int(sizes.median())} max {int(sizes.max())} "
        f"singletons {summ['singletons']}, usage freq median {usage_freq.median():.4f}, "
        f"{mixed} mix matrix types, {multilayer} span layers -> {own_path(args, C).name}")
    outj = Path(args.out).with_name(Path(args.out).stem + f"_C{C}.json")
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps({**summ, "hist": hist}, indent=1))


def phase_stream(args, dev):
    cfg = S67.SENSORS[args.sensor]
    assert cfg["K"] == 1
    corpus = Corpus(args)
    T, lo = corpus.T, args.min_pos
    n_pos = T - 1 - lo
    N_seq = len(corpus)
    SB = args.seq_per_batch
    n_batches = math.ceil(N_seq / SB)
    net = make_net(args, dev, cfg["mode"])
    atoms = DirAtoms(net, args, dev, blob=args._blob)
    J = atoms.J
    Cs = list(args.C)
    ck = zdir(args) / f"stream{run_stem(args)}_ckpt.pt"
    log(f"stream: {N_seq} seqs x {n_pos} positions in {n_batches} batches; J={J} direction atoms "
        f"(read {args.basis_read}, write {args.basis_write}); C={Cs}; chunk {args.chunk_events}, "
        f"decay {args.decay}")
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
        A = atoms.attr(net, idx, reward_fn)
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
                "decay": args.decay}
    live = st["live"] if st["live"] is not None else torch.arange(J, device=dev)
    for c in Cs:
        finish_C(args, c, st["own"][c], st["p"][c], st["acc"][c], st["top"][c], st["rms"], live,
                 st["dead"], atoms, st["hist"][c], st["n_events"], src_info, t0)
    if ck.exists():
        ck.unlink()
    log(f"stream done: {st['n_events']} events ({time.perf_counter() - t0:.0f}s)")


# ------------------------------------------------------------------- eval ---

def phase_eval(args, dev):
    """unitabl67's keep / drop protocol with direction atoms; masks are
    weight-space edits W - (W M) diag(removed) A."""
    cfg = S67.SENSORS[args.sensor]
    net = make_net(args, dev, cfg["mode"])
    atoms = DirAtoms(net, args, dev, blob=args._blob)
    J = atoms.J
    mass = atoms.mass(net)
    atoms.apply_keep(net, torch.zeros(J, dtype=torch.bool, device=dev))
    zerr = max(net.Wt[p].abs().max().item() for p in MODULES)
    atoms.apply_keep(net, torch.ones(J, dtype=torch.bool, device=dev))
    ferr = max((net.Wt[p] - net.W0[p]).abs().max().item() for p in MODULES)
    net.restore()
    log(f"{J} direction atoms; all-kept err {ferr:.2e}, none-kept max |W| {zerr:.2e}")
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

    a_cache = torch.zeros(len(samp), J, device=dev)
    ar = torch.arange(1, device=dev)
    for j, (b, t) in enumerate(samp):
        idx = eval_ids[b:b + 1]
        posf = torch.tensor([t], device=dev)

        def reward_fn(lg):
            return reward_at_positions(lg, ar, posf, "gold", idx)[0]
        a_cache[j] = atoms.attr(net, idx, reward_fn).sum(1)[0]
    res = {"format": "zdir67_eval_v1", "n_samples": len(samp), "base_token_ce": round(base_ce, 5),
           "J": J, "basis_read": args.basis_read, "basis_write": args.basis_write,
           "recon_err": ferr, "results": {}}
    for C in args.C:
        own = torch.load(own_path(args, C), map_location="cpu", weights_only=False)
        members = [m.to(dev) for m in own["members"]] + [own["inert"].to(dev)]
        n_comp = len(members)
        CM = torch.zeros(n_comp, J, dtype=torch.bool, device=dev)
        for c, m in enumerate(members):
            if m.numel():
                CM[c, m] = True
        score_attr = (a_cache @ CM.float().t()).t()
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
    net = make_net(args, "cpu", "plain")
    atoms = DirAtoms(net, args, "cpu", blob=args._blob)
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
    ap.add_argument("phase", choices=["stream", "eval", "report"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="_100m")
    ap.add_argument("--lab_tag", default="_1m")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--z_dir", default="")
    ap.add_argument("--out", default="out/zdir67.json")
    ap.add_argument("--sensor", default="gim")
    ap.add_argument("--scalar", choices=["pred", "gold"], default="pred")
    ap.add_argument("--basis_read", choices=["ica", "pca", "svd", "std"], default="ica",
                    help="basis of the input space of q/k/v/c_fc")
    ap.add_argument("--basis_write", choices=["ica", "pca", "svd", "std"], default="std",
                    help="basis of the input space of o_proj (head dims) / down_proj (neurons)")
    ap.add_argument("--dirbases", default="", help="override dirbases.pt path")
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
    # fit
    ap.add_argument("--C", type=int, nargs="+", default=[2048])
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--kmeans_iters", type=int, default=25)
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
    args = ap.parse_args(argv)
    global DIRBASES
    if args.dirbases:
        DIRBASES = Path(args.dirbases)
    args._blob = None
    return args


def main():
    args = parse()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    {"stream": phase_stream, "eval": phase_eval, "report": phase_report}[args.phase](args, dev)


if __name__ == "__main__":
    main()
