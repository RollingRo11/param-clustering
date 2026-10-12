"""Direction decomposition of the 67M VPD target.

Every decomposed weight matrix W [d_out, d_in] is split into rank-one pieces
indexed by directions in its INPUT activation space:

    W = sum_c (W m_c) a_c^T          with  A M = I   (A rows a_c, M columns m_c)

so the pieces sum to W exactly with nothing trained and no residual. On an
input x the piece contributes (W m_c)(a_c . x): a component whose activity
a_c . x is zero at a position is inert there. Ablating a set of components at
a position is a projection of that position's module input,

    x' = x - sum_{c removed} m_c (a_c . x),

which the model then runs through its own nonlinearities. Nothing fractional.

Bases (per input space; q/k/v of a layer read the same normed residual and so
share one space -- the components differ through W):
  ica  FastICA on the whitened module inputs: directions chosen for SPARSE
       usage across tokens (the candidate that should make ablatability fall
       out of the basis)
  pca  eigenvectors of the input covariance (orthonormal, dense usage expected)
  svd  right singular vectors of W (cofac's atoms; per module, orthonormal)
  std  coordinate basis (for down_proj this is the neuron basis)

Evaluation protocol (per position, not per target token): every position keeps
its own top-k components by its own importance, the sequence is run once with
those per-position masks, and KL(full || masked) is read at EVERY position.
Rankings: attr = first-order effect (a_c . x_s)(g_s . W m_c) with g the
cotangent of the summed predicted-token log-prob (label-free); act = the
component's contribution norm |a_c . x_s| ||W m_c|| (gradient-free); rand.
The VPD checkpoint is scored under the identical protocol (top-k by its
causal-importance logit at each position, plus its native gate logit > 0).

Phases
  fit   stream module inputs over the fit corpus: covariance + ICA subsample
        per space, FastICA, PCA, SVD; writes dirbases_{tag}.pt
  eval  per-position keep/drop curves, random controls, usage sparsity
  sel   single-component removal: KL at positions where the component is
        active vs where it is not (the collateral test)

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k python -u dirdecomp67.py fit
    python -u dirdecomp67.py eval --arms ica:attr,ica:act,svd:attr
    python -u dirdecomp67.py eval --arms vpd
    python -u dirdecomp67.py sel --decomps ica,svd,vpd
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

import sensor_study67 as S67
from sensor_study67 import MODULES, load67

DATA = Path(os.environ.get("COFAC_DATA", "/data"))
RUN = Path(os.environ.get("STREAM67_RUN", str(DATA / "geo67" / "stream8k")))
TOKBIN = Path(os.environ.get("DIR_TOKBIN", str(RUN / "pile_1b_uint16.bin")))
OUT = Path(os.environ.get("DIR_OUT", str(DATA / "geo67" / "dir")))
if os.environ.get("COFAC_DATA"):
    S67.CKPT = DATA / "target" / "model_step_99999.pt"
VPD_CKPT = DATA / "vpd" / "model_400000.pth"

N_LAYER = 4
BASES = ("ica", "pca", "svd", "std")
KEEP_GRID = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
             3072, 4096, 6144, 8192, 12288, 16384]


def log(m):
    print(f"[dir67] {m}", flush=True)


# ------------------------------------------------------------------ spaces --

def space_of(m: str) -> str:
    """Input-activation space a module reads from."""
    layer = m.split(".")[1]
    if m.endswith(("attn.q_proj", "attn.k_proj", "attn.v_proj")):
        return f"attn_in{layer}"
    if m.endswith("attn.o_proj"):
        return f"attn_out{layer}"
    if m.endswith("mlp.c_fc"):
        return f"mlp_in{layer}"
    if m.endswith("mlp.down_proj"):
        return f"mlp_out{layer}"
    raise ValueError(m)


SPACES = sorted({space_of(m) for m in MODULES})
# one module per space whose input we hook when collecting statistics
REP = {}
for _m in MODULES:
    REP.setdefault(space_of(_m), _m)


# -------------------------------------------------------------------- data --

def bin_batches(offset: int, n_seq: int, seq: int, stride: int, batch: int):
    """Yield int64 [b, seq] windows from the token memmap, starts spaced by
    `stride` from `offset` (stride 0 = spread evenly over the rest of the bin)."""
    arr = np.memmap(TOKBIN, dtype=np.uint16, mode="r")
    if stride <= 0:
        stride = (len(arr) - offset - seq) // n_seq
    starts = [offset + i * stride for i in range(n_seq)]
    assert starts[-1] + seq <= len(arr), "corpus runs past the bin"
    for i in range(0, n_seq, batch):
        rows = [np.asarray(arr[s:s + seq], dtype=np.int64)
                for s in starts[i:i + batch]]
        yield torch.from_numpy(np.stack(rows))


# --------------------------------------------------------------- FastICA ----

def fastica(Xw: torch.Tensor, seed: int, max_iter: int = 400, tol: float = 1e-5,
            chunk: int = 65536):
    """Symmetric FastICA (logcosh contrast) on whitened rows Xw [n, d].
    Returns the orthogonal unmixing matrix W [d, d] (rows = components in the
    whitened space), the iteration count and the final change."""
    n, d = Xw.shape
    g = torch.Generator(device=Xw.device).manual_seed(seed)
    W, _ = torch.linalg.qr(torch.randn(d, d, device=Xw.device, generator=g))
    W = W.T.contiguous()

    def sym_decorrelate(W):
        s, u = torch.linalg.eigh(W @ W.T)
        return (u * s.clamp_min(1e-12).rsqrt()) @ u.T @ W

    W = sym_decorrelate(W)
    it, delta = 0, float("inf")
    for it in range(1, max_iter + 1):
        acc = torch.zeros_like(W)
        gp = torch.zeros(d, device=Xw.device)
        for i in range(0, n, chunk):
            xb = Xw[i:i + chunk]
            S = torch.tanh(xb @ W.T)                       # [b, d]
            acc += S.T @ xb
            gp += (1.0 - S.pow(2)).sum(0)
        W_new = sym_decorrelate(acc / n - (gp / n)[:, None] * W)
        delta = float((1.0 - (W_new * W).sum(1).abs()).max())
        W = W_new
        if delta < tol:
            break
    return W, it, delta


# ------------------------------------------------------------------- fit ----

def phase_fit(args, dev):
    model = load67(dev, "plain")
    d_of = {sp: model.get_submodule(REP[sp]).weight.shape[1] for sp in SPACES}
    stats = {sp: {"n": 0, "sum": torch.zeros(d_of[sp], dtype=torch.float64, device=dev),
                  "xtx": torch.zeros(d_of[sp], d_of[sp], dtype=torch.float64, device=dev),
                  "sub": []} for sp in SPACES}
    p_keep = min(1.0, args.n_ica / (args.fit_seqs * args.seq))
    gsub = torch.Generator(device=dev).manual_seed(args.seed + 7)
    pre = {}
    hs = []
    for sp in SPACES:
        def hook(mod, inp, out, _sp=sp):
            pre[_sp] = inp[0].detach()
        hs.append(model.get_submodule(REP[sp]).register_forward_hook(hook))
    t0 = time.perf_counter()
    n_batches = 0
    with torch.no_grad():
        for ids in bin_batches(args.fit_offset, args.fit_seqs, args.seq,
                               args.fit_stride, args.batch):
            ids = ids.to(dev)
            model(ids)
            keep = torch.rand(ids.shape[0] * ids.shape[1], device=dev,
                              generator=gsub) < p_keep
            for sp in SPACES:
                x = pre[sp].reshape(-1, d_of[sp]).float()
                st = stats[sp]
                st["n"] += x.shape[0]
                st["sum"] += x.sum(0).double()
                st["xtx"] += (x.T @ x).double()
                st["sub"].append(x[keep].half().cpu())
            n_batches += 1
            if n_batches % 16 == 0:
                log(f"fit collect {n_batches * args.batch}/{args.fit_seqs} seqs "
                    f"({time.perf_counter() - t0:.0f}s)")
    for h in hs:
        h.remove()
    log(f"collected {stats[SPACES[0]]['n']} positions per space; ICA subsample "
        f"{sum(t.shape[0] for t in stats[SPACES[0]]['sub'])} ({time.perf_counter() - t0:.0f}s)")

    out = {"format": "dirbases_v1", "spaces": {}, "svd": {}, "space_of": {m: space_of(m) for m in MODULES},
           "config": vars(args)}
    for sp in SPACES:
        st = stats[sp]
        d = d_of[sp]
        mu = (st["sum"] / st["n"]).float()
        C = (st["xtx"] / st["n"] - torch.outer(mu.double(), mu.double())).float()
        C = 0.5 * (C + C.T)
        lam, E = torch.linalg.eigh(C)                    # ascending
        lam, E = lam.flip(0), E.flip(1)                  # descending
        floor = args.eig_floor * lam.mean()
        lam_r = lam.clamp_min(floor)
        n_floor = int((lam < floor).sum())
        # ZCA whitening K = E diag(lam^-1/2) E^T and its inverse
        K = (E * lam_r.rsqrt()) @ E.T
        Kinv = (E * lam_r.sqrt()) @ E.T
        X = torch.cat(st["sub"]).to(dev).float()
        Xw = (X - mu) @ K
        t1 = time.perf_counter()
        Wica, it, delta = fastica(Xw, args.seed, args.ica_iters, args.ica_tol)
        A_ica = Wica @ K                                  # rows a_c (original space)
        M_ica = Kinv @ Wica.T                             # columns m_c
        err = float((A_ica @ M_ica - torch.eye(d, device=dev)).abs().max())
        # order ICA components by mean squared activity on the subsample (descending)
        act = ((X @ A_ica.T) ** 2).mean(0)
        order = act.argsort(descending=True)
        A_ica, M_ica = A_ica[order].contiguous(), M_ica[:, order].contiguous()
        # excess kurtosis of the components on the subsample (sparsity check)
        Sc = (X - mu) @ A_ica.T
        Sc = Sc / Sc.std(0, keepdim=True).clamp_min(1e-12)
        kurt = (Sc.pow(4).mean(0) - 3.0)
        pcs = (X - mu) @ E
        pcs = pcs / pcs.std(0, keepdim=True).clamp_min(1e-12)
        kurt_pca = (pcs.pow(4).mean(0) - 3.0)
        log(f"{sp}: d={d} eig floor hit {n_floor}, ICA {it} iters delta {delta:.2e} "
            f"AM-I {err:.1e}, median excess kurtosis ica {float(kurt.median()):.2f} "
            f"pca {float(kurt_pca.median()):.2f} ({time.perf_counter() - t1:.0f}s)")
        out["spaces"][sp] = {
            "mu": mu.cpu(), "eig": lam.cpu(), "n": st["n"],
            "A_pca": E.T.contiguous().cpu(), "M_pca": E.contiguous().cpu(),
            "A_ica": A_ica.cpu(), "M_ica": M_ica.cpu(),
            "ica_iters": it, "ica_delta": delta, "am_err": err,
            "kurt_ica": kurt.cpu(), "kurt_pca": kurt_pca.cpu(),
            "n_eig_floor": n_floor}
        del X, Xw, Sc, pcs
        st["sub"] = None
        torch.cuda.empty_cache()
    for m in MODULES:
        W = model.get_submodule(m).weight.detach().float()
        # full_matrices: for a wide matrix (down_proj) the extra right singular
        # vectors span W's null space -- zero components, kept so every basis
        # has exactly d_in directions
        U, S, Vh = torch.linalg.svd(W, full_matrices=True)
        # W = sum_q s_q u_q v_q^T = sum_q (W v_q) v_q^T ; rows of Vh are v_q
        out["svd"][m] = {"A": Vh.contiguous().cpu(), "M": Vh.T.contiguous().cpu(),
                         "S": S.cpu()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"dirbases{args.tag}.pt"
    torch.save(out, path)
    summ = {sp: {"d": d_of[sp], "ica_iters": out["spaces"][sp]["ica_iters"],
                 "ica_delta": out["spaces"][sp]["ica_delta"],
                 "am_err": out["spaces"][sp]["am_err"],
                 "kurt_ica_median": float(out["spaces"][sp]["kurt_ica"].median()),
                 "kurt_pca_median": float(out["spaces"][sp]["kurt_pca"].median()),
                 "eig_top5_share": [round(float(v), 4) for v in
                                    (out["spaces"][sp]["eig"][:5] / out["spaces"][sp]["eig"].sum())],
                 "n_eig_floor": out["spaces"][sp]["n_eig_floor"]} for sp in SPACES}
    (OUT / f"dirfit{args.tag}.json").write_text(json.dumps(
        {"format": "dirfit_v1", "n_positions": stats[SPACES[0]]["n"],
         "spaces": summ, "config": vars(args)}, indent=1))
    log(f"wrote {path}")


# --------------------------------------------------------------- bases -----

class Bases:
    """Per (basis, module): A [d_in, d_in], M [d_in, d_in], WM = W M [d_out, d_in],
    wm_norm [d_in]. Shared-space bases are stored once per space."""

    def __init__(self, blob, model, dev, bases=BASES):
        self.dev = dev
        self.W = {m: model.get_submodule(m).weight.detach().float() for m in MODULES}
        self.d_in = {m: self.W[m].shape[1] for m in MODULES}
        self.A, self.M, self.WM, self.norm = {}, {}, {}, {}
        for b in bases:
            for m in MODULES:
                sp = space_of(m)
                if b == "ica":
                    A, M = blob["spaces"][sp]["A_ica"], blob["spaces"][sp]["M_ica"]
                elif b == "pca":
                    A, M = blob["spaces"][sp]["A_pca"], blob["spaces"][sp]["M_pca"]
                elif b == "svd":
                    A, M = blob["svd"][m]["A"], blob["svd"][m]["M"]
                elif b == "std":
                    A = M = torch.eye(self.d_in[m])
                else:
                    raise ValueError(b)
                A, M = A.to(dev).float(), M.to(dev).float()
                self.A[(b, m)], self.M[(b, m)] = A, M
                WM = self.W[m] @ M
                self.WM[(b, m)] = WM
                self.norm[(b, m)] = WM.norm(dim=0)
        self.offsets, o = {}, 0
        for m in MODULES:
            self.offsets[m] = o
            o += self.d_in[m]
        self.n_total = o


class Masker:
    """Per-position component masks applied as forward pre-hooks (ours) or
    forward hooks (VPD). mask[m] is a float tensor [B, T, n_m] of KEPT flags."""

    def __init__(self, model):
        self.model = model
        self.hs = []
        self.active = None       # (kind, basis, masks)

    def install_ours(self, bases: Bases, basis: str, masks: dict):
        self.remove()
        for m in MODULES:
            A, M = bases.A[(basis, m)], bases.M[(basis, m)]
            mk = masks[m]

            def pre_hook(mod, inp, _A=A, _M=M, _mk=mk):
                x = inp[0]
                z = x @ _A.T
                return (x - ((z * (1.0 - _mk)) @ _M.T),)
            self.hs.append(self.model.get_submodule(m).register_forward_pre_hook(pre_hook))

    def install_vpd(self, Vv: dict, Uv: dict, masks: dict):
        self.remove()
        for m in MODULES:
            mk = masks[m]

            def hook(mod, inp, out, _V=Vv[m], _U=Uv[m], _mk=mk):
                x = inp[0]
                return out - ((x @ _V) * (1.0 - _mk)) @ _U
            self.hs.append(self.model.get_submodule(m).register_forward_hook(hook))

    def remove(self):
        for h in self.hs:
            h.remove()
        self.hs = []


def split_masks(flat: torch.Tensor, sizes: list, names: list) -> dict:
    """flat [B, T, sum sizes] -> {name: [B, T, size]}"""
    parts = torch.split(flat, sizes, dim=-1)
    return dict(zip(names, parts))


def topk_masks(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Per-position top-k over the last axis -> float mask of kept flags."""
    n = scores.shape[-1]
    if k >= n:
        return torch.ones_like(scores)
    idx = scores.topk(k, dim=-1).indices
    return torch.zeros_like(scores).scatter_(-1, idx, 1.0)


def kl_rows(base_logp: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(base || model) per position, [B, T]."""
    lp = F.log_softmax(logits.float(), -1)
    return (base_logp.exp() * (base_logp - lp)).sum(-1)


def capture_pg(model, ids, scalar="pred"):
    """Module inputs and output cotangents of the summed objective, per position."""
    pre, post, hs = {}, {}, []
    for m in MODULES:
        def hook(mod, inp, out, _m=m):
            pre[_m] = inp[0]
            out.retain_grad()
            post[_m] = out
        hs.append(model.get_submodule(m).register_forward_hook(hook))
    logits = model(ids)
    for h in hs:
        h.remove()
    lg = logits[:, :-1].float()
    if scalar == "pred":
        reward = F.log_softmax(lg, -1).gather(-1, lg.argmax(-1, keepdim=True).detach()).sum()
    elif scalar == "gold":
        reward = F.log_softmax(lg, -1).gather(-1, ids[:, 1:, None]).sum()
    else:
        raise ValueError(scalar)
    grads = torch.autograd.grad(reward, [post[m] for m in MODULES])
    return ({m: pre[m].detach() for m in MODULES},
            {m: g.detach() for m, g in zip(MODULES, grads)}, logits.detach())


def load_vpd(dev):
    import compare_vpd as CV
    sd = torch.load(str(VPD_CKPT), map_location="cpu", weights_only=False)
    order_m = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    Uv = {m: sd[key[m] + ".U"].float().to(dev) for m in MODULES}   # [n_sub, d_out]
    Vv = {m: sd[key[m] + ".V"].float().to(dev) for m in MODULES}   # [d_in, n_sub]
    split = [Uv[n].shape[0] for n in order_m]
    ci_net = CV.CiFn({k: v.float().to(dev) for k, v in sd.items()
                      if k.startswith("ci_fn")}, order_m, split).to(dev).eval()
    return Uv, Vv, ci_net, order_m, split


def usage_stats(e: torch.Tensor):
    """e [P, n] nonnegative usage energies per position -> per-position n90, n99,
    participation ratio."""
    s, _ = e.sort(-1, descending=True)
    tot = s.sum(-1, keepdim=True).clamp_min(1e-30)
    cs = s.cumsum(-1) / tot
    n90 = (cs < 0.90).sum(-1) + 1
    n99 = (cs < 0.99).sum(-1) + 1
    pr = tot.squeeze(-1).pow(2) / s.pow(2).sum(-1).clamp_min(1e-30)
    return n90.float(), n99.float(), pr


# ------------------------------------------------------------------ eval ----

def phase_eval(args, dev):
    model = load67(dev, "plain")          # params keep requires_grad: capture_pg needs a graph
    arms = [a for a in args.arms.split(",") if a]
    need_ours = [a for a in arms if a != "vpd"]
    bases_used = sorted({a.split(":")[0] for a in need_ours})
    blob = torch.load(OUT / f"dirbases{args.tag}.pt", map_location="cpu",
                      weights_only=False) if need_ours else None
    B = Bases(blob, model, dev, bases=bases_used) if need_ours else None
    sizes = [model.get_submodule(m).weight.shape[1] for m in MODULES]
    if "vpd" in arms:
        Uv, Vv, ci_net, order_m, split = load_vpd(dev)
        n_vpd = sum(split)
        log(f"VPD: {n_vpd} subcomponents")
    masker = Masker(model)
    grid = [k for k in KEEP_GRID if k < (B.n_total if B else n_vpd)]
    grid_ours = grid + ([B.n_total] if B else [])
    grid_vpd = grid + ([n_vpd] if "vpd" in arms else [])
    s0, s1 = args.pos_min, args.seq - 1
    res = {}
    for a in arms:
        res[a] = {"k": (grid_vpd if a == "vpd" else grid_ours),
                  "keep": [], "drop": [], "keep_rand": [], "drop_rand": []}
    if "vpd" in arms:
        res["vpd"]["native"] = {"kl": [], "l0": []}
    usage = {b: {"n90": [], "n99": [], "pr": []} for b in bases_used}
    usage_mod = {b: {m: [] for m in MODULES} for b in bases_used}
    gen = torch.Generator(device=dev).manual_seed(args.seed + 99)
    t0 = time.perf_counter()
    n_batches = 0
    for ids in bin_batches(args.eval_offset, args.eval_seqs, args.seq,
                           args.eval_stride, args.batch):
        ids = ids.to(dev)
        with torch.enable_grad():
            pre, grad, base_logits = capture_pg(model, ids, args.scalar)
        base_logp = F.log_softmax(base_logits.float(), -1)[:, s0:s1]
        with torch.no_grad():
            scores = {}
            for b in bases_used:
                z = {m: pre[m] @ B.A[(b, m)].T for m in MODULES}          # [Bt, T, d_in]
                act = torch.cat([z[m].abs() * B.norm[(b, m)] for m in MODULES], -1)
                e = act[:, s0:s1].reshape(-1, act.shape[-1]).pow(2)
                n90, n99, pr = usage_stats(e)
                usage[b]["n90"].append(n90.cpu()); usage[b]["n99"].append(n99.cpu())
                usage[b]["pr"].append(pr.cpu())
                for m in MODULES:
                    em = (z[m][:, s0:s1].abs() * B.norm[(b, m)]).reshape(-1, sizes[MODULES.index(m)]).pow(2)
                    usage_mod[b][m].append(usage_stats(em)[0].cpu())
                attr = torch.cat([z[m] * (grad[m] @ B.WM[(b, m)]) for m in MODULES], -1).abs()
                scores[(b, "act")] = act
                scores[(b, "attr")] = attr
                del z
            if "vpd" in arms:
                logits_ci = ci_net({m: pre[m] for m in MODULES})
                ci = torch.cat([logits_ci[m] for m in order_m], -1)            # [Bt, T, n_vpd]
                # module order for masks = MODULES order; ci is in sorted order
                perm = []
                for m in MODULES:
                    o = sum(split[:order_m.index(m)])
                    perm.append(torch.arange(o, o + Uv[m].shape[0], device=dev))
                perm = torch.cat(perm)
                ci = ci[..., perm]
                sizes_vpd = [Uv[m].shape[0] for m in MODULES]

            def run(kind, basis, mask_flat, szs):
                masks = split_masks(mask_flat, szs, MODULES)
                if kind == "ours":
                    masker.install_ours(B, basis, masks)
                else:
                    masker.install_vpd(Vv, Uv, masks)
                lg = model(ids)[:, s0:s1]
                masker.remove()
                return kl_rows(base_logp, lg)

            for a in arms:
                if a == "vpd":
                    sc, kind, basis, szs, ks = ci, "vpd", None, sizes_vpd, grid_vpd
                else:
                    basis, rk = a.split(":")
                    sc, kind, szs, ks = scores[(basis, rk)], "ours", sizes, grid_ours
                rnd = torch.rand(sc.shape, device=dev, generator=gen)
                for ki, k in enumerate(ks):
                    mk = topk_masks(sc, k)
                    kl_keep = run(kind, basis, mk, szs)
                    kl_drop = run(kind, basis, 1.0 - mk, szs)
                    mr = topk_masks(rnd, k)
                    kl_kr = run(kind, basis, mr, szs)
                    kl_dr = run(kind, basis, 1.0 - mr, szs)
                    for name, v in (("keep", kl_keep), ("drop", kl_drop),
                                    ("keep_rand", kl_kr), ("drop_rand", kl_dr)):
                        if len(res[a][name]) <= ki:
                            res[a][name].append([])
                        res[a][name][ki].append(v.reshape(-1).cpu())
                if a == "vpd":
                    mk = (ci > 0).float()
                    v = run("vpd", None, mk, sizes_vpd)
                    res["vpd"]["native"]["kl"].append(v.reshape(-1).cpu())
                    res["vpd"]["native"]["l0"].append(mk[:, s0:s1].sum(-1).reshape(-1).cpu())
        n_batches += 1
        log(f"eval batch {n_batches}/{math.ceil(args.eval_seqs / args.batch)} "
            f"({time.perf_counter() - t0:.0f}s)")

    def summ(chunks):
        v = torch.cat(chunks)
        return {"mean": float(v.mean()), "median": float(v.median()),
                "p90": float(v.quantile(0.9)), "p10": float(v.quantile(0.1))}

    out = {"format": "direval_v1", "tag": args.tag, "arms": arms, "scalar": args.scalar,
           "eval_seqs": args.eval_seqs, "seq": args.seq, "pos_min": s0,
           "n_positions": int(sum(c.numel() for c in res[arms[0]]["keep"][0])),
           "n_total_ours": (B.n_total if B else None),
           "n_total_vpd": (n_vpd if "vpd" in arms else None),
           "protocol": "per-position top-k masks, KL(full||masked) at every position",
           "results": {}, "usage": {}}
    for a in arms:
        r = {"k": res[a]["k"]}
        for name in ("keep", "drop", "keep_rand", "drop_rand"):
            ss = [summ(ch) for ch in res[a][name]]
            r[name] = {key: [round(s[key], 5) for s in ss] for key in ("mean", "median", "p90", "p10")}
        if a == "vpd":
            r["native"] = {"kl": summ(res["vpd"]["native"]["kl"]),
                           "l0": summ(res["vpd"]["native"]["l0"])}
        out["results"][a] = r
    for b in bases_used:
        out["usage"][b] = {key: summ(usage[b][key]) for key in ("n90", "n99", "pr")}
        out["usage"][b]["per_module_n90_median"] = {
            m: float(torch.cat(usage_mod[b][m]).median()) for m in MODULES}
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT / f"direval{args.tag}_{args.arms.replace(':', '').replace(',', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")
    for a in arms:
        r = out["results"][a]
        log(f"{a}: " + "  ".join(f"k{k}:{v:.3f}" for k, v in zip(r["k"], r["keep"]["mean"])))


# ------------------------------------------------------------------- sel ----

def phase_sel(args, dev):
    """Remove ONE component everywhere in the sequence; compare KL at positions
    where it is active (in that position's top-`act_k` by activity / CI) vs not."""
    model = load67(dev, "plain")
    for p in model.parameters():
        p.requires_grad_(False)
    decomps = [d for d in args.decomps.split(",") if d]
    bases_used = [d for d in decomps if d != "vpd"]
    blob = torch.load(OUT / f"dirbases{args.tag}.pt", map_location="cpu",
                      weights_only=False) if bases_used else None
    B = Bases(blob, model, dev, bases=bases_used) if bases_used else None
    sizes = [model.get_submodule(m).weight.shape[1] for m in MODULES]
    if "vpd" in decomps:
        Uv, Vv, ci_net, order_m, split = load_vpd(dev)
        sizes_vpd = [Uv[m].shape[0] for m in MODULES]
        n_vpd = sum(split)
    masker = Masker(model)
    s0, s1 = args.pos_min, args.seq - 1
    # pass 1: base log-probs (kept on the GPU) and per-position active sets
    batches = []
    with torch.no_grad():
        for ids in bin_batches(args.eval_offset, args.eval_seqs, args.seq,
                               args.eval_stride, args.batch):
            ids = ids.to(dev)
            pre, hs = {}, []
            for m in MODULES:
                def hook(mod, inp, out, _m=m):
                    pre[_m] = inp[0]
                hs.append(model.get_submodule(m).register_forward_hook(hook))
            base_logp = F.log_softmax(model(ids).float(), -1)[:, s0:s1]
            for h in hs:
                h.remove()
            active = {}
            for b in bases_used:
                act = torch.cat([(pre[m] @ B.A[(b, m)].T).abs() * B.norm[(b, m)]
                                 for m in MODULES], -1)
                active[b] = topk_masks(act, args.act_k)[:, s0:s1].bool().cpu()
            if "vpd" in decomps:
                lc = ci_net({m: pre[m] for m in MODULES})
                ci = torch.cat([lc[m] for m in order_m], -1)
                perm = []
                for m in MODULES:
                    o = sum(split[:order_m.index(m)])
                    perm.append(torch.arange(o, o + Uv[m].shape[0], device=dev))
                ci = ci[..., torch.cat(perm)]
                active["vpd"] = (ci > 0)[:, s0:s1].cpu()
            batches.append((ids, base_logp, active))
    n_pos_total = sum(a[bases_used[0] if bases_used else "vpd"].shape[0]
                      * a[bases_used[0] if bases_used else "vpd"].shape[1]
                      for _, _, a in batches)
    gen = torch.Generator().manual_seed(args.seed + 5)
    out = {"format": "dirsel_v1", "tag": args.tag, "act_k": args.act_k,
           "eval_seqs": args.eval_seqs, "n_comp": args.n_comp,
           "n_positions": int(n_pos_total), "results": {}}
    for d in decomps:
        n = n_vpd if d == "vpd" else B.n_total
        szs = sizes_vpd if d == "vpd" else sizes
        own_count = torch.stack([a[d].reshape(-1, n).sum(0) for _, _, a in batches]).sum(0)
        eligible = (own_count >= args.min_own).nonzero().flatten()
        pick = eligible[torch.randperm(eligible.numel(), generator=gen)[:args.n_comp]]
        log(f"{d}: {eligible.numel()} components active on >= {args.min_own} positions; "
            f"testing {pick.numel()}")
        flat = torch.ones(args.batch, args.seq, n, device=dev)
        rows = []
        t0 = time.perf_counter()
        for j, c in enumerate(pick.tolist()):
            kl_own, kl_oth = [], []
            flat[..., c] = 0.0
            with torch.no_grad():
                for ids, base_logp, active in batches:
                    Bt = ids.shape[0]
                    masks = split_masks(flat[:Bt], szs, MODULES)
                    if d == "vpd":
                        masker.install_vpd(Vv, Uv, masks)
                    else:
                        masker.install_ours(B, d, masks)
                    kl = kl_rows(base_logp, model(ids)[:, s0:s1]).cpu()
                    masker.remove()
                    own = active[d][..., c]
                    kl_own.append(kl[own]); kl_oth.append(kl[~own])
            flat[..., c] = 1.0
            ko, kt = torch.cat(kl_own), torch.cat(kl_oth)
            rows.append({"comp": int(c), "n_own": int(ko.numel()),
                         "kl_own": float(ko.mean()), "kl_other": float(kt.mean()),
                         "kl_other_median": float(kt.median())})
            if (j + 1) % 32 == 0:
                log(f"{d} {j + 1}/{pick.numel()} ({time.perf_counter() - t0:.0f}s)")
        del flat
        own = torch.tensor([r["kl_own"] for r in rows])
        oth = torch.tensor([r["kl_other"] for r in rows])
        out["results"][d] = {
            "n": len(rows),
            "own_frac": float(own_count[pick].float().mean() / n_pos_total),
            "kl_own_median": float(own.median()), "kl_other_median": float(oth.median()),
            "kl_own_mean": float(own.mean()), "kl_other_mean": float(oth.mean()),
            "ratio_median": float((own / oth.clamp_min(1e-9)).median()),
            "rows": rows}
        r = out["results"][d]
        log(f"{d}: median own {r['kl_own_median']:.4f} other {r['kl_other_median']:.4f} "
            f"ratio {r['ratio_median']:.1f}")
    path = Path(args.out) if args.out else OUT / f"dirsel{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")


# ------------------------------------------------------------------ main ----

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "eval", "sel"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    # fit
    ap.add_argument("--fit_offset", type=int, default=0)
    ap.add_argument("--fit_seqs", type=int, default=2048)
    ap.add_argument("--fit_stride", type=int, default=512)
    ap.add_argument("--n_ica", type=int, default=262144,
                    help="positions subsampled for FastICA per space")
    ap.add_argument("--ica_iters", type=int, default=400)
    ap.add_argument("--ica_tol", type=float, default=1e-5)
    ap.add_argument("--eig_floor", type=float, default=1e-3,
                    help="whitening floor as a fraction of the mean eigenvalue")
    # eval / sel
    ap.add_argument("--arms", default="ica:attr,ica:act,pca:attr,pca:act,svd:attr,svd:act,std:attr,std:act")
    ap.add_argument("--decomps", default="ica,svd,vpd")
    ap.add_argument("--scalar", default="pred", choices=["pred", "gold"])
    ap.add_argument("--eval_offset", type=int, default=800_000_000)
    ap.add_argument("--eval_seqs", type=int, default=128)
    ap.add_argument("--eval_stride", type=int, default=100_000)
    ap.add_argument("--pos_min", type=int, default=32)
    ap.add_argument("--act_k", type=int, default=200,
                    help="sel: a component is 'active' at a position when it is in "
                         "that position's top-act_k by activity (VPD: its gate logit > 0)")
    ap.add_argument("--n_comp", type=int, default=256)
    ap.add_argument("--min_own", type=int, default=16)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def main():
    args = parse()
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log(f"phase {args.phase} device {dev} out {OUT}")
    {"fit": phase_fit, "eval": phase_eval, "sel": phase_sel}[args.phase](args, dev)


if __name__ == "__main__":
    main()
