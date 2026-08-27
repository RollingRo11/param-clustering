"""Co-factorization stages for the 67M VPD target. Runs INSIDE Beam
containers (cofac67 disk at /data). Driver: beam_cofac67.py.

Method (proposal sec 4, winning toy config): variant-B SVD atoms over the 24
matrices (J = 24*768 = 18,432), per-event attribution of the predicted-token
LOG-ODDS streamed through hooks -- a_iq = sigma_q sum_t (u_q^T d_t)(v_q^T h_t)
-- then M_bar ~= U S V^T with V on a (C+1)-simplex (residual/background C0),
U on a simplex, I-divergence objective."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, load67

import os as _os
DATA = Path(_os.environ.get("COFAC_DATA", "/data"))
RUN = DATA / "cofac67"
S67.CKPT = DATA / "target/model_step_99999.pt"


def atoms_prep(device="cuda"):
    """SVD of each decomposed matrix; saves per-module U,S,V + atom index."""
    model = load67(device, "plain")
    blob, index = {}, []
    for p in MODULES:
        W = model.get_submodule(p).weight.detach().float()      # [out, in]
        U, S, V = torch.linalg.svd(W, full_matrices=False)
        blob[p] = {"U": U.cpu(), "S": S.cpu(), "V": V.T.cpu()}  # V: [in, r]
        index += [(p, q) for q in range(S.numel())]
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({"svd": blob, "index": index}, RUN / "atoms.pt")
    return {"J": len(index),
            "per_module": {p: blob[p]["S"].numel() for p in MODULES},
            "recon_relerr_max": max(
                ((blob[p]["U"] @ torch.diag(blob[p]["S"]) @ blob[p]["V"].T
                  - model.get_submodule(p).weight.detach().float().cpu())
                 .norm() / model.get_submodule(p).weight.norm()).item()
                for p in MODULES)}


def _logodds(logits, y):
    """logit_y - logsumexp(other logits), rowwise."""
    ls = torch.logsumexp(logits, -1)
    ly = logits.gather(-1, y[:, None]).squeeze(1)
    # logsumexp excluding y: log(exp(ls) - exp(ly)) stably
    other = ls + torch.log1p(-torch.exp(ly - ls).clamp(max=1 - 1e-7))
    return ly - other


def _collect_batch(model, svd, ids, pos, device, bf16=False):
    """One batched forward+backward; returns A [B, J] fp32 and y [B].
    bf16 runs model fwd/bwd under autocast; projections accumulate fp32."""
    B = ids.shape[0]
    pre, post, hs = {}, {}, []
    for p in MODULES:
        mod = model.get_submodule(p)

        def hook(m, inp, out, _p=p):
            pre[_p] = inp[0]
            out.retain_grad()
            post[_p] = out
        hs.append(mod.register_forward_hook(hook))
    import contextlib
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if bf16
           else contextlib.nullcontext())
    with ctx:
        logits = model(ids)
    for h in hs:
        h.remove()
    rows = torch.arange(B, device=device)
    at = logits[rows, pos].float()
    y = at.argmax(-1)
    s = _logodds(at, y).sum()
    grads = torch.autograd.grad(s, [post[p] for p in MODULES])
    outs = []
    for p, g in zip(MODULES, grads):
        U, S, V = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                   svd[p]["V"].to(device))
        # a_bq = S_q * sum_t (delta_bt @ U)_q * (h_bt @ V)_q ; einsum over t
        # contracts in one matmul-like pass, accumulated fp32
        du = torch.einsum("btd,dr->btr", g.float(), U)
        hv = torch.einsum("bti,ir->btr", pre[p].detach().float(), V)
        outs.append(S[None] * (du * hv).sum(1))
        del du, hv
    for p in MODULES:
        pre[p] = post[p] = None
    return torch.cat(outs, dim=1), y                            # [B, J]


def verify(device="cuda"):
    """Hook-streamed attribution vs direct grad projection, 2 events."""
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids, _, _ = pile_data.load_pile_blocks(tok, 2, 512, seed=7,
                                           tokenizer_name=S67.TOKENIZER)
    ids = ids.to(device)
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    gen = torch.Generator().manual_seed(0)
    pos = torch.randint(64, 511, (2,), generator=gen).to(device)
    A, y = _collect_batch(model, svd, ids, pos, device)
    # direct: per event, dW s projected onto atoms of one attn + one mlp mod
    worst = 0.0
    for b in range(2):
        logits = model(ids[b:b + 1])
        at = logits[0, pos[b]].float()
        s = _logodds(at[None], y[b:b + 1]).squeeze()
        for p in ("h.1.attn.q_proj", "h.2.mlp.down_proj"):
            W = model.get_submodule(p).weight
            gW = torch.autograd.grad(s, W, retain_graph=True)[0].float()
            U, S, V = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                       svd[p]["V"].to(device))
            direct = S * torch.einsum("dr,di,ir->r", U, gW, V)
            j0 = sum(svd[q]["S"].numel() for q in MODULES
                     if MODULES.index(q) < MODULES.index(p))
            got = A[b, j0:j0 + S.numel()]
            rel = ((got - direct).norm() / direct.norm().clamp_min(1e-9)).item()
            worst = max(worst, rel)
    return {"worst_relerr": worst}


def collect_chunk(chunk_id: int, n_chunks: int, seqs_per_chunk: int = 512,
                  batch: int = 16, seq: int = 512, device="cuda",
                  sensor="plain"):
    """Resumable: computes A rows for this chunk's sequences, saves fp16.
    sensor: a sensor_study67 mode ("plain" = true gradients; "gim" = corrected
    backward — frozen RMS stats, tempered softmax jacobian, scaled qk/v/mlp
    credit). Forward pass identical across modes; same events/positions."""
    pref = "A_chunk" if sensor == "plain" else f"A_{sensor}_chunk"
    out_f = RUN / f"{pref}{chunk_id:04d}.pt"
    if out_f.exists():
        return {"chunk": chunk_id, "cached": True}
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    total = n_chunks * seqs_per_chunk
    ids_all, labels, _ = pile_data.load_pile_blocks(
        tok, total, seq, seed=0, tokenizer_name=S67.TOKENIZER)
    sl = slice(chunk_id * seqs_per_chunk, (chunk_id + 1) * seqs_per_chunk)
    ids_all = ids_all[sl]
    labels = labels[sl]
    model = load67(device, sensor)
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    gen = torch.Generator().manual_seed(1000 + chunk_id)
    pos_all = torch.randint(64, seq - 1, (ids_all.shape[0],), generator=gen)
    rowsA, rowsY = [], []
    for i in range(0, ids_all.shape[0], batch):
        A, y = _collect_batch(model, svd, ids_all[i:i + batch].to(device),
                              pos_all[i:i + batch].to(device), device)
        rowsA.append(A.cpu().half())
        rowsY.append(y.cpu())
    torch.save({"A": torch.cat(rowsA), "y": torch.cat(rowsY),
                "pos": pos_all, "labels": labels}, out_f)
    return {"chunk": chunk_id, "rows": int(sum(a.shape[0] for a in rowsA))}


# ------------------------------------------------------------------- fit ----

def _load_A(prefix="A_chunk"):
    chunks = sorted(RUN.glob(f"{prefix}*.pt"))
    assert chunks, "no collected chunks"
    As, ys, poss = [], [], []
    for f in chunks:
        d = torch.load(f, map_location="cpu", weights_only=False)
        As.append(d["A"].float())
        ys.append(d["y"])
        poss.append(d["pos"])
    return torch.cat(As), torch.cat(ys), torch.cat(poss), len(chunks)


def spectrum_stats(a_prefix="A_chunk", holdout_frac=0.125, device="cuda"):
    """Diagnostics of the matrix the fit actually sees (layer-RMS, abs,
    row-L1): singular-energy concentration raw and mean-centered, plus the
    rank-1 LS backbone share (fit_pinned's formula). Lets sensors be A/B'd
    before paying for a refit."""
    A, y, pos, _ = _load_A(a_prefix)
    N = A.shape[0]
    n_hold = int(N * holdout_frac)
    A_fit = A[:-n_hold].to(device)
    svd = torch.load(RUN / "atoms.pt", map_location="cpu", weights_only=False)
    sizes = [svd["svd"][p]["S"].numel() for p in MODULES]
    j0 = 0
    for sz in sizes:
        g = A_fit[:, j0:j0 + sz]
        A_fit[:, j0:j0 + sz] = g / g.pow(2).mean().sqrt().clamp_min(1e-12)
        j0 += sz
    M_bar = A_fit.abs()
    M_bar = M_bar / M_bar.sum(1, keepdim=True).clamp_min(1e-12)

    def stats(X):
        e = torch.linalg.svdvals(X).pow(2)
        tot = e.sum()
        cs = e.cumsum(0) / tot
        return {"top1_energy": float(e[0] / tot),
                "n50": int((cs < 0.5).sum().item()) + 1,
                "n90": int((cs < 0.9).sum().item()) + 1,
                "eff_rank_pr": float(tot.pow(2) / e.pow(2).sum())}

    raw = stats(M_bar)
    cen = stats(M_bar - M_bar.mean(0, keepdim=True))
    mu = M_bar.mean(0)
    u = mu / mu.norm()
    a0 = M_bar @ u
    beta = ((a0[:, None] * M_bar).sum(0) / a0.pow(2).sum()).clamp_min(0)
    back_share = float(a0.sum() * beta.sum() / M_bar.sum())
    return {"prefix": a_prefix, "N_fit": int(M_bar.shape[0]),
            "raw": raw, "centered": cen, "rank1_ls_share": back_share}


def fit(k_factors=2048, c_groups=1024, steps=3000, lr=2e-2, seed=0,
        holdout_frac=0.125, device="cuda", a_prefix="A_chunk",
        out_name="factorization.pt", variant="v2"):
    """v2 co-factorization (U-simplex, residual V, I-div) on collected A.
    variant: "v2" (default); "snorm" = S columns L1-normalized in-graph,
    everything else unchanged (equal per-component throughput in S);
    "s2v" = column-stochastic S with a FREE softplus V carrying all scale
    (V is the actual additive allocation; no residual column; saved "V" is
    the per-atom row-normalized allocation so mass stats/eval compare)."""
    A, y, pos, n_chunks = _load_A(a_prefix)
    N = A.shape[0]
    n_hold = int(N * holdout_frac)
    A_fit = A[:-n_hold].to(device)
    # layer-RMS normalize per module group (norm=layer, group=matrix)
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)
    sizes = [svd["svd"][p]["S"].numel() for p in MODULES]
    Ahat = A_fit.clone()
    j0 = 0
    for sz in sizes:
        g = Ahat[:, j0:j0 + sz]
        Ahat[:, j0:j0 + sz] = g / g.pow(2).mean().sqrt().clamp_min(1e-12)
        j0 += sz
    M = Ahat.abs()
    M_bar = M / M.sum(1, keepdim=True).clamp_min(1e-12)

    n, j = M_bar.shape
    g = torch.Generator().manual_seed(seed)
    Wu = (torch.rand(n, k_factors, generator=g) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Ws = (torch.rand(k_factors, c_groups, generator=g) * 0.5 + 0.2
          ).to(device).requires_grad_()
    if variant == "s2v":
        # free V initialized near uniform allocation with unit atom mass
        import math
        v0 = math.log(math.expm1(1.0 / c_groups))
        Wv = (torch.randn(j, c_groups, generator=g) * 0.05 + v0
              ).to(device).requires_grad_()
    else:
        Wv = (torch.randn(j, c_groups + 1, generator=g) * 0.05
              ).to(device).requires_grad_()
    opt = torch.optim.Adam([Wu, Ws, Wv], lr=lr)
    mass = M_bar.sum().clamp_min(1e-8)

    def factors():
        S = F.softplus(Ws)
        if variant in ("snorm", "s2v"):
            S = S / S.sum(0, keepdim=True).clamp_min(1e-8)
        if variant == "s2v":
            Vfull = F.softplus(Wv)
            return S, Vfull, Vfull
        Vfull = torch.softmax(Wv, dim=1)
        return S, Vfull, Vfull[:, :c_groups]

    log_hist = []
    for step in range(steps):
        U = torch.softmax(Wu, dim=1)
        S, Vfull, V = factors()
        M_hat = U @ S @ V.T
        loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                - M_bar + M_hat).sum() / mass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                rel = ((M_bar - M_hat).norm() / M_bar.norm()).item()
            log_hist.append(f"step {step} idiv {loss.item():.4e} rel {rel:.4f}")
            print(log_hist[-1], flush=True)
    with torch.no_grad():
        S, Vfull, V = factors()
        r2 = 1 - (M_bar - torch.softmax(Wu, 1) @ S @ V.T
                  ).pow(2).sum().item() / \
            (M_bar - M_bar.mean()).pow(2).sum().item()
        if variant == "s2v":
            V_save = (V / V.sum(1, keepdim=True).clamp_min(1e-12)).cpu()
            r_save = torch.zeros(j)
            resid_mean = 0.0
        else:
            V_save = V.cpu()
            r_save = Vfull[:, -1].cpu()
            resid_mean = Vfull[:, -1].mean().item()
    torch.save({"V": V_save, "r": r_save,
                "U": torch.softmax(Wu, 1).cpu(), "S": S.cpu(),
                "sizes": sizes, "n_hold": n_hold,
                "V_raw": (V.cpu() if variant == "s2v" else None),
                "config": {"K": k_factors, "C": c_groups, "steps": steps,
                           "a_prefix": a_prefix, "variant": variant}},
               RUN / out_name)
    return {"N_fit": n, "J": j, "r2_attr_euclid": r2, "variant": variant,
            "mean_residual": resid_mean, "log": log_hist[-3:]}


def fit_centered(k_factors=2048, c_groups=1024, steps=3000, lr=2e-2, seed=0,
                 holdout_frac=0.125, device="cuda"):
    """v3 prototype: the residual column RECONSTRUCTS (M_hat includes
    a_i * r) instead of being a pure loss sink, is initialized to the
    empirical mean attribution shape with a large logit head start, and
    carries a learned per-event backbone scale. Components can then only
    earn mass by explaining deviations from the backbone; since eval always
    keeps r, keep-top-k counts deviation components only."""
    A, y, pos, n_chunks = _load_A()
    N = A.shape[0]
    n_hold = int(N * holdout_frac)
    A_fit = A[:-n_hold].to(device)
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)
    sizes = [svd["svd"][p]["S"].numel() for p in MODULES]
    Ahat = A_fit.clone()
    j0 = 0
    for sz in sizes:
        g = Ahat[:, j0:j0 + sz]
        Ahat[:, j0:j0 + sz] = g / g.pow(2).mean().sqrt().clamp_min(1e-12)
        j0 += sz
    M = Ahat.abs()
    M_bar = M / M.sum(1, keepdim=True).clamp_min(1e-12)

    n, j = M_bar.shape
    gen = torch.Generator().manual_seed(seed)
    Wu = (torch.rand(n, k_factors, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Wv = torch.randn(j, c_groups + 1, generator=gen) * 0.05
    mu = M_bar.mean(0).cpu()
    Wv[:, -1] = (mu / mu.mean()).clamp(0.05, 20).log() + 7.0
    Wv = Wv.to(device).requires_grad_()
    Wa = torch.full((n,), 0.5413, device=device).requires_grad_()
    opt = torch.optim.Adam([Wu, Ws, Wv, Wa], lr=lr)
    mass = M_bar.sum().clamp_min(1e-8)
    log_hist = []
    for step in range(steps):
        U = torch.softmax(Wu, dim=1)
        S = F.softplus(Ws)
        Vfull = torch.softmax(Wv, dim=1)
        V = Vfull[:, :c_groups]
        r = Vfull[:, -1]
        a = F.softplus(Wa)
        M_hat = U @ S @ V.T + a[:, None] * r[None, :]
        loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                - M_bar + M_hat).sum() / mass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                rel = ((M_bar - M_hat).norm() / M_bar.norm()).item()
                rm = float(Vfull[:, -1].sum() / j)
            log_hist.append(f"step {step} idiv {loss.item():.4e} "
                            f"rel {rel:.4f} r_mass {rm:.3f}")
            print(log_hist[-1], flush=True)
    with torch.no_grad():
        Vfull = torch.softmax(Wv, dim=1)
        a = F.softplus(Wa)
    torch.save({"V": Vfull[:, :c_groups].cpu(), "r": Vfull[:, -1].cpu(),
                "U": torch.softmax(Wu, 1).cpu(), "S": F.softplus(Ws).cpu(),
                "a": a.cpu(), "sizes": sizes, "n_hold": n_hold,
                "config": {"K": k_factors, "C": c_groups, "steps": steps,
                           "centered": True}},
               RUN / "factorization_centered.pt")
    return {"N_fit": n, "J": j,
            "r_mean_mass": float(Vfull[:, -1].mean()),
            "a_mean": float(a.mean()), "log": log_hist[-3:]}


def fit_pinned(k_factors=2048, c_groups=1024, steps=3000, lr=2e-2, seed=0,
               holdout_frac=0.125, device="cuda", f_cap=0.9):
    """v3b: the backbone is PINNED, not learned. A rank-1 LS fit of M_bar
    (per-event coefficient a on the mean direction, per-atom loading beta)
    fixes each atom's backbone fraction f; the residual column IS f and V
    distributes only the remaining (1-f) of each atom. Drain is impossible
    and no component can become the backbone. M_hat adds a*beta^T with a
    free positive per-event scale initialized at the LS solution."""
    A, y, pos, n_chunks = _load_A()
    N = A.shape[0]
    n_hold = int(N * holdout_frac)
    A_fit = A[:-n_hold].to(device)
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)
    sizes = [svd["svd"][p]["S"].numel() for p in MODULES]
    Ahat = A_fit.clone()
    j0 = 0
    for sz in sizes:
        g = Ahat[:, j0:j0 + sz]
        Ahat[:, j0:j0 + sz] = g / g.pow(2).mean().sqrt().clamp_min(1e-12)
        j0 += sz
    M = Ahat.abs()
    M_bar = M / M.sum(1, keepdim=True).clamp_min(1e-12)
    n, j = M_bar.shape

    with torch.no_grad():
        mu = M_bar.mean(0)
        u = mu / mu.norm()
        a0 = M_bar @ u                                          # [n]
        beta = (a0[:, None] * M_bar).sum(0) / a0.pow(2).sum()   # [j]
        beta = beta.clamp_min(0)
        colsum = M_bar.sum(0)
        f = (beta * a0.sum() / colsum.clamp_min(1e-12)).clamp(0, f_cap)
        back_share = float((a0[:, None] * beta[None, :]).sum() / M_bar.sum())
    print(f"pinned backbone: mean f {f.mean():.3f}  "
          f"mass share {back_share:.3f}", flush=True)

    gen = torch.Generator().manual_seed(seed)
    Wu = (torch.rand(n, k_factors, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Wv = (torch.randn(j, c_groups, generator=gen) * 0.05
          ).to(device).requires_grad_()
    Wa = torch.log(torch.expm1(a0.clamp_min(1e-6))).requires_grad_()
    opt = torch.optim.Adam([Wu, Ws, Wv, Wa], lr=lr)
    mass = M_bar.sum().clamp_min(1e-8)
    keep1f = (1.0 - f)[:, None]
    log_hist = []
    for step in range(steps):
        U = torch.softmax(Wu, dim=1)
        S = F.softplus(Ws)
        V = keep1f * torch.softmax(Wv, dim=1)
        a = F.softplus(Wa)
        M_hat = U @ S @ V.T + a[:, None] * beta[None, :]
        loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                - M_bar + M_hat).sum() / mass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                rel = ((M_bar - M_hat).norm() / M_bar.norm()).item()
            log_hist.append(f"step {step} idiv {loss.item():.4e} rel {rel:.4f}")
            print(log_hist[-1], flush=True)
    with torch.no_grad():
        V = keep1f * torch.softmax(Wv, dim=1)
    torch.save({"V": V.cpu(), "r": f.cpu(),
                "U": torch.softmax(Wu, 1).cpu(), "S": F.softplus(Ws).cpu(),
                "a": F.softplus(Wa).detach().cpu(), "beta": beta.cpu(),
                "sizes": sizes, "n_hold": n_hold,
                "config": {"K": k_factors, "C": c_groups, "steps": steps,
                           "pinned": True, "f_cap": f_cap}},
               RUN / "factorization_pinned.pt")
    return {"N_fit": n, "J": j, "back_share": back_share,
            "f_mean": float(f.mean()), "log": log_hist[-3:]}


# ------------------------------------------------------------------ eval ----

def eval_klkeep(ks=(8, 16, 32, 64, 128, 256, 512, 1024), n_events=96,
                device="cuda", seed=0, fact_path=None, out_name="klkeep.json",
                a_prefix="A_chunk"):
    """KLKeep(k) on held-out events + matched-random null (sec 5.1)."""
    A, y, pos, _ = _load_A(a_prefix)
    fact = torch.load(fact_path or (RUN / "factorization.pt"),
                      map_location="cpu", weights_only=False)
    V = fact["V"].to(device)                                   # [J, C]
    C = V.shape[1]
    n_hold = fact["n_hold"]
    idx = torch.arange(A.shape[0] - n_hold, A.shape[0])[:n_events]
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    chunks = sorted(RUN.glob(f"{a_prefix}*.pt"))
    seqs_per = torch.load(chunks[0], map_location="cpu",
                          weights_only=False)["A"].shape[0]
    total = len(chunks) * seqs_per
    ids_all, _, _ = pile_data.load_pile_blocks(
        tok, total, 512, seed=0, tokenizer_name=S67.TOKENIZER)
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", weights_only=False)["svd"]
    z = (A[idx].to(device) @ V)                                # [E, C]
    gen = torch.Generator(device=device).manual_seed(seed)
    results = {k: {"kl": [], "kl_rand": []} for k in ks}
    params = {p: model.get_submodule(p).weight for p in MODULES}
    sizes = fact["sizes"]

    def kept_forward(w_atom, ids_row, position):
        # subtract removed mass: delta_m = U diag(sigma * omega_m) V^T
        orig = {}
        j0 = 0
        with torch.no_grad():
            for p, sz in zip(MODULES, sizes):
                om = 1.0 - w_atom[j0:j0 + sz]                  # removed frac
                Um, Sm, Vm = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                              svd[p]["V"].to(device))
                delta = Um @ torch.diag(Sm * om) @ Vm.T
                orig[p] = params[p].data.clone()
                params[p].data -= delta
                j0 += sz
            lg = model(ids_row[None].to(device))[0, position].float()
            for p in MODULES:
                params[p].data.copy_(orig[p])
        return lg

    with torch.no_grad():
        for e, i in enumerate(idx):
            ids_row = ids_all[i]
            position = int(pos[i])
            base = model(ids_row[None].to(device))[0, position].float()
            order = z[e].abs().argsort(descending=True)
            for k in ks:
                kept = order[:k]
                w = V[:, kept].sum(1) + fact["r"].to(device)   # [J]
                lg = kept_forward(w, ids_row, position)
                kl = F.kl_div(F.log_softmax(lg, -1),
                              F.log_softmax(base, -1),
                              log_target=True, reduction="sum").item()
                rk = torch.randperm(C, device=device, generator=gen)[:k]
                wr = V[:, rk].sum(1) + fact["r"].to(device)
                lgr = kept_forward(wr, ids_row, position)
                klr = F.kl_div(F.log_softmax(lgr, -1),
                               F.log_softmax(base, -1),
                               log_target=True, reduction="sum").item()
                results[k]["kl"].append(kl)
                results[k]["kl_rand"].append(klr)
    out = {str(k): {"kl_mean": float(torch.tensor(v["kl"]).mean()),
                    "kl_rand_mean": float(torch.tensor(v["kl_rand"]).mean())}
           for k, v in results.items()}
    (RUN / out_name).write_text(json.dumps(out, indent=1))
    return out


def eval_klkeep_big(ks=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
                    n_events=96, n_files=None, seqs_per_file=8192,
                    device="cuda", seed=0, fact_path=None,
                    out_name="klkeep_big.json"):
    """KLKeep(k) for the 1M-event fit. Holdout rows live in the tail chunk
    files; only the file(s) covering the first n_events are loaded, and the
    token ids come from the shard that produced them (seed 100+shard), not
    the 16k cache."""
    fact = torch.load(fact_path or (BIG / "factorization_big.pt"),
                      map_location="cpu", weights_only=False)
    V = fact["V"].to(device)
    C = V.shape[1]
    chunks = sorted(BIG.glob("A_chunk*.pt"))
    if n_files is None:
        n_files = len(chunks)
    N_all = n_files * seqs_per_file
    n_hold = fact["n_hold"]
    assert n_hold >= n_events and n_events <= seqs_per_file
    row0 = N_all - n_hold                     # first holdout row
    f0 = row0 // seqs_per_file
    off = row0 % seqs_per_file
    assert off + n_events <= seqs_per_file, "eval span crosses a chunk file"
    d = torch.load(chunks[f0], map_location="cpu", weights_only=False)
    A_ev = d["A"][off:off + n_events].float()
    pos = d["pos"][off:off + n_events]
    files_per_shard = SHARD_BLOCKS // seqs_per_file
    sh = f0 // files_per_shard
    ids_all, _, _ = _load_shard(sh)
    i0 = (f0 % files_per_shard) * seqs_per_file + off
    ids_ev = ids_all[i0:i0 + n_events]

    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", weights_only=False)["svd"]
    if fact.get("V2ch") is not None:          # centered two-channel fit:
        An = A_ev / fact["g_rms"][None, :]    # rank z in its convention
        An = An / An.abs().sum(1, keepdim=True).clamp_min(1e-12)
        Ac = An - fact["mu"][None, :]
        A2 = torch.cat([Ac.clamp_min(0), (-Ac).clamp_min(0)], dim=1)
        z = A2.to(device) @ fact["V2ch"].to(device)            # [E, C]
    else:
        if fact.get("col_scale") is not None:  # atom-normalized fits
            A_ev = A_ev / fact["col_scale"][None, :]
        z = A_ev.to(device) @ V                                # [E, C]
    gen = torch.Generator(device=device).manual_seed(seed)
    results = {k: {"kl": [], "kl_rand": []} for k in ks}
    params = {p: model.get_submodule(p).weight for p in MODULES}
    sizes = fact["sizes"]
    r_dev = fact["r"].to(device)

    def kept_forward(w_atom, ids_row, position):
        orig = {}
        j0 = 0
        with torch.no_grad():
            for p, sz in zip(MODULES, sizes):
                om = 1.0 - w_atom[j0:j0 + sz]
                Um, Sm, Vm = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                              svd[p]["V"].to(device))
                delta = Um @ torch.diag(Sm * om) @ Vm.T
                orig[p] = params[p].data.clone()
                params[p].data -= delta
                j0 += sz
            lg = model(ids_row[None].to(device))[0, position].float()
            for p in MODULES:
                params[p].data.copy_(orig[p])
        return lg

    with torch.no_grad():
        for e in range(n_events):
            ids_row = ids_ev[e]
            position = int(pos[e])
            base = model(ids_row[None].to(device))[0, position].float()
            order = z[e].abs().argsort(descending=True)
            for k in ks:
                w = V[:, order[:k]].sum(1) + r_dev
                lg = kept_forward(w, ids_row, position)
                kl = F.kl_div(F.log_softmax(lg, -1),
                              F.log_softmax(base, -1),
                              log_target=True, reduction="sum").item()
                rk = torch.randperm(C, device=device, generator=gen)[:k]
                wr = V[:, rk].sum(1) + r_dev
                lgr = kept_forward(wr, ids_row, position)
                klr = F.kl_div(F.log_softmax(lgr, -1),
                               F.log_softmax(base, -1),
                               log_target=True, reduction="sum").item()
                results[k]["kl"].append(kl)
                results[k]["kl_rand"].append(klr)
    out = {str(k): {"kl_mean": float(torch.tensor(v["kl"]).mean()),
                    "kl_rand_mean": float(torch.tensor(v["kl_rand"]).mean())}
           for k, v in results.items()}
    (BIG / out_name).write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------- compare ----

def compare_vpd(n_tokens=24, device="cuda", seed=12345):
    """Sec-7 comparison vs VPD on shared held-out tokens.

    Weight-space per token t, module m (compare_vpd.py protocol):
      M_ours(t,m) = sum_{c in top-L0(t)} components, L0 matched to VPD's
      M_vpd(t,m)  = sum_{s active} U_s outer V_s
    cosine per module + pooled, with rand_ours / rand_vpd / shuffled nulls;
    plus usage alignment: corr over tokens between our |z_c| and their
    per-module active counts.
    """
    import compare_vpd as CV
    fact = torch.load(RUN / "factorization.pt", map_location="cpu",
                      weights_only=False)
    Vf = fact["V"].to(device)                                  # [J, C]
    C = Vf.shape[1]
    svd = torch.load(RUN / "atoms.pt", weights_only=False)["svd"]
    sizes = fact["sizes"]

    sd = torch.load(str(DATA / "vpd/model_400000.pth"), map_location="cpu",
                    weights_only=False)
    order = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    Uv = {m: sd[key[m] + ".U"].float().to(device) for m in MODULES}
    Vv = {m: sd[key[m] + ".V"].float().to(device) for m in MODULES}
    split = [Uv[n].shape[0] for n in order]
    ci = CV.CiFn({k: v.float().to(device) for k, v in sd.items()
                  if k.startswith("ci_fn")}, order, split).to(device).eval()

    model = load67(device, "plain")
    W0 = {m: model.get_submodule(m).weight.detach().clone() for m in MODULES}

    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_all, _, _ = pile_data.load_pile_blocks(
        tok, 64, 512, seed=99, tokenizer_name=S67.TOKENIZER)
    ids_all = ids_all.to(device)
    gsel = torch.Generator().manual_seed(seed)
    samp = [(int(torch.randint(0, ids_all.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, 510, (1,), generator=gsel)))
            for _ in range(n_tokens)]

    def our_z(b, t):
        """our per-token component usage via one backward (logodds)."""
        pre, post, hs = {}, {}, []
        for p in MODULES:
            mod = model.get_submodule(p)

            def hook(mm, inp, out, _p=p):
                pre[_p] = inp[0]
                out.retain_grad()
                post[_p] = out
            hs.append(mod.register_forward_hook(hook))
        lg = model(ids_all[b:b + 1])
        for h in hs:
            h.remove()
        at = lg[0, t].float()
        y = at.argmax()
        s = _logodds(at[None], y[None]).squeeze()
        grads = torch.autograd.grad(s, [post[p] for p in MODULES])
        a = []
        for p, g in zip(MODULES, grads):
            U, S, V = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                       svd[p]["V"].to(device))
            du = torch.einsum("td,dr->tr", g[0].float(), U)
            hv = torch.einsum("ti,ir->tr", pre[p][0].detach().float(), V)
            a.append(S * (du * hv).sum(0))
        return torch.cat(a) @ Vf                               # [C]

    @torch.no_grad()
    def our_mask(comp_set):
        """sum of selected components per module, as weight tensors."""
        w = Vf[:, comp_set].sum(1)                             # [J]
        out, j0 = {}, 0
        for p, sz in zip(MODULES, sizes):
            U, S, V = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                       svd[p]["V"].to(device))
            out[p] = U @ torch.diag(S * w[j0:j0 + sz]) @ V.T
            j0 += sz
        return out

    @torch.no_grad()
    def vpd_active(b, t):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        model(ids_all[b:b + 1])
        for h in hs:
            h.remove()
        logits = ci({m: pre[m] for m in MODULES})
        return {m: (logits[m][0, t] > 0) for m in MODULES}

    @torch.no_grad()
    def vpd_mask(act):
        # [d_in, n_act] @ [n_act, d_out] -> transpose to [d_out, d_in]
        return {m: (Vv[m][:, act[m]] @ Uv[m][act[m]]).T for m in MODULES}

    def cos(a, b):
        num = sum((a[m] * b[m]).sum() for m in MODULES)
        return float(num / (math.sqrt(sum(a[m].pow(2).sum().item()
                                          for m in MODULES)) *
                            math.sqrt(sum(b[m].pow(2).sum().item()
                                          for m in MODULES)) + 1e-12))

    gen = torch.Generator().manual_seed(seed + 1)
    rows = []
    prev_ours = None
    for b, t in samp:
        act = vpd_active(b, t)
        L0 = int(sum(a.sum().item() for a in act.values()))
        z = our_z(b, t)
        # match OUR count to a comparable budget: top-k with k = min(C, L0)
        k = max(1, min(C, L0))
        top = z.abs().argsort(descending=True)[:k]
        m_ours = our_mask(top)
        m_vpd = vpd_mask(act)
        rand_ours = our_mask(torch.randperm(C, generator=gen)[:k].to(device))
        row = {"b": b, "t": t, "L0_vpd": L0, "k_ours": int(k),
               "cos": cos(m_ours, m_vpd),
               "cos_rand_ours": cos(rand_ours, m_vpd)}
        if prev_ours is not None:
            row["cos_shuffled"] = cos(prev_ours, m_vpd)
        prev_ours = m_ours
        rows.append(row)
        print(json.dumps(row), flush=True)
    agg = {kk: float(torch.tensor([float(r[kk]) for r in rows if kk in r]).mean())
           for kk in ("cos", "cos_rand_ours", "cos_shuffled", "L0_vpd")}
    out = {"rows": rows, "mean": agg, "C_ours": C}
    (RUN / "compare_vpd67.json").write_text(json.dumps(out, indent=1))
    return {"mean": agg, "n_tokens": len(rows)}


def eval_oracle(ks=(8, 16, 32, 64, 128, 256, 512, 1024), n_events=96,
                device="cuda"):
    """Oracle keep-top-k: order components per event by TRUE single-component
    ablation effect |dlogp(y)|, then KLKeep(k) with that ordering."""
    A, y, pos, _ = _load_A()
    fact = torch.load(RUN / "factorization.pt", map_location="cpu",
                      weights_only=False)
    V = fact["V"].to(device)
    C = V.shape[1]
    sizes = fact["sizes"]
    n_hold = fact["n_hold"]
    idx = torch.arange(A.shape[0] - n_hold, A.shape[0])[:n_events]
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    chunks = sorted(RUN.glob("A_chunk*.pt"))
    seqs_per = torch.load(chunks[0], map_location="cpu",
                          weights_only=False)["A"].shape[0]
    ids_all, _, _ = pile_data.load_pile_blocks(
        tok, len(chunks) * seqs_per, 512, seed=0,
        tokenizer_name=S67.TOKENIZER)
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", weights_only=False)["svd"]
    params = {p: model.get_submodule(p).weight for p in MODULES}
    E = len(idx)
    ev_ids = ids_all[idx].to(device)                          # [E, 512]
    ev_pos = torch.tensor([int(pos[i]) for i in idx], device=device)
    ev_y = torch.tensor([int(y[i]) for i in idx], device=device)
    rows = torch.arange(E, device=device)

    orc_f = RUN / "oracle_imp.pt"
    if orc_f.exists():
        imp = torch.load(orc_f, map_location=device, weights_only=True)
    else:
        with torch.no_grad():
            base_lp = F.log_softmax(model(ev_ids)[rows, ev_pos].float(), -1
                                    ).gather(1, ev_y[:, None]).squeeze(1)
            imp = torch.zeros(E, C, device=device)
            for c in range(C):
                w = V[:, c]                                    # this comp only
                orig, j0 = {}, 0
                for p, sz in zip(MODULES, sizes):
                    U_, S_, V_ = (svd[p]["U"].to(device),
                                  svd[p]["S"].to(device),
                                  svd[p]["V"].to(device))
                    delta = U_ @ torch.diag(S_ * w[j0:j0 + sz]) @ V_.T
                    orig[p] = params[p].data.clone()
                    params[p].data -= delta
                    j0 += sz
                lp = F.log_softmax(model(ev_ids)[rows, ev_pos].float(), -1
                                   ).gather(1, ev_y[:, None]).squeeze(1)
                for p in MODULES:
                    params[p].data.copy_(orig[p])
                imp[:, c] = (base_lp - lp).abs()
                if c % 128 == 0:
                    print(f"oracle {c}/{C}", flush=True)
        torch.save(imp.cpu(), orc_f)

    def kept_forward(w_atom, ids_row, position):
        orig, j0 = {}, 0
        with torch.no_grad():
            for p, sz in zip(MODULES, sizes):
                om = 1.0 - w_atom[j0:j0 + sz]
                U_, S_, V_ = (svd[p]["U"].to(device), svd[p]["S"].to(device),
                              svd[p]["V"].to(device))
                delta = U_ @ torch.diag(S_ * om) @ V_.T
                orig[p] = params[p].data.clone()
                params[p].data -= delta
                j0 += sz
            lg = model(ids_row[None])[0, position].float()
            for p in MODULES:
                params[p].data.copy_(orig[p])
        return lg

    results = {k: [] for k in ks}
    with torch.no_grad():
        for e in range(E):
            base = model(ev_ids[e:e + 1])[0, ev_pos[e]].float()
            order = imp[e].argsort(descending=True)
            for k in ks:
                w = V[:, order[:k]].sum(1) + fact["r"].to(device)
                lg = kept_forward(w, ev_ids[e], int(ev_pos[e]))
                kl = F.kl_div(F.log_softmax(lg, -1),
                              F.log_softmax(base, -1),
                              log_target=True, reduction="sum").item()
                results[k].append(kl)
    out = {str(k): {"kl_mean": float(torch.tensor(v).mean())}
           for k, v in results.items()}
    (RUN / "klkeep_oracle.json").write_text(json.dumps(out, indent=1))
    return out


def eval_vpd_klkeep(pcts=(0.78125, 1.5625, 3.125, 6.25, 12.5, 25, 50, 100),
                    n_events=96, device="cuda", seed=0):
    """KLKeep at percent-kept for THEIR decomposition on the SAME events:
    rank their 38,912 subcomponents per token by CI, keep top p%, subtract
    the rest from W (their residual/delta stays automatically)."""
    import compare_vpd as CV
    A, y, pos, _ = _load_A()
    fact = torch.load(RUN / "factorization.pt", map_location="cpu",
                      weights_only=False)
    n_hold = fact["n_hold"]
    idx = torch.arange(A.shape[0] - n_hold, A.shape[0])[:n_events]
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    chunks = sorted(RUN.glob("A_chunk*.pt"))
    seqs_per = torch.load(chunks[0], map_location="cpu",
                          weights_only=False)["A"].shape[0]
    ids_all, _, _ = pile_data.load_pile_blocks(
        tok, len(chunks) * seqs_per, 512, seed=0,
        tokenizer_name=S67.TOKENIZER)
    model = load67(device, "plain")
    params = {p: model.get_submodule(p).weight for p in MODULES}

    sd = torch.load(str(DATA / "vpd/model_400000.pth"), map_location="cpu",
                    weights_only=False)
    order_m = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    Uv = {m: sd[key[m] + ".U"].float().to(device) for m in MODULES}
    Vv = {m: sd[key[m] + ".V"].float().to(device) for m in MODULES}
    split = [Uv[n].shape[0] for n in order_m]
    ci_net = CV.CiFn({k: v.float().to(device) for k, v in sd.items()
                      if k.startswith("ci_fn")}, order_m,
                     split).to(device).eval()
    n_sub = sum(split)
    offs, o = {}, 0
    for m in order_m:
        offs[m] = o
        o += Uv[m].shape[0]

    E = len(idx)
    ev_ids = ids_all[idx].to(device)
    ev_pos = torch.tensor([int(pos[i]) for i in idx], device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def ci_at(e):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        model(ev_ids[e:e + 1])
        for h in hs:
            h.remove()
        logits = ci_net({m: pre[m] for m in MODULES})
        return torch.cat([logits[m][0, int(ev_pos[e])] for m in order_m])

    @torch.no_grad()
    def kept_forward(keep_mask, e):
        orig = {}
        for m in order_m:
            sl = slice(offs[m], offs[m] + Uv[m].shape[0])
            rm = ~keep_mask[sl]
            orig[m] = params[m].data.clone()
            if rm.any():
                params[m].data -= (Vv[m][:, rm] @ Uv[m][rm]).T
        lg = model(ev_ids[e:e + 1])[0, int(ev_pos[e])].float()
        for m in order_m:
            params[m].data.copy_(orig[m])
        return lg

    results = {p: {"kl": [], "kl_rand": []} for p in pcts}
    with torch.no_grad():
        for e in range(E):
            base = model(ev_ids[e:e + 1])[0, int(ev_pos[e])].float()
            ci = ci_at(e)
            order = ci.argsort(descending=True)
            for p in pcts:
                k = max(1, int(round(n_sub * p / 100)))
                keep = torch.zeros(n_sub, dtype=torch.bool, device=device)
                keep[order[:k]] = True
                lg = kept_forward(keep, e)
                kl = F.kl_div(F.log_softmax(lg, -1),
                              F.log_softmax(base, -1),
                              log_target=True, reduction="sum").item()
                keep_r = torch.zeros(n_sub, dtype=torch.bool, device=device)
                keep_r[torch.randperm(n_sub, device=device,
                                      generator=gen)[:k]] = True
                klr = F.kl_div(
                    F.log_softmax(kept_forward(keep_r, e), -1),
                    F.log_softmax(base, -1),
                    log_target=True, reduction="sum").item()
                results[p]["kl"].append(kl)
                results[p]["kl_rand"].append(klr)
            if e % 16 == 0:
                print(f"vpd klkeep event {e}/{E}", flush=True)
    out = {str(p): {"kl_mean": float(torch.tensor(v["kl"]).mean()),
                    "kl_rand_mean": float(torch.tensor(v["kl_rand"]).mean())}
           for p, v in results.items()}
    (RUN / "klkeep_vpd.json").write_text(json.dumps(out, indent=1))
    return out


def eval_vpd_klkeep_big(pcts=(0.78125, 1.5625, 3.125, 6.25, 12.5, 25, 50,
                              100),
                        n_events=96, seqs_per_file=8192, device="cuda",
                        seed=0):
    """eval_vpd_klkeep on the SAME 96 holdout events as eval_klkeep_big
    (1M-run tail chunk, shard-seeded ids) so the percent-kept curves share
    an event set."""
    import compare_vpd as CV
    # only n_hold is needed; prefer the frozen ep12 snapshot so this never
    # races a concurrent fit_big rewriting factorization_big.pt
    fp = BIG / "factorization_big_ep12.pt"
    if not fp.exists():
        fp = BIG / "factorization_big.pt"
    fact = torch.load(fp, map_location="cpu", weights_only=False)
    n_hold = fact["n_hold"]
    chunks = sorted(BIG.glob("A_chunk*.pt"))
    N_all = len(chunks) * seqs_per_file
    row0 = N_all - n_hold
    f0 = row0 // seqs_per_file
    off = row0 % seqs_per_file
    assert off + n_events <= seqs_per_file, "eval span crosses a chunk file"
    d = torch.load(chunks[f0], map_location="cpu", weights_only=False)
    pos_ev = d["pos"][off:off + n_events]
    files_per_shard = SHARD_BLOCKS // seqs_per_file
    sh = f0 // files_per_shard
    ids_all, _, _ = _load_shard(sh)
    i0 = (f0 % files_per_shard) * seqs_per_file + off
    ids_ev = ids_all[i0:i0 + n_events]

    model = load67(device, "plain")
    params = {p: model.get_submodule(p).weight for p in MODULES}
    sd = torch.load(str(DATA / "vpd/model_400000.pth"), map_location="cpu",
                    weights_only=False)
    order_m = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    Uv = {m: sd[key[m] + ".U"].float().to(device) for m in MODULES}
    Vv = {m: sd[key[m] + ".V"].float().to(device) for m in MODULES}
    split = [Uv[n].shape[0] for n in order_m]
    ci_net = CV.CiFn({k: v.float().to(device) for k, v in sd.items()
                      if k.startswith("ci_fn")}, order_m,
                     split).to(device).eval()
    n_sub = sum(split)
    offs, o = {}, 0
    for m in order_m:
        offs[m] = o
        o += Uv[m].shape[0]

    E = n_events
    ev_ids = ids_ev.to(device)
    ev_pos = pos_ev.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def ci_at(e):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        model(ev_ids[e:e + 1])
        for h in hs:
            h.remove()
        logits = ci_net({m: pre[m] for m in MODULES})
        return torch.cat([logits[m][0, int(ev_pos[e])] for m in order_m])

    @torch.no_grad()
    def kept_forward(keep_mask, e):
        orig = {}
        for m in order_m:
            sl = slice(offs[m], offs[m] + Uv[m].shape[0])
            rm = ~keep_mask[sl]
            orig[m] = params[m].data.clone()
            if rm.any():
                params[m].data -= (Vv[m][:, rm] @ Uv[m][rm]).T
        lg = model(ev_ids[e:e + 1])[0, int(ev_pos[e])].float()
        for m in order_m:
            params[m].data.copy_(orig[m])
        return lg

    results = {p: {"kl": [], "kl_rand": []} for p in pcts}
    with torch.no_grad():
        for e in range(E):
            base = model(ev_ids[e:e + 1])[0, int(ev_pos[e])].float()
            ci = ci_at(e)
            order = ci.argsort(descending=True)
            for p in pcts:
                k = max(1, int(round(n_sub * p / 100)))
                keep = torch.zeros(n_sub, dtype=torch.bool, device=device)
                keep[order[:k]] = True
                lg = kept_forward(keep, e)
                kl = F.kl_div(F.log_softmax(lg, -1),
                              F.log_softmax(base, -1),
                              log_target=True, reduction="sum").item()
                keep_r = torch.zeros(n_sub, dtype=torch.bool, device=device)
                keep_r[torch.randperm(n_sub, device=device,
                                      generator=gen)[:k]] = True
                klr = F.kl_div(
                    F.log_softmax(kept_forward(keep_r, e), -1),
                    F.log_softmax(base, -1),
                    log_target=True, reduction="sum").item()
                results[p]["kl"].append(kl)
                results[p]["kl_rand"].append(klr)
            if e % 16 == 0:
                print(f"vpd klkeep_big event {e}/{E}", flush=True)
    out = {str(p): {"kl_mean": float(torch.tensor(v["kl"]).mean()),
                    "kl_rand_mean": float(torch.tensor(v["kl_rand"]).mean())}
           for p, v in results.items()}
    (BIG / "klkeep_vpd_big.json").write_text(json.dumps(out, indent=1))
    return out


def component_stats_big():
    """Weight-mass statistics of the C=4096 factorization, from the file
    alone (CPU). Atom mass = V columns (each atom row softmax-splits one
    unit across components + residual, so masses are in 'atoms'); ranks
    are exact because each component's per-module delta is diagonal in
    that module's SVD basis."""
    fact = torch.load(BIG / "factorization_big.pt", map_location="cpu",
                      weights_only=False)
    V, r, sizes = fact["V"].float(), fact["r"].float(), fact["sizes"]
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    sig = torch.cat([svd[p]["S"].float() for p in MODULES])       # [J]
    J, C = V.shape

    m = V.sum(0)                                                  # atom mass
    order = m.argsort(descending=True)
    cum = m[order].cumsum(0) / m.sum()

    def n_for(frac):
        return int((cum < frac).sum().item()) + 1

    pr_atoms = m.pow(2) / V.pow(2).sum(0).clamp_min(1e-12)        # eff #atoms
    # per-module effective rank (participation ratio within module block)
    mod_mass = torch.zeros(len(MODULES), C)
    mod_rank = torch.zeros(len(MODULES), C)
    j0 = 0
    for mi, sz in enumerate(sizes):
        blk = V[j0:j0 + sz]
        mod_mass[mi] = blk.sum(0)
        mod_rank[mi] = blk.sum(0).pow(2) / blk.pow(2).sum(0).clamp_min(1e-12)
        j0 += sz
    mshare = mod_mass / mod_mass.sum(0, keepdim=True).clamp_min(1e-12)
    n_mod = (mshare > 0.05).sum(0).float()                        # modules >5%
    # mass-weighted mean effective rank across modules, per component
    eff_rank = (mshare * mod_rank).sum(0)

    fro2 = (V * sig[:, None].pow(2)).sum(0)                       # ||delta||_F^2
    fro_share = fro2 / sig.pow(2).sum()
    r_fro_share = float((r * sig.pow(2)).sum() / sig.pow(2).sum())

    def q(t, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
        return {str(p): round(float(torch.quantile(t, p)), 3) for p in ps}

    top = [{"component": int(c), "atom_mass": round(float(m[c]), 1),
            "eff_atoms": round(float(pr_atoms[c]), 1),
            "eff_rank_per_module": round(float(eff_rank[c]), 1),
            "n_modules": int(n_mod[c]),
            "fro_share": round(float(fro_share[c]), 5)}
           for c in order[:16].tolist()]
    out = {"J": J, "C": C,
           "residual_atom_mass": round(float(r.sum()), 1),
           "residual_fro_share": round(r_fro_share, 5),
           "components_for_50pct_mass": n_for(0.5),
           "components_for_90pct_mass": n_for(0.9),
           "components_for_99pct_mass": n_for(0.99),
           "n_components_over_1_atom": int((m > 1.0).sum()),
           "n_components_over_0p1_atom": int((m > 0.1).sum()),
           "atom_mass_quantiles": q(m),
           "eff_atoms_quantiles": q(pr_atoms),
           "eff_rank_per_module_quantiles": q(eff_rank),
           "n_modules_quantiles": q(n_mod),
           "mean_eff_rank_mass_weighted": round(
               float((eff_rank * m).sum() / m.sum()), 2),
           "top16_by_mass": top}
    (BIG / "component_stats.json").write_text(json.dumps(out, indent=1))
    return out


def interp_report_big(n_show=40, top_events=12, sample_events=8,
                      device="cuda", seed=0):
    """interp_report for the 1M/C=4096 run: usage streamed chunk-by-chunk,
    contexts decoded from the shard caches, plus per-token committee stats
    (effective components per token, top-16 cumulative share)."""
    from collections import Counter
    fact = torch.load(BIG / "factorization_big.pt", map_location="cpu",
                      weights_only=False)
    V = fact["V"].float().to(device)
    sizes = fact["sizes"]
    C = V.shape[1]
    chunks = sorted(BIG.glob("A_chunk*.pt"))
    K = top_events
    tot = torch.zeros(C, device=device)
    top_v = torch.full((K, C), -1.0, device=device)
    top_i = torch.zeros(K, C, dtype=torch.long, device=device)
    ys, poss, labels = [], [], []
    ent_sum, t16_sum, n_ev = 0.0, 0.0, 0
    for ci, f in enumerate(chunks):
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = (d["A"].to(device).float() @ V).abs()                # [B, C]
        tot += z.sum(0)
        v, i = z.topk(K, dim=0)                                  # [K, C]
        gi = i + ci * z.shape[0]
        top_v, sel = torch.cat([top_v, v]).topk(K, dim=0)
        top_i = torch.cat([top_i, gi]).gather(0, sel)
        sh = z / z.sum(1, keepdim=True).clamp_min(1e-12)
        ent_sum += float(torch.exp(
            -(sh * sh.clamp_min(1e-12).log()).sum(1)).sum())
        t16_sum += float(sh.topk(16, dim=1).values.sum(1).sum())
        n_ev += z.shape[0]
        ys.append(d["y"])
        poss.append(d["pos"])
        labels += list(d["labels"])
        if ci % 16 == 0:
            print(f"usage chunk {ci}/{len(chunks)}", flush=True)
    y = torch.cat(ys)
    pos = torch.cat(poss)

    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    shard_ids = {}

    def ids_of(g):
        ci = g // 8192
        sh = ci // (SHARD_BLOCKS // 8192)
        if sh not in shard_ids:
            shard_ids[sh] = _load_shard(sh)[0]
        off = (ci % (SHARD_BLOCKS // 8192)) * 8192 + g % 8192
        return shard_ids[sh][off]

    foot = torch.zeros(len(MODULES), C)
    Vc = V.cpu()
    j0 = 0
    for mi, sz in enumerate(sizes):
        foot[mi] = Vc[j0:j0 + sz].sum(0)
        j0 += sz
    foot = foot / foot.sum(0, keepdim=True).clamp_min(1e-12)

    base_counts = Counter(labels)
    N = len(labels)
    order = tot.argsort(descending=True).cpu()
    g = torch.Generator().manual_seed(seed)
    picks = list(order[:n_show - 16].tolist()) + \
        list(order[torch.randint(200, C, (16,), generator=g)].tolist())
    top_i_c, top_v_c = top_i.cpu(), top_v.cpu()
    report, stats_sel, stats_tok = [], [], []
    for c in picks:
        tops = top_i_c[:, c].tolist()
        subs = [labels[i] for i in tops]
        toks = [tok.decode([int(y[i])]) for i in tops]
        sc, cnt = Counter(subs).most_common(1)[0]
        sel = (cnt / K) / max(base_counts[sc] / N, 1e-9)
        tc, tcnt = Counter(toks).most_common(1)[0]
        stats_sel.append(cnt / K)
        stats_tok.append(tcnt / K)
        exs = []
        for i in tops[:5]:
            p = int(pos[i])
            ctx = tok.decode(ids_of(i)[max(0, p - 12):p + 1].tolist())
            exs.append({"subset": labels[i], "ctx": ctx[-90:],
                        "pred": tok.decode([int(y[i])])})
        fm = foot[:, c]
        fmod = [(MODULES[i], round(float(fm[i]), 3))
                for i in fm.argsort(descending=True)[:3].tolist()]
        report.append({"component": int(c),
                       "usage_share": round(float(tot[c] / tot.sum()), 5),
                       "top_subset": f"{sc} ({cnt}/{K}, {sel:.1f}x base)",
                       "top_pred_tok": f"{tc!r} ({tcnt}/{K})",
                       "module_footprint": fmod,
                       "examples": exs})

    # per-token committees on a few holdout events
    n_hold = fact["n_hold"]
    ev_rows = torch.arange(n_ev - n_hold, n_ev - n_hold + sample_events)
    committees = []
    for gidx in ev_rows.tolist():
        ci, off = gidx // 8192, gidx % 8192
        d = torch.load(chunks[ci], map_location="cpu", weights_only=False)
        zrow = (d["A"][off:off + 1].to(device).float() @ V).abs()[0]
        shr = zrow / zrow.sum().clamp_min(1e-12)
        vals, idxs = shr.topk(16)
        p = int(d["pos"][off])
        ctx = tok.decode(ids_of(gidx)[max(0, p - 12):p + 1].tolist())
        committees.append({
            "ctx": ctx[-90:], "pred": tok.decode([int(d["y"][off])]),
            "top16": [(int(i), round(float(v), 4))
                      for i, v in zip(idxs.tolist(), vals.tolist())],
            "top16_cum_share": round(float(vals.sum()), 4)})

    srt = tot.sort(descending=True).values
    out = {"components": report,
           "frac_subset_selective": float(
               (torch.tensor(stats_sel) >= 0.75).float().mean()),
           "frac_token_concentrated": float(
               (torch.tensor(stats_tok) >= 0.5).float().mean()),
           "usage_share_top": {str(k): round(
               float(srt[:k].sum() / srt.sum()), 4)
               for k in (1, 2, 4, 16, 64, 256)},
           "mean_effective_components_per_token": round(ent_sum / n_ev, 2),
           "mean_top16_share_per_token": round(t16_sum / n_ev, 4),
           "sample_committees": committees}
    (BIG / "interp_report_big.json").write_text(json.dumps(out, indent=1))
    return out


def component_examples_big(top_events=15, mass_min=0.1, ctx_tokens=32,
                           device="cuda"):
    """Raw top-activating text per component, no scoring lens: for every
    component holding > mass_min atoms, the top_events events ranked by
    that component's SHARE of the event's usage (distinctive activation,
    not global event size), decoded with a longer context window."""
    fact = torch.load(BIG / "factorization_big.pt", map_location="cpu",
                      weights_only=False)
    Vc = fact["V"].float()
    m = Vc.sum(0)
    sel = (m > mass_min).nonzero().flatten()
    V = Vc.to(device)
    sel_dev = sel.to(device)
    Cs = sel.numel()
    K = top_events
    chunks = sorted(BIG.glob("A_chunk*.pt"))
    top_v = torch.full((K, Cs), -1.0, device=device)
    top_i = torch.zeros(K, Cs, dtype=torch.long, device=device)
    ys, poss, labels = [], [], []
    usage = torch.zeros(V.shape[1], device=device)
    for ci, f in enumerate(chunks):
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = (d["A"].to(device).float() @ V).abs()
        usage += z.sum(0)
        sh = (z / z.sum(1, keepdim=True).clamp_min(1e-12))[:, sel_dev]
        v, i = sh.topk(K, dim=0)
        gi = i + ci * z.shape[0]
        top_v, sidx = torch.cat([top_v, v]).topk(K, dim=0)
        top_i = torch.cat([top_i, gi]).gather(0, sidx)
        ys.append(d["y"])
        poss.append(d["pos"])
        labels += list(d["labels"])
        if ci % 16 == 0:
            print(f"chunk {ci}/{len(chunks)}", flush=True)
    y = torch.cat(ys)
    pos = torch.cat(poss)
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    shard_ids = {}

    def ids_of(g):
        ci = g // 8192
        s = ci // (SHARD_BLOCKS // 8192)
        if s not in shard_ids:
            shard_ids[s] = _load_shard(s)[0]
        off = (ci % (SHARD_BLOCKS // 8192)) * 8192 + g % 8192
        return shard_ids[s][off]

    top_i_c, top_v_c = top_i.cpu(), top_v.cpu()
    usage = usage.cpu()
    out = []
    order = m[sel].argsort(descending=True)
    for oi in order.tolist():
        c = int(sel[oi])
        exs = []
        for r in range(K):
            g = int(top_i_c[r, oi])
            p = int(pos[g])
            ctx = tok.decode(ids_of(g)[max(0, p - ctx_tokens):p + 1].tolist())
            exs.append({"share": round(float(top_v_c[r, oi]), 4),
                        "subset": labels[g], "ctx": ctx[-160:],
                        "pred": tok.decode([int(y[g])])})
        out.append({"component": c,
                    "atom_mass": round(float(m[c]), 2),
                    "usage_share": round(float(usage[c] / usage.sum()), 5),
                    "examples": exs})
    (BIG / "component_examples.json").write_text(
        json.dumps({"components": out}, indent=1))
    return {"n_components": len(out)}


def interp_report(n_show=48, top_events=12, device="cpu"):
    """Interpretability profiles per component from collected data only:
    top-usage events (decoded context, predicted token, Pile subset),
    subset selectivity, predicted-token concentration, module footprint."""
    from collections import Counter
    A, y, pos, _ = _load_A()
    fact = torch.load(RUN / "factorization.pt", map_location="cpu",
                      weights_only=False)
    V = fact["V"].float()
    sizes = fact["sizes"]
    C = V.shape[1]
    labels = []
    for f in sorted(RUN.glob("A_chunk*.pt")):
        labels += list(torch.load(f, map_location="cpu",
                                  weights_only=False)["labels"])
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_all, _, _ = pile_data.load_pile_blocks(
        tok, A.shape[0], 512, seed=0, tokenizer_name=S67.TOKENIZER)

    # usage, chunked
    zs = []
    for i in range(0, A.shape[0], 2048):
        zs.append((A[i:i + 2048].float() @ V).abs())
    z = torch.cat(zs)                                          # [N, C]
    share = z / z.sum(1, keepdim=True).clamp_min(1e-12)
    tot = z.sum(0)

    # module footprint per component from V mass
    foot = torch.zeros(len(MODULES), C)
    j0 = 0
    for mi, sz in enumerate(sizes):
        foot[mi] = V[j0:j0 + sz].sum(0)
        j0 += sz
    foot = foot / foot.sum(0, keepdim=True).clamp_min(1e-12)

    base_counts = Counter(labels)
    N = len(labels)
    order = tot.argsort(descending=True)
    picks = list(order[:n_show - 16].tolist()) + \
        list(order[torch.randint(200, C, (16,))].tolist())
    report, stats_sel, stats_tok = [], [], []
    for c in picks:
        top = z[:, c].argsort(descending=True)[:top_events]
        subs = [labels[i] for i in top.tolist()]
        toks = [tok.decode([int(y[i])]) for i in top.tolist()]
        # selectivity: max subset share among top events vs base rate
        sc, cnt = Counter(subs).most_common(1)[0]
        sel = (cnt / top_events) / max(base_counts[sc] / N, 1e-9)
        tc, tcnt = Counter(toks).most_common(1)[0]
        stats_sel.append(cnt / top_events)
        stats_tok.append(tcnt / top_events)
        exs = []
        for i in top[:5].tolist():
            p = int(pos[i])
            ctx = tok.decode(ids_all[i, max(0, p - 12):p + 1].tolist())
            nxt = tok.decode([int(y[i])])
            exs.append({"subset": labels[i], "ctx": ctx[-90:], "pred": nxt})
        fm = foot[:, c]
        fmod = [(MODULES[i], round(float(fm[i]), 3))
                for i in fm.argsort(descending=True)[:3].tolist()]
        report.append({"component": int(c),
                       "usage_share": round(float(tot[c] / tot.sum()), 5),
                       "top_subset": f"{sc} ({cnt}/{top_events}, "
                                     f"{sel:.1f}x base)",
                       "top_pred_tok": f"{tc!r} ({tcnt}/{top_events})",
                       "module_footprint": fmod,
                       "examples": exs})
    out = {"components": report,
           "frac_subset_selective": float(
               (torch.tensor(stats_sel) >= 0.75).float().mean()),
           "frac_token_concentrated": float(
               (torch.tensor(stats_tok) >= 0.5).float().mean())}
    (RUN / "interp_report.json").write_text(json.dumps(out, indent=1))
    return {"n_profiled": len(report),
            "frac_subset_selective(>=75% one subset)":
                out["frac_subset_selective"],
            "frac_token_concentrated(>=50% one predicted token)":
                out["frac_token_concentrated"]}


BIG = RUN.parent / "cofac67_big"
SHARD_BLOCKS = 131_072          # 16 files x 8192; 8 shards = 1,048,576 events
N_SHARDS = 8


def prestage_shard(shard_id: int, seq=512):
    """Tokenize ONE 262k-block stratified shard (seed 100+shard). 1/4 the
    memory peak and scan length of the failed monolith; retryable."""
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids, labels, _ = pile_data.load_pile_blocks(
        tok, SHARD_BLOCKS, seq, seed=100 + shard_id, max_docs=1_200_000,
        tokenizer_name=S67.TOKENIZER)
    return {"shard": shard_id, "blocks": int(ids.shape[0]),
            "subsets": len(set(labels))}


def _load_shard(shard_id):
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    return pile_data.load_pile_blocks(
        tok, SHARD_BLOCKS, 512, seed=100 + shard_id, max_docs=1_200_000,
        tokenizer_name=S67.TOKENIZER, verbose=False)


def prestage_tokens(total_blocks=1_000_000, seq=512):
    """Tokenize the stratified Pile sample ONCE (CPU job) into the shared
    cache; collect chunks then hit the cache."""
    import pile_data
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids, labels, stats = pile_data.load_pile_blocks(
        tok, total_blocks, seq, seed=0, max_docs=4_000_000,
        tokenizer_name=S67.TOKENIZER)
    return {"blocks": int(ids.shape[0]), "seq": seq,
            "subsets": len(set(labels))}


def collect_chunk_big(chunk_id: int, n_chunks: int, seqs_per_chunk=2048,
                      batch=64, bf16=True, total_blocks=1_000_000,
                      device="cuda"):
    """Tuned resumable collection into cofac67_big/. Auto-halves batch on OOM."""
    BIG.mkdir(parents=True, exist_ok=True)
    out_f = BIG / f"A_chunk{chunk_id:05d}.pt"
    if out_f.exists():
        return {"chunk": chunk_id, "cached": True}
    import pile_data, time
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_all, labels, _ = pile_data.load_pile_blocks(
        tok, total_blocks, 512, seed=0, max_docs=4_000_000,
        tokenizer_name=S67.TOKENIZER, verbose=False)
    sl = slice(chunk_id * seqs_per_chunk, (chunk_id + 1) * seqs_per_chunk)
    ids_all, labels = ids_all[sl], labels[sl]
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    gen = torch.Generator().manual_seed(50_000 + chunk_id)
    pos_all = torch.randint(64, 511, (ids_all.shape[0],), generator=gen)
    rowsA, rowsY = [], []
    t0 = time.time()
    i = 0
    while i < ids_all.shape[0]:
        try:
            A, y = _collect_batch(model, svd,
                                  ids_all[i:i + batch].to(device),
                                  pos_all[i:i + batch].to(device),
                                  device, bf16=bf16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch = max(8, batch // 2)
            continue
        rowsA.append(A.cpu().half())
        rowsY.append(y.cpu())
        i += batch
    torch.save({"A": torch.cat(rowsA), "y": torch.cat(rowsY),
                "pos": pos_all, "labels": labels}, out_f)
    dt = time.time() - t0
    return {"chunk": chunk_id, "rows": int(sum(a.shape[0] for a in rowsA)),
            "secs": round(dt, 1), "batch_final": batch,
            "ms_per_event": round(1000 * dt / ids_all.shape[0], 1)}


def bench_collect(device="cuda"):
    """Tuning benchmark + numerics gate: fp32 vs bf16 on identical events."""
    import pile_data, time
    pile_data.CACHE = RUN / "piledata"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids, _, _ = pile_data.load_pile_blocks(tok, 256, 512, seed=3,
                                           tokenizer_name=S67.TOKENIZER)
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    gen = torch.Generator().manual_seed(1)
    pos = torch.randint(64, 511, (256,), generator=gen).to(device)
    ids = ids.to(device)
    name = torch.cuda.get_device_name()
    out = {"gpu": name}
    A_ref, _ = _collect_batch(model, svd, ids[:64], pos[:64], device,
                              bf16=False)
    A_bf, _ = _collect_batch(model, svd, ids[:64], pos[:64], device,
                             bf16=True)
    num = (A_ref - A_bf).norm() / A_ref.norm()
    corr = torch.nn.functional.cosine_similarity(
        A_ref.flatten(), A_bf.flatten(), dim=0)
    out["bf16_relerr"] = float(num)
    out["bf16_cos"] = float(corr)
    for bf16 in (False, True):
        for batch in (16, 32, 64):
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                n = 0
                for i in range(0, 192, batch):
                    _collect_batch(model, svd, ids[i:i + batch],
                                   pos[i:i + batch], device, bf16=bf16)
                    n += min(batch, 192 - i)
                torch.cuda.synchronize()
                out[f"ms_ev_b{batch}_{'bf16' if bf16 else 'fp32'}"] = \
                    round(1000 * (time.time() - t0) / n, 1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out[f"ms_ev_b{batch}_{'bf16' if bf16 else 'fp32'}"] = "OOM"
    return out


def collect_span(start_file: int, end_file: int, seqs_per_file=8192,
                 batch=64, total_blocks=1_000_000, device="cuda"):
    """Process file-chunks [start_file, end_file) in ONE container: model,
    SVD and the token cache load once, amortized over ~10 min of compute.
    fp32 (bf16 failed the numerics gate at 10% relerr). Resumable per file."""
    import pile_data, time
    BIG.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(start_file, end_file)
            if not (BIG / f"A_chunk{i:05d}.pt").exists()]
    if not todo:
        return {"span": [start_file, end_file], "cached": True}
    model = load67(device, "plain")
    svd = torch.load(RUN / "atoms.pt", map_location="cpu",
                     weights_only=False)["svd"]
    files_per_shard = SHARD_BLOCKS // seqs_per_file
    cur_shard, ids_all, labels_all = -1, None, None
    done = []
    for ci in todo:
        sh = ci // files_per_shard
        if sh != cur_shard:
            ids_all, labels_all, _ = _load_shard(sh)
            cur_shard = sh
        off = (ci % files_per_shard) * seqs_per_file
        sl = slice(off, off + seqs_per_file)
        ids, labels = ids_all[sl], labels_all[sl]
        if ids.shape[0] == 0:
            break
        gen = torch.Generator().manual_seed(50_000 + ci)
        pos = torch.randint(64, 511, (ids.shape[0],), generator=gen)
        rowsA, rowsY = [], []
        t0 = time.time()
        i, b = 0, batch
        while i < ids.shape[0]:
            try:
                A, y = _collect_batch(model, svd, ids[i:i + b].to(device),
                                      pos[i:i + b].to(device), device,
                                      bf16=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                b = max(8, b // 2)
                continue
            rowsA.append(A.cpu().half())
            rowsY.append(y.cpu())
            i += b
        torch.save({"A": torch.cat(rowsA), "y": torch.cat(rowsY),
                    "pos": pos, "labels": labels},
                   BIG / f"A_chunk{ci:05d}.pt")
        done.append({"file": ci, "secs": round(time.time() - t0, 1)})
        print(json.dumps(done[-1]), flush=True)
    return {"span": [start_file, end_file], "files_done": len(done),
            "detail": done[-3:]}


# --------------------------------------------------- scalable fit (1M) ----

def _load_A_big():
    chunks = sorted(BIG.glob("A_chunk*.pt"))
    assert chunks, "no big chunks"
    As, ys, poss = [], [], []
    for f in chunks:
        d = torch.load(f, map_location="cpu", weights_only=False)
        As.append(d["A"])                      # keep fp16 on cpu
        ys.append(d["y"])
        poss.append(d["pos"])
    return torch.cat(As), torch.cat(ys), torch.cat(poss), len(chunks)


def fit_big(k_factors=8192, c_groups=4096, epochs=12, lr=2e-2,
            row_batch=16384, seed=0, holdout_frac=0.0625, device="cuda",
            atom_norm="none", out_name=None):
    """Alternating minibatched v2 fit for N up to ~1M.

    U rows are independent given (S, V) under the I-div objective, so each
    epoch sweeps row-minibatches: the batch's U rows and the global (S, V)
    take an Adam step together, but U's optimizer state lives per-row on
    CPU and is swapped in with the batch. Layer-RMS + row normalization as
    in fit().

    atom_norm: per-ATOM column rescaling applied before everything else
    (proposal sec 4.3), targeting the shared static magnitude profile that
    feeds the mega-component:
      'dir'    -- divide out sigma_q (direction-normalized attribution
                  u^T grad v: sensitivity to the direction, not the mass
                  stored in it);
      'fisher' -- divide each atom by its RMS attribution over the fit
                  split (diagonal-Fisher-style: attribution relative to how
                  sensitive that atom typically is);
      'none'   -- original behavior."""
    A16, y, pos, n_chunks = _load_A_big()
    N_all = A16.shape[0]
    n_hold = int(N_all * holdout_frac)
    N = N_all - n_hold
    svdblob = torch.load(RUN / "atoms.pt", map_location="cpu",
                         weights_only=False)
    sizes = [svdblob["svd"][p]["S"].numel() for p in MODULES]
    J = sum(sizes)
    if atom_norm == "dir":
        col_scale = torch.cat([svdblob["svd"][p]["S"]
                               for p in MODULES]).float().clamp_min(1e-12)
    elif atom_norm == "fisher":
        cs = torch.zeros(J, device=device)
        for i in range(0, N, 65536):
            blk = A16[i:min(i + 65536, N)].to(device).float()
            cs += blk.pow(2).sum(0)
        col_scale = (cs / N).sqrt().clamp_min(1e-8).cpu()
    else:
        col_scale = torch.ones(J)
    cs_dev = col_scale.to(device)
    # per-group RMS over the FIT split, computed streaming on gpu
    g_rms, j0 = torch.zeros(J), 0
    for sz in sizes:
        acc = 0.0
        for i in range(0, N, 65536):
            blk = (A16[i:i + 65536, j0:j0 + sz].to(device).float()
                   / cs_dev[j0:j0 + sz])
            acc += blk.pow(2).sum().item()
        g_rms[j0:j0 + sz] = math.sqrt(acc / (N * sz)) + 1e-12
        j0 += sz

    gen = torch.Generator().manual_seed(seed)
    Wu_all = (torch.rand(N, k_factors, generator=gen) * 0.5 + 0.2)  # cpu fp32
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Wv = (torch.randn(J, c_groups + 1, generator=gen) * 0.05
          ).to(device).requires_grad_()
    opt_sv = torch.optim.Adam([Ws, Wv], lr=lr)
    u_m = torch.zeros_like(Wu_all)             # per-row Adam state, cpu
    u_v = torch.zeros_like(Wu_all)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_count = 0
    hist = []
    rms_dev = g_rms.to(device)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=gen)
        ep_loss, ep_n = 0.0, 0
        for bi in range(0, N, row_batch):
            rows = perm[bi:bi + row_batch]
            M = (A16[rows].to(device).float() / cs_dev).abs() / rms_dev
            M_bar = M / M.sum(1, keepdim=True).clamp_min(1e-12)
            Wu = Wu_all[rows].to(device).requires_grad_()
            U = torch.softmax(Wu, dim=1)
            S = F.softplus(Ws)
            Vfull = torch.softmax(Wv, dim=1)
            V = Vfull[:, :c_groups]
            M_hat = U @ S @ V.T
            mass = M_bar.sum().clamp_min(1e-8)
            loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                    - M_bar + M_hat).sum() / mass
            opt_sv.zero_grad(set_to_none=True)
            loss.backward()
            opt_sv.step()
            with torch.no_grad():              # manual Adam for the U rows
                g = Wu.grad
                m = u_m[rows].to(device)
                v = u_v[rows].to(device)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                step_count += 1
                mh = m / (1 - beta1 ** step_count)
                vh = v / (1 - beta2 ** step_count)
                Wu_new = Wu.detach() - lr * mh / (vh.sqrt() + eps)
                Wu_all[rows] = Wu_new.cpu()
                u_m[rows] = m.cpu()
                u_v[rows] = v.cpu()
            ep_loss += loss.item() * rows.numel()
            ep_n += rows.numel()
        hist.append(f"epoch {ep} idiv/row {ep_loss / ep_n:.5f}")
        print(hist[-1], flush=True)
    with torch.no_grad():
        Vfull = torch.softmax(Wv, dim=1)
    out = BIG / (out_name or "factorization_big.pt")
    torch.save({"V": Vfull[:, :c_groups].cpu(), "r": Vfull[:, -1].cpu(),
                "S": F.softplus(Ws).detach().cpu(),
                "g_rms": g_rms, "sizes": sizes, "n_hold": n_hold,
                "col_scale": (col_scale if atom_norm != "none" else None),
                "config": {"K": k_factors, "C": c_groups, "epochs": epochs,
                           "N_fit": N, "atom_norm": atom_norm}}, out)
    return {"N_fit": N, "J": J, "C": c_groups, "out": str(out),
            "hist": hist[-4:]}


def fit_big_pinned(k_factors=8192, c_groups=4096, epochs=12, lr=2e-2,
                   row_batch=16384, seed=0, holdout_frac=0.0625,
                   device="cuda", f_cap=0.9, out_name=None):
    """fit_big with the v3b PINNED backbone: the rank-1 LS backbone (per-
    event coefficient a on the mean direction, per-atom loading beta) is
    estimated streaming over the fit split, each atom's backbone fraction f
    is pinned at <= f_cap, V = (1-f) * softmax distributes only the
    remainder, and the per-event backbone scale a is trained with the same
    per-row CPU Adam as the U rows. Drain is impossible."""
    A16, y, pos, n_chunks = _load_A_big()
    N_all = A16.shape[0]
    n_hold = int(N_all * holdout_frac)
    N = N_all - n_hold
    svdblob = torch.load(RUN / "atoms.pt", map_location="cpu",
                         weights_only=False)
    sizes = [svdblob["svd"][p]["S"].numel() for p in MODULES]
    J = sum(sizes)
    g_rms, j0 = torch.zeros(J), 0
    for sz in sizes:
        acc = 0.0
        for i in range(0, N, 65536):
            blk = A16[i:min(i + 65536, N), j0:j0 + sz].to(device).float()
            acc += blk.pow(2).sum().item()
        g_rms[j0:j0 + sz] = math.sqrt(acc / (N * sz)) + 1e-12
        j0 += sz
    rms_dev = g_rms.to(device)

    def mbar_block(i, n=65536):
        M = A16[i:min(i + n, N)].to(device).float().abs() / rms_dev
        return M / M.sum(1, keepdim=True).clamp_min(1e-12)

    # streaming rank-1 LS backbone over the fit split
    colsum = torch.zeros(J, device=device)
    for i in range(0, N, 65536):
        colsum += mbar_block(i).sum(0)
    mu = colsum / N
    u = mu / mu.norm().clamp_min(1e-12)
    a0_all = torch.zeros(N)
    beta_acc = torch.zeros(J, device=device)
    a0_sq = a0_sum = 0.0
    for i in range(0, N, 65536):
        Mb = mbar_block(i)
        a0 = Mb @ u
        beta_acc += Mb.T @ a0
        a0_all[i:i + a0.numel()] = a0.cpu()
        a0_sq += float(a0.pow(2).sum())
        a0_sum += float(a0.sum())
    beta = (beta_acc / max(a0_sq, 1e-12)).clamp_min(0)
    f = (beta * a0_sum / colsum.clamp_min(1e-12)).clamp(0, f_cap)
    back_share = float(a0_sum * beta.sum() / colsum.sum())
    print(f"pinned backbone: mean f {f.mean():.3f}  "
          f"mass share {back_share:.3f}", flush=True)

    gen = torch.Generator().manual_seed(seed)
    Wu_all = (torch.rand(N, k_factors, generator=gen) * 0.5 + 0.2)  # cpu
    Wa_all = torch.log(torch.expm1(a0_all.clamp_min(1e-6)))
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Wv = (torch.randn(J, c_groups, generator=gen) * 0.05
          ).to(device).requires_grad_()
    opt_sv = torch.optim.Adam([Ws, Wv], lr=lr)
    u_m = torch.zeros_like(Wu_all)
    u_v = torch.zeros_like(Wu_all)
    a_m = torch.zeros(N)
    a_v = torch.zeros(N)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    keep1f = (1.0 - f)[:, None]
    step_count = 0
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(N, generator=gen)
        ep_loss, ep_n = 0.0, 0
        for bi in range(0, N, row_batch):
            rows = perm[bi:bi + row_batch]
            M = A16[rows].to(device).float().abs() / rms_dev
            M_bar = M / M.sum(1, keepdim=True).clamp_min(1e-12)
            Wu = Wu_all[rows].to(device).requires_grad_()
            Wa = Wa_all[rows].to(device).requires_grad_()
            U = torch.softmax(Wu, dim=1)
            a = F.softplus(Wa)
            S = F.softplus(Ws)
            V = keep1f * torch.softmax(Wv, dim=1)
            M_hat = U @ S @ V.T + a[:, None] * beta[None, :]
            mass = M_bar.sum().clamp_min(1e-8)
            loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                    - M_bar + M_hat).sum() / mass
            opt_sv.zero_grad(set_to_none=True)
            loss.backward()
            opt_sv.step()
            with torch.no_grad():              # manual Adam for U and a rows
                step_count += 1
                for W, all_t, m_all, v_all in (
                        (Wu, Wu_all, u_m, u_v), (Wa, Wa_all, a_m, a_v)):
                    g = W.grad
                    m = m_all[rows].to(device)
                    v = v_all[rows].to(device)
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    mh = m / (1 - beta1 ** step_count)
                    vh = v / (1 - beta2 ** step_count)
                    all_t[rows] = (W.detach()
                                   - lr * mh / (vh.sqrt() + eps)).cpu()
                    m_all[rows] = m.cpu()
                    v_all[rows] = v.cpu()
            ep_loss += loss.item() * rows.numel()
            ep_n += rows.numel()
        hist.append(f"epoch {ep} idiv/row {ep_loss / ep_n:.5f}")
        print(hist[-1], flush=True)
    with torch.no_grad():
        V = (keep1f * torch.softmax(Wv, dim=1)).cpu()
    out = BIG / (out_name or "factorization_big_pinned.pt")
    torch.save({"V": V, "r": f.cpu(), "S": F.softplus(Ws).detach().cpu(),
                "beta": beta.cpu(), "a": F.softplus(Wa_all).cpu(),
                "g_rms": g_rms, "sizes": sizes, "n_hold": n_hold,
                "config": {"K": k_factors, "C": c_groups, "epochs": epochs,
                           "N_fit": N, "pinned": True, "f_cap": f_cap}}, out)
    return {"N_fit": N, "J": J, "C": c_groups, "back_share": back_share,
            "mean_f": float(f.mean()), "out": str(out), "hist": hist[-4:]}


def fit_big_2ch(k_factors=8192, c_groups=4096, epochs=12, lr=2e-2,
                row_batch=16384, seed=0, holdout_frac=0.0625,
                device="cuda", out_name=None):
    """Centered two-channel v2 fit (proposal sec 4.3): rows are layer-RMS +
    L1-normalized SIGNED attributions with the mean event profile
    SUBTRACTED, split into positive/negative channels
    [max(Ac,0), max(-Ac,0)] and row-renormalized. The fit never sees the
    shared profile, and there is no backbone: every atom's membership is
    the average of its two channel rows (each a softmax over C+residual),
    so the decomposition is full. The mean is a known constant added back
    conceptually at reconstruction; kept-weight evals use the aggregated
    per-atom V exactly as before."""
    A16, y, pos, n_chunks = _load_A_big()
    N_all = A16.shape[0]
    n_hold = int(N_all * holdout_frac)
    N = N_all - n_hold
    svdblob = torch.load(RUN / "atoms.pt", map_location="cpu",
                         weights_only=False)
    sizes = [svdblob["svd"][p]["S"].numel() for p in MODULES]
    J = sum(sizes)
    g_rms, j0 = torch.zeros(J), 0
    for sz in sizes:
        acc = 0.0
        for i in range(0, N, 65536):
            blk = A16[i:min(i + 65536, N), j0:j0 + sz].to(device).float()
            acc += blk.pow(2).sum().item()
        g_rms[j0:j0 + sz] = math.sqrt(acc / (N * sz)) + 1e-12
        j0 += sz
    rms_dev = g_rms.to(device)

    def rows_norm(i0, i1):
        """Signed, layer-RMS'd, row-L1-normalized rows [i0:i1)."""
        An = A16[i0:i1].to(device).float() / rms_dev
        return An / An.abs().sum(1, keepdim=True).clamp_min(1e-12)

    # streaming mean event profile over the fit split
    mu = torch.zeros(J, device=device)
    for i in range(0, N, 65536):
        mu += rows_norm(i, min(i + 65536, N)).sum(0)
    mu /= N

    def m2_batch(rows_idx):
        An = A16[rows_idx].to(device).float() / rms_dev
        An = An / An.abs().sum(1, keepdim=True).clamp_min(1e-12)
        Ac = An - mu[None, :]
        M2 = torch.cat([Ac.clamp_min(0), (-Ac).clamp_min(0)], dim=1)
        return M2 / M2.sum(1, keepdim=True).clamp_min(1e-12)

    gen = torch.Generator().manual_seed(seed)
    Wu_all = (torch.rand(N, k_factors, generator=gen) * 0.5 + 0.2)  # cpu
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(device).requires_grad_()
    Wv = (torch.randn(2 * J, c_groups + 1, generator=gen) * 0.05
          ).to(device).requires_grad_()
    opt_sv = torch.optim.Adam([Ws, Wv], lr=lr)
    u_m = torch.zeros_like(Wu_all)
    u_v = torch.zeros_like(Wu_all)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_count = 0
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(N, generator=gen)
        ep_loss, ep_n = 0.0, 0
        for bi in range(0, N, row_batch):
            rows = perm[bi:bi + row_batch]
            M_bar = m2_batch(rows)
            Wu = Wu_all[rows].to(device).requires_grad_()
            U = torch.softmax(Wu, dim=1)
            S = F.softplus(Ws)
            Vfull = torch.softmax(Wv, dim=1)
            V = Vfull[:, :c_groups]
            M_hat = U @ S @ V.T
            mass = M_bar.sum().clamp_min(1e-8)
            loss = (M_bar * ((M_bar + 1e-8).log() - (M_hat + 1e-8).log())
                    - M_bar + M_hat).sum() / mass
            opt_sv.zero_grad(set_to_none=True)
            loss.backward()
            opt_sv.step()
            with torch.no_grad():
                g = Wu.grad
                m = u_m[rows].to(device)
                v = u_v[rows].to(device)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                step_count += 1
                mh = m / (1 - beta1 ** step_count)
                vh = v / (1 - beta2 ** step_count)
                Wu_all[rows] = (Wu.detach()
                                - lr * mh / (vh.sqrt() + eps)).cpu()
                u_m[rows] = m.cpu()
                u_v[rows] = v.cpu()
            ep_loss += loss.item() * rows.numel()
            ep_n += rows.numel()
        hist.append(f"epoch {ep} idiv/row {ep_loss / ep_n:.5f}")
        print(hist[-1], flush=True)
    with torch.no_grad():
        Vfull = torch.softmax(Wv, dim=1)
        V2 = Vfull[:, :c_groups]
        V_agg = 0.5 * (V2[:J] + V2[J:])                # [J, C]
        r_agg = 0.5 * (Vfull[:J, -1] + Vfull[J:, -1])
    out = BIG / (out_name or "factorization_big_2ch.pt")
    torch.save({"V": V_agg.cpu(), "r": r_agg.cpu(),
                "V2ch": V2.cpu(), "mu": mu.cpu(),
                "S": F.softplus(Ws).detach().cpu(),
                "g_rms": g_rms, "sizes": sizes, "n_hold": n_hold,
                "config": {"K": k_factors, "C": c_groups, "epochs": epochs,
                           "N_fit": N, "two_channel": True}}, out)
    return {"N_fit": N, "J": J, "C": c_groups, "out": str(out),
            "hist": hist[-4:]}
