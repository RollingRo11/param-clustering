"""VPD Section 4.3, ported to a decomposition whose components are not rank-1.

VPD writes the QK circuit as a sum over subcomponent PAIRS,

    W_QK^h = W_Q^{h T} W_K^h = sum_{c,c'} V_{Q,c} ((U^h_{Q,c})^T R_tau U^h_{K,c'}) V_{K,c'}^T

and reads off a StaticInteractionStrength(c, c', tau, h) from the middle scalar.
That scalar exists because each VPD subcomponent is rank-1: one read direction
V and one write direction U.

Our components are not rank-1 — they are shares of every weight ENTRY, with
measured stable rank around 7.5 — so the middle term is a matrix, not a scalar.
But the bilinear decomposition still holds EXACTLY, because the shares sum to 1
per entry:  W_Q = sum_c A_{Q,c}, hence

    W_Q^{h T} R_tau W_K^h = sum_{c,c'} A_{Q,c}^{h T} R_tau A_{K,c'}^h

and the natural scalar is the Frobenius norm of each pair's contribution:

    I(c,c',h,tau) = || A_{Q,c}^{h T} R_tau A_{K,c'}^h ||_F

which for rank-1 A collapses to |V_Q| |V_K| |U_Q^T R_tau U_K| — VPD's quantity.
Computing it never needs the d_model x d_model product:

    I^2 = tr( R_tau^T G_{Q,c} R_tau G_{K,c'} ),   G = A A^T   (64 x 64)

The question this answers is the one that matters for a partition: does a
component interact with ITSELF across q and k (self-contained circuits), or
with OTHER components (circuits that span components)? VPD found the latter —
previous-token behaviour in a q-subcomponent paired with a DIFFERENT
k-subcomponent, spread over heads. Diagonal mass fraction measures it directly.

    python3.12 qk_interactions.py --top_m 16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import geo1b  # noqa: F401
from german_vpd_1b import log


def rope_matrix(inv_freq, tau, HD, dev):
    """R_tau in HF Llama's rotate_half convention: pairs are (i, i + HD/2)."""
    half = HD // 2
    ang = inv_freq[:half].to(dev) * float(tau)
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.zeros(HD, HD, device=dev)
    idx = torch.arange(half, device=dev)
    R[idx, idx] = c
    R[idx, idx + half] = -s
    R[idx + half, idx] = s
    R[idx + half, idx + half] = c
    return R


def head_grams(bank, W, path, rows, comps, dev):
    """G[c] = A_c A_c^T for the given row block, for each component in comps."""
    sidx = bank["sidx"][path][:, rows, :].to(dev)          # [8, HD, D]
    swgt = bank["swgt"][path][:, rows, :].to(dev).float()
    Wb = W[rows, :]                                        # [HD, D]
    G = []
    for c in comps:
        share = ((sidx == c).float() * swgt).sum(0)        # [HD, D]
        A = share * Wb
        G.append(A @ A.t())
    del sidx, swgt
    return torch.stack(G)                                  # [m, HD, HD]


def head_mass(bank, W, path, rows, C, dev):
    """[C] owned squared mass in this row block."""
    sidx = bank["sidx"][path][:, rows, :].to(dev)
    swgt = bank["swgt"][path][:, rows, :].to(dev).float()
    w2 = (W[rows, :] ** 2)[None]
    contrib = swgt * w2
    out = torch.bincount(sidx.reshape(-1).long(),
                         weights=contrib.reshape(-1), minlength=C).float()
    del sidx, swgt, contrib
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top_m", type=int, default=16)
    ap.add_argument("--taus", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="qk_interactions.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    target = geo1b.load_target_1b(dev)
    cfg = target.hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    KV, HD = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    REP = H // KV
    inv_freq = target.hf.model.rotary_emb.inv_freq.detach().float()
    layers = args.layers if args.layers is not None else list(range(L))
    R = {t: rope_matrix(inv_freq, t, HD, dev) for t in args.taus}

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    M = args.top_m

    diag_frac, self_pairs, records = [], [], []
    tau_profile = {t: [] for t in args.taus}
    # keep the raw pair-interaction tensors: every downstream question (offset
    # tuning, head spread, how many pairs carry the mass) is a slice of these
    raw = {"I": [], "cq": [], "ck": [], "layer": [], "head": []}
    for l in layers:
        p = f"hf.model.layers.{l}.self_attn"
        Wq = target.get_submodule(f"{p}.q_proj").weight.detach().float()
        Wk = target.get_submodule(f"{p}.k_proj").weight.detach().float()
        for h in range(H):
            qr = slice(h * HD, (h + 1) * HD)
            kr = slice((h // REP) * HD, (h // REP + 1) * HD)
            mq = head_mass(bank, Wq, f"{p}.q_proj", qr, C, dev)
            mk = head_mass(bank, Wk, f"{p}.k_proj", kr, C, dev)
            cq = torch.topk(mq, M).indices.tolist()
            ck = torch.topk(mk, M).indices.tolist()
            GQ = head_grams(bank, Wq, f"{p}.q_proj", qr, cq, dev)
            GK = head_grams(bank, Wk, f"{p}.k_proj", kr, ck, dev)
            # I^2[c,c'] = tr(R^T G_Q R G_K)
            per_tau = []
            for t in args.taus:
                Rt = R[t]
                R_GQ_R = torch.einsum("ij,mjk,kl->mil", Rt.t(), GQ, Rt)
                I2 = torch.einsum("mij,nji->mn", R_GQ_R, GK).clamp_min(0)
                I = I2.sqrt()
                per_tau.append(I.cpu())
                tot = I.sum().clamp_min(1e-30)
                # shared components appear in both top-M lists; the "diagonal"
                # is every (c, c) pair, wherever it sits in the two orderings
                same = torch.zeros_like(I, dtype=torch.bool)
                for a, ca in enumerate(cq):
                    for b, cb in enumerate(ck):
                        if ca == cb:
                            same[a, b] = True
                df = float(I[same].sum() / tot) if same.any() else 0.0
                tau_profile[t].append(df)
                if t == 1:
                    diag_frac.append(df)
                    self_pairs.append(int(same.sum()))
                    flat = I.flatten()
                    k = min(3, flat.numel())
                    top = torch.topk(flat, k)
                    for v, fi in zip(top.values.tolist(), top.indices.tolist()):
                        a, b = fi // M, fi % M
                        records.append({
                            "layer": l, "head": h,
                            "q_component": cq[a], "k_component": ck[b],
                            "same": bool(cq[a] == ck[b]),
                            "strength": round(v / float(tot), 5)})
            raw["I"].append(torch.stack(per_tau))      # [n_tau, M, M]
            raw["cq"].append(torch.tensor(cq))
            raw["ck"].append(torch.tensor(ck))
            raw["layer"].append(l)
            raw["head"].append(h)
            del GQ, GK, mq, mk
        log(f"layer {l} done ({H} heads)")
    del bank

    df_t = torch.tensor(diag_frac)
    n_same = torch.tensor(self_pairs, dtype=torch.float)
    # chance: if interaction mass were spread evenly over the M*M cells, the
    # same-component cells would take n_same / M^2 of it
    chance = float((n_same / (M * M)).mean())
    cross = [r for r in records if not r["same"]]
    out = {
        "format": "qk_interactions_v1", "C": C, "top_m": M,
        "layers": layers, "taus": args.taus,
        "diagonal_mass_fraction_tau1": {
            "mean": round(float(df_t.mean()), 5),
            "median": round(float(df_t.median()), 5),
            "chance": round(chance, 5),
            "ratio": round(float(df_t.mean()) / max(chance, 1e-12), 3),
            "mean_same_component_cells": round(float(n_same.mean()), 2),
        },
        "tau_profile_diagonal_fraction": {
            str(t): round(float(torch.tensor(v).mean()), 5)
            for t, v in tau_profile.items()},
        "top_pairs": sorted(records, key=lambda r: -r["strength"])[:40],
        "frac_top_pairs_cross_component": round(
            len(cross) / max(len(records), 1), 4),
    }
    d = out["diagonal_mass_fraction_tau1"]
    log(f"same-component QK interaction mass: {d['mean']:.4f} vs chance "
        f"{d['chance']:.4f}  ({d['ratio']:.2f}x); "
        f"{100 * out['frac_top_pairs_cross_component']:.1f}% of the strongest "
        f"pairs are CROSS-component")
    log("diagonal fraction by RoPE offset: " + ", ".join(
        f"t={t}:{v:.4f}" for t, v in out["tau_profile_diagonal_fraction"].items()))
    torch.save({"I": torch.stack(raw["I"]), "cq": torch.stack(raw["cq"]),
                "ck": torch.stack(raw["ck"]),
                "layer": torch.tensor(raw["layer"]),
                "head": torch.tensor(raw["head"]), "taus": args.taus},
               run_dir / "qk_interactions_arrays.pt")
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
