"""If components do not group COMPOSING heads, what do they group?

virtual_heads.py finds co-owned head pairs compose at chance (Q 1.01x, K 0.93x,
V 1.01x) with a score validated at 2.05x on the real prev-token -> induction
circuit. So the partition is not cutting along virtual heads. But c3392 owns
four of the eight strongest induction heads, in three different layers, and no
previous-token head at all — which is a different cross-layer structure:
same ROLE in several layers, rather than a writer paired with its reader.

This tests that at the level of all 4096 components, on behavioural profiles
measured from attention patterns alone:

  role enrichment   is a component's mass-weighted mean induction (or
                    previous-token) score extreme against a permutation null?
  role similarity   are co-owned cross-layer head pairs closer in profile than
                    random cross-layer pairs?

The two tests are complementary: enrichment finds individual components that
collect a role, similarity asks whether the whole partition does it.

    python3.12 functional_grouping.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log
from induction4096 import repeated_batch
from validate_composition import attention_maps

FEATURES = ("induction", "prev_token", "bos", "self", "distance")


@torch.no_grad()
def profiles(hf, cfg, dev, span, n_seq, warmup):
    """[L*H, len(FEATURES)] behavioural profile per head, from attention only."""
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    S = span
    idx = repeated_batch(n_seq, S, 1000, 20000, cfg.bos_token_id, dev, 0)
    A = attention_maps(hf, idx, L, H)                      # [L,B,H,T,T]
    T = idx.shape[1]

    q = torch.arange(warmup + 1, T, device=dev)
    q2 = torch.arange(S + 1 + warmup, 2 * S, device=dev)
    feat = {
        "induction": A[:, :, :, q2, q2 - S + 1].mean(dim=(1, 3)),
        "prev_token": A[:, :, :, q, q - 1].mean(dim=(1, 3)),
        "bos": A[:, :, :, q, 0].mean(dim=(1, 3)),
        "self": A[:, :, :, q, q].mean(dim=(1, 3)),
    }
    # mean normalised distance back to the attended position
    kpos = torch.arange(T, device=dev).view(1, 1, 1, 1, T).float()
    qpos = torch.arange(T, device=dev).view(1, 1, 1, T, 1).float()
    d = ((qpos - kpos).clamp_min(0) / T)
    feat["distance"] = (A * d).sum(-1)[:, :, :, warmup + 1:].mean(dim=(1, 3))
    P = torch.stack([feat[f].reshape(-1) for f in FEATURES], dim=1)
    return P                                               # [L*H, F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--n_seq", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--top_heads", type=int, default=6)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="functional_grouping.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    target = geo1b.load_target_1b(dev)
    hf, cfg = target.hf, target.hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HD = cfg.hidden_size // H
    P = profiles(hf, cfg, dev, args.span, args.n_seq, args.warmup)
    log("behavioural profiles: " + ", ".join(
        f"{f} max {float(P[:, i].max()):.2f}" for i, f in enumerate(FEATURES)))

    # ---- per-(component, head) attention mass (q rows / o columns) ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    mass = torch.zeros(C, L * H, device=dev)
    for path in bank["modules"]:
        kind = path.rsplit(".", 1)[1]
        if kind not in ("q_proj", "o_proj"):
            continue
        l = int(path.split("layers.")[1].split(".")[0])
        w2 = target.get_submodule(path).weight.detach().float() ** 2
        sidx, swgt = bank["sidx"][path].to(dev), bank["swgt"][path].to(dev)
        for h in range(H):
            if kind == "q_proj":
                sl = (slice(None), slice(h * HD, (h + 1) * HD), slice(None))
                wsl = w2[h * HD:(h + 1) * HD]
            else:
                sl = (slice(None), slice(None), slice(h * HD, (h + 1) * HD))
                wsl = w2[:, h * HD:(h + 1) * HD]
            contrib = (swgt[sl].float() * wsl).reshape(swgt.shape[0], -1)
            mass[:, l * H + h] += torch.bincount(
                sidx[sl].reshape(-1).long(), weights=contrib.reshape(-1),
                minlength=C).float()
        del w2, sidx, swgt
    del bank
    share = mass / mass.sum(1).clamp_min(1e-30)[:, None]
    log("per-(component, head) mass done")

    g = torch.Generator(device=dev).manual_seed(args.seed)
    out = {"format": "functional_grouping_v1", "features": list(FEATURES),
           "enrichment": {}, "similarity": {}}

    # ---- role enrichment, per feature ----
    zs = {}
    for i, f in enumerate(FEATURES):
        v = P[:, i]
        obs = share @ v                                    # [C]
        # null: permute which HEAD carries which score, keep the mass pattern
        null = torch.stack([share @ v[torch.randperm(L * H, generator=g,
                                                     device=dev)]
                            for _ in range(args.perms)])   # [perms, C]
        mu, sd = null.mean(0), null.std(0).clamp_min(1e-9)
        z = (obs - mu) / sd
        zs[f] = z.cpu()
        order = torch.argsort(z, descending=True)[:8]
        out["enrichment"][f] = {
            "max_z": round(float(z.max()), 2),
            "n_components_z_gt_3": int((z > 3).sum()),
            "n_components_z_gt_5": int((z > 5).sum()),
            "top": [{"component": int(c), "z": round(float(z[c]), 2),
                     "score": round(float(obs[c]), 4),
                     "null_mean": round(float(mu[c]), 4)} for c in order],
        }
        r = out["enrichment"][f]
        log(f"{f:<11} enrichment: {r['n_components_z_gt_3']} components z>3, "
            f"{r['n_components_z_gt_5']} z>5, max z {r['max_z']:.1f} "
            f"(c{r['top'][0]['component']})")

    # ---- role similarity of co-owned cross-layer pairs ----
    Z = (P - P.mean(0)) / P.std(0).clamp_min(1e-9)
    layer_of = torch.arange(L * H, device=dev) // H
    obs_d, obs_w, pairs = [], [], []
    for c in range(C):
        top = torch.topk(share[c], args.top_heads)
        if float(top.values[1]) < 0.02:
            continue
        for a in range(args.top_heads):
            for b in range(a + 1, args.top_heads):
                i, j = int(top.indices[a]), int(top.indices[b])
                if layer_of[i] == layer_of[j]:
                    continue                    # cross-layer only
                obs_d.append(float((Z[i] - Z[j]).norm()))
                obs_w.append(float(top.values[a] * top.values[b]))
                pairs.append((i, j))
    od = torch.tensor(obs_d)
    ow = torch.tensor(obs_w)
    # null: random cross-layer head pairs
    ii = torch.randint(0, L * H, (200000,), generator=g, device=dev)
    jj = torch.randint(0, L * H, (200000,), generator=g, device=dev)
    keep = layer_of[ii] != layer_of[jj]
    nd = (Z[ii[keep]] - Z[jj[keep]]).norm(dim=1).cpu()
    out["similarity"] = {
        "n_pairs": len(obs_d),
        "co_owned_mean_distance": round(float(od.mean()), 4),
        "co_owned_mass_weighted": round(float((od * ow).sum() / ow.sum()), 4),
        "null_mean_distance": round(float(nd.mean()), 4),
        "null_p05": round(float(nd.quantile(0.05)), 4),
        "ratio": round(float(od.mean() / nd.mean()), 3),
        "frac_below_null_p05": round(float((od < nd.quantile(0.05))
                                           .float().mean()), 3),
    }
    s = out["similarity"]
    log(f"role similarity: co-owned cross-layer distance "
        f"{s['co_owned_mean_distance']:.3f} vs null {s['null_mean_distance']:.3f} "
        f"(ratio {s['ratio']:.2f}x, lower = more alike); "
        f"{100 * s['frac_below_null_p05']:.1f}% below null p05  "
        f"(n={s['n_pairs']})")

    # full arrays for the figure: z per component per feature, the head
    # profiles, and c3392's own mass map
    torch.save({"z": zs, "profiles": P.cpu(), "share_c3392": share[3392].cpu(),
                "features": list(FEATURES), "L": L, "H": H},
               run_dir / "functional_grouping_arrays.pt")

    # where does c3392 sit on induction enrichment?
    vi = FEATURES.index("induction")
    obs_ind = share @ P[:, vi]
    rank = int((obs_ind > obs_ind[3392]).sum())
    out["c3392_induction_rank"] = rank
    out["c3392_induction_score"] = round(float(obs_ind[3392]), 4)
    log(f"c3392 induction enrichment rank {rank} of {C} "
        f"(score {float(obs_ind[3392]):.4f}, mean "
        f"{float(obs_ind.mean()):.4f})")

    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
