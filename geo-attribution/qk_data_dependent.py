"""VPD's DataDependentInteractionStrength: static QK interaction x co-activation.

The static term in qk_interactions.py asks what the WEIGHTS allow. VPD's
data-dependent version multiplies it by the subcomponents' activations at the
query and key tokens, which asks what actually happens on text:

    DD(c, c', tau, t, h) = phi_t(c) * phi_{t-tau}(c') * I(c, c', tau, h)

Our activation is the frozen fingerprint posterior p_t(c) — the same quantity
the streaming assignment uses — so the port is direct. Averaged over positions,

    DD(c, c', tau, h) = I(c, c', tau, h) * < p_t(c) p_{t-tau}(c') >_t

This matters here more than it would for VPD. Our components are DEFINED by
co-activation geometry, so it is entirely possible for the weights to interact
broadly across components while the pairs that are ever simultaneously active
are mostly within-component. Static says what could interact; data-dependent
says what does. If the diagonal fraction jumps, the partition is real at
runtime even though it is not visible in the weights alone.

    python3.12 qk_data_dependent.py --taus 1 2 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--arrays", default="qk_interactions_arrays.pt")
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--first_pos", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", default="qk_data_dependent.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    arrays = torch.load(run_dir / args.arrays, weights_only=False)
    taus_all = list(arrays["taus"])
    I_all, CQ, CK = arrays["I"], arrays["cq"], arrays["ck"]
    LAY, HEAD = arrays["layer"], arrays["head"]
    M = CQ.shape[1]

    # ---- per-token component posteriors on consecutive positions ----
    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    idx = data["pile_eval"][:args.n_blocks].to(dev)
    bank_meta = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                           weights_only=True, map_location="cpu", mmap=True)
    C = int(bank_meta["C"])
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    sm = load_stream_model(run_dir / "stream_model.pt", dev)

    T = idx.shape[1]
    pos1 = torch.arange(args.first_pos, T - 2, device=dev)
    B = idx.shape[0]
    pos = pos1[None].expand(B, -1)
    bi = torch.arange(B, device=dev)[:, None].expand_as(pos)
    phi, _ = pass_features(cfg, cap, idx, pos, bi, spec, scales, dim,
                           return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize((x - sm["mean"]) @ sm["projector"], dim=1)
    post = torch.softmax((y @ sm["centroids"].t()) / args.temperature, dim=1)
    post = post.view(B, pos1.numel(), C)                 # [B, P, C]
    log(f"posteriors {tuple(post.shape)} over consecutive positions")

    out = {"format": "qk_data_dependent_v1", "C": C, "top_m": M,
           "n_blocks": args.n_blocks, "results": {}}
    per_head = {}
    for tau in args.taus:
        ti = taus_all.index(tau)
        # co-activation at offset tau: <p_t(c) p_{t-tau}(c')>
        a = post[:, tau:, :].reshape(-1, C)              # query side, token t
        b = post[:, :-tau, :].reshape(-1, C)             # key side, t - tau
        coact = (a.t() @ b) / a.shape[0]                 # [C, C]

        stat_d, dd_d, dd_ratio, keep_lh = [], [], [], []
        for n in range(I_all.shape[0]):
            I = I_all[n, ti].to(dev)                     # [M, M]
            cq, ck = CQ[n].to(dev), CK[n].to(dev)
            same = cq[:, None] == ck[None, :]
            if not bool(same.any()):
                continue
            co = coact[cq][:, ck]                        # [M, M]
            DD = I * co
            st = float(I[same].sum() / I.sum().clamp_min(1e-30))
            dv = float(DD[same].sum() / DD.sum().clamp_min(1e-30))
            stat_d.append(st)
            dd_d.append(dv)
            dd_ratio.append(dv / max(st, 1e-12))
            keep_lh.append((int(LAY[n]), int(HEAD[n])))
        s_t, d_t = torch.tensor(stat_d), torch.tensor(dd_d)
        out["results"][str(tau)] = {
            "n_heads": len(stat_d),
            "static_diagonal_fraction": round(float(s_t.mean()), 5),
            "data_dependent_diagonal_fraction": round(float(d_t.mean()), 5),
            "amplification": round(float(d_t.mean() / s_t.mean()), 3),
            "median_amplification": round(
                float(torch.tensor(dd_ratio).median()), 3),
            "frac_heads_amplified": round(
                float((torch.tensor(dd_ratio) > 1).float().mean()), 4),
        }
        r = out["results"][str(tau)]
        log(f"tau={tau}: static diagonal {r['static_diagonal_fraction']:.4f} "
            f"-> data-dependent {r['data_dependent_diagonal_fraction']:.4f}  "
            f"({r['amplification']:.2f}x, median per-head "
            f"{r['median_amplification']:.2f}x, "
            f"{100 * r['frac_heads_amplified']:.0f}% of heads amplified)")

        per_head[str(tau)] = {"lh": keep_lh, "static": stat_d,
                              "dd": dd_d, "ratio": dd_ratio}
    torch.save(per_head, run_dir / "qk_data_dependent_perhead.pt")
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
