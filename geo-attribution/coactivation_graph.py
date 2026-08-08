"""The runtime interaction graph: which components fire together?

Everything measured so far is about weights — where a component's mass sits and
how it multiplies with another's. But two components also "interact" in the
plainest sense if they are active on the same token, because then their edits
to the residual stream are added together on that token.

    Coact(c, c') = < p_t(c) p_t(c') >_t

on the frozen fingerprint posterior, the same quantity the streaming assignment
uses. Three questions:

  structure   is the graph modular (tight cliques of components that always
              co-fire) or diffuse (everything mildly co-fires with everything)?
  null        marginals alone predict a lot of co-activation, so the comparison
              is against independent firing, p(c) p(c'), computed from the same
              posteriors with the temporal pairing destroyed.
  placement   do components that fire together also SIT together — overlapping
              weight mass in the same matrices? If co-firing and co-location
              are unrelated, the decomposition's runtime structure and its
              weight structure are two different graphs.

    python3.12 coactivation_graph.py --n_blocks 16
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
    ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--top_pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="coactivation_graph.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    blocks = torch.cat([data["pile_eval"], data["bio_retain_eval"]])
    idx = blocks[:args.n_blocks].to(dev)
    bank_meta = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                           weights_only=True, map_location="cpu", mmap=True)
    C = int(bank_meta["C"])
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    sm = load_stream_model(run_dir / "stream_model.pt", dev)

    T = idx.shape[1]
    pos1 = torch.arange(8, T - 2, device=dev)
    posts = []
    for s in range(0, idx.shape[0], 4):
        b = idx[s:s + 4]
        pos = pos1[None].expand(b.shape[0], -1)
        bi = torch.arange(b.shape[0], device=dev)[:, None].expand_as(pos)
        phi, _ = pass_features(cfg, cap, b, pos, bi, spec, scales, dim,
                               return_pg=False)
        x = phi.clamp(-6e4, 6e4).half().float()
        y = F.normalize((x - sm["mean"]) @ sm["projector"], dim=1)
        posts.append(torch.softmax((y @ sm["centroids"].t())
                                   / args.temperature, dim=1))
        del phi, x, y
    P = torch.cat(posts)                                  # [N, C]
    N = P.shape[0]
    log(f"{N} token positions, C={C}")

    marg = P.mean(0)                                       # p(c)
    Coact = (P.t() @ P) / N                                # <p_t(c) p_t(c')>
    Indep = marg[:, None] * marg[None, :]                  # independent firing
    eye = torch.eye(C, dtype=torch.bool, device=dev)
    off = ~eye

    lift = Coact / Indep.clamp_min(1e-30)
    # how concentrated is a component's co-activation? participation ratio over
    # partners, so 1 = fires with exactly one other, C = fires with everything
    Q = Coact.clone()
    Q[eye] = 0
    qn = Q / Q.sum(1, keepdim=True).clamp_min(1e-30)
    partners = 1.0 / (qn ** 2).sum(1).clamp_min(1e-30)

    # active components per token, two conventional thresholds
    active = {f"p>{t}": float((P > t).sum(1).float().mean())
              for t in (0.01, 0.02, 0.05, 0.10)}
    pr_tok = float((1.0 / (P ** 2).sum(1)).mean())

    out = {
        "format": "coactivation_graph_v1", "C": C, "n_positions": N,
        "active_components_per_token": active,
        "participation_ratio_per_token": round(pr_tok, 3),
        "coactivation_partners": {
            "mean": round(float(partners.mean()), 2),
            "median": round(float(partners.median()), 2),
            "p10": round(float(partners.quantile(0.10)), 2),
            "p90": round(float(partners.quantile(0.90)), 2),
        },
        "lift_over_independence": {
            "mean_offdiag": round(float(lift[off].mean()), 4),
            "median_offdiag": round(float(lift[off].median()), 4),
            "p99": round(float(lift[off].quantile(0.99)), 3),
            "max": round(float(lift[off].max()), 2),
            "frac_pairs_lift_gt_10": round(
                float((lift[off] > 10).float().mean()), 6),
        },
    }
    log(f"active per token: {active}; participation ratio {pr_tok:.2f}")
    log(f"co-activation partners (PR): mean {out['coactivation_partners']['mean']:.1f}, "
        f"median {out['coactivation_partners']['median']:.1f}")
    log(f"lift over independence: median {out['lift_over_independence']['median_offdiag']:.3f}, "
        f"p99 {out['lift_over_independence']['p99']:.1f}, "
        f"max {out['lift_over_independence']['max']:.0f}")

    # ---- top co-activating pairs, for the causal follow-up ----
    Cf = Coact.clone()
    Cf[eye] = -1
    flat = Cf.flatten()
    top = torch.topk(flat, args.top_pairs * 2)
    pairs, seen = [], set()
    for v, fi in zip(top.values.tolist(), top.indices.tolist()):
        a, b = fi // C, fi % C
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append({"a": a, "b": b, "coact": round(v, 8),
                      "lift": round(float(lift[a, b]), 3)})
        if len(pairs) >= args.top_pairs:
            break
    out["top_pairs"] = pairs
    log(f"top pair c{pairs[0]['a']} x c{pairs[0]['b']}: "
        f"coact {pairs[0]['coact']:.2e}, lift {pairs[0]['lift']:.1f}")

    torch.save({"coact": Coact.cpu(), "marg": marg.cpu(),
                "partners": partners.cpu()},
               run_dir / "coactivation_arrays.pt")
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
