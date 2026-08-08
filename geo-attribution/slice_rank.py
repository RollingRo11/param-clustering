"""Effective rank of a component's per-matrix slice  s_c ⊙ W_m.

Ownership in this decomposition is a per-ENTRY fractional mask (soft_s=8
owners per weight entry, shares summing to 1), not a low-rank factorisation —
so unlike APD/VPD, where each component is rank-1 by construction, nothing
here forces the per-matrix pieces to be low rank. This measures what they
actually are.

Reported per slice, against the same metrics on the unmasked W_m:
  stable rank     ||A||_F^2 / ||A||_2^2     — mass-weighted, robust
  effective rank  exp(spectral entropy)     — "how many directions carry it"
  numerical rank  singular values above tol
  occupancy       fraction of entries the component owns at all

  cd geo-attribution && python3.12 slice_rank.py --components 12 --device cuda:0
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import geo1b  # noqa: F401


def rank_metrics(A: torch.Tensor) -> tuple[float, float, int, int]:
    s = torch.linalg.svdvals(A.float()).double()
    # float32 singular values can underflow to 0 after normalising, and
    # 0*log(0) then poisons the entropy with NaN — do it in float64 and drop
    # the exact zeros.
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0, 0.0, 0, min(A.shape)
    stable = ((s ** 2).sum() / s[0] ** 2).item()
    p = s / s.sum()
    p = p[p > 0]
    effective = torch.exp(-(p * p.log()).sum()).item()
    tol = s[0] * max(A.shape) * torch.finfo(torch.float32).eps
    return stable, effective, int((s > tol).sum()), min(A.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--components", type=int, default=12,
                    help="random components to sample")
    ap.add_argument("--probe_matrices", type=int, default=12,
                    help="matrices examined per component to find its heaviest")
    ap.add_argument("--slices_per_component", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    root = (args.artifact_root and __import__("pathlib").Path(args.artifact_root)) \
        or geo1b.SHM_ROOT
    run_dir = root / args.tag
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    C = int(bank["C"])
    mods = list(bank["modules"])
    target = geo1b.load_target_1b(dev)
    rng = random.Random(args.seed)
    comps = rng.sample(range(C), args.components)
    print(f"C={C}, soft_s={bank.get('soft_s')}, {len(mods)} matrices; "
          f"sampling {len(comps)} components\n")

    rows = []
    for c in comps:
        probe = rng.sample(range(len(mods)), args.probe_matrices)
        masses = []
        for mi in probe:
            path = mods[mi]
            sidx = bank["sidx"][path].to(dev)
            swgt = bank["swgt"][path].to(dev)
            share = ((sidx == c) * swgt).sum(0).float()
            W = target.get_submodule(path).weight.detach().float().to(dev)
            masses.append((float(((share * W) ** 2).sum()), mi, share, W))
            del sidx, swgt
        masses.sort(key=lambda t: -t[0])
        for mass, mi, share, W in masses[:args.slices_per_component]:
            if mass == 0.0:
                continue
            A = share * W
            sr, er, nr, full = rank_metrics(A)
            wsr, wer, _, _ = rank_metrics(W)
            rows.append(dict(c=c, path=mods[mi], stable=sr, eff=er, num=nr,
                             full=full, w_stable=wsr, w_eff=wer,
                             occ=float((share > 0).float().mean())))
            print(f"  c{c:<5} {mods[mi].split('layers.')[1]:<28} "
                  f"occ {100*rows[-1]['occ']:5.2f}%  stable {sr:8.1f}  "
                  f"eff {er:8.1f}  num {nr:>5}/{full}")
        del masses
        torch.cuda.empty_cache()

    print(f"\n=== means over {len(rows)} (component, matrix) slices ===")
    print(f"{'':<38}{'slice':>10}{'full W_m':>12}")
    print(f"  {'stable rank':<36}{np.mean([r['stable'] for r in rows]):>10.1f}"
          f"{np.mean([r['w_stable'] for r in rows]):>12.1f}")
    print(f"  {'effective rank exp(entropy)':<36}"
          f"{np.mean([r['eff'] for r in rows]):>10.1f}"
          f"{np.mean([r['w_eff'] for r in rows]):>12.1f}")
    print(f"  {'numerical rank':<36}{np.mean([r['num'] for r in rows]):>10.1f}")
    print(f"  {'max possible = min(dim)':<36}"
          f"{np.mean([r['full'] for r in rows]):>10.1f}")
    print(f"  {'entry occupancy':<36}"
          f"{100*np.mean([r['occ'] for r in rows]):>9.2f}%")
    print(f"\n  numerical rank / min(dim)        "
          f"{np.mean([r['num'] / r['full'] for r in rows]):.3f}")
    print(f"  slice eff rank / full-matrix eff  "
          f"{np.mean([r['eff'] / r['w_eff'] for r in rows]):.3f}")
    print(f"\n  rank-1 would be 1.0; anything near min(dim) means the slice is "
          f"as high-rank as the matrix it came from.")


if __name__ == "__main__":
    main()
