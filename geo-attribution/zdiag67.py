"""Diagnostics of a stored zcofac67 Z: how much low-rank structure do the
per-event atom-attribution rows have, and which atoms carry it?

For each column scaling (raw, rms^0.5, rms^1 = unit RMS) the rows are
L2-normalised and the top singular values of the N x J matrix are taken by
randomised subspace iteration; reported as cumulative explained energy at
several ranks against the chance level k / J.  Per atom: excess kurtosis
over events (sparse usage -> large; Gaussian noise -> 0), mean / RMS
(consistent sign -> large), usage frequency at 3 RMS, by unit type.

    python -u zdiag67.py --lab_tag _1m --out out/zdiag67_1m.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import zcofac67 as Zc


def log(m):
    print(f"[zdiag] {m}", flush=True)


def topk_energy(Zl, ks, dev, seed=0, iters=4):
    """Cumulative energy fraction of the top-k right singular directions."""
    N, J = Zl.shape
    kmax = max(ks)
    g = torch.Generator(device=dev).manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(J, min(kmax + 64, N, J), generator=g, device=dev))[0]
    for _ in range(iters):
        Q = torch.linalg.qr(Zl.t() @ (Zl @ Q))[0]
    B = Zl @ Q                                          # [N, k+64]
    s = torch.linalg.svdvals(B)                         # descending
    e = s.pow(2)
    tot = float(Zl.pow(2).sum())
    cs = e.cumsum(0) / tot
    return {int(k): round(float(cs[k - 1]), 4) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lab_tag", default="_1m")
    ap.add_argument("--sensor", default="gim")
    ap.add_argument("--scalar", default="pred")
    ap.add_argument("--z_dir", default="")
    ap.add_argument("--out", default="out/zdiag67.json")
    ap.add_argument("--max_events", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    files = Zc.z_files(args)
    d = torch.load(files[0], map_location="cpu", weights_only=False)
    Z16 = d["Z"]
    N_all, J = Z16.shape
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(N_all, generator=g)[:args.max_events].sort().values
    Z = Z16[sel].to(dev).float()
    N = Z.shape[0]
    utype = np.array(d["unit_type"])
    ulayer = np.array(d["unit_layer"])
    log(f"Z {N_all} x {J}; using {N} events")
    t0 = time.perf_counter()

    # per-atom statistics (raw attribution, nats)
    mean = Z.mean(0)
    rms = Z.pow(2).mean(0).sqrt()
    std = Z.std(0)
    zc = (Z - mean) / std.clamp_min(1e-12)
    kurt = zc.pow(4).mean(0) - 3.0
    use3 = (Z.abs() > 3 * rms).float().mean(0)
    absmean = Z.abs().mean(0)
    res = {"format": "zdiag67_v1", "N": N, "J": J, "lab_tag": args.lab_tag,
           "by_type": {}, "energy": {}, "chance": {}}
    for ty in ("mlp", "qk", "ov"):
        m = torch.from_numpy(utype == ty).to(dev)
        res["by_type"][ty] = {
            "n": int(m.sum()),
            "rms_q": Zc.U.qdict(rms[m].cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "kurtosis_q": Zc.U.qdict(kurt[m].cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "mean_over_rms_q": Zc.U.qdict((mean[m] / rms[m].clamp_min(1e-12)).abs().cpu().numpy(), (0.1, 0.5, 0.9)),
            "usage3rms_q": Zc.U.qdict(use3[m].cpu().numpy(), (0.1, 0.5, 0.9)),
            "share_of_raw_energy": round(float(Z[:, m].pow(2).sum() / Z.pow(2).sum()), 4)}
    for l in range(4):
        m = torch.from_numpy(ulayer == l).to(dev)
        res["by_type"][f"layer{l}"] = {
            "share_of_raw_energy": round(float(Z[:, m].pow(2).sum() / Z.pow(2).sum()), 4),
            "kurtosis_median": round(float(kurt[m].median()), 3)}
    # energy share of the top atoms by rms and by kurtosis
    order = rms.argsort(descending=True)
    tot = float(Z.pow(2).sum())
    res["raw_energy_share_top_atoms_by_rms"] = {
        int(k): round(float(Z[:, order[:k]].pow(2).sum() / tot), 4) for k in (64, 256, 1024, 4096)}
    hk = kurt.argsort(descending=True)
    res["atoms_with_kurtosis_over"] = {t: int((kurt > t).sum()) for t in (1, 3, 10, 30, 100)}
    log(f"per-atom stats done ({time.perf_counter() - t0:.0f}s)")

    ks = [k for k in (4, 16, 64, 256, 1024, 2048, 4096) if k + 8 <= min(N, J)]
    ks = ks or [max(1, min(N, J) // 2)]
    res["chance"] = {int(k): round(k / J, 4) for k in ks}
    for name, alpha in (("raw", 0.0), ("rms_half", 0.5), ("rms_full", 1.0)):
        scale = rms.clamp_min(1e-2 * rms[rms > 0].median()).pow(alpha)
        Zl = Z / scale
        Zl = Zl / Zl.norm(dim=1, keepdim=True).clamp_min(1e-30)
        res["energy"][name] = topk_energy(Zl, ks, dev, args.seed)
        # how much of each row's energy sits in the 1024 highest-kurtosis atoms
        res["energy"][name + "_share_top1024_kurtosis_atoms"] = round(
            float(Zl[:, hk[:1024]].pow(2).sum() / Zl.pow(2).sum()), 4)
        log(f"{name}: top-k energy {res['energy'][name]} ({time.perf_counter() - t0:.0f}s)")
        del Zl
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log("DIAG " + json.dumps({k: res[k] for k in ("energy", "chance", "atoms_with_kurtosis_over")}))


if __name__ == "__main__":
    main()
