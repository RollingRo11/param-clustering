"""Do gradient attributions predict causal necessity? And can they be made to?

The clustering is co-attribution by construction, for a reason that is easy to
miss: streaming_decomposition.py line 101 does F.normalize(X @ projector) and
then runs SPHERICAL k-means. L2-normalising the feature vector throws away its
magnitude — and magnitude is precisely the part that says "this weight matters".
What survives is the DIRECTION of the attribution vector, i.e. which weights
light up together. So the objective is co-attribution whatever the features mean.

But there is a second, deeper reason, and this measures it. The current feature
is dR/dW * W, which is exactly the FIRST-ORDER estimate of what happens if you
delete that weight: ablation moves W by -W, so the linear prediction of the
change is -dR/dW * W. That estimate is only as good as the model is linear over
a full-size step, and this model is not remotely linear over one — scaling a
component by 4 moves log p by 5 nats while deleting it entirely moves it by
0.002.

The principled fix is already standard: integrate the gradient along the path
from the ablated model to the real one. By the fundamental theorem of calculus

    integral_0^1 (dR/dt)(t) dt  =  R(full) - R(ablated)

so a path-integrated attribution EQUALS the finite ablation effect instead of
approximating it. cap.wscale (geo67.py:168) already walks that path.

One catch worth knowing: GIM's modified backward is not the gradient of
anything, so integrating it does not telescope. Causal-necessity features want
TRUE gradients along the path, which is what this measures.

    python3.12 causal_features.py --ig_k 16 --sample 96
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--ig_k", type=int, default=16)
    ap.add_argument("--sample", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="causal_features.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    t00 = time.perf_counter()

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    ids = torch.cat([data["pile_eval"],
                     data["bio_retain_eval"]])[:args.n_blocks].to(dev)
    target = geo1b.load_target_1b(dev)

    def reward():
        """The pipeline's objective: 1 unit on each ground-truth next logit."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            lg = target(ids)
        return lg[:, :-1].float().gather(-1, ids[:, 1:, None]).sum()

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    C = int(bank["C"])
    mods = list(bank["modules"])
    S, Wt, W0 = {}, {}, {}
    for p in mods:
        S[p] = (bank["sidx"][p].to(dev), bank["swgt"][p].to(dev))
        w = target.get_submodule(p).weight
        W0[p], Wt[p] = w.detach().clone(), w
        w.requires_grad_(True)
    del bank
    for q in target.parameters():
        q.requires_grad_(False)
    for p in mods:
        Wt[p].requires_grad_(True)
    log(f"bank resident ({time.perf_counter() - t00:.0f}s, "
        f"{torch.cuda.memory_allocated(dev) / 2**30:.0f} GiB)")

    # ---- per-component dR/dalpha along the weight-scaling path ----
    # dR/dalpha_c = sum over c's owned entries of dR/dW * (s_c * W)
    steps = [(j + 1) / args.ig_k for j in range(args.ig_k)]
    A = torch.zeros(args.ig_k, C, device=dev, dtype=torch.float64)
    for j, t in enumerate(steps):
        with torch.no_grad():
            for p in mods:
                Wt[p].copy_(W0[p] * t)
        target.zero_grad(set_to_none=True)
        reward().backward()
        with torch.no_grad():
            for p in mods:
                gw = (Wt[p].grad * W0[p])          # dR/dW * W (original W)
                A[j] += torch.bincount(
                    S[p][0].reshape(-1).int(),
                    weights=(S[p][1].float() * gw[None]).reshape(-1).double(),
                    minlength=C)
        target.zero_grad(set_to_none=True)
        log(f"  path step {j + 1}/{args.ig_k} (wscale {t:.3f})")
    with torch.no_grad():
        for p in mods:
            Wt[p].copy_(W0[p])
            Wt[p].requires_grad_(False)

    first_order = A[-1].cpu().numpy()             # gradient at the real weights
    ig = {k: A[-k:].mean(0).cpu().numpy() if k > 1 else first_order
          for k in (1, 2, 4, 8, args.ig_k) if k <= args.ig_k}
    # IG over the whole path is the mean of dR/dalpha across all steps
    ig["full"] = A.mean(0).cpu().numpy()

    # ---- ground truth: actually ablate each sampled component ----
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    mass = np.zeros(C)
    with torch.no_grad():
        for p in mods:
            mass += torch.bincount(
                S[p][0].reshape(-1).int(),
                weights=(S[p][1].float() * (W0[p] ** 2)[None]).reshape(-1).double(),
                minlength=C).cpu().numpy()
    # sample across the mass range so the correlation is not driven by one tail
    order = np.argsort(-mass)
    picks = sorted(set(order[np.linspace(0, C - 1, args.sample).astype(int)]
                       .tolist()))
    with torch.no_grad():
        base_R = float(reward())
        actual = []
        for c in picks:
            for p in mods:
                m = (S[p][1] * (S[p][0] == c)).sum(0, dtype=torch.float32)
                Wt[p].copy_(W0[p] * (1.0 - m))
            actual.append(base_R - float(reward()))
        for p in mods:
            Wt[p].copy_(W0[p])
    actual = np.array(actual)
    log(f"measured {len(picks)} true ablations "
        f"({time.perf_counter() - t00:.0f}s)")

    out = {"format": "causal_features_v1", "C": C, "ig_k": args.ig_k,
           "n_sampled": len(picks), "base_reward": base_R,
           "note": "attribution predicts R(full) - R(ablated); "
                   "sign convention matches `actual`",
           "correlations": {}}
    for name, vec in [("first_order_grad_x_weight", first_order),
                      ("ig_2", ig.get(2)), ("ig_4", ig.get(4)),
                      ("ig_8", ig.get(8)), (f"ig_{args.ig_k}_full", ig["full"])]:
        if vec is None:
            continue
        pred = vec[picks]
        out["correlations"][name] = {
            "pearson": round(float(np.corrcoef(pred, actual)[0, 1]), 4),
            "spearman": round(spearman(pred, actual), 4),
            "sum_ratio_vs_total": round(
                float(vec.sum() / max(base_R - 0.0, 1e-9)), 4),
        }
        r = out["correlations"][name]
        log(f"{name:<26} pearson {r['pearson']:+.4f}  "
            f"spearman {r['spearman']:+.4f}  "
            f"completeness Σattr/R {r['sum_ratio_vs_total']:.4f}")
    out["sample"] = [{"component": int(c), "actual": round(float(a), 6),
                      "first_order": round(float(first_order[c]), 6),
                      "ig_full": round(float(ig["full"][c]), 6),
                      "mass": float(mass[c])}
                     for c, a in zip(picks, actual)]
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
