"""Per-token attribution gating: scale each component by its attribution to the token.

At position t, score every component by how much it contributes to predicting
the next token, then run the model with each component's owned weight mass
scaled by that score, and read the log-probability of the true next token.

Weights are shared across positions, so per-token gating means one weight build
and one forward PER POSITION. That is the honest version of the experiment and
it is what makes it expensive; everything here is arranged so the cost is a few
passes over GPU-resident memory rather than any per-component materialisation.

Score variants:
  attr        the real thing — GIM attribution of component c at token t,
              sum over c's owned entries of |dReward/dW * W| with the same
              modified backward the decomposition was built from
  posterior   the frozen fingerprint posterior p_t(c), i.e. which centroid the
              token's attribution PATTERN matches. This is what the streaming
              assignment uses, and it is a different object from `attr`
  shuffled    `attr` taken from a DIFFERENT random position — the control that
              decides whether gating works because the attribution is
              informative or merely because some components are always useful
  mass        global owned weight mass, position-independent — the other control

Gate shapes:
  topk        keep the k highest-scoring components, ablate the rest
  soft        scale every component by score / max(score), continuously

    python3.12 attribution_gating.py --positions 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import geo67
from collect_fast_impl import pass_features, setup_model, objective
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
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[8, 32, 128, 512, 2048, 4096])
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="attribution_gating.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    t00 = time.perf_counter()

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    ids = data["pile_eval"][8:9, :args.seq].to(dev)      # a mid-difficulty block
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    pos_list = sorted(torch.randperm(args.seq - 2, generator=g)[
        :args.positions].add(1).tolist())

    # ---- GIM sensor for attribution; same backward the bank was built with ----
    bank_meta = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                           weights_only=True, map_location="cpu", mmap=True)
    C = int(bank_meta["C"])
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    sm = load_stream_model(run_dir / "stream_model.pt", dev)

    # fingerprint posteriors at the sampled positions
    P = torch.tensor(pos_list, device=dev)[None]
    bi = torch.zeros_like(P)
    phi, _ = pass_features(cfg, cap, ids, P, bi, spec, scales, dim,
                           return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize((x - sm["mean"]) @ sm["projector"], dim=1)
    post = torch.softmax((y @ sm["centroids"].t()) / args.temperature, dim=1)

    # per-position GIM pre/post caches, so attribution is one outer product away
    cap.wscale = 1.0
    cap.target.zero_grad(set_to_none=True)
    from collect_fast_impl import model_pass
    with model_pass(dev, cfg.bf16, cfg.fused_attention):
        logits, cache = cap.run(ids)
        reward = objective(logits, ids, cfg.scalar)
        posts = [cache[p]["post"] for p in geo67.MODULES]
        gposts = torch.autograd.grad(reward, posts)
    PRE = {p: cache[p]["pre"].detach()[0] for p in geo67.MODULES}
    GRD = {p: gg.detach()[0] for p, gg in zip(geo67.MODULES, gposts)}
    del cache, gposts, logits, cap
    torch.cuda.empty_cache()
    log(f"GIM caches ready ({time.perf_counter() - t00:.0f}s)")

    # ---- bank + originals resident on GPU ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    mods = list(bank["modules"])
    target = geo1b.load_target_1b(dev)
    S, Wt, W0 = {}, {}, {}
    mass = torch.zeros(C, device=dev, dtype=torch.float64)
    for p in mods:
        S[p] = (bank["sidx"][p].to(dev), bank["swgt"][p].to(dev))
        w = target.get_submodule(p).weight
        W0[p], Wt[p] = w.detach().clone(), w
        mass += torch.bincount(S[p][0].reshape(-1).int(),
                               weights=(S[p][1].float()
                                        * (W0[p] ** 2)[None]).reshape(-1).double(),
                               minlength=C)
    del bank
    log(f"bank resident ({time.perf_counter() - t00:.0f}s, "
        f"{torch.cuda.memory_allocated(dev) / 2**30:.0f} GiB)")

    @torch.no_grad()
    def logprob_at(t):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            lg = target(ids[:, :t + 1])
        return float(F.log_softmax(lg[0, -1].float(), -1)[ids[0, t + 1]])

    @torch.no_grad()
    def attribution(t):
        """|dReward/dW * W| summed over each component's owned entries."""
        a = torch.zeros(C, device=dev, dtype=torch.float64)
        for p in mods:
            M = (GRD[p][t][:, None] * PRE[p][t][None, :] * W0[p]).abs()
            a += torch.bincount(S[p][0].reshape(-1).int(),
                                weights=(S[p][1].float() * M[None]
                                         ).reshape(-1).double(), minlength=C)
        return a

    @torch.no_grad()
    def apply_gate(gate):
        """gate: [C] float in [0,1]; scale each component's owned mass."""
        for p in mods:
            gsum = (S[p][1] * gate[S[p][0].int()]).sum(0, dtype=torch.float32)
            keep = (S[p][1]).sum(0, dtype=torch.float32)
            Wt[p].copy_(W0[p] * (1.0 - keep + gsum))

    @torch.no_grad()
    def restore():
        for p in mods:
            Wt[p].copy_(W0[p])

    restore()
    base = {t: logprob_at(t) for t in pos_list}
    log(f"baseline mean log p = {sum(base.values()) / len(base):+.4f}")

    scores = {}
    t0 = time.perf_counter()
    for i, t in enumerate(pos_list):
        scores[t] = attribution(t)
    log(f"attribution for {len(pos_list)} positions in "
        f"{time.perf_counter() - t0:.0f}s")
    shuffle = pos_list[1:] + pos_list[:1]

    variants = {}
    for i, t in enumerate(pos_list):
        variants.setdefault("attr", {})[t] = scores[t]
        variants.setdefault("posterior", {})[t] = post[i].double()
        variants.setdefault("shuffled", {})[t] = scores[shuffle[i]]
        variants.setdefault("mass", {})[t] = mass

    results = {}
    for vname, per_pos in variants.items():
        for shape in ("topk", "soft"):
            key = f"{vname}_{shape}"
            ks = args.ks if shape == "topk" else [None]
            for k in ks:
                per, t1 = {}, time.perf_counter()
                for t in pos_list:
                    s = per_pos[t]
                    if shape == "topk":
                        gate = torch.zeros(C, device=dev)
                        gate[torch.topk(s, min(k, C)).indices] = 1.0
                    else:
                        gate = (s / s.max().clamp_min(1e-30)).float()
                    apply_gate(gate)
                    per[t] = logprob_at(t) - base[t]
                tot = sum(per.values())
                results.setdefault(key, []).append(
                    {"k": k, "delta_logprob": round(tot / len(pos_list), 5),
                     "per_position": {str(t): round(v, 5)
                                      for t, v in per.items()}})
                log(f"  {key:<18} k={str(k):<5} Δlogp "
                    f"{tot / len(pos_list):+8.4f}  "
                    f"({time.perf_counter() - t1:.0f}s)")
    restore()

    out = {"format": "attribution_gating_v1", "C": C,
           "positions": pos_list, "ks": args.ks,
           "baseline_mean_logprob": round(sum(base.values()) / len(base), 5),
           "sensor": {"name": "gim", "tau": cfg.gim_tau, "ig_k": 1,
                      "objective": cfg.scalar,
                      "feature": "dReward/dW * |W| / sqrt(R*q)"},
           "results": results}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
