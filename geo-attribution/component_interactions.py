"""Do the parameters inside one component actually interact?

The decomposition assigns every weight ENTRY a share, over 112 matrices, with
no notion of layers, neurons or circuits in the objective. So "do a component's
parameters belong together" has a ground-truth version that needs no
interpretation, because some parameters are structurally coupled whatever any
decomposition says:

  MLP neuron   in SwiGLU, neuron i is gate_proj row i, up_proj row i and
               down_proj column i. Its output is act(gate_i . x) * (up_i . x),
               written out through down[:, i]. Three matrices, one unit.

  OV dimension o_proj column j pairs with v_proj row j inside a head: the same
               coordinate of the head's output. (Under GQA query head h reads
               kv head h // 4.)

If a component's mass on gate row i predicts its mass on up row i above chance,
the component is collecting whole neurons rather than unrelated weights.

The null matters more than the statistic. Both mass vectors concentrate on
neurons with large weights, so raw mass would correlate for reasons that have
nothing to do with the partition. Two controls:

  share    divide each neuron's mass by the mass ALL components put there, so
           the global weight-magnitude profile cancels
  permute  compare component c's gate profile against a DIFFERENT component's
           up profile, which preserves every marginal and destroys only the
           pairing

    python3.12 component_interactions.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import geo1b  # noqa: F401
from german_vpd_1b import log


def owned_mass(bank, target, path, dev, C, axis):
    """[C, N] squared owned weight mass per row (axis=0) or column (axis=1)."""
    w2 = target.get_submodule(path).weight.detach().float().to(dev) ** 2
    sidx = bank["sidx"][path].to(dev)
    swgt = bank["swgt"][path].to(dev)
    N = w2.shape[axis]
    out = torch.zeros(C, N, device=dev)
    # accumulate over the other axis in slabs; sidx/swgt are [8, out, in]
    step = 512
    for s in range(0, N, step):
        sl = slice(s, min(s + step, N))
        if axis == 0:
            idx, wgt, ww = sidx[:, sl, :], swgt[:, sl, :], w2[sl, :]
        else:
            idx, wgt, ww = sidx[:, :, sl], swgt[:, :, sl], w2[:, sl]
        contrib = wgt.float() * ww[None]                    # [8, a, b]
        n = contrib.shape[1] if axis == 0 else contrib.shape[2]
        # bin by (component, position-along-axis)
        pos = (torch.arange(n, device=dev).view(1, -1, 1) if axis == 0
               else torch.arange(n, device=dev).view(1, 1, -1))
        pos = pos.expand_as(idx)
        flat = (idx.long() * n + pos).reshape(-1)
        out[:, sl] = torch.bincount(
            flat, weights=contrib.reshape(-1), minlength=C * n
        ).view(C, n).float()
        del idx, wgt, ww, contrib, pos, flat
    del w2, sidx, swgt
    return out


def coherence(A, B, g, dev):
    """Cosine between matched profiles vs the same against a shuffled partner.

    A, B are [C, N] SHARES (each column already normalised by the total mass
    all components put on that index), so a component that simply owns big
    weights does not score.
    """
    An = torch.nn.functional.normalize(A, dim=1)
    Bn = torch.nn.functional.normalize(B, dim=1)
    obs = (An * Bn).sum(1)
    perm = torch.randperm(A.shape[0], generator=g, device=dev)
    null = (An * Bn[perm]).sum(1)
    return obs, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="component_interactions.json")
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
    layers = args.layers if args.layers is not None else list(range(L))
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    g = torch.Generator(device=dev).manual_seed(args.seed)

    def share_of(m):
        return m / m.sum(0, keepdim=True).clamp_min(1e-30)

    acc = {k: {"obs": [], "null": []} for k in
           ("gate-up", "gate-down", "up-down", "v-o")}
    # layer-shifted control: the SAME component's gate profile against its own
    # up profile from a different layer. Same component, same two matrices,
    # every marginal preserved -- only the neuron correspondence is gone. If
    # the within-layer score is just "components have a consistent signature",
    # this scores the same.
    shifted = {"obs": []}
    keep_gate = {}
    for l in layers:
        p = f"hf.model.layers.{l}"
        gate = share_of(owned_mass(bank, target, f"{p}.mlp.gate_proj", dev, C, 0))
        up = share_of(owned_mass(bank, target, f"{p}.mlp.up_proj", dev, C, 0))
        down = share_of(owned_mass(bank, target, f"{p}.mlp.down_proj", dev, C, 1))
        for name, (A, B) in (("gate-up", (gate, up)), ("gate-down", (gate, down)),
                             ("up-down", (up, down))):
            o, n = coherence(A, B, g, dev)
            acc[name]["obs"].append(o.cpu())
            acc[name]["null"].append(n.cpu())
        if keep_gate:
            prev_l, prev_gate = next(iter(keep_gate.items()))
            An = torch.nn.functional.normalize(prev_gate, dim=1)
            Bn = torch.nn.functional.normalize(up, dim=1)
            shifted["obs"].append((An * Bn).sum(1).cpu())
        keep_gate = {l: gate.clone()}
        del gate, up, down

        # OV: o_proj column (h*HD + d) pairs with v_proj row ((h//REP)*HD + d)
        v = owned_mass(bank, target, f"{p}.self_attn.v_proj", dev, C, 0)
        o_ = owned_mass(bank, target, f"{p}.self_attn.o_proj", dev, C, 1)
        v_exp = torch.cat([v[:, (h // REP) * HD:(h // REP + 1) * HD]
                           for h in range(H)], dim=1)      # [C, H*HD]
        ov_o, ov_n = coherence(share_of(v_exp), share_of(o_), g, dev)
        acc["v-o"]["obs"].append(ov_o.cpu())
        acc["v-o"]["null"].append(ov_n.cpu())
        del v, o_, v_exp
        log(f"layer {l} done")
    del bank

    sh = torch.stack(shifted["obs"]) if shifted["obs"] else None
    out = {"format": "component_interactions_v1", "C": C,
           "layers": layers, "pairs": {},
           "layer_shifted_gate_up": (None if sh is None else {
               "mean": round(float(sh.mean()), 5),
               "median": round(float(sh.median()), 5)})}
    for name, d in acc.items():
        o = torch.stack(d["obs"])            # [n_layers, C]
        n = torch.stack(d["null"])
        out["pairs"][name] = {
            "observed_mean": round(float(o.mean()), 5),
            "null_mean": round(float(n.mean()), 5),
            "ratio": round(float(o.mean() / n.mean().clamp_min(1e-12)), 3),
            "frac_component_layers_above_null_p99":
                round(float((o > n.quantile(0.99)).float().mean()), 4),
            "median_observed": round(float(o.median()), 5),
            "median_null": round(float(n.median()), 5),
        }
        r = out["pairs"][name]
        log(f"{name:<10} cosine {r['observed_mean']:.4f} vs null "
            f"{r['null_mean']:.4f}  ratio {r['ratio']:.1f}x  "
            f"{100 * r['frac_component_layers_above_null_p99']:.1f}% above "
            f"null p99")

    if sh is not None:
        gu = torch.stack(acc["gate-up"]["obs"])
        log(f"layer-shifted gate-up control: {float(sh.mean()):.4f} "
            f"vs within-layer {float(gu.mean()):.4f} "
            f"({float(gu.mean() / sh.mean().clamp_min(1e-12)):.1f}x)")
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    torch.save({k: {kk: torch.stack(vv) for kk, vv in d.items()}
                for k, d in acc.items()},
               run_dir / "component_interactions_arrays.pt")
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
