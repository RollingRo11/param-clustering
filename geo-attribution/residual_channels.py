"""Do a component's parameters at different layers talk through the residual?

component_interactions.py shows components collect whole MLP neurons: within a
layer, gate row i / up row i / down column i are owned together at 138x the
shuffled-partner null and 89x a layer-shifted control. That is a WITHIN-layer
interaction.

The cross-layer question is different. Every layer communicates through the
residual stream, so if a component's parameters interact across layers, the
residual channels it WRITES at layer l should be the channels it READS at some
later layer l'.

  write   rows of o_proj and down_proj (their output axis is the residual)
  read    columns of q/k/v/gate/up  (their input axis is the residual)

Scored as cosine between a component's write-share profile at l and its
read-share profile at l' > l, against the same quantity for a different
component (which preserves every marginal and destroys only the identity).

Caveat worth stating: the residual basis is not privileged under rotation, so
coordinate overlap is a weaker notion than subspace alignment. It is meaningful
here only to the extent the residual basis is de facto privileged in a trained
transformer, which is a real but partial effect.

    python3.12 residual_channels.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import geo1b  # noqa: F401
from german_vpd_1b import log
from component_interactions import owned_mass

WRITE = ("self_attn.o_proj", "mlp.down_proj")     # residual on the ROW axis
READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "mlp.gate_proj", "mlp.up_proj")           # residual on the COLUMN axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="residual_channels.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    target = geo1b.load_target_1b(dev)
    L = target.hf.config.num_hidden_layers
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    g = torch.Generator(device=dev).manual_seed(args.seed)

    def share(m):
        return torch.nn.functional.normalize(
            m / m.sum(0, keepdim=True).clamp_min(1e-30), dim=1)

    w_prof, r_prof = [], []
    for l in range(L):
        p = f"hf.model.layers.{l}"
        w = sum(owned_mass(bank, target, f"{p}.{m}", dev, C, 0) for m in WRITE)
        r = sum(owned_mass(bank, target, f"{p}.{m}", dev, C, 1) for m in READ)
        w_prof.append(share(w))
        r_prof.append(share(r))
        log(f"layer {l} residual profiles done")
    del bank

    perm = torch.randperm(C, generator=g, device=dev)
    by_gap, obs_all, null_all = {}, [], []
    for l in range(L):
        for l2 in range(l + 1, L):
            o = (w_prof[l] * r_prof[l2]).sum(1)
            n = (w_prof[l] * r_prof[l2][perm]).sum(1)
            gap = l2 - l
            by_gap.setdefault(gap, {"o": [], "n": []})
            by_gap[gap]["o"].append(o.mean().item())
            by_gap[gap]["n"].append(n.mean().item())
            obs_all.append(o.cpu())
            null_all.append(n.cpu())
    O, N = torch.stack(obs_all), torch.stack(null_all)
    out = {
        "format": "residual_channels_v1", "C": C, "L": L,
        "overall": {
            "observed_mean": round(float(O.mean()), 5),
            "null_mean": round(float(N.mean()), 5),
            "ratio": round(float(O.mean() / N.mean().clamp_min(1e-12)), 3),
            "frac_above_null_p99": round(float((O > N.quantile(0.99))
                                               .float().mean()), 4),
        },
        "by_layer_gap": {
            str(k): {"observed": round(sum(v["o"]) / len(v["o"]), 5),
                     "null": round(sum(v["n"]) / len(v["n"]), 5),
                     "ratio": round((sum(v["o"]) / len(v["o"]))
                                    / max(sum(v["n"]) / len(v["n"]), 1e-12), 3)}
            for k, v in sorted(by_gap.items())},
    }
    r = out["overall"]
    log(f"write(l) vs read(l') cosine {r['observed_mean']:.4f} vs null "
        f"{r['null_mean']:.4f}  ratio {r['ratio']:.2f}x  "
        f"{100 * r['frac_above_null_p99']:.1f}% above null p99")
    for k in ("1", "2", "4", "8", "15"):
        if k in out["by_layer_gap"]:
            b = out["by_layer_gap"][k]
            log(f"  gap {k:>2} layers: {b['observed']:.4f} vs {b['null']:.4f} "
                f"({b['ratio']:.2f}x)")
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
