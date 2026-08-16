"""Re-cluster a trained VPD run's subcomponents into a new component count.

Training (vpd_toy.py) is independent of the final component count: it learns
8 x c_per_module rank-one subcomponents + the CI function. This loads
vpd_state.pt, recomputes CI profiles, k-means them into --n_components
model-spanning components and writes a components.pt for ablation curves.

    python vpd_recluster.py --state out/vpd_C100_ddp/vpd_state.pt \
        --n_components 600 --out out/vpd_C600_ddp
"""
import argparse
from pathlib import Path

import torch

from induction_model import InductionModel, gen_batch
from prev_method import kmeans
from vpd_toy import MODULES, CITransformer, install_components, clear_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--n_components", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.state, weights_only=True, map_location=dev)
    c_per = state["ci_fn"]["proj_out.weight"].shape[0] // len(MODULES)
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    wrappers = install_components(model, c_per)
    model.to(dev)
    for n, w in wrappers.items():
        w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
        w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
    ci_fn = CITransformer({n: 16 for n in MODULES}, c_per).to(dev)
    ci_fn.load_state_dict(state["ci_fn"])
    ci_fn.eval()

    gen = torch.Generator(device=dev).manual_seed(args.seed + 7)
    with torch.no_grad():
        profiles = []
        for _ in range(8):
            seq, _, _ = gen_batch(512, dev, gen)
            clear_masks(wrappers)
            model(seq)
            ci_lower, _ = ci_fn({n: w.last_input for n, w in wrappers.items()})
            profiles.append(torch.cat(
                [ci_lower[n].flatten(0, 1).T for n in sorted(MODULES)]))
        raw = torch.cat(profiles, dim=1)
        norms = raw.norm(dim=1, keepdim=True)
        alive = norms.squeeze(1) > 1e-3
        print(f"subcomponents alive: {int(alive.sum())}/{raw.shape[0]}")
        prof = torch.where(alive[:, None], raw / norms.clamp_min(1e-12),
                           torch.zeros_like(raw))
        lab = kmeans(prof, args.n_components, iters=25, seed=args.seed)

        comps = {}
        total_ci = torch.zeros(args.n_components, device=dev)
        total_ci.index_add_(0, lab, raw.sum(1))
        dump_cluster = int(total_ci.argmin())
        for mi, path in enumerate(sorted(MODULES)):
            w = wrappers[path]
            sub = torch.einsum("co,ic->coi", w.U, w.V)
            lab_m = lab[mi * w.C:(mi + 1) * w.C]
            dense = torch.zeros(args.n_components, *w.W_target.shape,
                                device=dev)
            dense.index_add_(0, lab_m, sub)
            dense[dump_cluster] += w.weight_delta()
            comps[path + ".weight"] = dense
            err = (dense.sum(0) - w.W_target).abs().max().item()
            assert err < 1e-4, (path, err)
        n_empty = int((torch.bincount(lab, minlength=args.n_components)
                       == 0).sum())
        print(f"{args.n_components} components ({n_empty} empty clusters), "
              f"residual folded into cluster {dump_cluster}")

    torch.save({"components": {n: c.cpu() for n, c in comps.items()},
                "labels": lab.cpu(),
                "config": {k: str(v) for k, v in vars(args).items()}},
               args.out / "components.pt")
    import shutil
    shutil.copy(args.state, args.out / "vpd_state.pt")
    print(f"saved {args.out}/components.pt (+ vpd_state.pt copy)")


if __name__ == "__main__":
    main()
