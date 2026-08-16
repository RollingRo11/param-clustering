"""2-GPU parallel per-token oracle ablation curve.

Same output as ablation_curve.py (json + figures), computed with heavy
batching across both GPUs:

  phase A  oracle importance: components sharded across ranks; each rank
           evaluates its components as one flat (component x event) vmapped
           batch in large chunks instead of C sequential forwards.
  phase B  curve: K grid points sharded across ranks; each K evaluates all
           events with per-event ablated weights in one flat batch.

Ranks exchange results with torch.distributed (gloo, file rendezvous).

    python curve2gpu.py --components out/vpd_C100/components.pt \
        --orders per_example_asc:oracle
"""
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.func import functional_call, vmap

from induction_model import VOCAB, InductionModel
from atoms import AtomBasis, collect_attributions, collect_grads, make_events
from ablation_curve import build_ranks, plot_acc, plot_ce

CHUNK = 32768   # flat (variant x event) sequences per vmapped forward


def worker(rank: int, world: int, args, rdv_file: str):
    dist.init_process_group("gloo", init_method=f"file://{rdv_file}",
                            rank=rank, world_size=world)
    dev = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.components is not None:
        blob = torch.load(args.components, weights_only=True, map_location=dev)
        comps = {n: t.float() for n, t in blob["components"].items()}
        basis, V = None, None
        run_dir = args.components.parent
    else:
        fact = torch.load(args.run / "factorization.pt", weights_only=False,
                          map_location=dev)
        V = fact["V"].to(dev)
        matrices = sorted(set(fact["atom_matrix"]),
                          key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
        comps = basis.components(V)
        run_dir = args.run
    n_comp = next(iter(comps.values())).shape[0]

    events = make_events(model, args.n_seq, "final", seed=args.seed + 1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    m_tok = events["m_token"]
    n_ev = seq.shape[0]

    params = {n: p.detach() for n, p in model.named_parameters()}
    in_dims = ({n: (0 if n in comps else None) for n in params}, 0)

    def fwd(pdict, s):
        return functional_call(model, pdict, (s.unsqueeze(0),))[0, -1]

    batched_fwd = vmap(fwd, in_dims=in_dims)

    @torch.no_grad()
    def flat_eval(deltas: dict[str, torch.Tensor], n_var: int) -> torch.Tensor:
        """Final-position logits for n_var weight variants x all events.
        deltas: name -> [n_var, out, in]. Returns [n_var, n_ev, vocab]."""
        seq_flat = seq.repeat(n_var, 1)
        out = []
        for i in range(0, n_var * n_ev, CHUNK):
            sl = slice(i, min(i + CHUNK, n_var * n_ev))
            var_idx = torch.arange(sl.start, sl.stop, device=dev) // n_ev
            pdict = dict(params)
            for name, d in deltas.items():
                pdict[name] = params[name] - d[var_idx]
            out.append(batched_fwd(pdict, seq_flat[sl]))
        return torch.cat(out).view(n_var, n_ev, -1)

    # -- phase A: oracle importance, components sharded by rank -------------
    with torch.no_grad():
        base_logits = model(seq)[:, -1]
        base_lp = F.log_softmax(base_logits, -1).gather(
            1, m_tok[:, None]).squeeze(1)
    my_comps = list(range(rank, n_comp, world))
    imp_local = torch.zeros(len(my_comps), n_ev, device=dev)
    for i in range(0, len(my_comps), args.comp_batch):
        cb = my_comps[i: i + args.comp_batch]
        logits = flat_eval({n: c[cb] for n, c in comps.items()}, len(cb))
        lp = F.log_softmax(logits, -1).gather(
            2, m_tok[None, :, None].expand(len(cb), -1, -1)).squeeze(2)
        imp_local[i: i + len(cb)] = (base_lp[None] - lp).abs()
    gathered: list = [None] * world
    dist.all_gather_object(gathered, imp_local.cpu())
    imp_oracle = torch.zeros(n_ev, n_comp, device=dev)
    for r, sh in enumerate(gathered):
        imp_oracle[:, list(range(r, n_comp, world))] = sh.to(dev).T
    if rank == 0:
        print(f"phase A done: oracle importance [{n_ev}, {n_comp}]",
              flush=True)

    imps = {"oracle": imp_oracle}

    def ci_importance():
        """Canonical causal-importance ordering: member subcomponents'
        max-over-position g, summed per component via the saved labels.
        Dispatches on the state format (spd_state.pt or vpd_state.pt)."""
        spd_path = args.components.parent / "spd_state.pt"
        if spd_path.exists():
            return spd_ci_importance(spd_path)
        from vpd_toy import (MODULES, CITransformer, install_components,
                             clear_masks)
        state = torch.load(args.components.parent / "vpd_state.pt",
                           weights_only=True, map_location=dev)
        lab = torch.load(args.components, weights_only=True,
                         map_location=dev)["labels"].to(dev)
        c_per = state["ci_fn"]["proj_out.weight"].shape[0] // len(MODULES)
        wrapped = InductionModel().to(dev)
        wrapped.load_state_dict(torch.load(args.ckpt)["state_dict"])
        wrapped.eval()
        wrappers = install_components(wrapped, c_per)
        wrapped.to(dev)
        for n, w in wrappers.items():
            w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
            w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
        ci_fn = CITransformer({n: 16 for n in MODULES}, c_per).to(dev)
        ci_fn.load_state_dict(state["ci_fn"])
        ci_fn.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, n_ev, 1024):
                clear_masks(wrappers)
                wrapped(seq[i: i + 1024])
                ci_lower, _ = ci_fn({n: w.last_input
                                     for n, w in wrappers.items()})
                scores.append(torch.cat([ci_lower[n].amax(dim=1)
                                         for n in sorted(MODULES)], dim=1))
            imp = torch.zeros(n_ev, n_comp, device=dev)
            imp.index_add_(1, lab, torch.cat(scores))
        return imp

    def spd_ci_importance(spd_path):
        from spd_toy import MODULES as SPD_MODULES, MatrixGate, install
        import torch.nn as nn
        state = torch.load(spd_path, weights_only=True, map_location=dev)
        lab = torch.load(args.components, weights_only=True,
                         map_location=dev)["labels"].to(dev)
        c_per = int(state["c_per_module"])
        wrapped = InductionModel().to(dev)
        wrapped.load_state_dict(torch.load(args.ckpt)["state_dict"])
        wrapped.eval()
        wrappers = install(wrapped, c_per)
        wrapped.to(dev)
        for n, w in wrappers.items():
            w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
            w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
        gates = nn.ModuleDict({n.replace(".", "_"): MatrixGate(c_per)
                               for n in SPD_MODULES}).to(dev)
        gates.load_state_dict(state["gates"])
        gates.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, n_ev, 1024):
                for w in wrappers.values():
                    w.mode, w.mask = "target", None
                wrapped(seq[i: i + 1024])
                scores.append(torch.cat(
                    [gates[n.replace(".", "_")](wrappers[n].last_input)[0]
                     .amax(dim=1) for n in sorted(SPD_MODULES)], dim=1))
            imp = torch.zeros(n_ev, n_comp, device=dev)
            imp.index_add_(1, lab, torch.cat(scores))
        return imp

    def attr_importance():
        """Canonical for gradient-based decompositions: |grad_W s_i . C_c|,
        valid for any additive components."""
        matrices = list(comps)
        grads = collect_grads(model, matrices, seq, pos, y, chunk=2048)
        z = torch.zeros(n_ev, n_comp, device=dev)
        for name in matrices:
            z += torch.einsum("noi,coi->nc", grads[name], comps[name])
        return z.abs()

    def importance(score):
        if score not in imps:
            if score == "ci":
                imps[score] = ci_importance()
            elif score == "attr":
                imps[score] = attr_importance()
            else:
                assert basis is not None, \
                    f"score '{score}' needs a factorization run"
                z = collect_attributions(model, basis, seq, pos, y,
                                         score=score) @ V
                imps[score] = z.abs()
        return imps[score]

    # -- phase B: curve, K grid sharded by rank ------------------------------
    ks = list(range(n_comp + 1))
    curves = {}
    for spec in args.orders:
        order, _, score = spec.partition(":")
        rk = build_ranks(order, importance(score or "logp"), args.seed)
        my_ks = ks[rank::world]
        rows = []
        for i in range(0, len(my_ks), args.k_batch):
            kb = my_ks[i: i + args.k_batch]
            mask = (rk[None] < torch.tensor(kb, device=dev)[:, None, None]
                    ).float()                           # [Kb, N, C]
            deltas = {n: (mask @ c.reshape(n_comp, -1)).reshape(
                len(kb) * n_ev, *c.shape[1:])
                for n, c in comps.items()}
            seq_flat = seq.repeat(len(kb), 1)
            outs = []
            for j in range(0, len(kb) * n_ev, CHUNK):
                sl = slice(j, min(j + CHUNK, len(kb) * n_ev))
                pdict = dict(params)
                for name, d in deltas.items():
                    pdict[name] = params[name] - d[sl]
                outs.append(batched_fwd(pdict, seq_flat[sl]))
            logits = torch.cat(outs).view(len(kb), n_ev, -1)
            ce = F.cross_entropy(logits.flatten(0, 1),
                                 m_tok.repeat(len(kb)),
                                 reduction="none").view(len(kb), n_ev).mean(1)
            acc = (logits.argmax(-1) == m_tok[None]).float().mean(1)
            rows += [{"k": k, "ce": round(ce[t].item(), 6),
                      "acc": round(acc[t].item(), 5)}
                     for t, k in enumerate(kb)]
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if rank == 0:
            allrows = sorted((r for g in gathered for r in g),
                             key=lambda r: r["k"])
            base_ce = allrows[0]["ce"]
            for r in allrows:
                r["delta"] = round(r["ce"] - base_ce, 6)
            curves[spec] = allrows
            print(f"phase B done: {spec} ({len(allrows)} K points)",
                  flush=True)

    if rank == 0:
        base_ce = next(iter(curves.values()))[0]["ce"]

        def acc_budget(curve, floor):
            ok = [r["k"] for r in curve if all(
                r2["acc"] >= floor for r2 in curve if r2["k"] <= r["k"])]
            return max(ok) if ok else 0

        out = {"format": "parfact_ablation_curve_v2", "C": n_comp,
               "run": str(run_dir), "n_events": n_ev,
               "base_ce": round(base_ce, 8),
               "uniform_ce": round(math.log(VOCAB), 5), "curves": curves,
               "components_removable_keeping_acc": {
                   spec: {f"{fl}": acc_budget(c, fl)
                          for fl in (0.99, 0.95, 0.9)}
                   for spec, c in curves.items()}}
        (run_dir / "ablation_curve.json").write_text(json.dumps(out, indent=1))
        for spec, b in out["components_removable_keeping_acc"].items():
            print(f"{spec:<26} removable keeping acc: " +
                  "  ".join(f">={fl}: {v} ({100 * v / n_comp:.0f}%)"
                            for fl, v in b.items()), flush=True)
        plot_acc(out, run_dir / "ablation_curve_acc.png")
        plot_ce(out, run_dir / "ablation_curve.png")
        print(f"wrote {run_dir}/ablation_curve.json + figures", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path(__file__).parent / "out/B_layer_K400_C100_long")
    ap.add_argument("--components", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--orders", nargs="+", default=["per_example_asc:oracle"])
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--comp_batch", type=int, default=16)
    ap.add_argument("--k_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", type=int, default=min(2, torch.cuda.device_count()))
    args = ap.parse_args()
    rdv = tempfile.NamedTemporaryFile(delete=False, suffix=".rdv")
    rdv.close()
    os.unlink(rdv.name)
    mp.spawn(worker, args=(args.world, args, rdv.name), nprocs=args.world,
             join=True)


if __name__ == "__main__":
    main()
