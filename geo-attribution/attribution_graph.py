"""VPD Section 5: attribution graphs, as causal graphs over (component, layer).

VPD defines attr(c' -> c) = (d a_c / d a_c')* a_c' g_c', a gradient through
subcomponent activations with stop-gradients on everything else. That needs two
things we do not have: rank-1 subcomponents with a scalar activation a_c, and a
learned causal-importance gate g_c. Our components are shares of every weight
entry and have no gate.

So the edge is defined causally instead, which is the thing the gradient is a
first-order approximation of anyway:

    node   a whole component c, perturbed in all 112 matrices at once.
    DE(c)  direct effect = drop in log p(target token) under the perturbation.
    edge   e(c' -> c) = DE(c) - DE(c | c' already perturbed), for depth(c') <
           depth(c). Reads as: how much of c's effect depends on c' having
           run. A large positive edge means c is downstream of c'.

The perturbation is SCALING, not ablation, and that is a real difference from
VPD. VPD trains its subcomponents so that ablating them is meaningful — that is
what the causal-importance gate and the adversarial-ablation objective buy. Our
components are a geometric partition of already-trained weights with no such
objective, and it shows: zeroing a whole component moves log p(target) by about
0.002 nats, indistinguishable from noise, while scaling the same component by 4
moves it by up to 5 nats. So the graph is built on the intervention this
decomposition actually responds to.

Direction comes from each component's DEPTH — the mass-weighted mean layer of
its owned parameters. Components span layers, so there is no exact causal
order; the centroid is the honest approximation and it is reported per node so
a reader can see how separated two nodes actually are.

Pruning (VPD 5.2) is by direct effect: take the components active on the prompt
by fingerprint posterior, score every (component, layer) node, keep the top-K.

    python3.12 attribution_graph.py --case pronoun
    python3.12 attribution_graph.py --case induction
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_permatrix import PerMatrixEditor
from german_vpd_1b import log, ranking_args

CASES = {
    # VPD case study 1: gendered possessive pronoun
    "pronoun": {"prompt": "The princess lost her crown. The princess said that",
                "target": " she",
                "note": "gendered pronoun agreement"},
    # VPD case study 2: bracket closing
    "bracket": {"prompt": "def f(x):\n    return (x + 1",
                "target": ")",
                "note": "closing an open bracket"},
    # our own validated circuit: c3392 is the induction mechanism, c108 the
    # readout, so a correct graph should surface them and link them
    "induction": {"prompt": "Serial: 8QK4T2M91XZ. Please confirm the serial: 8QK",
                  "target": "4",
                  "note": "copy a novel string seen earlier in context"},
}


@torch.no_grad()
def target_logprob(fwd, ids, tgt_id):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        lg = fwd(ids)
    return float(F.log_softmax(lg[0, -1].float(), -1)[tgt_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--case", default="pronoun", choices=sorted(CASES))
    ap.add_argument("--cand_per_pos", type=int, default=4)
    ap.add_argument("--max_cands", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--force", type=int, nargs="*", default=None,
                    help="components to add to the candidate pool regardless "
                         "of posterior (used to test whether a known circuit "
                         "is recovered)")
    ap.add_argument("--min_depth_gap", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="the intervention. alpha=0 ablates; this "
                         "decomposition barely responds to that (0.002 nats), "
                         "so the default AMPLIFIES the component's owned mass "
                         "instead, which moves log p by nats.")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    case = CASES[args.case]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    ids = torch.tensor([tok.encode(case["prompt"])], device=dev)
    tgt = tok.encode(case["target"], add_special_tokens=False)[0]
    log(f"case '{args.case}': {case['prompt']!r} -> {case['target']!r} "
        f"({ids.shape[1]} tokens)")

    # ---- pruning step 1: which components are even active here? ----
    bank_meta = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                           weights_only=True, map_location="cpu", mmap=True)
    C = int(bank_meta["C"])
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    sm = load_stream_model(run_dir / "stream_model.pt", dev)
    T = ids.shape[1]
    pos1 = torch.arange(1, T, device=dev)
    pos = pos1[None]
    bi = torch.zeros_like(pos)
    phi, _ = pass_features(cfg, cap, ids, pos, bi, spec, scales, dim,
                           return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize((x - sm["mean"]) @ sm["projector"], dim=1)
    post = torch.softmax((y @ sm["centroids"].t()) / args.temperature, dim=1)
    tops = torch.topk(post, args.cand_per_pos, dim=1).indices
    counts = torch.bincount(tops.reshape(-1), minlength=C).float()
    weight = post.sum(0)
    cands = torch.topk(counts * 1000 + weight, args.max_cands).indices.tolist()
    for c in (args.force or []):
        if c not in cands:
            cands.append(int(c))
    tok_strs = [tok.decode([int(t)]) for t in ids[0]]
    log(f"{len(cands)} candidate components active on this prompt")
    del cap, phi, x, y

    # ---- score every (component, layer) node by direct effect ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)
    L = target.hf.config.num_hidden_layers

    def depth_of(ed, c_slot):
        """Mass-weighted mean layer of a component's owned parameters."""
        num = den = 0.0
        for i, m in enumerate(ed.modules):
            l = int(m.split("layers.")[1].split(".")[0])
            w = float(ed.slices[m][c_slot].square().sum())
            num += l * w
            den += w
        return num / max(den, 1e-30)

    node_scores = []
    t0 = time.perf_counter()
    for c in cands:
        ed = PerMatrixEditor(target, bank, [c], dev)
        n_mod = len(ed.modules)
        base = target_logprob(lambda z: ed.logits(z, None), ids, tgt)
        lp = target_logprob(
            lambda z: ed.logits(
                z, torch.full((1, n_mod), args.alpha, device=dev)), ids, tgt)
        node_scores.append({"component": c, "direct_effect": base - lp,
                            "depth": round(depth_of(ed, 0), 3),
                            "mass_fraction": ed.mass_fraction[0]})
        ed.alpha = None
        del ed
        torch.cuda.empty_cache()
    node_scores.sort(key=lambda r: -abs(r["direct_effect"]))
    nodes = node_scores[:args.nodes]
    log(f"scored {len(node_scores)} components in "
        f"{time.perf_counter() - t0:.0f}s; keeping {len(nodes)}")
    for n in nodes:
        log(f"   c{n['component']:<5} depth {n['depth']:5.2f}  "
            f"DE {n['direct_effect']:+.4f}")

    # ---- edges: does the downstream node still matter without the upstream? ----
    edges = []
    for nd in nodes:
        for nu in nodes:
            if nu["component"] == nd["component"]:
                continue
            if nu["depth"] >= nd["depth"] - args.min_depth_gap:
                continue                  # direction is depth order
            comps = [nu["component"], nd["component"]]
            ed = PerMatrixEditor(target, bank, comps, dev)
            n_mod = len(ed.modules)

            def alpha(ku, kd):
                a = torch.ones(2, n_mod, device=dev)
                if ku:
                    a[0, :] = args.alpha
                if kd:
                    a[1, :] = args.alpha
                return a

            base = target_logprob(lambda z: ed.logits(z, None), ids, tgt)
            lp_u = target_logprob(lambda z: ed.logits(z, alpha(1, 0)), ids, tgt)
            lp_d = target_logprob(lambda z: ed.logits(z, alpha(0, 1)), ids, tgt)
            lp_b = target_logprob(lambda z: ed.logits(z, alpha(1, 1)), ids, tgt)
            de_d = base - lp_d                       # downstream effect alone
            de_d_given_u = lp_u - lp_b               # downstream effect after u
            edges.append({"from": f"c{nu['component']}",
                          "to": f"c{nd['component']}",
                          "from_component": nu["component"],
                          "to_component": nd["component"],
                          "from_depth": nu["depth"], "to_depth": nd["depth"],
                          "weight": round(de_d - de_d_given_u, 6),
                          "de_downstream": round(de_d, 6),
                          "de_downstream_given_upstream":
                              round(de_d_given_u, 6)})
            ed.alpha = None
            del ed
            torch.cuda.empty_cache()
    del bank
    edges.sort(key=lambda e: -abs(e["weight"]))
    log(f"{len(edges)} directed edges; strongest:")
    for e in edges[:8]:
        log(f"   {e['from']:>8} -> {e['to']:<8} w {e['weight']:+.5f} "
            f"(DE {e['de_downstream']:+.4f} -> {e['de_downstream_given_upstream']:+.4f})")

    catalog = json.loads((run_dir / "catalog_prop1b_1B.json").read_text())
    out = {"format": "attribution_graph_v1", "case": args.case,
           "intervention_alpha": args.alpha,
           "prompt": case["prompt"], "target": case["target"],
           "note": case["note"], "tokens": tok_strs,
           "base_logprob": target_logprob(lambda z: target(z), ids, tgt),
           "nodes": [{**n, "label": catalog[str(n["component"])]["label"],
                      "category": catalog[str(n["component"])]["category"],
                      "id": f"c{n['component']}"}
                     for n in nodes],
           "edges": edges,
           "all_node_scores": node_scores[:60]}
    name = args.out or f"attribution_graph_{args.case}.json"
    (run_dir / name).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / name}")


if __name__ == "__main__":
    main()
