"""Keep-top-k ablation curves (canonical + oracle) for v1, v2, and SPD on the
same events, per the updated proposal's sec 5.1 protocol. Each method's
candidate components are ablated least-important-first per event; anything
not listed as a candidate (v2's background C0, undecomposed embed/unembed)
always stays, so the v2 curve automatically follows the C0-retained rule.

Writes curves_<tag>.json per method into --out_dir.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from induction_model import VOCAB, InductionModel
from atoms import AtomBasis, collect_attributions, make_events
from ablation_curve import build_ranks


def curves_for(model, comps, canonical_imp, events, dev, chunk=1024,
               orders=("canonical", "oracle"), seed=0, log=print):
    seq, m_tok = events["seq"], events["m_token"]
    n_comp = next(iter(comps.values())).shape[0]
    params = {n: p.detach() for n, p in model.named_parameters()}
    in_dims = ({n: (0 if n in comps else None) for n in params}, 0)

    def fwd(pdict, s):
        return functional_call(model, pdict, (s.unsqueeze(0),))[0, -1]
    batched_fwd = vmap(fwd, in_dims=in_dims)

    @torch.no_grad()
    def oracle_imp():
        base = F.log_softmax(model(seq)[:, -1], -1).gather(
            1, m_tok[:, None]).squeeze(1)
        imp = torch.zeros(seq.shape[0], n_comp, device=dev)
        for c in range(n_comp):
            pdict = dict(params)
            for name, ct in comps.items():
                pdict[name] = params[name] - ct[c]
            lg = functional_call(model, pdict, (seq,))[:, -1]
            lp = F.log_softmax(lg, -1).gather(1, m_tok[:, None]).squeeze(1)
            imp[:, c] = (base - lp).abs()
        return imp

    @torch.no_grad()
    def ce_at(rank, k):
        ces, hits = [], []
        for i in range(0, seq.shape[0], chunk):
            sl = slice(i, i + chunk)
            mask = (rank[sl] < k).float()
            pdict = dict(params)
            for name, ct in comps.items():
                delta = (mask @ ct.reshape(n_comp, -1)).reshape(
                    -1, *ct.shape[1:])
                pdict[name] = params[name] - delta
            logits = batched_fwd(pdict, seq[sl])
            ces.append(F.cross_entropy(logits, m_tok[sl], reduction="none"))
            hits.append(logits.argmax(-1) == m_tok[sl])
        return (torch.cat(ces).mean().item(),
                torch.cat(hits).float().mean().item())

    base_ce, base_acc = ce_at(torch.zeros(seq.shape[0], n_comp,
                                          dtype=torch.long, device=dev), 0)
    out = {"C": n_comp, "base_ce": base_ce, "base_acc": base_acc,
           "uniform_ce": math.log(VOCAB), "curves": {}}
    for order in orders:
        imp = canonical_imp if order == "canonical" else oracle_imp()
        rank = build_ranks("per_example_asc", imp, seed)
        curve = []
        for k in range(n_comp + 1):
            ce, acc = ce_at(rank, k)
            curve.append({"k": k, "ce": round(ce, 6), "acc": round(acc, 5),
                          "delta": round(ce - base_ce, 6)})
            if k % 100 == 0 or k == n_comp:
                log(f"  {order:<10} K={k:<4} CE {ce:9.4f} acc {acc:.4f}")
        out["curves"][order] = curve
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--spd", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    [p.requires_grad_(False) for p in model.parameters()]
    events = make_events(model, args.n_seq, "final", seed=args.seed + 1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]

    # ---- v1 and v2: canonical = |z| from log p ----
    for tag, run in (("v1", args.v1), ("v2", args.v2)):
        fact = torch.load(run / "factorization.pt", weights_only=False,
                          map_location=dev)
        V = fact["V"].to(dev)
        matrices = sorted(set(fact["atom_matrix"]),
                          key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
        comps = {n: t.to(dev) for n, t in
                 basis.components(V).items()}
        z = collect_attributions(model, basis, seq, pos, y, score="logp") @ V
        print(f"== {tag} ==", flush=True)
        blob = curves_for(model, comps, z.abs(), events, dev, seed=args.seed)
        (args.out_dir / f"curves_{tag}.json").write_text(json.dumps(blob))

    # ---- SPD: canonical = causal importance at circuit positions ----
    from spd_analysis import load_spd
    from spd_toy import MODULES
    smodel, wrappers, gates, c_per = load_spd(args.spd, args.ckpt, dev)
    sp = {n: t.float().to(dev) for n, t in torch.load(
        args.spd / "components.pt", weights_only=True,
        map_location=dev)["components"].items()}
    from induction_model import gen_batch
    B = seq.shape[0]
    rows = torch.arange(B, device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)
    seq2, s_pos, _ = gen_batch(args.n_seq, dev, gen)
    assert (seq2 == seq).all(), "event seqs out of sync with gen_batch"
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        smodel(seq)
        g = torch.cat([gates[n.replace(".", "_")](wrappers[n].last_input)[0]
                       for n in sorted(MODULES)], dim=2)
        final = torch.full_like(s_pos, seq.shape[1] - 1)
        ci = torch.stack([g[rows, p] for p in (s_pos, s_pos + 1, final)]
                         ).amax(0)
    print("== spd ==", flush=True)
    blob = curves_for(model, sp, ci, events, dev, seed=args.seed)
    (args.out_dir / "curves_spd.json").write_text(json.dumps(blob))
    print("done")


if __name__ == "__main__":
    main()
