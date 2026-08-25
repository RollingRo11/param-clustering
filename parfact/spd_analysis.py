"""Active-subcomponent census + circuit structure, following Christensen &
Riggs Smith's evaluation of the toy induction model.

Reproduces their Table 1a/1b logic on our SPD run:
  - per matrix and token type (s1 = first s, m = s1+1, s2 = final, random),
    the average number of subcomponents with causal importance g > 0.5;
  - the total number of UNIQUE subcomponents active anywhere (they report 7).

Then probes whether our active subcomponents implement their 2-step circuit:
  - layer-0 attention m -> s1 (previous-token step) and layer-1 attention
    s2 -> m (match step), before/after ablating the active subcomponents of
    the matrices said to carry each step;
  - positional-alignment check for K0: their claim is that K0 "aligns the
    representation of token n with the positional encoding of n+1", i.e. the
    Shortformer score (W_Q p_{n+1}) . (W_K p_n) is superdiagonal-dominant,
    and the active K0 subcomponents carry that structure;
  - which subcomponents dominate the two attention logits.

    python spd_analysis.py --run out/spd_C600
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from induction_model import InductionModel, gen_batch
from spd_toy import MODULES, MatrixGate, install

TOKEN_TYPES = ("s1", "m", "s2", "random")


def load_spd(run: Path, ckpt: Path, dev: str):
    state = torch.load(run / "spd_state.pt", weights_only=True,
                       map_location=dev)
    c_per = int(state["c_per_module"])
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(ckpt)["state_dict"])
    model.eval()
    wrappers = install(model, c_per)
    model.to(dev)
    for n, w in wrappers.items():
        w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
        w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
    gates = nn.ModuleDict({n.replace(".", "_"): MatrixGate(c_per)
                           for n in MODULES}).to(dev)
    gates.load_state_dict(state["gates"])
    gates.eval()
    return model, wrappers, gates, c_per


@torch.no_grad()
def clean_forward(sd, seq):
    """Manual forward on the target weights; returns per-layer attention
    probabilities and the residual streams feeding each layer."""
    d = sd["embed.weight"].shape[1]
    x = sd["embed.weight"][seq]
    p = sd["pos.weight"][: seq.shape[1]]
    attn_probs, layer_inputs = [], []
    n = seq.shape[1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=seq.device), 1)
    for l in range(2):
        layer_inputs.append(x)
        q = (x + p) @ sd[f"layers.{l}.wq.weight"].T
        k = (x + p) @ sd[f"layers.{l}.wk.weight"].T
        v = x @ sd[f"layers.{l}.wv.weight"].T
        scores = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(
            mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        attn_probs.append(probs)
        x = x + probs @ v
    logits = x @ sd["unembed.weight"].T
    return logits, attn_probs, layer_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path(__file__).parent / "out/spd_C600")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    model, wrappers, gates, c_per = load_spd(args.run, args.ckpt, dev)
    comps = {n: t.float().to(dev) for n, t in torch.load(
        args.run / "components.pt", weights_only=True,
        map_location=dev)["components"].items()}

    gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)
    seq, s_pos, m_tok = gen_batch(args.n_seq, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    m_pos = s_pos + 1
    rand_pos = torch.randint(4, S - 1, (B,), device=dev, generator=gen)
    coll = (rand_pos == s_pos) | (rand_pos == m_pos)
    rand_pos = torch.where(coll, (rand_pos + 2) % (S - 5) + 4, rand_pos)
    pos_of = {"s1": s_pos, "m": m_pos,
              "s2": torch.full_like(s_pos, S - 1), "random": rand_pos}

    # -- causal importances everywhere --------------------------------------
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        model(seq)
        g = {n: gates[n.replace(".", "_")](wrappers[n].last_input)[0]
             for n in MODULES}                      # name -> [B, S, C]

    # -- Table 1a analog: avg active per matrix x token type -----------------
    print(f"avg active subcomponents (g > {args.thresh}) per matrix/token "
          f"type, {B} sequences:")
    print(f"{'matrix':<14}" + "".join(f"{t:>9}" for t in TOKEN_TYPES))
    unique = {}
    for nm in sorted(MODULES):
        act = g[nm] > args.thresh                   # [B, S, C]
        cells = []
        for t in TOKEN_TYPES:
            cells.append(act[rows, pos_of[t]].float().sum(-1).mean().item())
        unique[nm] = torch.nonzero(act.any(0).any(0)).squeeze(-1)
        print(f"{nm:<14}" + "".join(f"{c:9.2f}" for c in cells)
              + f"   unique anywhere: {len(unique[nm])} -> "
              + str(sorted(unique[nm].tolist()))[:60])
    total_unique = sum(len(v) for v in unique.values())
    print(f"\nTable-1b analog: TOTAL unique active subcomponents = "
          f"{total_unique} / {c_per * len(MODULES)}")

    # -- circuit probes ------------------------------------------------------
    sd = {k: v.detach().clone() for k, v in
          InductionModel().to(dev).state_dict().items()}
    sd.update(torch.load(args.ckpt)["state_dict"])
    sd = {k: v.to(dev) for k, v in sd.items()}
    logits, probs, layer_in = clean_forward(sd, seq)
    base_acc = (logits[rows, -1].argmax(-1) == m_tok).float().mean().item()
    a0 = probs[0][rows, m_pos, s_pos].mean().item()      # m attends to s1
    a1 = probs[1][rows, -1, m_pos].mean().item()         # s2 attends to m
    print(f"\nclean model: acc {base_acc:.4f}; attention m->s1 (L0) "
          f"{a0:.3f}; s2->m (L1) {a1:.3f}")

    def ablate(matrix_comp_ids: dict[str, torch.Tensor]):
        sd2 = {k: v.clone() for k, v in sd.items()}
        for nm, ids in matrix_comp_ids.items():
            if len(ids):
                sd2[nm + ".weight"] -= comps[nm + ".weight"][ids].sum(0)
        lg, pr, _ = clean_forward(sd2, seq)
        return ((lg[rows, -1].argmax(-1) == m_tok).float().mean().item(),
                pr[0][rows, m_pos, s_pos].mean().item(),
                pr[1][rows, -1, m_pos].mean().item())

    mat_offset = {nm: i * c_per for i, nm in enumerate(sorted(MODULES))}
    groups = {
        "L0 K+Q+V (prev-token step)": ["layers.0.wk", "layers.0.wq",
                                       "layers.0.wv"],
        "L1 Q+K (match step)": ["layers.1.wq", "layers.1.wk"],
        "L1 V (copy step)": ["layers.1.wv"],
    }
    print("\nablating each group's ACTIVE subcomponents (their component ids "
          "in the global 0..599 numbering):")
    for label, mats in groups.items():
        ids = {nm: unique[nm] + mat_offset[nm] for nm in mats}
        glob = sorted((unique[nm]).tolist() for nm in mats)
        acc, aa0, aa1 = ablate(ids)
        print(f"  {label:<28} acc {acc:.4f}  m->s1 {aa0:.3f}  "
              f"s2->m {aa1:.3f}")

    # -- K0 positional alignment (their central L0 claim) --------------------
    wq0, wk0 = sd["layers.0.wq.weight"], sd["layers.0.wk.weight"]
    p = sd["pos.weight"]
    M = (p @ wq0.T) @ (p @ wk0.T).T / math.sqrt(16)      # [S, S] pos-only
    diag1 = M.diagonal(-1).mean().item()
    off = (M.tril(-2).sum() / max(1, (S - 2) * (S - 1) // 2)).item()
    print(f"\nK0/Q0 positional score (W_Q p_i).(W_K p_j): mean at j=i-1 "
          f"{diag1:.2f} vs mean elsewhere below diag {off:.2f}")
    k0_ids = unique["layers.0.wk"] - mat_offset["layers.0.wk"]
    w = wrappers["layers.0.wk"]
    for c in k0_ids.tolist():
        kc = torch.outer(w.U[c], w.V[:, c])              # [d_out, d_in]
        Mc = (p @ wq0.T) @ (p @ kc).T / math.sqrt(16)
        print(f"  K0 subcomp {c + mat_offset['layers.0.wk']}: its share of "
              f"the j=i-1 positional score {Mc.diagonal(-1).mean().item():.2f}")

    # -- who carries the two attention logits --------------------------------
    print("\ntop subcomponent contributions to the attention logits:")
    x1 = layer_in[1]
    q1 = (x1 + p[:S]) @ sd["layers.1.wq.weight"].T
    xk = (x1 + p[:S])[rows, m_pos]                       # key input at m
    wk1 = wrappers["layers.1.wk"]
    contrib = (q1[rows, -1] @ wk1.U.T) * (xk @ wk1.V) / math.sqrt(16)
    top = contrib.abs().mean(0).topk(5)
    for v, c in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  L1 K logit s2->m: subcomp {c + mat_offset['layers.1.wk']} "
              f"mean |contrib| {v:.2f}")

    out = {"unique_per_matrix": {nm: unique[nm].tolist() for nm in unique},
           "total_unique": total_unique, "thresh": args.thresh,
           "base": {"acc": base_acc, "attn_m_s1": a0, "attn_s2_m": a1}}
    (args.run / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.run}/analysis.json")


if __name__ == "__main__":
    main()
