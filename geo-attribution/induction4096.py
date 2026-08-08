"""Is there an induction component in the C=4096 decomposition?

Induction is the one circuit with a clean behavioural probe: on a random token
sequence repeated twice, the second copy is predictable ONLY by "find the
earlier occurrence of the current token and copy what followed it". So:

  stage rank  - contrast the attribution-fingerprint posterior at second-copy
                positions against first-copy positions of the SAME tokens.
                Everything except induction is held fixed by construction.
  stage edit  - the ranking is correlational. Scale each candidate component's
                owned weight mass and check that induction loss moves while
                loss on ordinary text does not. Attribution rank has repeatedly
                failed to predict causal importance in this decomposition, so
                this stage is the actual answer.

  python3.12 induction4096.py rank --device cuda:1
  python3.12 induction4096.py edit --components 1 2 3 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args
from german67 import ENGLISH


def repeated_batch(n_seq, span, vocab_lo, vocab_hi, bos, device, seed):
    """[BOS, R, R] — the second copy is predictable only by induction."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(vocab_lo, vocab_hi, (n_seq, span), generator=g)
    idx = torch.cat([torch.full((n_seq, 1), bos), r, r], dim=1)
    return idx.to(device)


def control_batch(tokenizer, seq_len, device):
    ids = []
    for s in ENGLISH:
        ids += tokenizer.encode(s, add_special_tokens=False)
        ids += tokenizer.encode("\n\n", add_special_tokens=False)
    count = max(len(ids) // seq_len, 1)
    return torch.tensor(ids[:count * seq_len],
                        dtype=torch.long).view(count, seq_len).to(device)


def ce_at(model_logits, idx, mask):
    logp = F.log_softmax(model_logits[:, :-1].float(), -1)
    tgt = idx[:, 1:]
    nll = -logp.gather(-1, tgt[..., None]).squeeze(-1)
    return nll[mask[:, :-1]].mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["rank", "edit", "matrix"])
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--span", type=int, default=64)
    parser.add_argument("--n_seq", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8,
                        help="positions skipped at the start of each copy")
    parser.add_argument("--rank_temperature", type=float, default=0.05)
    parser.add_argument("--candidate_k", type=int, default=12)
    parser.add_argument("--components", type=int, nargs="+", default=None)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    parser.add_argument("--matrix_alphas", type=float, nargs="+",
                        default=[16.0, 64.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = args.device
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    meta = {k: bank[k] for k in ("format", "C", "modules", "sensor",
                                 "gim_tau", "scalar") if k in bank}
    cfg = ranking_args(meta)
    cap = setup_model(cfg, device)
    bos = cap.target.hf.config.bos_token_id
    S = args.span
    idx = repeated_batch(args.n_seq, S, 1000, 20000, bos, device, args.seed)
    # position p predicts idx[p+1]; copy-1 spans [1, S], copy-2 spans [S+1, 2S]
    first = torch.zeros_like(idx, dtype=torch.bool)
    second = torch.zeros_like(idx, dtype=torch.bool)
    first[:, 1 + args.warmup:S] = True
    second[:, S + 1 + args.warmup:2 * S] = True

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=True):
        logits = cap.target(idx)
    ce1, ce2 = ce_at(logits, idx, first), ce_at(logits, idx, second)
    log(f"induction check: copy-1 CE {ce1:.3f} -> copy-2 CE {ce2:.3f} "
        f"(drop {ce1 - ce2:.3f} nats) — the model does induction")
    del logits
    torch.cuda.empty_cache()

    if args.stage == "rank":
        spec, scales, dim = load_spec(run_dir, device)
        stream = load_stream_model(run_dir / "stream_model.pt", device)

        def posterior(mask):
            bi, pos = mask.nonzero(as_tuple=True)
            phi, _ = pass_features(cfg, cap, idx, pos[None], bi[None],
                                   spec, scales, dim, return_pg=False)
            x = phi.clamp(-6e4, 6e4).half().float()
            y = F.normalize((x - stream["mean"]) @ stream["projector"], dim=1)
            sims = y @ stream["centroids"].t()
            return torch.softmax(sims / args.rank_temperature, dim=1).mean(0)

        p2, p1 = posterior(second), posterior(first)
        contrast = (p2 - p1).cpu()
        order = contrast.argsort(descending=True)[:args.candidate_k]
        catalog = {}
        cat_path = Path(__file__).parent / "out/catalog_prop1b_C4096.json"
        if cat_path.exists():
            catalog = json.loads(cat_path.read_text())
        log("top induction-contrast components "
            "(copy-2 posterior minus copy-1 posterior):")
        rows = []
        for c in order.tolist():
            row = {"component": c, "contrast": float(contrast[c]),
                   "p_second": float(p2[c]), "p_first": float(p1[c]),
                   "label": catalog.get(str(c), {}).get("label", "")}
            rows.append(row)
            log(f"  c{c:<5} contrast {row['contrast']:+.4f}  "
                f"copy2 {row['p_second']:.4f} vs copy1 {row['p_first']:.4f}  "
                f"{row['label'][:58]}")
        out = {"copy1_ce": ce1, "copy2_ce": ce2, "candidates": rows}
        (run_dir / "induction_rank.json").write_text(json.dumps(out, indent=1))
        log(f"wrote {run_dir / 'induction_rank.json'}")
        return

    if args.stage == "matrix":
        # Which of the 112 matrices does the component route induction through?
        # One matrix is perturbed at a time; the other 111 stay at identity.
        from german_permatrix import PerMatrixEditor
        comps = args.components or [108]
        ctrl = control_batch(tokenizer, 96, device)
        cm = torch.ones_like(ctrl, dtype=torch.bool)
        cm[:, :4] = False
        editor = PerMatrixEditor(cap.target, bank, comps, device)
        mods = list(editor.modules)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            lg = editor.logits(idx, None)
            b2 = ce_at(lg, idx, second)
            del lg
            lc = editor.logits(ctrl, None)
            bc = ce_at(lc, ctrl, cm)
            del lc
        log(f"base copy2 CE {b2:.3f}, control CE {bc:.3f}")
        rows = []
        for a in args.matrix_alphas:
            for mi, path in enumerate(mods):
                alpha = torch.ones(len(comps), len(mods), device=device)
                alpha[:, mi] = a
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=True):
                    lg = editor.logits(idx, alpha)
                    d2 = ce_at(lg, idx, second)
                    del lg
                    lc = editor.logits(ctrl, alpha)
                    cc = ce_at(lc, ctrl, cm)
                    del lc
                rows.append({"alpha": a, "matrix": mi, "path": path,
                             "layer": int(path.split("layers.")[1].split(".")[0]),
                             "kind": path.rsplit(".", 1)[1],
                             "d_induction": d2 - b2, "d_control": cc - bc})
            top = sorted([r for r in rows if r["alpha"] == a],
                         key=lambda r: -r["d_induction"])[:6]
            log(f"alpha={a:g} — matrices with the largest induction damage:")
            for r in top:
                log(f"  L{r['layer']:<2} {r['kind']:<10} "
                    f"induction {r['d_induction']:+7.3f}  "
                    f"control {r['d_control']:+6.3f}")
        out = {"components": comps, "base_copy2_ce": b2, "base_control_ce": bc,
               "rows": rows}
        (run_dir / "induction_matrix.json").write_text(json.dumps(out, indent=1))
        log(f"wrote {run_dir / 'induction_matrix.json'}")
        return

    # ---- causal stage ----
    from german_permatrix import PerMatrixEditor
    comps = args.components
    if comps is None:
        rows = json.loads((run_dir / "induction_rank.json").read_text())
        comps = [r["component"] for r in rows["candidates"][:3]]
    log(f"causal sweep on components {comps}")
    ctrl = control_batch(tokenizer, 96, device)
    editor = PerMatrixEditor(cap.target, bank, comps, device)
    n_mod = len(editor.modules)
    results = []
    for slot, c in enumerate(comps):
        for a in args.alphas:
            alpha = torch.ones(len(comps), n_mod, device=device)
            alpha[slot] = a
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True):
                lg = editor.logits(idx, alpha)
                d1, d2 = ce_at(lg, idx, first), ce_at(lg, idx, second)
                del lg
                lc = editor.logits(ctrl, alpha)
                cm = torch.ones_like(ctrl, dtype=torch.bool)
                cm[:, :4] = False
                cc = ce_at(lc, ctrl, cm)
                del lc
            results.append({"component": c, "alpha": a, "copy1_ce": d1,
                            "copy2_ce": d2, "induction_gap": d1 - d2,
                            "control_ce": cc})
            log(f"  c{c} a={a:+.1f}: copy2 CE {d2:6.3f} "
                f"(Δ{d2 - ce2:+.3f})  induction gap {d1 - d2:6.3f} "
                f"(base {ce1 - ce2:.3f})  control CE {cc:6.3f}")
    editor.logits(idx, None)
    (run_dir / "induction_edit.json").write_text(json.dumps(
        {"base": {"copy1_ce": ce1, "copy2_ce": ce2}, "arms": results}, indent=1))
    log(f"wrote {run_dir / 'induction_edit.json'}")


if __name__ == "__main__":
    main()
