"""Unlearn a hazardous capability by scaling ONE component's weight mass.

Same edit as the German demonstration: W' = W + (alpha_m - 1) * (s_c * W), one
scalar per matrix, 112 scalars total, Adam. Nothing else in the model moves.

    forget   relu(ln 4 - CE4(correct letter))   on 300 WMDP-bio TRAIN questions
    retain   lam * KL(base || edit)             on benign-bio, MMLU-bio and Pile

The relu ceiling is ln(4): the objective pushes the four-way answer
distribution to uniform — chance — and stops rewarding damage past it, exactly
as the German objective stopped at ln(V).

What makes this a test rather than a demo is where it is scored. The edit never
sees the 823 held-out WMDP-bio questions, and removal that does not transfer to
them is memorisation of 300 items, not unlearning of a capability. The retain
panel is adversarial on purpose: benign biology shares nearly all of its
vocabulary with the hazardous set, so a lesion that just breaks biology is
supposed to show up there.

    python3.12 wmdp_edit.py --components 3203 3317 304 1499
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from german_permatrix import PerMatrixEditor
from german_vpd_1b import log

LETTERS = ["A", "B", "C", "D"]
CHANCE_CE = math.log(4)
INVERT_INIT = -12.0


class MCQ:
    """Right-padded question batch; scored at each row's true last token.

    Right padding rather than left: the wrapper takes no attention mask, and
    under causal attention a real token can never attend to padding that comes
    after it. Left padding would silently let every question attend to a run of
    pad tokens.
    """

    def __init__(self, items, tok, device, pad):
        enc = [tok.encode(p) for p, _ in items]
        n = max(len(e) for e in enc)
        self.idx = torch.tensor([e + [pad] * (n - len(e)) for e in enc],
                                device=device)
        self.last = torch.tensor([len(e) - 1 for e in enc], device=device)
        self.ans = torch.tensor([a for _, a in items], device=device)
        self.rows = torch.arange(len(enc), device=device)

    def __len__(self):
        return self.idx.shape[0]

    def slice(self, s, e):
        out = object.__new__(MCQ)
        out.idx, out.last = self.idx[s:e], self.last[s:e]
        out.ans = self.ans[s:e]
        out.rows = torch.arange(out.idx.shape[0], device=self.idx.device)
        return out


def letter_logits(logits, batch, letter_ids):
    """[b, 4] logits over the answer letters at each row's final real token."""
    at = logits[batch.rows, batch.last]
    return at[:, letter_ids].float()


def ce4(logits, batch, letter_ids):
    return F.cross_entropy(letter_logits(logits, batch, letter_ids),
                           batch.ans)


@torch.no_grad()
def mcq_eval(fwd, batch, letter_ids, chunk=16):
    """(accuracy, CE4) with no gradient, in chunks."""
    hit, tot, ce = 0, 0, 0.0
    for s in range(0, len(batch), chunk):
        b = batch.slice(s, s + chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            lg = fwd(b.idx)
        z = letter_logits(lg, b, letter_ids)
        hit += int((z.argmax(-1) == b.ans).sum())
        ce += float(F.cross_entropy(z, b.ans, reduction="sum"))
        tot += len(b)
    return hit / tot, ce / tot


def ce_each(logits, idx):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        idx[:, 1:].reshape(-1), reduction="none").view(idx.shape[0], -1).mean(1)


@torch.no_grad()
def text_ce(fwd, idx, chunk=8):
    out = []
    for s in range(0, idx.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            lg = fwd(idx[s:s + chunk])
        out.append(ce_each(lg, idx[s:s + chunk]))
    return float(torch.cat(out).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=geo1b.SHM_ROOT / "run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--components", type=int, nargs="+", default=[3203])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--lams", type=float, nargs="+", default=[10.0, 100.0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval_steps", type=int, nargs="+",
                    default=[25, 50, 100, 200, 400])
    ap.add_argument("--train_batch", type=int, default=8)
    ap.add_argument("--retain_batch", type=int, default=2)
    ap.add_argument("--mmlu_limit", type=int, default=800)
    ap.add_argument("--out", default="wmdp_edit.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    pad = tok.eos_token_id or 0
    letter_ids = [tok.encode(" " + l, add_special_tokens=False)[-1]
                  for l in LETTERS]
    data = torch.load(args.run_dir / args.data, weights_only=False,
                      map_location="cpu")

    from wmdp_eval import load_items
    mmlu_items = load_items("mmlu", args.mmlu_limit)
    sets = {
        "bio_train": MCQ(data["mcq_wmdp-bio"]["train"], tok, dev, pad),
        "bio_dev": MCQ(data["mcq_wmdp-bio"]["dev"], tok, dev, pad),
        "bio_eval": MCQ(data["mcq_wmdp-bio"]["eval"], tok, dev, pad),
        "cyber_eval": MCQ(data["mcq_wmdp-cyber"]["eval"], tok, dev, pad),
        "chem_dev": MCQ(data["mcq_wmdp-chem"]["dev"], tok, dev, pad),
        "mmlu": MCQ(mmlu_items, tok, dev, pad),
    }
    texts = {k: data[f"{k}_eval"].to(dev) for k in
             ("bio_retain", "cyber_retain", "pile")}
    # retain blocks the objective trains on, disjoint from the *_eval above
    retain_train = torch.cat([data["bio_retain_rank"], data["mmlu_bio_rank"],
                              data["pile_rank"]]).to(dev)

    bank = torch.load(args.run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)
    ed = PerMatrixEditor(target, bank, args.components, dev)
    del bank
    n_mod = len(ed.modules)
    k = len(args.components)
    # alpha is (k, n_modules): every component gets its own gain in every
    # matrix, so k=4 is 448 independent scalars, not 112 shared ones.
    log(f"components {args.components}: {k} x {n_mod} = {k * n_mod} scalars, "
        f"mass fraction {sum(ed.mass_fraction):.6f} "
        f"({[round(m, 7) for m in ed.mass_fraction]})")

    base = lambda idx: ed.logits(idx, None)
    log("scoring the unedited model")
    t0 = time.perf_counter()
    baseline = {}
    for name, b in sets.items():          # not `k` — that is the component count
        acc, c4 = mcq_eval(base, b, letter_ids)
        baseline[name] = {"acc": round(acc, 4), "ce4": round(c4, 4),
                          "n": len(b)}
        log(f"  base {name:<11} acc {acc:.4f}  ce4 {c4:.4f}  (n={len(b)})")
    for name, idx in texts.items():
        baseline[name] = {"ce": round(text_ce(base, idx), 4)}
        log(f"  base {name:<11} CE {baseline[name]['ce']:.4f}")
    # frozen teacher for the KL term
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=True):
        teacher = torch.cat([F.log_softmax(base(retain_train[s:s + 4])[:, :-1]
                                           .float(), -1).cpu()
                             for s in range(0, retain_train.shape[0], 4)])
    log(f"baseline in {time.perf_counter() - t0:.0f}s")

    results, alphas = [], {}
    for lr in args.lrs:
        for lam in args.lams:
            alpha = torch.nn.Parameter(
                torch.full((k, n_mod), INVERT_INIT, device=dev))
            opt = torch.optim.Adam([alpha], lr=lr)
            tag = f"k{k} lr={lr:g} lam={lam:g}"
            t1 = time.perf_counter()
            for step in range(1, args.steps + 1):
                s = ((step - 1) * args.train_batch) % len(sets["bio_train"])
                fb = sets["bio_train"].slice(s, s + args.train_batch)
                if len(fb) == 0:
                    continue
                r = ((step - 1) * args.retain_batch) % retain_train.shape[0]
                rb = retain_train[r:r + args.retain_batch]
                tb = teacher[r:r + args.retain_batch].to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=True):
                    forget = F.relu(
                        CHANCE_CE - ce4(ed.logits(fb.idx, alpha), fb,
                                        letter_ids))
                    lp = F.log_softmax(
                        ed.logits(rb, alpha)[:, :-1].float(), -1)
                    kl = F.kl_div(lp, tb, log_target=True,
                                  reduction="batchmean") / lp.shape[1]
                    loss = forget + lam * kl
                loss.backward()
                opt.step()
                with torch.no_grad():
                    alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
                    alpha.clamp_(-50.0, 100.0)
                if step in args.eval_steps:
                    a = alpha.detach()
                    fwd = lambda idx: ed.logits(idx, a)
                    row = {"components": args.components, "lr": lr,
                           "lam": lam, "step": step, "mcq": {}, "text": {}}
                    for name, b in sets.items():
                        acc, c4 = mcq_eval(fwd, b, letter_ids)
                        row["mcq"][name] = {"acc": round(acc, 4),
                                            "ce4": round(c4, 4)}
                    for name, idx in texts.items():
                        c = text_ce(fwd, idx)
                        row["text"][name] = {
                            "ce": round(c, 4),
                            "delta": round(c - baseline[name]["ce"], 4)}
                    row["seconds"] = time.perf_counter() - t1
                    # keep the edit itself, not just its score: re-measuring a
                    # checkpoint under a new metric should not need a re-train
                    key = f"lr{lr:g}_lam{lam:g}_s{step}"
                    alphas[key] = a.clone().cpu()
                    row["alpha_key"] = key
                    results.append(row)
                    m = row["mcq"]
                    log(f"{tag} step {step}: bio_eval {m['bio_eval']['acc']:.3f} "
                        f"(base {baseline['bio_eval']['acc']:.3f}) | "
                        f"mmlu {m['mmlu']['acc']:.3f} "
                        f"(base {baseline['mmlu']['acc']:.3f}) | "
                        f"cyber {m['cyber_eval']['acc']:.3f} | "
                        f"bio-retain ΔCE {row['text']['bio_retain']['delta']:+.3f} "
                        f"pile ΔCE {row['text']['pile']['delta']:+.3f}")
            ed.alpha = None

    out = {"format": "wmdp_component_unlearn_v1",
           "components": args.components,
           "n_scalars": k * n_mod,
           "mass_fraction": sum(ed.mass_fraction),
           "mass_fraction_each": ed.mass_fraction,
           "objective": "relu(ln4 - CE4_correct_letter) + lam * KL(base||edit)",
           "chance_ce4": CHANCE_CE,
           "baseline": baseline, "points": results}
    (args.run_dir / args.out).write_text(json.dumps(out, indent=1))
    apath = args.run_dir / args.out.replace(".json", "_alpha.pt")
    torch.save({"components": args.components, "modules": ed.modules,
                "alphas": alphas}, apath)
    log(f"wrote {args.run_dir / args.out} and {apath}")


if __name__ == "__main__":
    main()
