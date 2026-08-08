"""Per-matrix component scalars: k components x 112 per-matrix gains.

The full-mask result showed chance-level German removal needs ~1,700 global
component scalars — but a global scalar is the crudest use of a component.
Here each of k components gets ONE SCALAR PER DECOMPOSED MATRIX (112), so an
edit of "just k components" carries k x 112 degrees of freedom of layer-wise
amplify/suppress/invert structure:

    W'_m = W_m + sum_i (alpha[i, m] - 1) * (component_i slice of W_m)

Swept over k in {1, 2, 4, 8, 16} with the multilingual objective
(relu(logV - deCE) + lam_en*KL_en + lam_rom*mean KL_romance). Components are
ordered causal-winners-first; rows beyond k stay frozen at identity. Inits
broadcast each component's best-known global scalar across its matrices.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from german_vpd_1b import ComponentEditor, ce_each, log, prepare_data
from german_vpd_multi import ROMANCE, train_blocks

# Causal screen winners first, then the full-mask fine-tune's top movers.
COMPONENT_ORDER = [1668, 42, 2207, 1406, 3634, 3600, 493, 2161,
                   2318, 3749, 928, 2111, 749, 2520, 2921, 2200]
INIT_SCALAR = {1668: 13.0, 42: 10.0, 2207: 3.0, 1406: 3.2, 3634: 2.1,
               3600: 7.7, 493: -5.8, 2161: 7.3, 2318: 0.8, 3749: 7.5,
               928: 5.4, 2111: -4.3, 749: -2.7, 2520: 4.7, 2921: 4.5,
               2200: -2.3}
PROMPTS = {
    "de": "Die Lösung liegt natürlich",
    "en": "And, of course, the solution does",
    "fr": "Cependant, le projet",
    "es": "Sin embargo, el proyecto",
    "it": "La ridestinazione delle risorse",
}


class PerMatrixEditor(ComponentEditor):
    """alpha of shape (k, n_modules): one gain per component per matrix."""

    def _linear_forward(self, linear):
        def forward(x):
            if self.alpha is None:
                return F.linear(x, linear.weight, linear.bias)
            slices = self.slices[linear._component_edit_path]
            gains = self.alpha[:, linear._module_index] - 1.0
            edited = linear.weight
            for slot in range(gains.numel()):
                edited = edited + gains[slot] * slices[slot]
            return F.linear(x, edited, linear.bias)
        return forward

    def _install(self):
        super()._install()
        for index, path in enumerate(self.modules):
            self.target.get_submodule(path)._module_index = index

    def logits(self, idx, alpha):
        self.alpha = alpha  # (k, n_modules) tensor or None
        return self.target(idx)

    def apply_in_place(self, alpha):
        saved = {}
        with torch.no_grad():
            for index, path in enumerate(self.modules):
                weight = self.target.get_submodule(path).weight
                saved[path] = weight.detach().clone()
                for slot in range(alpha.shape[0]):
                    weight.add_(self.slices[path][slot],
                                alpha=float(alpha[slot, index]) - 1.0)
        return saved

    def restore(self, saved):
        with torch.no_grad():
            for path, weight in saved.items():
                self.target.get_submodule(path).weight.copy_(weight)


def language_metrics(editor, idx, alpha, cache, key):
    idx = idx.to(editor.device)
    if key not in cache:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            base = editor.logits(idx, None)
        cache[key] = (ce_each(base, idx).mean().item(),
                      F.log_softmax(base[:, :-1].float(), -1))
        del base
    base_ce, base_logp = cache[key]
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        edited = editor.logits(idx, alpha)
    edited_ce = ce_each(edited, idx).mean().item()
    edited_logp = F.log_softmax(edited[:, :-1].float(), -1)
    kl = (base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean().item()
    del edited, edited_logp
    return {"base_ce": base_ce, "edited_ce": edited_ce,
            "delta_ce": edited_ce - base_ce, "kl_from_base": kl}


def train_k(editor, data, k, args, ceiling):
    total = len(editor.components)
    modules = len(editor.modules)
    init = torch.ones(total, modules, device=editor.device)
    for slot in range(k):
        init[slot] = INIT_SCALAR[editor.components[slot]]
    alpha = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([alpha], lr=args.lr)
    blocks = train_blocks(data)
    base_logp = {}
    for lang in ("en",) + ROMANCE:
        rows = []
        for block in blocks[lang]:
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                base = editor.logits(block[None].to(editor.device), None)
            rows.append(F.log_softmax(base[:, :-1].float(), -1))
            del base
        base_logp[lang] = rows
    counts = {lang: len(blocks[lang]) for lang in blocks}
    tag = f"k={k}"
    for step in range(args.steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in blocks}
        chosen["de"] = step % counts["de"]
        idx = torch.stack([blocks[lang][chosen[lang]] for lang in
                           ("de", "en", "fr", "es", "it")]).to(editor.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = editor.logits(idx, alpha)
            german_ce = ce_each(logits[:1], idx[:1]).squeeze(0)
            kls = []
            for j, lang in enumerate(("en",) + ROMANCE, start=1):
                edited_logp = F.log_softmax(
                    logits[j:j + 1, :-1].float(), -1)
                kls.append(F.kl_div(
                    edited_logp, base_logp[lang][chosen[lang]],
                    log_target=True, reduction="none").sum(-1).mean())
            romance = (kls[1] + kls[2] + kls[3]) / 3.0
            loss = (F.relu(ceiling - german_ce) + args.lam_en * kls[0]
                    + args.lam_rom * romance)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=args.alpha_max,
                              neginf=args.alpha_min)
            alpha.clamp_(args.alpha_min, args.alpha_max)
            alpha[k:] = 1.0  # only the first k components are edited
        if step % args.log_every == 0 or step + 1 == args.steps:
            log(f"{tag} step {step:04d}: deCE={german_ce.item():.2f} "
                f"enKL={kls[0].item():.3f} romKL={romance.item():.3f}")
        del logits
    return alpha.detach()


def worker(editor, data, k_values, args, ceiling):
    cache = {}
    eval_sets = {
        "de_dev": data["de_dev"],
        "en_dev": data["en_dev"],
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    rows = []
    for k in k_values:
        t0 = time.perf_counter()
        alpha = train_k(editor, data, k, args, ceiling)
        row = {"k": k,
               "components": editor.components[:k],
               "eval": {name: language_metrics(editor, idx, alpha, cache,
                                               name)
                        for name, idx in eval_sets.items()},
               "alpha": alpha.cpu(),
               "elapsed_seconds": time.perf_counter() - t0}
        e = row["eval"]
        log(f"RESULT k={k}: de={e['german_europarl']['delta_ce']:+.3f} "
            f"en_pile={e['english_pile']['delta_ce']:+.3f} "
            f"en_euro={e['english_europarl']['delta_ce']:+.3f} "
            f"fr/es/it={e['french_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['spanish_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['italian_europarl_heldout']['delta_ce']:+.3f} "
            f"({row['elapsed_seconds']:.0f}s)")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--lam_en", type=float, default=10.0)
    parser.add_argument("--lam_rom", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--alpha_min", type=float, default=-50.0)
    parser.add_argument("--alpha_max", type=float, default=100.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]
    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    target = geo1b.load_target_1b(devices[0])
    editor = PerMatrixEditor(target, bank, COMPONENT_ORDER, devices[0])
    del bank
    editors = [editor] + [editor.replicate(dev) for dev in devices[1:]]
    for twin in editors[1:]:
        for index, path in enumerate(twin.modules):
            twin.target.get_submodule(path)._module_index = index
    ceiling = math.log(editor.target.hf.config.vocab_size)

    # Balance: big-k runs on device 0, small-k on device 1.
    order = sorted(args.k_values, reverse=True)
    queues = [order[i::len(editors)] for i in range(len(editors))]
    results = []
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(worker, ed, data, queue, args, ceiling)
                   for ed, queue in zip(editors, queues) if queue]
        for future in futures:
            results.extend(future.result())
    results.sort(key=lambda row: row["k"])

    # Rollouts + literal surgery for the best k meeting soft budgets.
    def score(row):
        e = row["eval"]
        ok = (e["english_pile"]["delta_ce"] < 0.3
              and max(e[f"{l}_europarl_heldout" if l == "french" else l]
                      ["delta_ce"] for l in
                      ("french_europarl_heldout", "spanish_europarl_heldout",
                       "italian_europarl_heldout")) < 1.0)
        return (ok, e["german_europarl"]["delta_ce"])

    best = max(results, key=score)
    editor0 = editors[0]
    alpha = best["alpha"].to(editor0.device)
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        expression = editor0.logits(
            data["de_eval"][:1, :32].to(editor0.device), alpha).float()
    saved = editor0.apply_in_place(alpha)
    editor0.alpha = None
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        literal = editor0.target(
            data["de_eval"][:1, :32].to(editor0.device)).float()
    error = (expression - literal).abs()
    verification = {"max_abs_logit_error": error.max().item(),
                    "mean_abs_logit_error": error.mean().item()}
    log(f"literal surgery (k={best['k']}): "
        f"max |dlogit|={verification['max_abs_logit_error']:.3e}")

    def generate(prompt):
        ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=editor0.device)
        for _ in range(args.max_new_tokens):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                logits = editor0.target(ids[:, -512:])
            ids = torch.cat(
                [ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokenizer.decode(ids[0].tolist())

    rollout_out = {lang: {"prompt": p, "edited": generate(p)}
                   for lang, p in PROMPTS.items()}
    editor0.restore(saved)

    output = args.run_dir / "german_permatrix.json"
    output.write_text(json.dumps({
        "format": "german_permatrix_v1",
        "component_order": COMPONENT_ORDER,
        "dof_per_k": {str(row["k"]): row["k"] * len(editor.modules)
                      for row in results},
        "objective": ("relu(logV-deCE) + lam_en*KL_en "
                      "+ lam_rom*mean(KL_fr,KL_es,KL_it)"),
        "lam_en": args.lam_en, "lam_rom": args.lam_rom,
        "results": [{key: value for key, value in row.items()
                     if key != "alpha"} for row in results],
        "best_k": best["k"],
        "literal_weight_edit_verification": verification,
        "rollouts": rollout_out,
    }, indent=2))
    torch.save({
        "format": "permatrix_alpha_adapter_v1",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "component_order": COMPONENT_ORDER,
        "results": [{"k": row["k"], "alpha": row["alpha"]}
                    for row in results],
        "best_k": best["k"],
    }, args.run_dir / "german_permatrix_adapter.pt")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
