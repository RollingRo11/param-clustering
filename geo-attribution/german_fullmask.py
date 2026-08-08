"""Full-width component-mask fine-tune: one scalar per ALL C components.

Decomposition-native counterpart of the guided-LoRA result: the only
trainable degrees of freedom are the C per-component mask scalars of the
softpart decomposition, applied through the exact ownership encoding

    W' = W  *  (1 + sum_slots swgt_s * (alpha[sidx_s] - 1))

so every candidate edit is a literal rescaling of component-owned weight
mass — no free directions. A custom autograd op computes the per-entry
multiplier on the fly from the (sidx, swgt) bank and scatter-adds the
gradient back into alpha, so no per-component slices are ever materialized
and the full C-dimensional mask trains in one piece.

Objective (multilingual preserve):
    relu(log V - German CE) + lam_en*KL_en + lam_rom*mean(KL_fr,KL_es,KL_it)

Because a single component holds ~1e-5 of squared mass, the landscape near
alpha=1 is flat below batch noise; configs therefore train from BOTH the
identity init and a seeded init (best guarded sweep edit + mild amplification
of the German-ranked components).
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
import geo67
from german_vpd_1b import ce_each, fmt_vec, log, prepare_data
from german_vpd_multi import ROMANCE, train_blocks

PROMPTS = {
    "de": "Die Lösung liegt natürlich",
    "en": "And, of course, the solution does",
    "fr": "Cependant, le projet",
    "es": "Sin embargo, el proyecto",
    "it": "La ridestinazione delle risorse",
}


class MaskedWeight(torch.autograd.Function):
    """W' = W * (1 + sum_s swgt_s * (alpha[sidx_s] - 1)); grad to alpha only."""

    @staticmethod
    def forward(ctx, alpha, weight, sidx, swgt):
        multiplier = torch.ones_like(weight)
        shifted = alpha - 1.0
        for s in range(sidx.shape[0]):
            multiplier += swgt[s].float() * shifted[sidx[s].long()]
        ctx.save_for_backward(weight, sidx, swgt)
        ctx.alpha_len = alpha.numel()
        return weight * multiplier

    @staticmethod
    def backward(ctx, grad):
        weight, sidx, swgt = ctx.saved_tensors
        grad_weight = (grad.float() * weight).contiguous()
        grad_alpha = torch.zeros(
            ctx.alpha_len, device=grad.device, dtype=torch.float32)
        for s in range(sidx.shape[0]):
            grad_alpha.index_add_(
                0, sidx[s].long().flatten(),
                (grad_weight * swgt[s].float()).flatten())
        return grad_alpha, None, None, None


class FullMaskEditor:
    """All-C mask editor over the softpart bank's ownership encoding."""

    def __init__(self, target, bank, device):
        if bank.get("format") != "softpart":
            raise ValueError("full-mask editor requires a softpart bank")
        self.target = target
        self.device = device
        self.C = int(bank["C"])
        self.modules = list(bank["modules"])
        self.encoding = {}
        for number, path in enumerate(self.modules, 1):
            self.encoding[path] = (
                bank["sidx"][path].to(device),
                bank["swgt"][path].to(device))
            if number % 32 == 0 or number == len(self.modules):
                log(f"loaded ownership encoding {number}/{len(self.modules)}")
        self.alpha = None
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        for path in self.modules:
            linear = target.get_submodule(path)
            linear._fm_path = path
            linear.forward = self._forward(linear)

    def _forward(self, linear):
        def forward(x):
            if self.alpha is None:
                return F.linear(x, linear.weight, linear.bias)
            sidx, swgt = self.encoding[linear._fm_path]
            edited = MaskedWeight.apply(self.alpha, linear.weight, sidx, swgt)
            return F.linear(x, edited, linear.bias)
        return forward

    def logits(self, idx, alpha):
        self.alpha = alpha
        return self.target(idx)

    def apply_in_place(self, alpha):
        """Literal surgery: returns saved weights for restore."""
        saved = {}
        with torch.no_grad():
            for path in self.modules:
                weight = self.target.get_submodule(path).weight
                saved[path] = weight.detach().clone()
                sidx, swgt = self.encoding[path]
                multiplier = torch.ones_like(weight)
                shifted = alpha - 1.0
                for s in range(sidx.shape[0]):
                    multiplier += swgt[s].float() * shifted[sidx[s].long()]
                weight.mul_(multiplier)
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


def seeded_alpha(args, C, components_path, device):
    alpha = torch.ones(C, device=device)
    sweep_path = args.run_dir / "german_topk_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        for component, value in zip(sweep["components"], sweep["best_alpha"]):
            alpha[component] = float(value)
        log(f"seed: sweep best {fmt_vec(sweep['best_alpha'])} on "
            f"{sweep['components']}")
    ranking = json.loads(components_path.read_text())
    for row in ranking["inspected_candidates"]:
        component = row["component"]
        if alpha[component] == 1.0:
            alpha[component] = args.seed_amplify
    return alpha


def train_config(editor, blocks, base_logp, config, args, ceiling):
    alpha = torch.nn.Parameter(config["init"].clone())
    optimizer = torch.optim.Adam([alpha], lr=args.lr)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    tag = config["tag"]
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
            loss = (F.relu(ceiling - german_ce)
                    + config["lam_en"] * kls[0]
                    + config["lam_rom"] * romance)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=args.alpha_max,
                              neginf=args.alpha_min)
            alpha.clamp_(args.alpha_min, args.alpha_max)
        if step % args.log_every == 0 or step + 1 == args.steps:
            moved = int((alpha.detach() - config["init"]).abs()
                        .gt(0.05).sum().item())
            log(f"{tag} step {step:04d}: deCE={german_ce.item():.2f} "
                f"enKL={kls[0].item():.3f} romKL={romance.item():.3f} "
                f"moved={moved}")
        del logits
    return alpha.detach()


def worker(editor, data, configs, args, ceiling):
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
    cache = {}
    eval_sets = {
        "de_dev": data["de_dev"],
        "en_dev": data["en_dev"],
        "fr_guard": data["fr_eval"][1:2],
        "es_guard": data["es_eval"][1:2],
        "it_guard": data["it_eval"][1:2],
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    rows = []
    for config in configs:
        t0 = time.perf_counter()
        alpha = train_config(editor, blocks, base_logp, config, args, ceiling)
        row = {"tag": config["tag"], "lam_en": config["lam_en"],
               "lam_rom": config["lam_rom"], "seed_kind": config["seed_kind"]}
        row["eval"] = {name: language_metrics(editor, idx, alpha, cache, name)
                       for name, idx in eval_sets.items()}
        row["alpha_stats"] = {
            "moved_gt_0.05": int((alpha - config["init"]).abs()
                                 .gt(0.05).sum().item()),
            "min": float(alpha.min()), "max": float(alpha.max()),
        }
        row["alpha"] = alpha.cpu()
        row["elapsed_seconds"] = time.perf_counter() - t0
        e = row["eval"]
        log(f"{config['tag']}: de={e['german_europarl']['delta_ce']:+.3f} "
            f"en_pile={e['english_pile']['delta_ce']:+.3f} "
            f"fr/es/it={e['french_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['spanish_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['italian_europarl_heldout']['delta_ce']:+.3f} "
            f"moved={row['alpha_stats']['moved_gt_0.05']} "
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
    parser.add_argument("--configs", type=float, nargs="+",
                        default=[10.0, 10.0, 30.0, 30.0],
                        help="flat (lam_en, lam_rom) pairs; each runs with "
                             "identity and seeded inits")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed_amplify", type=float, default=2.0)
    parser.add_argument("--en_budget", type=float, default=0.1)
    parser.add_argument("--romance_guard", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=-50.0)
    parser.add_argument("--alpha_max", type=float, default=100.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    pairs = [(args.configs[i], args.configs[i + 1])
             for i in range(0, len(args.configs), 2)]
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]

    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    editors = []
    for device in devices:
        target = geo1b.load_target_1b(device)
        editors.append(FullMaskEditor(target, bank, device))
    del bank
    C = editors[0].C
    ceiling = math.log(editors[0].target.hf.config.vocab_size)

    ranking_path = args.run_dir / "german_vpd_ranking.json"
    configs = []
    for lam_en, lam_rom in pairs:
        for seed_kind in ("identity", "seeded"):
            init = (torch.ones(C) if seed_kind == "identity"
                    else seeded_alpha(args, C, ranking_path, "cpu"))
            configs.append({
                "tag": f"{seed_kind},l_en={lam_en:g},l_rom={lam_rom:g}",
                "lam_en": lam_en, "lam_rom": lam_rom,
                "seed_kind": seed_kind, "init": init})
    for config in configs:
        config["init"] = config["init"].to(devices[0])
    log(f"full-mask fine-tune: C={C}, {len(configs)} configs")

    queues = [configs[i::len(editors)] for i in range(len(editors))]
    for editor, queue in zip(editors, queues):
        for config in queue:
            config["init"] = config["init"].to(editor.device)
    results = []
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(worker, editor, data, queue, args, ceiling)
                   for editor, queue in zip(editors, queues) if queue]
        for future in futures:
            results.extend(future.result())

    def guarded(row):
        e = row["eval"]
        romance = max(e[f"{lang}_guard"]["delta_ce"] for lang in ROMANCE)
        return (e["en_dev"]["delta_ce"] < args.en_budget
                and romance < args.romance_guard)

    eligible_rows = [row for row in results if guarded(row)]
    selected = (max(eligible_rows,
                    key=lambda row: row["eval"]["de_dev"]["edited_ce"])
                if eligible_rows else None)

    rollout_out, verification = None, None
    if selected is not None:
        editor = editors[0]
        alpha = selected["alpha"].to(editor.device)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            expression = editor.logits(
                data["de_eval"][:1, :32].to(editor.device), alpha).float()
        saved = editor.apply_in_place(alpha)
        editor.alpha = None
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            literal = editor.target(
                data["de_eval"][:1, :32].to(editor.device)).float()
        error = (expression - literal).abs()
        verification = {"max_abs_logit_error": error.max().item(),
                        "mean_abs_logit_error": error.mean().item()}
        log(f"literal surgery check: max |dlogit|="
            f"{verification['max_abs_logit_error']:.3e}")

        def generate(prompt):
            ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                device=editor.device)
            for _ in range(args.max_new_tokens):
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=True):
                    logits = editor.target(ids[:, -512:])
                ids = torch.cat(
                    [ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            return tokenizer.decode(ids[0].tolist())

        rollout_out = {lang: {"prompt": p, "edited": generate(p)}
                       for lang, p in PROMPTS.items()}
        editor.restore(saved)
        log(f"selected {selected['tag']}")

    output = args.run_dir / "german_fullmask.json"
    output.write_text(json.dumps({
        "format": "german_fullmask_v1",
        "C": C,
        "objective": ("relu(logV-deCE) + lam_en*KL_en "
                      "+ lam_rom*mean(KL_fr,KL_es,KL_it)"),
        "dof": C,
        "lr": args.lr, "steps": args.steps,
        "configs": [{key: value for key, value in row.items()
                     if key != "alpha"} for row in results],
        "selected_tag": selected["tag"] if selected else None,
        "literal_weight_edit_verification": verification,
        "rollouts": rollout_out,
    }, indent=2))
    torch.save({
        "format": "fullmask_alpha_adapter_v1",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "results": [{"tag": row["tag"], "alpha": row["alpha"]}
                    for row in results],
        "selected_tag": selected["tag"] if selected else None,
    }, args.run_dir / "german_fullmask_adapter.pt")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
