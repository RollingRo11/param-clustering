"""Component-guided LoRA: free directions confined to German-owned rows.

Plain LoRA proves a clean German-removal weight direction exists but says
nothing about WHERE it lives; component rescaling is localized but capped at
graded dampening. This runner ties the two together: rank-r adapters whose
write-side (B) rows are masked to the rows that carry the causal German
components' owned mass, trained with the multilingual objective

    relu(log V - German CE) + lam_en*KL_en + lam_rom*mean(KL_fr,KL_es,KL_it)

An unmasked control with the same objective separates the effect of the
Romance term from the effect of the localization. Guarded selection and
held-out evaluation match the component-edit experiments. A row mask keeps
the adapter exactly rank-r: mask ⊙ (B A) = (mask ⊙ B) A.
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
from german_vpd_1b import ce_each, log, prepare_data
from german_vpd_multi import ROMANCE, train_blocks

MASKED_ROLLOUT_PROMPTS = {
    "de": "Die Lösung liegt natürlich",
    "en": "And, of course, the solution does",
    "fr": "Cependant, le projet",
    "es": "Sin embargo, el proyecto",
    "it": "La ridestinazione delle risorse",
}


def german_row_masks(bank_path: Path, components: list[int], coverage: float,
                     target, device: str) -> tuple[dict, dict]:
    """Binary write-row masks covering `coverage` of the components' mass."""
    bank = torch.load(bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    comp = torch.tensor(components)
    masks, stats = {}, {}
    for path in geo67.MODULES:
        weight = target.get_submodule(path).weight.detach()
        sidx = bank["sidx"][path].to(device)
        swgt = bank["swgt"][path].to(device)
        share = (torch.isin(sidx, comp.to(device)) * swgt).sum(0).float()
        owned = (share * weight.square()).sum(1)          # per-row G mass
        del sidx, swgt, share
        order = owned.argsort(descending=True)
        cum = owned[order].cumsum(0)
        total = cum[-1].clamp_min(1e-30)
        keep = int((cum < coverage * total).sum().item()) + 1
        mask = torch.zeros_like(owned)
        mask[order[:keep]] = 1.0
        masks[path] = mask
        stats[path] = {"rows_kept": keep, "rows_total": owned.numel()}
    kept = sum(s["rows_kept"] for s in stats.values())
    total_rows = sum(s["rows_total"] for s in stats.values())
    log(f"row masks: {kept}/{total_rows} rows "
        f"({100 * kept / total_rows:.1f}%) carry {coverage:.0%} of "
        f"components' owned mass")
    return masks, stats


class GuidedLora:
    """Rank-r adapters with an optional fixed write-row mask."""

    def __init__(self, target, modules, rank, device, seed, masks=None):
        self.target = target
        self.modules = list(modules)
        self.device = device
        self.enabled = False
        self.masks = masks
        self.params = []
        self.ab = {}
        generator = torch.Generator().manual_seed(seed)
        for path in self.modules:
            linear = target.get_submodule(path)
            out_dim, in_dim = linear.weight.shape
            a = (torch.randn(rank, in_dim, generator=generator)
                 / math.sqrt(in_dim)).to(device).requires_grad_(True)
            b = torch.zeros(out_dim, rank, device=device, requires_grad=True)
            self.ab[path] = (a, b)
            self.params += [a, b]
            linear._glora_path = path
            linear.forward = self._forward(linear)
        for parameter in target.parameters():
            parameter.requires_grad_(False)

    def _forward(self, linear):
        def forward(x):
            out = F.linear(x, linear.weight, linear.bias)
            if self.enabled:
                a, b = self.ab[linear._glora_path]
                if self.masks is not None:
                    b = b * self.masks[linear._glora_path][:, None]
                out = out + F.linear(F.linear(x, a), b).to(out.dtype)
            return out
        return forward

    def logits(self, idx, enabled):
        self.enabled = enabled
        return self.target(idx)

    def reset(self):
        with torch.no_grad():
            for _, b in self.ab.values():
                b.zero_()

    def state(self):
        return {path: (a.detach().cpu().clone(), b.detach().cpu().clone())
                for path, (a, b) in self.ab.items()}

    def load(self, state):
        with torch.no_grad():
            for path, (a, b) in self.ab.items():
                a.copy_(state[path][0])
                b.copy_(state[path][1])


def metrics(model, idx, cache, key):
    idx = idx.to(model.device)
    if key not in cache:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            base = model.logits(idx, False)
        cache[key] = (ce_each(base, idx).mean().item(),
                      F.log_softmax(base[:, :-1].float(), -1))
        del base
    base_ce, base_logp = cache[key]
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        edited = model.logits(idx, True)
    edited_ce = ce_each(edited, idx).mean().item()
    edited_logp = F.log_softmax(edited[:, :-1].float(), -1)
    kl = (base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean().item()
    del edited, edited_logp
    return {"base_ce": base_ce, "edited_ce": edited_ce,
            "delta_ce": edited_ce - base_ce, "kl_from_base": kl}


def train_config(model, blocks, base_logp, lam_en, lam_rom, lr, steps,
                 ceiling, tag, log_every):
    model.reset()
    optimizer = torch.optim.Adam(model.params, lr=lr)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    for step in range(steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in blocks}
        chosen["de"] = step % counts["de"]
        idx = torch.stack([blocks[lang][chosen[lang]] for lang in
                           ("de", "en", "fr", "es", "it")]).to(model.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model.logits(idx, True)
            german_ce = ce_each(logits[:1], idx[:1]).squeeze(0)
            kls = []
            for j, lang in enumerate(("en",) + ROMANCE, start=1):
                edited_logp = F.log_softmax(
                    logits[j:j + 1, :-1].float(), -1)
                kls.append(F.kl_div(
                    edited_logp, base_logp[lang][chosen[lang]],
                    log_target=True, reduction="none").sum(-1).mean())
            loss = (F.relu(ceiling - german_ce) + lam_en * kls[0]
                    + lam_rom * (kls[1] + kls[2] + kls[3]) / 3.0)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step + 1 == steps:
            log(f"{tag} step {step:04d}: deCE={german_ce.item():.2f} "
                f"enKL={kls[0].item():.3f} "
                f"romKL={((kls[1]+kls[2]+kls[3])/3).item():.3f}")


def worker(model, data, configs, args, ceiling):
    blocks = train_blocks(data)
    base_logp = {}
    for lang in ("en",) + ROMANCE:
        rows = []
        for block in blocks[lang]:
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                base = model.logits(block[None].to(model.device), False)
            rows.append(F.log_softmax(base[:, :-1].float(), -1))
            del base
        base_logp[lang] = rows
    cache = {}
    eval_sets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "de_dev": data["de_dev"],
        "en_dev": data["en_dev"],
        "fr_guard": data["fr_eval"][1:2],
        "es_guard": data["es_eval"][1:2],
        "it_guard": data["it_eval"][1:2],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    rows = []
    for config in configs:
        tag = (f"{config['variant']},l_en={config['lam_en']:g},"
               f"l_rom={config['lam_rom']:g}")
        t0 = time.perf_counter()
        train_config(model, blocks, base_logp, config["lam_en"],
                     config["lam_rom"], args.lr, args.steps, ceiling, tag,
                     args.log_every)
        row = dict(config)
        row["eval"] = {name: metrics(model, idx, cache, name)
                       for name, idx in eval_sets.items()}
        row["state"] = model.state()
        row["elapsed_seconds"] = time.perf_counter() - t0
        e = row["eval"]
        log(f"{tag}: de={e['german_europarl']['delta_ce']:+.3f} "
            f"en_pile={e['english_pile']['delta_ce']:+.3f} "
            f"fr/es/it={e['french_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['spanish_europarl_heldout']['delta_ce']:+.3f}/"
            f"{e['italian_europarl_heldout']['delta_ce']:+.3f} "
            f"({row['elapsed_seconds']:.0f}s)")
        rows.append(row)
    return rows


def rollouts(model, tokenizer, max_new_tokens):
    def generate(prompt, enabled):
        ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=model.device)
        for _ in range(max_new_tokens):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.logits(ids[:, -512:], enabled)
            ids = torch.cat(
                [ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokenizer.decode(ids[0].tolist())

    return {lang: {"prompt": prompt, "base": generate(prompt, False),
                   "edited": generate(prompt, True)}
            for lang, prompt in MASKED_ROLLOUT_PROMPTS.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--components", type=int, nargs="+",
                        default=[1668, 42, 2207, 3634, 2318],
                        help="causal German components whose rows gate LoRA")
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--configs", type=float, nargs="+",
                        default=[10.0, 10.0, 10.0, 30.0, 30.0, 30.0,
                                 100.0, 100.0],
                        help="flat (lam_en, lam_rom) pairs for the MASKED "
                             "variant; controls use the first two pairs")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--en_budget", type=float, default=0.1)
    parser.add_argument("--romance_guard", type=float, default=1.0)
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

    models, mask_stats = [], None
    for device in devices:
        target = geo1b.load_target_1b(device)
        masks, stats = german_row_masks(
            args.bank_path, args.components, args.coverage, target, device)
        mask_stats = stats
        models.append({
            "masked": GuidedLora(target, geo67.MODULES, args.rank, device,
                                 args.seed, masks=masks),
        })
    # The unmasked control shares each device's target; build lazily per run.
    configs = ([{"variant": "masked", "lam_en": le, "lam_rom": lr_}
                for le, lr_ in pairs]
               + [{"variant": "plain", "lam_en": le, "lam_rom": lr_}
                  for le, lr_ in pairs[:2]])
    ceiling = math.log(models[0]["masked"].target.hf.config.vocab_size)
    log(f"{len(configs)} configs (masked x{len(pairs)}, plain control x2)")

    # Round-robin configs across devices; plain variant runs by clearing the
    # mask on the same adapter object.
    queues = [configs[i::len(models)] for i in range(len(models))]

    def run_queue(bundle, queue):
        model = bundle["masked"]
        rows = []
        for config in queue:
            model.masks, saved = (
                (None, model.masks) if config["variant"] == "plain"
                else (model.masks, model.masks))
            rows.extend(worker(model, data, [config], args, ceiling))
            model.masks = saved
        return rows

    results = []
    with ThreadPoolExecutor(len(models)) as pool:
        futures = [pool.submit(run_queue, bundle, queue)
                   for bundle, queue in zip(models, queues) if queue]
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
    outputs = None
    if selected is not None:
        bundle = models[0]
        model = bundle["masked"]
        model.load(selected["state"])
        model.masks = None if selected["variant"] == "plain" else model.masks
        outputs = rollouts(model, tokenizer, args.max_new_tokens)
        log(f"selected {selected['variant']} l_en={selected['lam_en']:g} "
            f"l_rom={selected['lam_rom']:g}")

    def strip(row):
        return {key: value for key, value in row.items() if key != "state"}

    output = args.run_dir / "german_lora_guided.json"
    output.write_text(json.dumps({
        "format": "german_lora_guided_v1",
        "components": args.components,
        "coverage": args.coverage,
        "mask_rows": {path: stats for path, stats in mask_stats.items()},
        "rank": args.rank,
        "objective": ("relu(logV-deCE) + lam_en*KL_en "
                      "+ lam_rom*mean(KL_fr,KL_es,KL_it)"),
        "configs": [strip(row) for row in results],
        "selected": strip(selected) if selected else None,
        "rollouts": outputs,
    }, indent=2))
    if selected is not None:
        torch.save({
            "format": "german_lora_guided_adapter_v1",
            "model": geo1b.model_identity(),
            "components": args.components,
            "variant": selected["variant"],
            "lam_en": selected["lam_en"], "lam_rom": selected["lam_rom"],
            "state": selected["state"],
            "result": str(output),
        }, args.run_dir / "german_lora_guided_adapter.pt")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
