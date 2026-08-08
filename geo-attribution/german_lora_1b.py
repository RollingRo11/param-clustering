"""LoRA r=1 German-removal benchmark matching german_vpd_1b's protocol.

The post benchmarks its decomposition edits against LoRA fine-tuned with the
SAME objective: maximize German CE capped at chance while KL-regularizing
English logits to the original model. This runner reproduces that baseline at
1B against the streamed-decomposition scalar edits:

- rank-r (default 1) adapters B A^T on every decomposed linear (the same 112
  matrices the component edits touch); B starts at zero so training starts at
  the identity edit, mirroring the alpha=1 mask init;
- identical protocol data (2,048 German training tokens, paired English KL
  blocks, held-out Europarl dev, the same seven eval sets);
- a (learning rate x KL lambda) grid trained in parallel across GPUs, every
  config dev- and eval-scored so the full removal-vs-leak frontier is
  reported, with the winner chosen by the same constrained dev rule.

LoRA has no flat-landscape problem (gradients flow through fresh directions),
so no warm start is used — plain Adam from the identity, as in the post.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import geo1b  # noqa: E402
import geo67  # noqa: E402
from german_vpd_1b import (  # noqa: E402
    PROTOCOL_URL, ce_each, log, prepare_data)


class LoraModel:
    """Rank-r adapters on the decomposed linears with an enable switch."""

    def __init__(self, target, modules: list[str], rank: int, device: str,
                 seed: int):
        self.target = target
        self.modules = list(modules)
        self.device = device
        self.rank = rank
        self.enabled = False
        self.params: list[torch.Tensor] = []
        generator = torch.Generator().manual_seed(seed)
        self.ab: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for path in self.modules:
            linear = target.get_submodule(path)
            out_dim, in_dim = linear.weight.shape
            a = torch.randn(rank, in_dim, generator=generator) / \
                math.sqrt(in_dim)
            a = a.to(device).requires_grad_(True)
            b = torch.zeros(out_dim, rank, device=device,
                            requires_grad=True)
            self.ab[path] = (a, b)
            self.params += [a, b]
            linear._lora_path = path
            linear.forward = self._linear_forward(linear)
        for parameter in target.parameters():
            parameter.requires_grad_(False)

    def _linear_forward(self, linear):
        def forward(x):
            out = F.linear(x, linear.weight, linear.bias)
            if self.enabled:
                a, b = self.ab[linear._lora_path]
                out = out + F.linear(F.linear(x, a), b).to(out.dtype)
            return out
        return forward

    def logits(self, idx: torch.Tensor, enabled: bool) -> torch.Tensor:
        self.enabled = enabled
        return self.target(idx)

    def state(self) -> dict:
        return {path: (a.detach().cpu().clone(), b.detach().cpu().clone())
                for path, (a, b) in self.ab.items()}

    def load(self, state: dict) -> None:
        with torch.no_grad():
            for path, (a, b) in self.ab.items():
                a.copy_(state[path][0])
                b.copy_(state[path][1])

    def reset(self) -> None:
        with torch.no_grad():
            for a, b in self.ab.values():
                b.zero_()


def dataset_metrics(model: LoraModel, idx: torch.Tensor,
                    base_cache: dict, key: str) -> dict:
    idx = idx.to(model.device)
    if key not in base_cache:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            base = model.logits(idx, enabled=False)
        base_cache[key] = (
            ce_each(base, idx).mean().item(),
            F.log_softmax(base[:, :-1].float(), -1))
        del base
    base_ce, base_logp = base_cache[key]
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        edited = model.logits(idx, enabled=True)
    edited_ce = ce_each(edited, idx).mean().item()
    edited_logp = F.log_softmax(edited[:, :-1].float(), -1)
    kl = (base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean().item()
    del edited, edited_logp
    return {"base_ce": base_ce, "edited_ce": edited_ce,
            "delta_ce": edited_ce - base_ce, "kl_from_base": kl}


def train_config(model: LoraModel, data, lr: float, lam: float,
                 steps: int, log_every: int, ceiling: float,
                 en_logp_all: torch.Tensor, tag: str) -> list[dict]:
    model.reset()
    optimizer = torch.optim.Adam(model.params, lr=lr)
    de_train = data["de_train"]
    en_train = data["en_train"]
    history = []
    for step in range(steps):
        de = de_train[step % len(de_train)][None].to(model.device)
        en_index = (step * 7) % len(en_train)
        en = en_train[en_index][None].to(model.device)
        optimizer.zero_grad(set_to_none=True)
        idx = torch.cat([de, en])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model.logits(idx, enabled=True)
            german_ce = ce_each(logits[:1], idx[:1]).squeeze(0)
            edited_logp = F.log_softmax(logits[1:, :-1].float(), -1)
            english_kl = F.kl_div(
                edited_logp, en_logp_all[en_index:en_index + 1],
                log_target=True, reduction="none").sum(-1).mean()
            loss = F.relu(ceiling - german_ce) + lam * english_kl
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step + 1 == steps:
            row = {"step": step, "german_ce": german_ce.item(),
                   "english_kl": english_kl.item()}
            history.append(row)
            log(f"{tag} step {step:04d}: deCE={row['german_ce']:.3f} "
                f"enKL={row['english_kl']:.4f}")
        del logits
    return history


def worker(model: LoraModel, data, configs: list[dict], args,
           ceiling: float) -> list[dict]:
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        base = model.logits(data["en_train"].to(model.device), enabled=False)
    en_logp_all = F.log_softmax(base[:, :-1].float(), -1)
    del base
    base_cache: dict = {}
    eval_sets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "french_europarl": data["fr_eval"],
        "spanish_europarl": data["es_eval"],
        "italian_europarl": data["it_eval"],
        "codeparrot": data["code_eval"],
    }
    rows = []
    for config in configs:
        tag = f"lr={config['lr']:g},lambda={config['kl_lambda']:g}"
        t0 = time.perf_counter()
        history = train_config(
            model, data, config["lr"], config["kl_lambda"], args.steps,
            args.log_every, ceiling, en_logp_all, tag)
        row = dict(config)
        row["history"] = history
        row["dev_de"] = dataset_metrics(
            model, data["de_dev"], base_cache, "de_dev")
        row["dev_en"] = dataset_metrics(
            model, data["en_dev"], base_cache, "en_dev")
        row["eligible"] = (
            row["dev_en"]["delta_ce"] < args.max_dev_en_damage)
        row["eval"] = {name: dataset_metrics(model, idx, base_cache, name)
                       for name, idx in eval_sets.items()}
        row["state"] = model.state()
        row["elapsed_seconds"] = time.perf_counter() - t0
        log(f"{tag}: dev dCE de/en={row['dev_de']['delta_ce']:+.3f}/"
            f"{row['dev_en']['delta_ce']:+.3f}, eval de/en_pile="
            f"{row['eval']['german_europarl']['delta_ce']:+.3f}/"
            f"{row['eval']['english_pile']['delta_ce']:+.3f} "
            f"({row['elapsed_seconds']:.0f}s)")
        rows.append(row)
    return rows


def rollouts(model: LoraModel, tokenizer, max_new_tokens: int) -> dict:
    prompts = {
        "de": "Die Lösung liegt natürlich",
        "en": "And, of course, the solution does",
        "fr": "Cependant, le projet",
        "es": "Sin embargo, el proyecto",
        "it": "La ridestinazione delle risorse",
    }

    def generate(prompt: str, enabled: bool) -> str:
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

    return {language: {"prompt": prompt,
                       "base": generate(prompt, False),
                       "edited": generate(prompt, True)}
            for language, prompt in prompts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1m_stream")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--lrs", type=float, nargs="+",
                        default=[3e-3, 1e-2, 3e-2])
    parser.add_argument("--kl_lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 30.0, 100.0])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--max_dev_en_damage", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    args.output = args.run_dir / f"german_lora_r{args.rank}.json"
    args.adapter_output = args.run_dir / f"german_lora_r{args.rank}_best.pt"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]

    models = []
    for device in devices:
        target = geo1b.load_target_1b(device)
        models.append(LoraModel(
            target, geo67.MODULES, args.rank, device, args.seed))
    parameters = sum(p.numel() for p in models[0].params)
    ceiling = math.log(models[0].target.hf.config.vocab_size)
    log(f"LoRA r={args.rank} on {len(geo67.MODULES)} matrices: "
        f"{parameters} trainable parameters; devices {devices}")

    configs = [{"lr": lr, "kl_lambda": lam}
               for lr in args.lrs for lam in args.kl_lambdas]
    queues = [configs[i::len(models)] for i in range(len(models))]
    with ThreadPoolExecutor(len(models)) as pool:
        futures = [pool.submit(worker, model, data, queue, args, ceiling)
                   for model, queue in zip(models, queues) if queue]
        rows = [row for future in futures for row in future.result()]

    eligible = [row for row in rows if row["eligible"]]
    if eligible:
        selected = max(
            eligible, key=lambda row: row["dev_de"]["edited_ce"])
        selection = ("maximum dev German CE subject to dev English delta CE "
                     f"< {args.max_dev_en_damage:g}")
    else:
        selected = max(rows, key=lambda row: (
            row["dev_de"]["delta_ce"]
            - 10 * max(0, row["dev_en"]["delta_ce"]
                       - args.max_dev_en_damage)))
        selection = "penalized fallback; no config met English constraint"
    log(f"selected lr={selected['lr']:g} lambda={selected['kl_lambda']:g}: "
        f"dev dCE de/en={selected['dev_de']['delta_ce']:+.3f}/"
        f"{selected['dev_en']['delta_ce']:+.3f}")

    models[0].load(selected["state"])
    outputs = rollouts(models[0], tokenizer, args.max_new_tokens)

    def strip(row):
        return {key: value for key, value in row.items() if key != "state"}

    result = {
        "format": "german_lora_benchmark_v1",
        "protocol": {
            "kind": "rank-r LoRA fine-tune benchmark on decomposed linears",
            "objective":
                "relu(log(vocab)-German_CE) + lambda*KL(base_en||edit_en)",
            "german_reference": PROTOCOL_URL,
            "model": geo1b.model_identity(),
            "rank": args.rank,
            "trainable_parameters": parameters,
            "modules": len(geo67.MODULES),
        },
        "chance_ce": ceiling,
        "steps": args.steps,
        "german_token_budget": args.train_tokens,
        "configs": [strip(row) for row in rows],
        "selection_rule": selection,
        "selected": strip(selected),
        "rollouts": outputs,
    }
    args.output.write_text(json.dumps(result, indent=2))
    torch.save({
        "format": "german_lora_adapter_v1",
        "model": geo1b.model_identity(),
        "rank": args.rank,
        "lr": selected["lr"],
        "kl_lambda": selected["kl_lambda"],
        "state": selected["state"],
        "result": str(args.output),
    }, args.adapter_output)
    log(f"wrote result {args.output}")
    log(f"wrote best adapter {args.adapter_output}")


if __name__ == "__main__":
    main()
