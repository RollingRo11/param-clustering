"""VPD-style few-scalar German edit for the streamed 1B decomposition.

This is the protocol-matched successor to german67.py. The legacy experiment
ranks several components and sweeps hand-picked scale values. This runner:

1. ranks German-vs-English components on held-out attribution fingerprints;
2. selects one or a few components specific to German rather than
   multilingual;
3. learns one scalar alpha_i per selected component in

       W' = W + sum_i (alpha_i - 1) W_component_i

   by maximizing capped German CE while regularizing English logits to the
   original model with KL (the post's single-mask and 16-mask fine-tunes are
   the k=1 and k>1 cases); and
4. reports held-out German, Pile English, French, Spanish, Italian, and code
   effects plus an in-place weight-edit equivalence check.

The streamed decomposition contains cross-layer, fractional weight components
rather than VPD's per-matrix rank-1 atoms. Scaling a handful of such
components is nevertheless the same kind of constrained weight-space
fine-tune: the only trainable degrees of freedom amplify, suppress, or invert
existing parameter components. No activation steering or inference-time token
gate is used.

Optimization notes for the 1B run (the 67M post needed none of this):

- The components hold ~1e-5 of squared weight mass, so the loss landscape is
  flat to below batch noise near the protocol init alpha=1 and Adam alone
  random-walks. A warm-start scan of the SAME objective over an alpha grid
  seeds each KL-coefficient config before gradient refinement.
- The (English KL, German destruction) frontier is non-convex here, so
  lambda-scalarized optima cluster at its ends. Selection therefore
  dev-screens the trained alphas together with the scan grid and a local
  refinement grid under the post's rule: maximize dev German CE subject to
  the dev English damage threshold.
- Because F.linear is linear in the weight, per-batch-row alpha vectors
  evaluate many candidate edits in one forward (exactly, by linearity);
  training, scanning, screening, evaluation, and rollouts shard across all
  visible GPUs. The selected alpha vector is still verified against literal
  parameter surgery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import itertools
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import geo1b  # noqa: E402,F401 - installs the 1B target bindings in geo67
from collect_fast_impl import pass_features, setup_model  # noqa: E402
from geo1m import load_spec  # noqa: E402
from streaming_decomposition import load_stream_model  # noqa: E402


EUROPARL = "Helsinki-NLP/europarl"
CODEPARROT = "codeparrot/codeparrot-clean-valid"
PROTOCOL_URL = (
    "https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/"
    "exploration-fine-tuning-with-parameter-decomposition"
)
VPD_URL = "https://www.goodfire.com/research/interpreting-lm-parameters"


def log(message: str) -> None:
    print(f"[german-vpd] {message}", flush=True)


def fmt_vec(values) -> str:
    return "[" + ",".join(f"{v:+.3g}" for v in values) + "]"


def blocks(tokens: list[int], count: int, seq_len: int) -> torch.Tensor:
    need = count * seq_len
    if len(tokens) < need:
        raise ValueError(f"only collected {len(tokens)} tokens; need {need}")
    return torch.tensor(tokens[:need], dtype=torch.long).view(count, seq_len)


def paired_europarl(tokenizer, config: str, languages: tuple[str, str],
                    token_count: int) -> dict[str, list[int]]:
    """Stream aligned Europarl documents until both languages have enough."""
    from datasets import load_dataset

    stream = load_dataset(EUROPARL, config, split="train", streaming=True)
    result = {language: [] for language in languages}
    eos = tokenizer.eos_token_id
    for example in stream:
        translation = example["translation"]
        for language in languages:
            text = translation[language]
            result[language].extend(
                tokenizer.encode(text, add_special_tokens=False))
            if eos is not None:
                result[language].append(eos)
        if all(len(result[language]) >= token_count for language in languages):
            break
    return result


def code_tokens(tokenizer, token_count: int) -> list[int]:
    from datasets import load_dataset

    stream = load_dataset(CODEPARROT, split="train", streaming=True)
    result: list[int] = []
    eos = tokenizer.eos_token_id
    for example in stream:
        result.extend(tokenizer.encode(
            example["content"], add_special_tokens=False))
        if eos is not None:
            result.append(eos)
        if len(result) >= token_count:
            break
    return result


def prepare_data(args, tokenizer) -> dict[str, torch.Tensor]:
    """Create and cache disjoint train/dev/eval blocks from protocol datasets."""
    if args.data_cache.exists() and not args.refresh_data:
        cached = torch.load(args.data_cache, weights_only=True,
                            map_location="cpu")
        if (cached.get("seq_len") == args.seq_len
                and cached.get("train_tokens") == args.train_tokens
                and cached.get("eval_blocks") == args.eval_blocks):
            log(f"reusing cached protocol data {args.data_cache}")
            return cached

    rank_blocks = 1
    train_blocks = math.ceil(args.train_tokens / args.seq_len)
    dev_blocks = 2
    total_de = rank_blocks + train_blocks + dev_blocks + args.eval_blocks
    need_de = total_de * args.seq_len
    log(f"streaming {need_de} aligned German/English Europarl tokens")
    de_en = paired_europarl(
        tokenizer, "de-en", ("de", "en"), need_de)
    de_all = blocks(de_en["de"], total_de, args.seq_len)
    en_all = blocks(de_en["en"], total_de, args.seq_len)

    offset = 0
    data: dict[str, object] = {
        "format": "german_vpd_protocol_data_v1",
        "seq_len": args.seq_len,
        "train_tokens": args.train_tokens,
        "eval_blocks": args.eval_blocks,
        "sources": {
            "de_en": f"{EUROPARL}:de-en/train",
            "fr": f"{EUROPARL}:en-fr/train",
            "es": f"{EUROPARL}:en-es/train",
            "it": f"{EUROPARL}:en-it/train",
            "code": f"{CODEPARROT}:train",
            "pile": str(geo1b.BIN_PATH),
        },
    }
    data["de_rank"] = de_all[offset:offset + rank_blocks]
    data["en_rank"] = en_all[offset:offset + rank_blocks]
    offset += rank_blocks
    data["de_train"] = de_all[offset:offset + train_blocks]
    data["en_train"] = en_all[offset:offset + train_blocks]
    offset += train_blocks
    data["de_dev"] = de_all[offset:offset + dev_blocks]
    data["en_dev"] = en_all[offset:offset + dev_blocks]
    offset += dev_blocks
    data["de_eval"] = de_all[offset:offset + args.eval_blocks]
    data["en_europarl_eval"] = en_all[offset:offset + args.eval_blocks]

    # These blocks play the role of component-label inspection in the post:
    # they prevent selecting a generic "foreign language" component.
    for language, config in (("fr", "en-fr"), ("es", "en-es"),
                             ("it", "en-it")):
        count = (1 + args.eval_blocks) * args.seq_len
        pair = paired_europarl(tokenizer, config, ("en", language), count)
        all_blocks = blocks(pair[language], 1 + args.eval_blocks, args.seq_len)
        data[f"{language}_rank"] = all_blocks[:1]
        data[f"{language}_eval"] = all_blocks[1:]

    data["code_eval"] = blocks(
        code_tokens(tokenizer, args.eval_blocks * args.seq_len),
        args.eval_blocks, args.seq_len)

    # Collection consumed the beginning of this deterministic stream. Reserve
    # its tail as a disjoint English monitor.
    pile = np.memmap(geo1b.BIN_PATH, dtype=np.uint32, mode="r")
    needed = args.eval_blocks * args.seq_len
    if len(pile) < needed:
        raise ValueError(f"Pile token stream is shorter than {needed} tokens")
    data["pile_en_eval"] = torch.from_numpy(
        np.array(pile[-needed:], dtype=np.int64, copy=True)
    ).view(args.eval_blocks, args.seq_len)

    args.data_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.data_cache)
    log(f"cached protocol data at {args.data_cache}")
    return data


def ranking_args(bank: dict) -> SimpleNamespace:
    return SimpleNamespace(
        sensor=bank.get("sensor", "gim"),
        gim_tau=float(bank.get("gim_tau", 2.0)),
        compile=False,
        compile_mode="default",
        ig_k=1,
        bf16=True,
        # flash has no backward kernel for the IG sensors; only GIM's custom
        # backward can ride the fused path
        fused_attention=bank.get("sensor", "gim") == "gim",
        scalar=bank.get("scalar", "equal_reward"),
    )


def rank_components(args, data, bank_meta: dict) -> dict:
    """Rank with the frozen attribution fingerprints used by streaming.

    Exhaustive exact attribution of 2,048 billion-weight fractional slices
    would repeatedly materialize the whole bank. The frozen cluster posterior
    is the scalable causal-importance proxy for this decomposition.
    """
    device = args.device
    cfg = ranking_args(bank_meta)
    cap = setup_model(cfg, device)
    spec, scales, dim = load_spec(args.run_dir, device)
    stream_model = load_stream_model(
        args.run_dir / "stream_model.pt", device)

    languages = ["de", "en", "fr", "es", "it"]
    idx = torch.cat([data[f"{language}_rank"] for language in languages]).to(
        device)
    available = torch.arange(4, args.seq_len - 2)
    generator = torch.Generator().manual_seed(args.seed)
    selected = available[
        torch.randperm(available.numel(), generator=generator)[
            :min(args.rank_positions, available.numel())]]
    pos = selected[None].expand(len(languages), -1).to(device)
    bi = torch.arange(len(languages), device=device)[:, None].expand_as(pos)
    t0 = time.perf_counter()
    phi, _ = pass_features(
        cfg, cap, idx, pos, bi, spec, scales, dim, return_pg=False)
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize(
        (x - stream_model["mean"]) @ stream_model["projector"], dim=1)
    similarities = y @ stream_model["centroids"].t()
    posterior = torch.softmax(
        similarities / args.rank_temperature, dim=1)
    posterior = posterior.view(
        len(languages), pos.shape[1], -1).mean(1)
    labels = similarities.argmax(1).view(len(languages), -1)
    hard = torch.stack([
        torch.bincount(row, minlength=bank_meta["C"]).float() / row.numel()
        for row in labels
    ])

    lang_index = {language: i for i, language in enumerate(languages)}
    de = posterior[lang_index["de"]]
    en = posterior[lang_index["en"]]
    contrast = de - en
    candidates = contrast.argsort(descending=True)[:args.candidate_k]

    rows = []
    for component in candidates.tolist():
        activity = {
            language: float(posterior[lang_index[language], component])
            for language in languages
        }
        hard_activity = {
            language: float(hard[lang_index[language], component])
            for language in languages
        }
        foreign_max = max(activity[x] for x in ("en", "fr", "es", "it"))
        rows.append({
            "component": component,
            "de_minus_en": activity["de"] - activity["en"],
            "german_specificity": activity["de"] - foreign_max,
            "posterior_activity": activity,
            "hard_frequency": hard_activity,
        })
    rows.sort(key=lambda row: row["german_specificity"], reverse=True)
    chosen = int(rows[0]["component"])
    result = {
        "method": "frozen attribution-fingerprint cluster posterior",
        "temperature": args.rank_temperature,
        "positions_per_language": int(pos.shape[1]),
        "top_german_minus_english": candidates.tolist(),
        "inspected_candidates": rows,
        "selected_component": chosen,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.ranking_output.write_text(json.dumps(result, indent=2))
    log(f"selected component {chosen}; top German-specific candidates "
        f"{[row['component'] for row in rows[:5]]} "
        f"({result['elapsed_seconds']:.1f}s)")

    del cap, phi, x, y, similarities, posterior, stream_model, spec, scales
    gc.collect()
    torch.cuda.empty_cache()
    return result


class ComponentEditor:
    """Differentiable, exact weight-space scaling of softpart components.

    ``alpha`` is one scalar per selected component. A 1-D alpha of length k
    is the literal scalar edit (W' formed in fp32 before autocast); a 2-D
    alpha of shape (batch, k) evaluates a different candidate edit on every
    batch row through the linearity of F.linear in the weight.
    """

    def __init__(self, target, bank: dict, components: list[int],
                 device: str):
        if bank.get("format") != "softpart":
            raise ValueError("scalar editor currently requires a softpart bank")
        self.target = target
        self.device = device
        self.components = [int(component) for component in components]
        self.modules = list(bank["modules"])
        self.slices: dict[str, torch.Tensor] = {}
        self.alpha: torch.Tensor | None = None
        mass = [0.0] * len(self.components)
        total_mass = 0.0

        for number, path in enumerate(self.modules, 1):
            linear = target.get_submodule(path)
            sidx = bank["sidx"][path].to(device, non_blocking=True)
            swgt = bank["swgt"][path].to(device, non_blocking=True)
            weight = linear.weight.detach()
            per_component = []
            for slot, component in enumerate(self.components):
                share = ((sidx == component) * swgt).sum(0).float()
                component_weight = (weight * share).contiguous()
                per_component.append(component_weight)
                mass[slot] += component_weight.square().sum().item()
            self.slices[path] = torch.stack(per_component)
            total_mass += weight.square().sum().item()
            del sidx, swgt, per_component
            if number % 16 == 0 or number == len(self.modules):
                log(f"materialized components {self.components}: "
                    f"{number}/{len(self.modules)} matrices")
        self.mass_fraction = [m / total_mass for m in mass]
        self._install()

    def _install(self) -> None:
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        for path in self.modules:
            linear = self.target.get_submodule(path)
            linear._component_edit_path = path
            linear._component_edit_original_forward = linear.forward
            linear.forward = self._linear_forward(linear)

    def replicate(self, device: str) -> "ComponentEditor":
        """Second editor on another GPU, reusing the materialized slices.

        Preserves the concrete subclass so overridden forwards survive."""
        twin = object.__new__(type(self))
        twin.target = geo1b.load_target_1b(device)
        twin.device = device
        twin.components = list(self.components)
        twin.modules = list(self.modules)
        twin.alpha = None
        twin.mass_fraction = list(self.mass_fraction)
        twin.slices = {
            path: self.slices[path].to(device) for path in self.modules}
        twin._install()
        log(f"replicated component editor on {device}")
        return twin

    def _coerce(self, alpha) -> torch.Tensor | None:
        if alpha is None:
            return None
        k = len(self.components)
        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(
                alpha, dtype=torch.float32, device=self.device)
        if alpha.ndim == 0:
            alpha = alpha.expand(k).clone()
        if alpha.ndim == 1 and alpha.numel() != k:
            raise ValueError(f"alpha vector must have {k} entries")
        if alpha.ndim == 2 and alpha.shape[1] != k:
            raise ValueError(f"batched alpha must be (batch, {k})")
        return alpha

    def _linear_forward(self, linear):
        def forward(x):
            if self.alpha is None:
                return F.linear(x, linear.weight, linear.bias)
            scale = self.alpha - 1.0
            slices = self.slices[linear._component_edit_path]
            if scale.ndim == 2:
                # By linearity of F.linear in the weight, this equals the
                # per-row weight edit W + sum_i scale[b, i] * W_component_i
                # for every batch row in one pass; used to train/screen many
                # candidate edits at once. The single selected alpha vector
                # is still checked against literal parameter surgery below.
                if scale.shape[0] != x.shape[0]:
                    raise ValueError(
                        "batched alpha needs one row per batch row")
                out = F.linear(x, linear.weight, linear.bias)
                shape = (-1,) + (1,) * (x.ndim - 1)
                for slot in range(scale.shape[1]):
                    out = out + (scale[:, slot].view(shape).to(out.dtype)
                                 * F.linear(x, slices[slot]))
                return out
            # Form W' before autocast reaches F.linear. This is the literal
            # parameter-space operation and therefore has the same BF16
            # rounding as an in-place edit of the fp32 master weights.
            edited_weight = linear.weight
            for slot in range(scale.numel()):
                edited_weight = edited_weight + scale[slot] * slices[slot]
            return F.linear(x, edited_weight, linear.bias)
        return forward

    def logits(self, idx: torch.Tensor, alpha=None) -> torch.Tensor:
        self.alpha = self._coerce(alpha)
        return self.target(idx)

    def verify_in_place(self, idx: torch.Tensor, alpha) -> dict:
        """Compare the differentiable expression to literal parameter surgery."""
        vector = self._coerce(alpha)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            expression = self.logits(idx, vector).float()
            self.alpha = None
            deltas = (vector - 1.0).tolist()
            for path in self.modules:
                weight = self.target.get_submodule(path).weight
                for slot, delta in enumerate(deltas):
                    weight.add_(self.slices[path][slot], alpha=delta)
            literal = self.target(idx).float()
            for path in self.modules:
                weight = self.target.get_submodule(path).weight
                for slot, delta in enumerate(deltas):
                    weight.add_(self.slices[path][slot], alpha=-delta)
        error = (expression - literal).abs()
        return {
            "max_abs_logit_error": error.max().item(),
            "mean_abs_logit_error": error.mean().item(),
            "tokens_checked": int(idx.numel()),
        }


def base_logits(editor: ComponentEditor,
                tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    result = []
    for idx in tensors:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            result.append(editor.logits(idx.to(editor.device), None)
                          .detach().bfloat16().cpu())
    return result


def ce_each(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), idx[:, 1:],
        reduction="none")
    return loss.mean(1)


def forward_metrics(editor: ComponentEditor, idx: torch.Tensor,
                    original: torch.Tensor, alpha) -> dict:
    idx = idx.to(editor.device)
    original = original.to(editor.device)
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        edited = editor.logits(idx, alpha)
    base_ce = ce_each(original, idx).mean().item()
    edited_ce = ce_each(edited, idx).mean().item()
    base_logp = F.log_softmax(original[:, :-1].float(), -1)
    edited_logp = F.log_softmax(edited[:, :-1].float(), -1)
    kl = F.kl_div(
        edited_logp, base_logp, log_target=True, reduction="sum"
    ).item() / (idx.shape[0] * (idx.shape[1] - 1))
    return {
        "base_ce": base_ce,
        "edited_ce": edited_ce,
        "delta_ce": edited_ce - base_ce,
        "kl_from_base": kl,
    }


def _sweep_worker(editor: ComponentEditor, idx_blocks: torch.Tensor,
                  alphas: list[list[float]], rows_per_chunk: int) -> dict:
    device = editor.device
    idx = idx_blocks.to(device)
    n, seq = idx.shape
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        base = editor.logits(idx, None)
    base_ce = ce_each(base, idx).mean().item()
    base_logp = F.log_softmax(base[:, :-1].float(), -1)
    base_p = base_logp.exp()
    del base
    ce_out: list[float] = []
    kl_out: list[float] = []
    per = max(1, rows_per_chunk // n)
    for start in range(0, len(alphas), per):
        chunk = alphas[start:start + per]
        batch = idx.repeat(len(chunk), 1)
        alpha_batch = torch.tensor(
            chunk, dtype=torch.float32,
            device=device).repeat_interleave(n, dim=0)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            logits = editor.logits(batch, alpha_batch)
        ce_out.extend(
            ce_each(logits, batch).view(len(chunk), n).mean(1).tolist())
        for a in range(len(chunk)):
            logp = F.log_softmax(
                logits[a * n:(a + 1) * n, :-1].float(), -1)
            kl_out.append(
                (base_p * (base_logp - logp)).sum(-1).mean().item())
            del logp
        del logits
    return {"base_ce": base_ce, "ce": ce_out, "kl": kl_out}


def alpha_sweep(editors: list[ComponentEditor], idx_blocks: torch.Tensor,
                alphas: list[list[float]], rows_per_chunk: int) -> dict:
    """Mean CE and KL-from-base for many alpha vectors, sharded over editors."""
    parts: list[list[list[float]]] = [[] for _ in editors]
    for i, alpha in enumerate(alphas):
        parts[i % len(editors)].append(alpha)
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [
            pool.submit(_sweep_worker, editor, idx_blocks, part,
                        rows_per_chunk)
            for editor, part in zip(editors, parts) if part]
        results = [future.result() for future in futures]
    ce = [0.0] * len(alphas)
    kl = [0.0] * len(alphas)
    cursors = [0] * len(results)
    for i in range(len(alphas)):
        slot = i % len(editors)
        ce[i] = results[slot]["ce"][cursors[slot]]
        kl[i] = results[slot]["kl"][cursors[slot]]
        cursors[slot] += 1
    return {"base_ce": results[0]["base_ce"], "ce": ce, "kl": kl}


def scan_vectors(args, k: int) -> list[list[float]]:
    """Alpha-vector grid for the warm-start scan: full 1-D grid at k=1, a
    coarse joint product grid at k=2, and — because the product grid grows as
    grid**k — a tied-alpha grid plus one-component-at-a-time deviations from
    the identity for k>=3 (Adam then unties the scalars from the seed)."""
    if k == 1:
        return [[alpha] for alpha in args.init_grid]
    coarse = [-20.0, -12.0, -8.0, -4.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0,
              5.5, 6.0, 6.5, 8.0, 12.0, 20.0]
    if k == 2:
        return [list(combo) for combo in itertools.product(coarse, repeat=k)]
    vectors = [[alpha] * k for alpha in args.init_grid]
    for slot in range(k):
        for alpha in coarse:
            if alpha != 1.0:
                vector = [1.0] * k
                vector[slot] = alpha
                vectors.append(vector)
    return vectors


def warm_start_scan(args, editors: list[ComponentEditor], data,
                    ceiling: float) -> dict:
    """Evaluate the training objective on an alpha grid to seed each config.

    The objective is a handful of scalars, and near alpha=1 this
    decomposition's landscape is flat to well below the training-batch noise
    floor (the components hold ~1e-5 of squared weight mass), so Adam from
    the protocol init of 1.0 random-walks instead of finding the destructive
    basin. Scanning the SAME capped-destruction + KL objective over a coarse
    grid and starting each config at its per-lambda argmin keeps the
    trainable degrees of freedom, loss, and dev-set selection rule unchanged;
    only the optimizer gains restarts, which the 67M post never needed.
    """
    t0 = time.perf_counter()
    grid = scan_vectors(args, len(editors[0].components))
    de = alpha_sweep(editors, data["de_train"], grid, args.sweep_rows)
    en = alpha_sweep(editors, data["en_train"], grid, args.sweep_rows)
    rows = [{"alpha": alpha, "german_ce": dce, "english_kl": ekl,
             "destroy": max(0.0, ceiling - dce)}
            for alpha, dce, ekl in zip(grid, de["ce"], en["kl"])]
    inits = []
    for regularizer in args.kl_lambdas:
        best = min(rows, key=lambda row: (
            row["destroy"] + regularizer * row["english_kl"]))
        inits.append(best["alpha"])
    top = sorted(rows, key=lambda row: row["destroy"])[:8]
    log(f"warm-start scan over {len(grid)} vectors; most destructive: "
        + ", ".join(f"{fmt_vec(row['alpha'])}:deCE={row['german_ce']:.2f}/"
                    f"enKL={row['english_kl']:.3f}" for row in top))
    log("warm-start inits " + ", ".join(
        f"lambda={lam:g}->{fmt_vec(alpha)}"
        for lam, alpha in zip(args.kl_lambdas, inits)))
    return {"grid": rows, "inits": inits,
            "elapsed_seconds": time.perf_counter() - t0}


def _train_worker(editor: ComponentEditor, alphas: torch.nn.Parameter,
                  lambdas: torch.Tensor, de: torch.Tensor, en: torch.Tensor,
                  en_logp: torch.Tensor, ceiling: float) -> tuple:
    device = editor.device
    configs = alphas.shape[0]
    idx = torch.cat([de.to(device), en.to(device)])
    batch = idx.repeat(configs, 1)
    alpha_batch = alphas.repeat_interleave(2, dim=0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        logits = editor.logits(batch, alpha_batch)
        german_ce = ce_each(logits[0::2], batch[0::2])
        edited_logp = F.log_softmax(logits[1::2, :-1].float(), -1)
        english_kl = F.kl_div(
            edited_logp, en_logp.expand(configs, -1, -1), log_target=True,
            reduction="none").sum(-1).mean(-1)
        destroy = F.relu(ceiling - german_ce)
        per_config = destroy + lambdas * english_kl
        loss = per_config.sum()
    loss.backward()
    return (german_ce.detach().cpu(), english_kl.detach().cpu(),
            per_config.detach().cpu())


def refinement_vectors(center: list[float]) -> list[list[float]]:
    """Local product grid around the best eligible candidate."""
    offsets = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    if len(center) > 3:  # product grid explodes; vary one coordinate at a time
        vectors = []
        for slot in range(len(center)):
            for offset in offsets:
                vector = list(center)
                vector[slot] += offset
                vectors.append(vector)
        return vectors
    return [[c + o for c, o in zip(center, combo)]
            for combo in itertools.product(offsets, repeat=len(center))]


def train_scalars(args, editors: list[ComponentEditor], data) -> dict:
    """Train one alpha vector per KL coefficient, all configs batched per pass.

    Configs are split across the editor replicas (one per GPU); within each
    replica every config's German and English rows share a single batched
    forward/backward through the per-row-alpha linearity identity.
    """
    de_train = data["de_train"]
    en_train = data["en_train"]
    k = len(editors[0].components)
    configs = len(args.kl_lambdas)
    ceiling = math.log(editors[0].target.hf.config.vocab_size)
    warm = warm_start_scan(args, editors, data, ceiling)

    # Contiguous config split across replicas.
    share = math.ceil(configs / len(editors))
    groups = []
    for slot, editor in enumerate(editors):
        lo, hi = slot * share, min((slot + 1) * share, configs)
        if lo >= hi:
            continue
        groups.append({
            "editor": editor,
            "alphas": torch.nn.Parameter(torch.tensor(
                warm["inits"][lo:hi], dtype=torch.float32,
                device=editor.device)),
            "lambdas": torch.tensor(
                args.kl_lambdas[lo:hi], device=editor.device),
        })
    optimizer = torch.optim.Adam(
        [group["alphas"] for group in groups], lr=args.lr)

    # Per-device base English log-probs for every training block.
    for group in groups:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            base = group["editor"].logits(
                en_train.to(group["editor"].device), None)
        group["en_logp"] = F.log_softmax(base[:, :-1].float(), -1)
        del base

    history = []
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(len(groups))
    for step in range(args.steps):
        de = de_train[step % len(de_train)][None]
        en_index = (step * 7) % len(en_train)
        en = en_train[en_index][None]
        optimizer.zero_grad(set_to_none=True)
        futures = [
            pool.submit(_train_worker, group["editor"], group["alphas"],
                        group["lambdas"], de, en,
                        group["en_logp"][en_index:en_index + 1], ceiling)
            for group in groups]
        outputs = [future.result() for future in futures]
        optimizer.step()
        with torch.no_grad():
            for group in groups:
                group["alphas"].nan_to_num_(
                    nan=1.0, posinf=args.alpha_max, neginf=args.alpha_min)
                group["alphas"].clamp_(args.alpha_min, args.alpha_max)
        if step % args.log_every == 0 or step + 1 == args.steps:
            alpha_rows = torch.cat(
                [group["alphas"].detach().cpu() for group in groups])
            german_ce = torch.cat([out[0] for out in outputs])
            english_kl = torch.cat([out[1] for out in outputs])
            per_config = torch.cat([out[2] for out in outputs])
            row = {
                "step": step,
                "alpha": alpha_rows.tolist(),
                "german_ce": german_ce.tolist(),
                "english_kl": english_kl.tolist(),
                "loss": per_config.tolist(),
            }
            history.append(row)
            log(f"step {step:04d}: " + ", ".join(
                f"lambda={lam:g} alpha={fmt_vec(alpha)} "
                f"deCE={dece:.3f} enKL={enkl:.4f}"
                for lam, alpha, dece, enkl in zip(
                    args.kl_lambdas, row["alpha"], row["german_ce"],
                    row["english_kl"])))
    pool.shutdown()
    trained = torch.cat(
        [group["alphas"].detach().cpu() for group in groups]).tolist()

    # Select on held-out Europarl dev, requiring bounded English damage.
    # Candidates are the trained alphas PLUS the warm-start grid and a local
    # refinement grid: the lambda-scalarized objective can only reach the
    # convex hull of the (KL, destruction) frontier, which is non-convex for
    # this decomposition, so the constrained optimum need not be any
    # config's minimizer. The rule itself is unchanged from the post:
    # maximize dev German CE subject to dev English delta CE below threshold.
    def dev_rows(entries):
        vectors = [row["alpha"] for row in entries]
        dev_de = alpha_sweep(editors, data["de_dev"], vectors,
                             args.sweep_rows)
        dev_en = alpha_sweep(editors, data["en_dev"], vectors,
                             args.sweep_rows)
        for row, dce, dkl, ece, ekl in zip(
                entries, dev_de["ce"], dev_de["kl"], dev_en["ce"],
                dev_en["kl"]):
            row["dev_de"] = {"base_ce": dev_de["base_ce"], "edited_ce": dce,
                             "delta_ce": dce - dev_de["base_ce"],
                             "kl_from_base": dkl}
            row["dev_en"] = {"base_ce": dev_en["base_ce"], "edited_ce": ece,
                             "delta_ce": ece - dev_en["base_ce"],
                             "kl_from_base": ekl}
            row["eligible"] = (
                row["dev_en"]["delta_ce"] < args.max_dev_en_damage)
        return entries

    candidates = [
        {"source": "trained", "kl_lambda": regularizer, "alpha": vector}
        for regularizer, vector in zip(args.kl_lambdas, trained)]
    seen = {tuple(round(a, 4) for a in row["alpha"]) for row in candidates}
    scan_extra = scan_vectors(args, k)
    if k == 1:
        scan_extra = scan_extra + [[alpha] for alpha in args.select_grid]
    for vector in scan_extra:
        key = tuple(round(a, 4) for a in vector)
        if key not in seen:
            seen.add(key)
            candidates.append(
                {"source": "dev_screen", "kl_lambda": None, "alpha": vector})
    candidates = dev_rows(candidates)

    def pick(rows):
        eligible = [row for row in rows if row["eligible"]]
        if eligible:
            return max(eligible, key=lambda row: row["dev_de"]["edited_ce"])
        return max(rows, key=lambda row: (
            row["dev_de"]["delta_ce"]
            - 10 * max(0, row["dev_en"]["delta_ce"]
                       - args.max_dev_en_damage)))

    best = pick(candidates)
    refined = [
        {"source": "refine", "kl_lambda": None, "alpha": vector}
        for vector in refinement_vectors(best["alpha"])
        if tuple(round(a, 4) for a in vector) not in seen]
    if refined:
        candidates.extend(dev_rows(refined))
    selected = pick(candidates)
    if any(row["eligible"] for row in candidates):
        selection = (
            "maximum dev German CE subject to dev English delta CE < "
            f"{args.max_dev_en_damage:g}, over trained + screened + refined "
            "candidates")
    else:
        selection = "penalized fallback; no candidate met English constraint"
    result = {
        "objective": "relu(log(vocab)-German_CE) + lambda*KL(base_en||edit_en)",
        "chance_ce": ceiling,
        "learning_rate": args.lr,
        "steps": args.steps,
        "german_token_budget": args.train_tokens,
        "warm_start": warm,
        "configs": [row for row in candidates if row["source"] == "trained"],
        "dev_candidates": candidates,
        "selection_rule": selection,
        "selected": selected,
        "history": history,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    lam = selected.get("kl_lambda")
    log(f"selected source={selected['source']}"
        + (f", lambda={lam:g}" if lam is not None else "")
        + f", alpha={fmt_vec(selected['alpha'])}; "
        f"dev dCE de/en={selected['dev_de']['delta_ce']:+.3f}/"
        f"{selected['dev_en']['delta_ce']:+.3f}")
    return result


def evaluate(args, editors: list[ComponentEditor], data, alpha) -> dict:
    datasets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "french_europarl": data["fr_eval"],
        "spanish_europarl": data["es_eval"],
        "italian_europarl": data["it_eval"],
        "codeparrot": data["code_eval"],
    }

    # One worker per editor, each with its own dataset queue, so no editor is
    # ever driven from two threads at once (editor.alpha is shared state).
    def work(editor: ComponentEditor, items):
        rows = {}
        for name, idx in items:
            original = base_logits(editor, [idx])[0]
            rows[name] = forward_metrics(editor, idx, original, alpha)
        return rows

    queues = [[] for _ in editors]
    for i, item in enumerate(datasets.items()):
        queues[i % len(editors)].append(item)
    result = {}
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(work, editor, queue)
                   for editor, queue in zip(editors, queues) if queue]
        for future in futures:
            result.update(future.result())
    for name in datasets:
        row = result[name]
        log(f"eval {name}: CE {row['base_ce']:.3f} -> "
            f"{row['edited_ce']:.3f} ({row['delta_ce']:+.3f}), "
            f"KL={row['kl_from_base']:.4f}")
    return {name: result[name] for name in datasets}


def greedy_rollouts(editors: list[ComponentEditor], tokenizer, alpha,
                    max_new_tokens: int) -> dict:
    prompts = {
        "de": "Die Lösung liegt natürlich",
        "en": "And, of course, the solution does",
        "fr": "Cependant, le projet",
        "es": "Sin embargo, el proyecto",
        "it": "La ridestinazione delle risorse",
    }

    def generate(editor: ComponentEditor, prompt: str, edit_alpha) -> str:
        ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=editor.device)
        for _ in range(max_new_tokens):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                logits = editor.logits(ids[:, -512:], edit_alpha)
            token = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, token], dim=1)
        return tokenizer.decode(ids[0].tolist())

    jobs = [(language, variant)
            for language in prompts for variant in ("base", "edited")]
    queues = [[] for _ in editors]
    for i, job in enumerate(jobs):
        queues[i % len(editors)].append(job)

    def work(editor: ComponentEditor, queue):
        return {
            (language, variant): generate(
                editor, prompts[language],
                None if variant == "base" else alpha)
            for language, variant in queue}

    outputs = {}
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(work, editor, queue)
                   for editor, queue in zip(editors, queues) if queue]
        for future in futures:
            outputs.update(future.result())
    return {
        language: {
            "prompt": prompt,
            "base": outputs[(language, "base")],
            "edited": outputs[(language, "edited")],
        }
        for language, prompt in prompts.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1m_stream")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=None,
                        help="editor replica devices; default all visible "
                             "GPUs")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--rank_positions", type=int, default=256)
    parser.add_argument("--rank_temperature", type=float, default=0.05)
    parser.add_argument("--candidate_k", type=int, default=16)
    parser.add_argument("--kl_lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--init_grid", type=float, nargs="+",
                        default=[-20.0, -15.0, -12.0, -10.0, -8.0, -6.0,
                                 -4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0,
                                 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0,
                                 10.0, 12.0, 15.0, 20.0],
                        help="alpha grid for the warm-start objective scan "
                             "(per component; joint grid is coarser)")
    parser.add_argument("--select_grid", type=float, nargs="+",
                        default=[float(round(x, 2)) for x in
                                 list(np.arange(4.6, 6.61, 0.1))
                                 + list(np.arange(-13.0, -9.4, 0.5))],
                        help="extra dev-screen alphas near the constraint "
                             "boundary (k=1 only; k>1 uses local refinement)")
    parser.add_argument("--sweep_rows", type=int, default=32,
                        help="max batched rows per multi-alpha forward")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_dev_en_damage", type=float, default=0.1)
    parser.add_argument("--alpha_min", type=float, default=-50.0)
    parser.add_argument("--alpha_max", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--refresh_data", action="store_true")
    parser.add_argument("--rank_only", action="store_true",
                        help="prepare data, write ranking JSON, and stop")
    parser.add_argument("--component", type=int,
                        help="skip ranking and edit this single component")
    parser.add_argument("--components", type=int, nargs="+",
                        help="skip ranking and jointly edit these components")
    parser.add_argument("--screen_alpha", type=float,
                        help="evaluate this fixed alpha on dev German/English "
                             "and stop before training (applied to every "
                             "selected component)")
    parser.add_argument("--screen_alphas", type=float, nargs="+",
                        help="evaluate several fixed alphas without rebuilding "
                             "the selected components")
    args = parser.parse_args()
    if args.component is not None and args.components is not None:
        parser.error("pass --component or --components, not both")
    if args.component is not None:
        args.components = [args.component]
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    args.ranking_output = args.run_dir / "german_vpd_ranking.json"
    suffix = ("" if args.components is None or len(args.components) == 1
              else f"_k{len(args.components)}")
    args.output = args.run_dir / f"german_vpd_{args.banks_tag}{suffix}.json"
    args.adapter_output = (
        args.run_dir / f"german_vpd_{args.banks_tag}{suffix}_adapter.pt")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not args.bank_path.exists():
        raise FileNotFoundError(args.bank_path)
    if not (args.run_dir / "spec.pt").exists():
        raise FileNotFoundError(args.run_dir / "spec.pt")
    if not (args.run_dir / "stream_model.pt").exists():
        raise FileNotFoundError(args.run_dir / "stream_model.pt")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    bank_meta = {
        key: bank[key] for key in (
            "format", "C", "modules", "sensor", "gim_tau", "scalar",
            "soft_T", "soft_s") if key in bank
    }

    if args.components is None:
        ranking = rank_components(args, data, bank_meta)
        components = [ranking["selected_component"]]
    else:
        components = args.components
        ranking = {
            "method": "explicit --components override",
            "selected_components": components,
        }
    if args.rank_only:
        log(f"rank-only run complete: {args.ranking_output}")
        return

    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]

    # Ranking uses GIM's modified backward. Load a fresh, unpatched target for
    # the scalar fine-tune so alpha receives ordinary model gradients.
    target = geo1b.load_target_1b(devices[0])
    editor = ComponentEditor(target, bank, components, devices[0])
    del bank
    gc.collect()
    editors = [editor] + [editor.replicate(dev) for dev in devices[1:]]
    log(f"components {components} squared-weight mass fractions "
        + ", ".join(f"{fraction:.6e}"
                    for fraction in editor.mass_fraction)
        + f"; editors on {devices}")
    screen_alphas = (args.screen_alphas if args.screen_alphas is not None
                     else ([args.screen_alpha]
                           if args.screen_alpha is not None else None))
    if screen_alphas is not None:
        de_original = base_logits(editor, [data["de_dev"]])[0]
        en_original = base_logits(editor, [data["en_dev"]])[0]
        for alpha in screen_alphas:
            screen = {
                "components": components,
                "alpha": alpha,
                "weight_mass_fractions": editor.mass_fraction,
                "dev_de": forward_metrics(
                    editor, data["de_dev"], de_original, alpha),
                "dev_en": forward_metrics(
                    editor, data["en_dev"], en_original, alpha),
                "literal_weight_edit_verification": editor.verify_in_place(
                    data["de_dev"][:1, :32].to(devices[0]), alpha),
            }
            names = "-".join(str(c) for c in components)
            screen_path = args.run_dir / (
                f"german_vpd_screen_c{names}_a{alpha:g}.json")
            screen_path.write_text(json.dumps(screen, indent=2))
            log(f"screen alpha={alpha:+g} dCE de/en="
                f"{screen['dev_de']['delta_ce']:+.3f}/"
                f"{screen['dev_en']['delta_ce']:+.3f}; "
                f"wrote {screen_path}")
        return
    training = train_scalars(args, editors, data)
    alpha = [float(a) for a in training["selected"]["alpha"]]
    evaluation = evaluate(args, editors, data, alpha)
    verification = editor.verify_in_place(
        data["de_eval"][:1, :32].to(devices[0]), alpha)
    log("literal weight-edit equivalence: "
        f"max |delta logit|={verification['max_abs_logit_error']:.3e}")
    rollouts = greedy_rollouts(
        editors, tokenizer, alpha, args.max_new_tokens)

    result = {
        "format": "german_vpd_scalar_edit_result_v2",
        "protocol": {
            "kind": "few-component scalar weight-space fine-tune",
            "formula": "W' = W + sum_i (alpha_i - 1) * W_component_i",
            "vpd_reference": VPD_URL,
            "german_reference": PROTOCOL_URL,
            "model": geo1b.model_identity(),
            "bank": str(args.bank_path),
            "bank_metadata": bank_meta,
            "selection_note": (
                "Streaming cluster posterior substitutes for exhaustive exact "
                "all-component causal-importance ranking; candidates are then "
                "filtered for German-vs-other-language specificity."),
        },
        "components": components,
        "component_weight_mass_fractions": editor.mass_fraction,
        "alpha": alpha,
        "ranking": ranking,
        "training": training,
        "evaluation": evaluation,
        "literal_weight_edit_verification": verification,
        "rollouts": rollouts,
    }
    args.output.write_text(json.dumps(result, indent=2))
    torch.save({
        "format": "softpart_component_scalar_adapter_v2",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "components": components,
        "alpha": alpha,
        "formula": "W' = W + sum_i (alpha_i - 1) * W_component_i",
        "result": str(args.output),
    }, args.adapter_output)
    log(f"wrote result {args.output}")
    log(f"wrote replayable scalar adapter {args.adapter_output}")


if __name__ == "__main__":
    main()
