"""ROAD-67: exact-rotation decomposition trained with packed ablations.

ROAD (Rotation-Only Ablation Decomposition) starts from the exact thin SVD of
each Goodfire-scope matrix. Learned orthogonal butterfly rotations change the
rank-one shards while preserving their all-on sum exactly. The 18,432 local
shard slots are distributed exactly evenly across 4,096 global components;
rotations learn what functional directions occupy those fixed slots.

A small router reads frozen target residuals.  Training packs a top-k mask,
two stochastic supersets, and a hard-negative superset into each student pass.
Most updates reconstruct hidden states; occasional updates distill full output
distributions.  There is no Delta loss, learned residual, or multi-step PGD.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Parfact implementations live in the sibling parfact directory.
import sys as _parfact_sys
from pathlib import Path as _ParfactPath
for _parfact_dir in (_ParfactPath(__file__).resolve().parent / "parfact",
                     _ParfactPath(__file__).resolve().parent.parent / "parfact"):
    if _parfact_dir.is_dir():
        _parfact_sys.path.insert(0, str(_parfact_dir))
        break

import parfact67 as legacy
import parfact67_streaming as stream


FORMAT = "road67_exact_rotation_v1"


def log(message: str) -> None:
    print(f"[road67] {message}", flush=True)


def config_json(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class Butterfly(nn.Module):
    """A cheap exactly orthogonal transform over rank coordinates."""

    def __init__(self, rank: int, transforms: int, seed: int):
        super().__init__()
        if rank % 2:
            raise ValueError("butterfly rank must be even")
        self.rank = rank
        self.angles = nn.Parameter(torch.zeros(transforms, rank // 2))
        generator = torch.Generator().manual_seed(seed)
        permutations = [torch.randperm(rank, generator=generator)
                        for _ in range(transforms)]
        self.register_buffer("permutations", torch.stack(permutations),
                             persistent=False)
        self.register_buffer("inverses", torch.stack(
            [torch.argsort(value) for value in permutations]),
            persistent=False)

    @staticmethod
    def _one(x: torch.Tensor, angle: torch.Tensor,
             permutation: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        pair = x[..., permutation].reshape(*shape[:-1], shape[-1] // 2, 2)
        left, right = pair[..., 0], pair[..., 1]
        cosine, sine = angle.cos(), angle.sin()
        rotated = torch.stack((cosine * left - sine * right,
                               sine * left + cosine * right), -1)
        return rotated.reshape(shape)[..., inverse]

    def forward(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        indices = (range(self.angles.shape[0] - 1, -1, -1) if transpose
                   else range(self.angles.shape[0]))
        for index in indices:
            angle = -self.angles[index] if transpose else self.angles[index]
            x = self._one(x, angle, self.permutations[index],
                          self.inverses[index])
        return x


class RoutedLinear(nn.Module):
    """Exact SVD matrix whose rotated rank-one terms have global gates."""

    def __init__(self, left: torch.Tensor, value: torch.Tensor,
                 right: torch.Tensor, global_offset: int, components: int,
                 transforms: int, seed: int):
        super().__init__()
        root = value.float().clamp_min(0).sqrt()
        self.register_buffer("left_factor", left.float() * root)
        self.register_buffer("right_factor", right.float().T * root)
        rank = value.numel()
        self.rotation = Butterfly(rank, transforms, seed + global_offset)
        index = torch.arange(rank, dtype=torch.int64) + global_offset
        # The odd multiplier is a permutation modulo power-of-two C. Across
        # all 18,432 sequential shards this gives exactly 4 or 5 shards per
        # component. Rotations learn what occupies a slot; IDs cannot collapse.
        owner = (index * 104729 + seed) % components
        self.register_buffer("owner", owner, persistent=False)
        self.register_buffer("singular_value", value.float(), persistent=False)
        self.gates: torch.Tensor | None = None

    def hard_owner(self) -> torch.Tensor:
        return self.owner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gates is None:
            raise RuntimeError("ROAD gates were not set before forward")
        coefficient = self.rotation(x @ self.right_factor)
        local_gate = self.gates[..., self.owner]
        coefficient = self.rotation(coefficient * local_gate, transpose=True)
        return coefficient @ self.left_factor.T


class CheapRouter(nn.Module):
    def __init__(self, hidden: int, layers: int, width: int, components: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(hidden, width, bias=False) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, components)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        result = sum(F.gelu(layer(state)) for layer, state in
                     zip(self.projections, states, strict=True))
        return self.head(self.norm(result / math.sqrt(len(states))))


def replace_module(model: nn.Module, path: str, value: nn.Module) -> None:
    parent, name = path.rsplit(".", 1)
    setattr(model.get_submodule(parent), name, value)


class RoadSystem(nn.Module):
    def __init__(self, target: nn.Module, basis: stream.AtomBasis,
                 components: int, transforms: int, router_width: int,
                 seed: int):
        super().__init__()
        self.student = copy.deepcopy(target)
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        self.routed = nn.ModuleDict()
        for module_index, path in enumerate(basis.paths):
            routed = RoutedLinear(
                basis.left[path], basis.value[path], basis.right[path],
                basis.offsets[module_index], components, transforms,
                seed)
            replace_module(self.student, path, routed)
            self.routed[path.replace(".", "-")] = routed
        hidden = target.cfg["n_embd"]
        self.router = CheapRouter(hidden, target.cfg["n_layer"], router_width,
                                  components)
        self.components = components

    def set_gates(self, gates: torch.Tensor) -> None:
        for module in self.routed.values():
            module.gates = gates

    def forward_states(self, ids: torch.Tensor,
                       gates: torch.Tensor) -> list[torch.Tensor]:
        self.set_gates(gates)
        length = ids.shape[1]
        hidden = self.student.wte(ids)
        cos, sin = self.student.cos[:, :, :length], self.student.sin[:, :, :length]
        states = []
        for block in self.student.h:
            hidden = block(hidden, cos, sin)
            states.append(hidden)
        states.append(self.student.ln_f(hidden))
        return states

    def trainable_payload(self) -> dict:
        return {
            "router": self.router.state_dict(),
            "modules": {name: {
                "angles": module.rotation.angles.detach().cpu(),
            } for name, module in self.routed.items()},
        }

    def load_trainable_payload(self, payload: dict) -> None:
        self.router.load_state_dict(payload["router"])
        with torch.no_grad():
            for name, value in payload["modules"].items():
                self.routed[name].rotation.angles.copy_(value["angles"])

    def hard_owners(self) -> dict[str, torch.Tensor]:
        return {name.replace("-", "."): module.hard_owner().cpu()
                for name, module in self.routed.items()}


@torch.no_grad()
def target_states(model: nn.Module, ids: torch.Tensor,
                  need_logits: bool) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    length = ids.shape[1]
    hidden = model.wte(ids)
    cos, sin = model.cos[:, :, :length], model.sin[:, :, :length]
    states = []
    for block in model.h:
        hidden = block(hidden, cos, sin)
        states.append(hidden)
    final = model.ln_f(hidden)
    return states + [final], model.lm_head(final) if need_logits else None


def topk_mask(logits: torch.Tensor, keep: int, temperature: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = logits.topk(keep, dim=-1).values[..., -1:].detach()
    soft = torch.sigmoid((logits - threshold) / temperature)
    hard = logits >= threshold
    straight = hard.to(soft.dtype).detach() - soft.detach() + soft
    return straight, hard


def packed_masks(base: torch.Tensor, hard: torch.Tensor,
                 adversarial: torch.Tensor | None,
                 generator: torch.Generator, count: int) -> list[torch.Tensor]:
    random = torch.rand(base.shape, device=base.device, generator=generator)
    first_extra = (~hard) & (random < .25)
    second_extra = (~hard) & (random >= .25) & (random < .75)
    first = base + (1 - base) * first_extra
    second = base + (1 - base) * second_extra
    if adversarial is None:
        adversarial = base + (1 - base) * ((~hard) & (random >= .5))
    candidates = ([base, adversarial] if count == 2 else
                  [base, first, adversarial] if count == 3 else
                  [base, first, second, adversarial])
    return candidates


def normalized_state_loss(actual: list[torch.Tensor],
                          target: list[torch.Tensor], masks: int,
                          batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    per_mask = torch.zeros(masks, device=actual[0].device)
    for observed, expected in zip(actual, target, strict=True):
        observed = observed.float().view(masks, batch, *observed.shape[1:])
        scale = expected.float().square().mean().clamp_min(1e-6)
        per_mask += (observed - expected.float()[None]).square().mean(
            tuple(range(1, observed.ndim))) / scale
    per_mask /= len(actual)
    return per_mask.mean() + per_mask.max(), per_mask.detach()


def output_kl(student_logits: torch.Tensor, target_logits: torch.Tensor,
              masks: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (masks, batch) + student_logits.shape[1:]
    student_logp = F.log_softmax(student_logits.float().view(shape), -1)
    target_logp = F.log_softmax(target_logits.float(), -1)
    per_token = F.kl_div(student_logp, target_logp[None], log_target=True,
                         reduction="none").sum(-1)
    per_mask = per_token.mean(tuple(range(1, per_token.ndim)))
    return per_mask.mean() + per_mask.max(), per_mask.detach()


def rotation_mass_balance(system: RoadSystem) -> torch.Tensor:
    """Balance rotated rank-one mass across fixed global component slots."""
    device = next(system.parameters()).device
    mass = torch.zeros(system.components, device=device)
    for module in system.routed.values():
        rank = module.singular_value.numel()
        rotation = module.rotation(torch.eye(rank, device=device))
        diagonal_energy = (module.singular_value[:, None]
                           * rotation.square()).sum(0)
        mass.scatter_add_(0, module.owner, diagonal_energy.square())
    ratio = mass / mass.mean().clamp_min(1e-12)
    return (ratio - 1).square().mean() + 10 * F.relu(ratio - 2).square().mean()


def component_usage(base: torch.Tensor, keep: int) -> torch.Tensor:
    usage = base.mean((0, 1))
    expected = keep / base.shape[-1]
    return ((usage - expected) / max(expected, 1e-6)).square().mean()


def make_adversarial(system: RoadSystem, ids: torch.Tensor,
                     target: list[torch.Tensor], base: torch.Tensor,
                     hard: torch.Tensor, add: int) -> torch.Tensor:
    probe = hard.float().detach().requires_grad_(True)
    actual = system.forward_states(ids, probe)
    loss, _ = normalized_state_loss(actual, target, 1, ids.shape[0])
    gradient = torch.autograd.grad(loss, probe)[0].detach()
    score = gradient.masked_fill(hard, -torch.inf)
    chosen = score.topk(min(add, score.shape[-1]), dim=-1).indices
    extra = torch.zeros_like(hard).scatter_(-1, chosen, True)
    return base + (1 - base) * extra


def token_bank(path: Path, count: int) -> np.ndarray:
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    if count > len(raw):
        raise ValueError("token file is shorter than --target-tokens")
    result = np.asarray(raw[:count], dtype=np.uint16).copy()
    del raw
    return result


def save_checkpoint(args, system: RoadSystem, optimizer, scheduler, step: int,
                    tokens: int, history: list[dict]) -> None:
    legacy.atomic_torch_save({
        "format": FORMAT, "config": config_json(args), "step": step,
        "tokens": tokens, "history": history,
        "trainable": system.trainable_payload(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }, args.run / "checkpoint.pt")


def run(args) -> None:
    args.run.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    target = legacy.load_model(args.checkpoint, args.device, "grad").eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    basis = stream.load_basis(args.basis, args.device)
    system = RoadSystem(target, basis, args.components, args.transforms,
                        args.router_width, args.seed).to(args.device)
    trainable = [parameter for parameter in system.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = math.ceil(args.target_tokens / (args.batch_sequences * args.context))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=args.lr * .1)
    start_step = processed = 0
    history: list[dict] = []
    checkpoint = args.run / "checkpoint.pt"
    if checkpoint.exists():
        raw = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if raw.get("format") != FORMAT or raw.get("config") != config_json(args):
            raise ValueError("incompatible ROAD checkpoint")
        system.load_trainable_payload(raw["trainable"])
        optimizer.load_state_dict(raw["optimizer"])
        scheduler.load_state_dict(raw["scheduler"])
        start_step, processed, history = raw["step"], raw["tokens"], raw["history"]
        log(f"resumed step {start_step:,}/{total_steps:,}, tokens={processed:,}")

    data = token_bank(args.tokens, args.target_tokens)
    log(f"loaded {data.size:,} training tokens into RAM once")
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    generator = torch.Generator(device=args.device).manual_seed(
        args.seed + start_step)
    started = time.perf_counter()
    start_tokens = processed
    system.train()
    for step in range(start_step, total_steps):
        first = step * args.batch_sequences * args.context
        last = min(first + args.batch_sequences * args.context, args.target_tokens)
        raw_ids = data[first:last]
        if raw_ids.size < args.context:
            break
        sequences = raw_ids.size // args.context
        ids = torch.from_numpy(raw_ids[:sequences * args.context].astype(
            np.int64)).view(sequences, args.context).to(args.device)
        need_output = step % args.output_every == 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_states, teacher_logits = target_states(target, ids, need_output)
        router_logits = system.router(teacher_states[:-1])
        progress = step / max(total_steps - 1, 1)
        keep = round(args.keep_start * (1 - progress) + args.keep_end * progress)
        base, hard = topk_mask(router_logits, keep, args.temperature)
        adversarial = None
        if step % args.adversary_every == 0:
            adversarial = make_adversarial(
                system, ids, teacher_states, base, hard, args.adversary_add)
        masks = packed_masks(base, hard, adversarial, generator,
                             args.masks_per_step)
        packed = torch.cat(masks)
        packed_ids = ids.repeat(len(masks), 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            observed = system.forward_states(packed_ids, packed)
            state_loss, state_each = normalized_state_loss(
                observed, teacher_states, len(masks), sequences)
            loss = state_loss
            kl_each = torch.zeros(len(masks), device=args.device)
            if need_output:
                student_logits = system.student.lm_head(observed[-1])
                kl_loss, kl_each = output_kl(
                    student_logits, teacher_logits, len(masks), sequences)
                loss = loss + args.output_weight * kl_loss
            usage_loss = component_usage(base, keep)
            capacity_loss = rotation_mass_balance(system)
            loss = (loss + args.usage_weight * usage_loss
                    + args.capacity_weight * capacity_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()
        processed = last

        if (step + 1) % args.log_steps == 0 or step + 1 == total_steps:
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = (processed - start_tokens) / elapsed
            row = {"step": step + 1, "tokens": processed, "keep": keep,
                   "rate": rate, "loss": float(loss.detach()),
                   "state_masks": state_each.cpu().tolist(),
                   "kl_masks": kl_each.cpu().tolist(),
                   "usage": float(usage_loss.detach()),
                   "capacity": float(capacity_loss.detach())}
            history.append(row)
            log(f"step {step+1:,}/{total_steps:,} tokens={processed:,} "
                f"keep={keep} rate={rate:,.0f}/s loss={row['loss']:.4f} "
                f"state={row['state_masks'][0]:.4f} kl={row['kl_masks'][0]:.4f}")
        if ((step + 1) % args.checkpoint_steps == 0 or stop_requested
                or step + 1 == total_steps):
            save_checkpoint(args, system, optimizer, scheduler, step + 1,
                            processed, history)
        if stop_requested:
            log("checkpointed after scheduler signal")
            return

    artifact = {
        "format": FORMAT, "config": config_json(args),
        "components": args.components, "local_rank_one_shards": basis.count,
        "scope": list(basis.paths), "basis": str(args.basis),
        "checkpoint": str(args.checkpoint), "tokens": processed,
        "trainable": system.trainable_payload(),
        "hard_owners": system.hard_owners(),
        "exact_all_on_reconstruction": True, "background_component": False,
        "random_ablation_training": True,
    }
    legacy.atomic_torch_save(artifact, args.run / "road.pt")
    atomic_json({
        "format": FORMAT, "complete": True, "tokens": processed,
        "steps": total_steps, "components": args.components,
        "local_rank_one_shards": basis.count,
        "parameter_faithfulness": "exact by orthogonal rotation",
        "background_component": False,
        "masks_per_step": args.masks_per_step, "multistep_pgd": False,
        "config": config_json(args), "history": history,
    }, args.run / "manifest.json")
    log("ROAD training complete")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--tokens", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--basis", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, default=100_000_000)
    result.add_argument("--context", type=int, default=512)
    result.add_argument("--batch-sequences", type=int, default=4)
    result.add_argument("--components", type=int, default=4096)
    result.add_argument("--transforms", type=int, default=8)
    result.add_argument("--router-width", type=int, default=128)
    result.add_argument("--keep-start", type=int, default=2048)
    result.add_argument("--keep-end", type=int, default=512)
    result.add_argument("--temperature", type=float, default=0.5)
    result.add_argument("--adversary-every", type=int, default=8)
    result.add_argument("--adversary-add", type=int, default=512)
    result.add_argument("--masks-per-step", type=int, choices=(2, 3, 4),
                        default=2)
    result.add_argument("--output-every", type=int, default=8)
    result.add_argument("--output-weight", type=float, default=0.2)
    result.add_argument("--usage-weight", type=float, default=0.01)
    result.add_argument("--capacity-weight", type=float, default=0.05)
    result.add_argument("--lr", type=float, default=2e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--checkpoint-steps", type=int, default=200)
    result.add_argument("--log-steps", type=int, default=20)
    result.add_argument("--seed", type=int, default=271828)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
