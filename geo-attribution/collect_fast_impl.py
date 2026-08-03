"""Optimized 1B random-feature collection and matched collection benchmark.

This is the collection front-end for ``geo1m.py``.  It writes the same feature,
metadata, and exact-subset files, so the existing cluster/extract stages remain
unchanged.  A spec can be reused with ``--spec_tag`` and a completed compatible
collection can be reused with ``--reuse_fingerprints``.

Examples
--------
Optimized collection::

  torchrun --nproc_per_node=2 collect_fast.py collect --tag run_gim \
    --spec_tag full1m --n_positions 1048576

Matched quick benchmark::

  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile baseline \
    --spec_tag full1m --pos_per_seq 64
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
    --spec_tag full1m --pos_per_seq 64
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import numpy as np
import torch

import geo1b  # noqa: F401 — installs 1B target/data and full GIM in geo67
import geo67
from collection_runtime import (compile_model, file_sha256, model_pass,
                                stable_fingerprint)
from geo1m import features_from, load_spec

SHM = Path("/dev/shm/geo1b")


def equal_reward(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Give every observed next token one unit of output reward.

    The output cotangent is 1 at each ground-truth next-token logit and 0 at
    every other logit.  Unlike CE, confident and surprising tokens therefore
    receive equal upstream reward.  This matches GIM's answer-direction setup.
    """
    return logits[:, :-1].float().gather(-1, idx[:, 1:, None]).sum()


def objective(logits: torch.Tensor, idx: torch.Tensor, scalar: str):
    if scalar == "equal_reward":
        return equal_reward(logits, idx)
    return geo67.scalar_sum(logits, idx, scalar)


def apply_profile(args):
    """Resolve named benchmark profiles without hiding individual switches."""
    if args.profile == "baseline":
        args.sensor = "ig"
        args.ig_k = 2
        args.scalar = "ce"
        args.bf16 = False
        args.compile = False
        args.fused_attention = False
    elif args.profile == "optimized":
        args.sensor = "gim"
        args.ig_k = 1
        args.scalar = "equal_reward"
        args.bf16 = True
        args.compile = True
        args.fused_attention = True
    if args.sensor == "gim" and args.ig_k != 1:
        raise ValueError("GIM replaces IG and therefore requires --ig_k 1")
    if args.fused_attention and not args.bf16:
        raise ValueError("Flash SDPA collection requires --bf16")


def setup_model(args, device):
    target = geo67.load_target(device)
    floating = {p.dtype for p in target.parameters() if p.is_floating_point()}
    if floating != {torch.float32}:
        raise RuntimeError(f"expected fp32 master parameters, found {floating}")
    if args.sensor == "gim":
        geo67.apply_gim(target, args.gim_tau)
    cap = geo67.Capture(target)
    cap.target = compile_model(cap.target, args.compile, args.compile_mode)
    return cap


def sampled_batch(loader, gen, device, pos_per_seq):
    idx = next(loader).to(device)
    batch, length = idx.shape
    if pos_per_seq > length - 6:
        raise ValueError(f"pos_per_seq={pos_per_seq} exceeds {length - 6}")
    pos = torch.stack([
        torch.randperm(length - 6, generator=gen)[:pos_per_seq] + 4
        for _ in range(batch)
    ]).to(device)
    bi = torch.arange(batch, device=device)[:, None].expand(-1, pos_per_seq)
    return idx, pos, bi


def pass_features(args, cap, idx, pos, bi, spec, scales, dim, keep_subset=0):
    batch, samples = pos.shape
    phi = torch.zeros(batch * samples, dim, device=idx.device)
    subset = ({p: {"p": [], "g": []} for p in geo67.MODULES}
              if keep_subset else None)
    for k in range(1, args.ig_k + 1):
        cap.wscale = k / args.ig_k
        cap.target.zero_grad(set_to_none=True)
        with model_pass(idx.device, args.bf16, args.fused_attention):
            logits, cache = cap.run(idx)
            reward = objective(logits, idx, args.scalar)
            posts = [cache[p]["post"] for p in geo67.MODULES]
            gposts = torch.autograd.grad(reward, posts)
        gs, ps = {}, {}
        for path, grad in zip(geo67.MODULES, gposts, strict=True):
            pre = cache[path]["pre"].detach()
            ps[path] = pre[bi, pos].reshape(batch * samples, -1)
            gs[path] = grad[bi, pos].reshape(batch * samples, -1)
        phi += features_from(spec, scales, dim, gs, ps, idx.device) / args.ig_k
        if subset is not None:
            for path in geo67.MODULES:
                subset[path]["p"].append(
                    ps[path].reshape(batch, samples, -1)[:, :keep_subset]
                    .reshape(batch * keep_subset, -1).bfloat16().cpu())
                subset[path]["g"].append(
                    gs[path].reshape(batch, samples, -1)[:, :keep_subset]
                    .reshape(batch * keep_subset, -1).bfloat16().cpu())
    return phi, subset


def config_payload(args, spec_path, world):
    return {
        "format": 2,
        "model": geo1b.MODEL_ID,
        "spec_sha256": file_sha256(spec_path),
        "feat_dim": args.feat_dim,
        "sensor": args.sensor,
        "gim_tau": args.gim_tau if args.sensor == "gim" else None,
        "ig_k": args.ig_k,
        "scalar": args.scalar,
        "master_dtype": "float32",
        "pass_dtype": "bfloat16" if args.bf16 else "float32",
        "compiled": args.compile,
        "fused_attention": args.fused_attention,
        "n_positions": args.n_positions,
        "pos_per_seq": args.pos_per_seq,
        "sub_per_seq": args.sub_per_seq,
        "seq_len": args.seq_len,
        "batch_seqs": args.batch_seqs,
        "seed": args.seed,
        "world": world,
    }


def reusable_rank(args, rank, payload, n_local, dim):
    manifest = args.dir / f"fingerprint_rank{rank}.json"
    feat = args.dir / f"feat_rank{rank}.f16"
    meta = args.dir / f"meta_rank{rank}.pt"
    subset = args.dir / f"collect_rank{rank}.pt"
    if not manifest.exists() or not feat.exists() or not meta.exists():
        return False
    old = json.loads(manifest.read_text())
    if old.get("id") != stable_fingerprint(payload):
        return False
    if feat.stat().st_size != n_local * dim * np.dtype(np.float16).itemsize:
        return False
    return args.sub_per_seq == 0 or subset.exists()


def stage_collect(args):
    ddp, rank, world, device = geo67.ddp_setup()
    args.dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.spec_dir / "spec.pt"
    if not spec_path.exists():
        raise FileNotFoundError(f"missing reusable feature spec: {spec_path}")
    local_spec = args.dir / "spec.pt"
    if local_spec.resolve() != spec_path.resolve() and rank == 0:
        shutil.copyfile(spec_path, local_spec)
    spec, scales, dim = load_spec(args.spec_dir, device)
    if args.feat_dim is None:
        args.feat_dim = dim
    elif args.feat_dim != dim:
        raise ValueError(f"--feat_dim {args.feat_dim} != spec D={dim}")
    n_local = args.n_positions // world
    payload = config_payload(args, spec_path, world)
    fingerprint_id = stable_fingerprint(payload)
    if args.reuse_fingerprints and reusable_rank(args, rank, payload, n_local, dim):
        geo67.log(f"reusing compatible fingerprints {fingerprint_id} on rank {rank}")
        if ddp:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    cap = setup_model(args, device)
    import nano_param_decomp.pile_4L as p4l
    loader = p4l.make_loader(args.batch_seqs * world, args.seq_len, rank, world,
                             "train", args.seed)
    gen = torch.Generator().manual_seed(args.seed + 7 * rank)
    rows_per_batch = args.batch_seqs * args.pos_per_seq
    n_batches = math.ceil(n_local / rows_per_batch)
    feat = np.memmap(args.dir / f"feat_rank{rank}.f16", dtype=np.float16,
                     mode="w+", shape=(n_local, dim))
    keep = args.sub_per_seq
    sub_store = ({p: {"p": [[] for _ in range(args.ig_k)],
                      "g": [[] for _ in range(args.ig_k)]}
                  for p in geo67.MODULES} if keep else None)
    sub_mask = torch.zeros(n_local, dtype=torch.bool)
    toks, nexts, sub_toks, sub_nexts = [], [], [], []
    written = 0
    t0 = time.perf_counter()
    for batch_idx in range(n_batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        phi, subset = pass_features(args, cap, idx, pos, bi, spec, scales, dim,
                                    keep_subset=keep)
        take = min(phi.shape[0], n_local - written)
        feat[written:written + take] = phi[:take].clamp(-6e4, 6e4) \
            .float().cpu().numpy().astype(np.float16)
        tok = idx[bi, pos].reshape(-1).cpu()[:take]
        nxt = idx[bi, (pos + 1).clamp(max=idx.shape[1] - 1)].reshape(-1).cpu()[:take]
        toks.append(tok)
        nexts.append(nxt)
        if keep:
            mask = torch.zeros(phi.shape[0], dtype=torch.bool)
            mask.reshape(idx.shape[0], args.pos_per_seq)[:, :keep] = True
            sub_mask[written:written + take] = mask[:take]
            sub_toks.append(tok[mask[:take]])
            sub_nexts.append(nxt[mask[:take]])
            for path in geo67.MODULES:
                for k in range(args.ig_k):
                    sub_store[path]["p"][k].append(subset[path]["p"][k])
                    sub_store[path]["g"][k].append(subset[path]["g"][k])
        written += take
        if batch_idx % 16 == 0:
            geo67.log(f"collect-fast batch {batch_idx}/{n_batches} "
                      f"({time.perf_counter() - t0:.0f}s)")
    feat.flush()

    if keep:
        subset_n = int(sub_mask.sum())
        out = {"modules": geo67.MODULES, "ig_k": args.ig_k,
               "sensor": args.sensor, "gim_tau": args.gim_tau,
               "scalar": args.scalar, "tok": torch.cat(sub_toks)[:subset_n],
               "next": torch.cat(sub_nexts)[:subset_n], "n": subset_n}
        for path in geo67.MODULES:
            out[path] = {
                "p": torch.stack([torch.cat(sub_store[path]["p"][k])[:subset_n]
                                  for k in range(args.ig_k)]),
                "g": torch.stack([torch.cat(sub_store[path]["g"][k])[:subset_n]
                                  for k in range(args.ig_k)]),
            }
        torch.save(out, args.dir / f"collect_rank{rank}.pt")
    torch.save({"sub_mask": sub_mask, "n_local": n_local,
                "tok": torch.cat(toks), "next": torch.cat(nexts),
                "fingerprint_id": fingerprint_id},
               args.dir / f"meta_rank{rank}.pt")
    rank_manifest = {"id": fingerprint_id, "rank": rank, "config": payload,
                     "seconds": time.perf_counter() - t0,
                     "positions": n_local}
    (args.dir / f"fingerprint_rank{rank}.json").write_text(
        json.dumps(rank_manifest, indent=2, sort_keys=True))
    geo67.log(f"collect-fast done: {n_local} fingerprints, id={fingerprint_id}, "
              f"{rank_manifest['seconds']:.1f}s")
    if ddp:
        torch.distributed.barrier()
    if rank == 0:
        combined = {"id": fingerprint_id, "config": payload,
                    "ranks": world, "reusable_across_C": True}
        (args.dir / "fingerprints.json").write_text(
            json.dumps(combined, indent=2, sort_keys=True))
    if ddp:
        torch.distributed.destroy_process_group()


def stage_benchmark(args):
    ddp, rank, world, device = geo67.ddp_setup()
    spec, scales, dim = load_spec(args.spec_dir, device)
    cap = setup_model(args, device)
    import nano_param_decomp.pile_4L as p4l
    loader = p4l.make_loader(args.batch_seqs * world, args.seq_len, rank, world,
                             "train", args.seed)
    gen = torch.Generator().manual_seed(args.seed + 7 * rank)
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    warm_start = time.perf_counter()
    for _ in range(args.warmup_batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        pass_features(args, cap, idx, pos, bi, spec, scales, dim)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.benchmark_batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        pass_features(args, cap, idx, pos, bi, spec, scales, dim)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    timing = torch.tensor([seconds, warm_seconds], device=device)
    if ddp:
        torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
    positions = args.benchmark_batches * args.batch_seqs * args.pos_per_seq * world
    result = {
        "profile": args.profile or "custom",
        "sensor": args.sensor,
        "ig_k": args.ig_k,
        "scalar": args.scalar,
        "bf16": args.bf16,
        "compile": args.compile,
        "fused_attention": args.fused_attention,
        "world": world,
        "batch_seqs_per_gpu": args.batch_seqs,
        "seq_len": args.seq_len,
        "pos_per_seq": args.pos_per_seq,
        "warmup_batches": args.warmup_batches,
        "timed_batches": args.benchmark_batches,
        "warmup_seconds_max_rank": timing[1].item(),
        "seconds_max_rank": timing[0].item(),
        "positions": positions,
        "positions_per_second": positions / timing[0].item(),
        "sequence_tokens_per_second": (args.benchmark_batches * args.batch_seqs
                                       * args.seq_len * world / timing[0].item()),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "benchmark"])
    ap.add_argument("--tag", default="run_gim")
    ap.add_argument("--spec_tag", default="full1m")
    ap.add_argument("--profile", choices=["baseline", "optimized"])
    ap.add_argument("--sensor", choices=["ig", "gim"], default="gim")
    ap.add_argument("--ig_k", type=int, default=1)
    ap.add_argument("--gim_tau", type=float, default=2.0)
    ap.add_argument("--scalar", choices=["ce", "logp_pred", "logit_pred",
                                         "equal_reward"], default="equal_reward")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile_mode", default="reduce-overhead")
    ap.add_argument("--fused_attention", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--reuse_fingerprints", action="store_true")
    ap.add_argument("--feat_dim", type=int)
    ap.add_argument("--n_positions", type=int, default=1048576)
    ap.add_argument("--pos_per_seq", type=int, default=64)
    ap.add_argument("--sub_per_seq", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup_batches", type=int, default=2)
    ap.add_argument("--benchmark_batches", type=int, default=8)
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/circuit-decomp/geo-attribution/out/benchmark.json"))
    args = ap.parse_args()
    apply_profile(args)
    args.dir = SHM / args.tag
    args.spec_dir = SHM / args.spec_tag
    torch.manual_seed(args.seed)
    {"collect": stage_collect, "benchmark": stage_benchmark}[args.stage](args)


if __name__ == "__main__":
    main()
