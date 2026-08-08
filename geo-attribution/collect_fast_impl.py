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

  python collect_fast.py spec --tag benchmark_spec --feat_dim 16384
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile baseline \
    --spec_tag benchmark_spec --synthetic_data --pos_per_seq 64 \
    --benchmark_batches 32 --output out/benchmark_baseline_b200.json
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
    --spec_tag benchmark_spec --synthetic_data --pos_per_seq 64 \
    --benchmark_batches 32 --output out/benchmark_optimized_b200.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401 — installs 1B target/data and full GIM in geo67
import geo67
from collection_runtime import (file_sha256, linear_kernel, model_pass,
                                stable_fingerprint)
from geo1m import features_from, load_spec
from streaming_decomposition import (ModuleReservoirWriter, assign_features,
                                     fit_stream_model, load_stream_model,
                                     local_reservoir_quota,
                                     reservoir_updates)

HERE = Path(__file__).resolve().parent
COLLECTOR_FORMAT = 3


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
    if getattr(args, "master_dtype", "float32") == "bfloat16":
        # Storage-level bf16 halves weight-read bandwidth and removes the
        # per-pass autocast cast. Sensor fidelity must be confirmed with the
        # ab_master stage before production use.
        target.to(torch.bfloat16)
    if args.sensor == "gim":
        geo67.apply_gim(target, args.gim_tau)
    # Compile only the pure linear kernel. Cache mutation stays eager so every
    # captured pre/post tensor remains connected to the ordinary autograd graph.
    cap = geo67.Capture(
        target, linear_kernel=linear_kernel(args.compile, args.compile_mode))
    return cap


def make_loader(args, cap, rank, world):
    vocab_size = cap.target.hf.config.vocab_size
    return geo1b.make_loader_1b(
        args.batch_seqs * world, args.seq_len, rank, world, "train", args.seed,
        data_path=args.data_path, synthetic=args.synthetic_data,
        vocab_size=vocab_size, order=args.data_order)


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


def features_direct(spec, scales, pres, grads, rows):
    """Gather ONLY the spec'd (o, i) coordinates straight from the live
    (batch, seq, d) pass tensors.

    The reference path first materialized every sampled position's full
    ~700k-value (p, g) row for all 112 matrices (multi-GiB of gather traffic
    per batch at high pos_per_seq) and then immediately reduced it to D
    sampled coordinates. Composing the two gathers reads and writes only
    rows x D values. The elementwise product order matches the reference
    exactly (bf16 g*p, then fp32 scale), so phi is bit-identical.
    """
    outs = []
    rows_col = rows[:, None]
    for path, (o_idx, i_idx) in spec.items():
        if o_idx.numel() == 0:
            continue
        g2 = grads[path].flatten(0, 1)
        p2 = pres[path].flatten(0, 1)
        outs.append(g2[rows_col, o_idx[None, :]]
                    * p2[rows_col, i_idx[None, :]]
                    * scales[path][None, :])
    return torch.cat(outs, dim=1)


class DetailGatherer:
    """Lazy row-level access to complete (p, g) vectors of one batch.

    Holds references to the pass's cached activations and output gradients so
    the caller can first decide WHICH rows it needs (reservoir acceptance is
    a handful of rows per batch once streams mature) and only then pay for
    gathering their full per-module vectors.
    """

    def __init__(self, passes, rows):
        self.passes = passes  # per IG step: (pres, grads) module dicts
        self.rows = rows

    def gather(self, sources: torch.Tensor) -> dict:
        rows = self.rows[sources.to(self.rows.device)]
        out = {}
        for path in geo67.MODULES:
            ps, gs = [], []
            for pres, grads in self.passes:
                ps.append(pres[path].flatten(0, 1)[rows])
                gs.append(grads[path].flatten(0, 1)[rows])
            out[path] = {"p": torch.stack(ps), "g": torch.stack(gs)}
        return out


def pass_features(args, cap, idx, pos, bi, spec, scales, dim, keep_subset=0,
                  return_pg=False):
    batch, samples = pos.shape
    seq = idx.shape[1]
    rows = (bi * seq + pos).reshape(-1)
    phi = torch.zeros(batch * samples, dim, device=idx.device)
    subset = ({p: {"p": [], "g": []} for p in geo67.MODULES}
              if keep_subset else None)
    keep_rows = (rows.view(batch, samples)[:, :keep_subset].reshape(-1)
                 if keep_subset else None)
    passes = []
    for k in range(1, args.ig_k + 1):
        cap.wscale = k / args.ig_k
        cap.target.zero_grad(set_to_none=True)
        with model_pass(idx.device, args.bf16, args.fused_attention):
            logits, cache = cap.run(idx)
            reward = objective(logits, idx, args.scalar)
            posts = [cache[p]["post"] for p in geo67.MODULES]
            gposts = torch.autograd.grad(reward, posts)
        pres = {path: cache[path]["pre"].detach() for path in geo67.MODULES}
        grads = {path: grad.detach()
                 for path, grad in zip(geo67.MODULES, gposts, strict=True)}
        phi += features_direct(spec, scales, pres, grads, rows) / args.ig_k
        if subset is not None:
            for path in geo67.MODULES:
                subset[path]["p"].append(
                    pres[path].flatten(0, 1)[keep_rows].bfloat16().cpu())
                subset[path]["g"].append(
                    grads[path].flatten(0, 1)[keep_rows].bfloat16().cpu())
        if return_pg:
            passes.append((pres, grads))
    if return_pg:
        return phi, subset, DetailGatherer(passes, rows)
    return phi, subset


def config_payload(args, spec_path, world):
    import transformers
    if args.synthetic_data:
        data = {"kind": "synthetic_uniform_v1"}
    else:
        if not args.data_path.exists():
            raise FileNotFoundError(f"missing token stream: {args.data_path}")
        data = {"kind": "uint32_token_stream",
                "bytes": args.data_path.stat().st_size,
                "sha256": file_sha256(args.data_path)}
    return {
        "format": COLLECTOR_FORMAT,
        "model": geo1b.model_identity(),
        "data": data,
        "spec_sha256": file_sha256(spec_path),
        "feat_dim": args.feat_dim,
        "sensor": args.sensor,
        "gim_tau": args.gim_tau if args.sensor == "gim" else None,
        "ig_k": args.ig_k,
        "scalar": args.scalar,
        "master_dtype": args.master_dtype,
        "pass_dtype": "bfloat16" if args.bf16 else "float32",
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "fused_attention": args.fused_attention,
        "runtime": {"torch": torch.__version__,
                    "transformers": transformers.__version__},
        "n_positions": args.n_positions,
        "pos_per_seq": args.pos_per_seq,
        "sub_per_seq": args.sub_per_seq,
        "data_order": args.data_order,
        "seq_len": args.seq_len,
        "batch_seqs": args.batch_seqs,
        "seed": args.seed,
        "world": world,
    }


def stage_spec(args):
    """Create a deterministic, unbiased W^2 feature proposal.

    The production spec can still be supplied with ``--spec_tag``. This stage
    provides a self-contained benchmark/recovery path when the old tmpfs spec
    is unavailable. Weight-only importance sampling has higher variance than
    the activation-statistics proposal in ``geo1m.py`` but estimates the same
    attribution kernel and performs the same amount of feature projection work.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run the spec stage with python, not torchrun")
    dim = args.feat_dim or 16384
    identity = geo1b.model_identity()
    spec_path = args.dir / "spec.pt"
    if args.reuse_spec and spec_path.exists():
        saved = torch.load(spec_path, weights_only=True, map_location="cpu")
        if (saved.get("D") == dim
                and saved.get("proposal") == "weight_squared"
                and saved.get("model") == identity
                and saved.get("seed") == args.seed):
            geo67.log(f"reusing compatible feature spec {spec_path}")
            return

    device = "cuda"
    target = geo67.load_target(device)
    masses = {}
    for path in geo67.MODULES:
        weight = target.get_submodule(path).weight.detach().float()
        masses[path] = weight.square().sum().item()
    total = sum(masses.values())
    quotas = {path: dim * masses[path] / total for path in geo67.MODULES}
    alloc = {path: math.floor(quotas[path]) for path in geo67.MODULES}
    remainder = dim - sum(alloc.values())
    for path in sorted(geo67.MODULES,
                       key=lambda p: quotas[p] - alloc[p],
                       reverse=True)[:remainder]:
        alloc[path] += 1

    generator = torch.Generator(device=device).manual_seed(args.seed)
    spec, scales = {}, {}
    for path in geo67.MODULES:
        count = alloc[path]
        weight = target.get_submodule(path).weight.detach().float()
        if count == 0:
            spec[path] = (torch.zeros(0, dtype=torch.int32),
                          torch.zeros(0, dtype=torch.int32))
            scales[path] = torch.zeros(0)
            continue
        flat = weight.square().flatten()
        coord = torch.multinomial(flat, count, replacement=True,
                                  generator=generator)
        rows = torch.div(coord, weight.shape[1], rounding_mode="floor")
        cols = coord.remainder(weight.shape[1])
        # q(o,i)=W_oi^2/sum(W^2), so |W|/sqrt(R*q)=sqrt(sum(W^2)/R).
        scales[path] = torch.full(
            (count,), math.sqrt(masses[path] / count), dtype=torch.float32)
        spec[path] = (rows.int().cpu(), cols.int().cpu())

    args.dir.mkdir(parents=True, exist_ok=True)
    payload = {"spec": spec, "scales": scales, "D": dim, "alloc": alloc,
               "seed": args.seed, "proposal": "weight_squared",
               "model": identity, "format": 1}
    torch.save(payload, spec_path)
    manifest = {"id": stable_fingerprint({
                    "D": dim, "alloc": alloc, "seed": args.seed,
                    "proposal": "weight_squared", "model": identity}),
                "D": dim, "proposal": "weight_squared", "model": identity,
                "path": str(spec_path)}
    (args.dir / "spec.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    geo67.log(f"created W^2 feature spec D={dim}: {spec_path}")


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
    if args.n_positions % world:
        raise ValueError(f"n_positions={args.n_positions} must divide world={world}")
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
    loader = make_loader(args, cap, rank, world)
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

    # Feature rows leave the GPU and hit the memmap from a side thread so the
    # device-to-host copy and page-cache write overlap the next model pass.
    from concurrent.futures import ThreadPoolExecutor
    writer_pool = ThreadPoolExecutor(1)
    pending: list = []

    def write_rows(offset: int, take: int, phi16: torch.Tensor):
        feat[offset:offset + take] = phi16.cpu().numpy()

    for batch_idx in range(n_batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        phi, subset = pass_features(args, cap, idx, pos, bi, spec, scales, dim,
                                    keep_subset=keep)
        take = min(phi.shape[0], n_local - written)
        while len(pending) >= 2:
            pending.pop(0).result()
        pending.append(writer_pool.submit(
            write_rows, written, take, phi[:take].clamp(-6e4, 6e4).half()))
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
    for future in pending:
        future.result()
    writer_pool.shutdown()
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


def _atomic_torch_save(payload: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def stage_fit_stream(args):
    """Learn the frozen streaming projection/centroids from a bounded pilot."""
    if "RANK" in os.environ:
        raise RuntimeError("run fit_stream with python, not torchrun")
    pilot_dir = args.artifact_root / args.pilot_tag
    model_path = args.dir / "stream_model.pt"
    if args.resume and model_path.exists():
        old = torch.load(model_path, weights_only=True, map_location="cpu")
        cfg = old.get("config", {})
        current_ids = [
            json.loads(path.read_text()).get("id")
            for path in sorted(pilot_dir.glob("fingerprint_rank*.json"))
        ]
        spec_path = pilot_dir / "spec.pt"
        compatible = (
            old.get("format") == "frozen_stream_decomposition_v2"
            and cfg.get("pilot_dir") == str(pilot_dir.resolve())
            and cfg.get("pilot_fingerprints") == current_ids
            and spec_path.exists()
            and cfg.get("spec_sha256") == file_sha256(spec_path)
            and cfg.get("C") == args.C
            and cfg.get("embed_dim") == min(
                args.embed_dim, int(cfg.get("feature_dim", 0)),
                int(cfg.get("pilot_positions", 0)) - 1)
            and cfg.get("storage_dtype") == "float32"
            and cfg.get("seed") == args.seed
            and cfg.get("pca_iters") == args.pca_iters
            and cfg.get("kmeans_iters") == args.kmeans_iters
            and int(cfg.get("pilot_positions", args.pilot_max_positions + 1))
                <= args.pilot_max_positions)
        if compatible:
            geo67.log(f"reusing compatible stream model {old['id']}")
            return
    fit_stream_model(
        pilot_dir, model_path, C=args.C, embed_dim=args.embed_dim,
        seed=args.seed, kmeans_iters=args.kmeans_iters,
        pca_iters=args.pca_iters,
        pilot_max_positions=args.pilot_max_positions, log=geo67.log)


def _save_stream_state(path: Path, *, payload_id: str, processed: int,
                       loader, pos_gen, reservoir_gen, seen: torch.Tensor,
                       labels: torch.Tensor, toks: torch.Tensor,
                       nexts: torch.Tensor, quota: int, complete: bool,
                       seconds: float):
    _atomic_torch_save({
        "format": "stream_checkpoint_v1", "id": payload_id,
        "processed": processed, "loader": loader.state_dict(),
        "position_generator_state": pos_gen.get_state(),
        "reservoir_generator_state": reservoir_gen.get_state(),
        "cluster_counts": seen, "labels": labels, "tok": toks,
        "next": nexts, "quota": quota, "complete": complete,
        "seconds": seconds,
    }, path)


def stage_stream(args):
    """Assign arbitrary-scale inputs online and retain bounded detailed data."""
    ddp, rank, world, device = geo67.ddp_setup()
    args.dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.spec_dir / "spec.pt"
    model_path = args.artifact_root / args.stream_model_tag / "stream_model.pt"
    if not spec_path.exists():
        raise FileNotFoundError(f"missing reusable feature spec: {spec_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing frozen stream model: {model_path}; run fit_stream first")
    model = load_stream_model(model_path, device)
    if model["config"]["spec_sha256"] != file_sha256(spec_path):
        raise ValueError("stream model and collection feature specs differ")
    if int(model["config"]["C"]) != args.C:
        raise ValueError(f"--C {args.C} != stream model C={model['config']['C']}")
    spec, scales, dim = load_spec(args.spec_dir, device)
    if int(model["config"]["feature_dim"]) != dim:
        raise ValueError("stream model feature dimension differs from spec")
    args.feat_dim = dim
    if rank == 0:
        local_spec = args.dir / "spec.pt"
        if local_spec.resolve() != spec_path.resolve():
            shutil.copyfile(spec_path, local_spec)

    collector = config_payload(args, spec_path, world)
    payload = {
        "format": "stream_collection_v1", "collector": collector,
        "stream_model_id": model["id"], "C": args.C,
        "reservoir_per_cluster": args.reservoir_per_cluster,
        "world": world,
    }
    payload_id = stable_fingerprint(payload)
    meta_path = args.dir / f"stream_meta_rank{rank}.pt"
    checkpoint_path = args.dir / f"stream_checkpoint_rank{rank}.pt"
    collection_path = args.dir / "stream_collection.json"
    if args.resume and meta_path.exists() and collection_path.exists():
        old = torch.load(meta_path, weights_only=True, map_location="cpu")
        collection = json.loads(collection_path.read_text())
        if (old.get("id") == payload_id and old.get("complete")
                and collection.get("id") == payload_id):
            geo67.log(f"reusing complete stream shard {payload_id} on rank {rank}")
            if ddp:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            return

    cap = setup_model(args, device)
    loader = make_loader(args, cap, rank, world)
    pos_gen = torch.Generator().manual_seed(args.seed + 7 * rank)
    reservoir_gen = torch.Generator().manual_seed(args.seed + 32452843 * (rank + 1))
    dims = {
        path: {"p": int(cap.target.get_submodule(path).in_features),
               "g": int(cap.target.get_submodule(path).out_features)}
        for path in geo67.MODULES
    }
    quota = local_reservoir_quota(args.reservoir_per_cluster, world, rank)
    capacity = args.C * quota
    resumed = args.resume and checkpoint_path.exists()
    writer = ModuleReservoirWriter(
        args.dir, rank, list(geo67.MODULES), dims, args.ig_k,
        args.C, quota, resume=resumed)
    if resumed:
        state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if state.get("id") != payload_id:
            raise ValueError(
                f"incompatible stream checkpoint {checkpoint_path}; pass "
                "--no-resume to start this shard from scratch")
        processed = int(state["processed"])
        loader.load_state_dict(state["loader"])
        pos_gen.set_state(state["position_generator_state"])
        reservoir_gen.set_state(state["reservoir_generator_state"])
        seen = state["cluster_counts"].long()
        slot_labels = state["labels"].long()
        slot_toks = state["tok"].long()
        slot_nexts = state["next"].long()
        prior_seconds = float(state.get("seconds", 0.0))
        geo67.log(f"resuming stream shard at {processed:,} positions")
    else:
        processed = 0
        seen = torch.zeros(args.C, dtype=torch.int64)
        slot_labels = torch.full((capacity,), -1, dtype=torch.int64)
        slot_toks = torch.zeros(capacity, dtype=torch.int64)
        slot_nexts = torch.zeros(capacity, dtype=torch.int64)
        prior_seconds = 0.0

    n_local = args.n_positions // world + int(rank < args.n_positions % world)
    rows_per_batch = args.batch_seqs * args.pos_per_seq
    n_batches = math.ceil(max(0, n_local - processed) / rows_per_batch)
    t0 = time.perf_counter()
    for batch_idx in range(n_batches):
        idx, pos, bi = sampled_batch(loader, pos_gen, device, args.pos_per_seq)
        phi, _, gatherer = pass_features(
            args, cap, idx, pos, bi, spec, scales, dim, return_pg=True)
        take = min(phi.shape[0], n_local - processed)
        labels = assign_features(phi[:take], model)
        # Decide reservoir acceptance BEFORE touching detailed vectors: once
        # reservoirs mature, Algorithm-R accepts only a few rows per batch,
        # so gathering full (p, g) rows lazily removes the dominant per-batch
        # gather traffic of the eager path.
        sources, destinations = reservoir_updates(
            labels, seen, quota, reservoir_gen)
        if sources.numel():
            detailed = gatherer.gather(sources)
            writer.write(detailed, torch.arange(sources.numel()), destinations)
            labels_cpu = labels.cpu()
            tok = idx[bi, pos].reshape(-1).cpu()[:take]
            nxt = idx[bi, (pos + 1).clamp(max=idx.shape[1] - 1)] \
                .reshape(-1).cpu()[:take]
            slot_labels[destinations] = labels_cpu[sources]
            slot_toks[destinations] = tok[sources]
            slot_nexts[destinations] = nxt[sources]
        del gatherer
        processed += take
        elapsed = prior_seconds + time.perf_counter() - t0
        checkpoint_due = ((batch_idx + 1) % args.checkpoint_batches == 0
                          or processed == n_local)
        if checkpoint_due:
            writer.flush()
            _save_stream_state(
                checkpoint_path, payload_id=payload_id, processed=processed,
                loader=loader, pos_gen=pos_gen, reservoir_gen=reservoir_gen,
                seen=seen, labels=slot_labels, toks=slot_toks,
                nexts=slot_nexts, quota=quota, complete=False,
                seconds=elapsed)
        if batch_idx == 0 or (batch_idx + 1) % 16 == 0:
            rate = processed / max(elapsed, 1e-9)
            geo67.log(f"stream batch {batch_idx + 1}/{n_batches}: "
                      f"{processed:,}/{n_local:,} ({rate:,.0f} pos/s)")

    writer.flush()
    elapsed = prior_seconds + time.perf_counter() - t0
    _save_stream_state(
        meta_path, payload_id=payload_id, processed=processed, loader=loader,
        pos_gen=pos_gen, reservoir_gen=reservoir_gen, seen=seen,
        labels=slot_labels, toks=slot_toks, nexts=slot_nexts, quota=quota,
        complete=True, seconds=elapsed)
    valid = int((slot_labels >= 0).sum())
    geo67.log(f"stream shard complete: {processed:,} assigned, "
              f"{valid:,} detailed reservoir rows, {elapsed:.1f}s")
    if ddp:
        torch.distributed.barrier()
    if rank == 0:
        parts = [torch.load(args.dir / f"stream_meta_rank{r}.pt",
                            weights_only=True, map_location="cpu")
                 for r in range(world)]
        labels = torch.cat([p["labels"][p["labels"] >= 0] for p in parts])
        toks = torch.cat([p["tok"][p["labels"] >= 0] for p in parts])
        nexts = torch.cat([p["next"][p["labels"] >= 0] for p in parts])
        sizes = sum((p["cluster_counts"] for p in parts),
                    torch.zeros(args.C, dtype=torch.int64))
        reservoir_sizes = torch.bincount(labels, minlength=args.C)
        _atomic_torch_save(
            {"labels": labels, "C": args.C, "sizes": sizes,
             "reservoir_sizes": reservoir_sizes, "tok": toks, "next": nexts,
             "stream_model_id": model["id"]},
            args.dir / f"labels_C{args.C}.pt")
        manifest = {
            "format": "module_shards_v1", "id": payload_id,
            "stream_model_id": model["id"], "modules": list(geo67.MODULES),
            "dims": dims, "ig_k": args.ig_k, "sensor": args.sensor,
            "gim_tau": args.gim_tau, "scalar": args.scalar, "C": args.C,
            "world": world, "positions": int(sizes.sum()),
            "reservoir_rows": int(labels.numel()),
            "reservoir_per_cluster": args.reservoir_per_cluster,
            "rank_quotas": [int(p["quota"]) for p in parts],
            "collector": collector,
        }
        tmp = args.dir / "stream_collection.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp, args.dir / "stream_collection.json")
        geo67.log(f"stream collection ready: {int(sizes.sum()):,} assignments, "
                  f"{labels.numel():,} extraction rows")
    if ddp:
        torch.distributed.destroy_process_group()


def stage_benchmark(args):
    ddp, rank, world, device = geo67.ddp_setup()
    spec_path = args.spec_dir / "spec.pt"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"missing feature spec: {spec_path}; create one with "
            f"`python collect_fast.py spec --tag {args.spec_tag}`")
    spec, scales, dim = load_spec(args.spec_dir, device)
    cap = setup_model(args, device)
    loader = make_loader(args, cap, rank, world)
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
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    timing = torch.tensor([seconds, warm_seconds, peak_gib], device=device)
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
        "compile_mode": args.compile_mode if args.compile else None,
        "fused_attention": args.fused_attention,
        "attention_backend": "flash_only" if args.fused_attention else "auto",
        "master_dtype": args.master_dtype,
        "pass_dtype": "bfloat16" if args.bf16 else "float32",
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
        "peak_gpu_gib_max_rank": timing[2].item(),
        "feature_dim": dim,
        "feature_spec_sha256": file_sha256(spec_path),
        "input": ("synthetic_uniform_v1" if args.synthetic_data
                  else str(args.data_path)),
        "model": geo1b.model_identity(),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except ImportError:
        pass
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


def stage_selftest(args):
    """Prove the direct sparse feature path bit-identical to the reference.

    One captured forward/backward supplies the pass tensors; the reference
    path (full per-position row materialization, then column gather) and the
    direct path (composed gather) are then evaluated on the same tensors, so
    any difference is a real indexing bug rather than kernel nondeterminism.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run the selftest stage with python, not torchrun")
    device = "cuda"
    spec, scales, dim = load_spec(args.spec_dir, device)
    cap = setup_model(args, device)
    loader = make_loader(args, cap, 0, 1)
    gen = torch.Generator().manual_seed(args.seed)
    idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
    batch, samples = pos.shape
    seq = idx.shape[1]
    with model_pass(device, args.bf16, args.fused_attention):
        logits, cache = cap.run(idx)
        reward = objective(logits, idx, args.scalar)
        posts = [cache[p]["post"] for p in geo67.MODULES]
        gposts = torch.autograd.grad(reward, posts)
    pres = {path: cache[path]["pre"].detach() for path in geo67.MODULES}
    grads = {path: grad.detach()
             for path, grad in zip(geo67.MODULES, gposts, strict=True)}
    ps = {path: pres[path][bi, pos].reshape(batch * samples, -1)
          for path in geo67.MODULES}
    gs = {path: grads[path][bi, pos].reshape(batch * samples, -1)
          for path in geo67.MODULES}
    phi_reference = features_from(spec, scales, dim, gs, ps, device)
    rows = (bi * seq + pos).reshape(-1)
    phi_direct = features_direct(spec, scales, pres, grads, rows)
    phi_ok = torch.equal(phi_reference, phi_direct)

    picks = torch.randperm(batch * samples, generator=gen)[:8].to(device)
    gathered = DetailGatherer([(pres, grads)], rows).gather(picks)
    detail_ok = all(
        torch.equal(gathered[path]["p"][0], ps[path][picks])
        and torch.equal(gathered[path]["g"][0], gs[path][picks])
        for path in geo67.MODULES)
    result = {"phi_bit_identical": phi_ok, "detail_rows_bit_identical": detail_ok,
              "rows": batch * samples, "feature_dim": dim}
    geo67.log(f"selftest: {json.dumps(result)}")
    if not (phi_ok and detail_ok):
        raise SystemExit("SELFTEST FAILED")


def stage_spec_subset(args):
    """Derive a lower-D spec from an existing one, exactly unbiased.

    Each module's sampled coordinates are i.i.d. draws from its proposal, so
    a prefix of them is itself a valid sample; only the 1/sqrt(R) factor in
    the per-feature scales must be re-normalized to the new count.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run spec_subset with python, not torchrun")
    parent_path = args.spec_dir / "spec.pt"
    parent = torch.load(parent_path, weights_only=True, map_location="cpu")
    d_old = int(parent["D"])
    d_new = args.feat_dim
    if not d_new or not (0 < d_new < d_old):
        raise ValueError(f"--feat_dim must be in (0, {d_old})")
    quotas = {path: d_new * count / d_old
              for path, count in parent["alloc"].items()}
    alloc = {path: math.floor(q) for path, q in quotas.items()}
    remainder = d_new - sum(alloc.values())
    for path in sorted(alloc, key=lambda p: quotas[p] - alloc[p],
                       reverse=True)[:remainder]:
        alloc[path] += 1
    spec, scales = {}, {}
    for path, (rows, cols) in parent["spec"].items():
        old_count = rows.numel()
        count = min(alloc[path], old_count)
        spec[path] = (rows[:count].clone(), cols[:count].clone())
        if count:
            scales[path] = (parent["scales"][path][:count]
                            * math.sqrt(old_count / count)).clone()
        else:
            scales[path] = torch.zeros(0)
    args.dir.mkdir(parents=True, exist_ok=True)
    payload = {"spec": spec, "scales": scales, "D": d_new, "alloc": alloc,
               "seed": parent.get("seed"),
               "proposal": f"subset_of:{parent.get('proposal', 'unknown')}",
               "parent_spec_sha256": file_sha256(parent_path),
               "model": parent.get("model"), "format": 1}
    torch.save(payload, args.dir / "spec.pt")
    geo67.log(f"spec_subset: D={d_old} -> {d_new} at {args.dir / 'spec.pt'}")


def load_reservoir_rows(root: Path, rank: int, count: int, device: str,
                        seed: int | None = None):
    """Occupied reservoir slots as full (p, g) row tensors.

    With ``seed`` the rows are drawn uniformly from ALL occupied slots; slots
    are laid out cluster-major, so a prefix would cover only the lowest
    cluster ids and produce a cluster-correlated, unrepresentative sample.
    """
    reservoir = root / f"reservoir_rank{rank}"
    manifest = json.loads((reservoir / "manifest.json").read_text())
    meta = torch.load(root / f"stream_meta_rank{rank}.pt", weights_only=True,
                      map_location="cpu")
    valid = (meta["labels"] >= 0).nonzero(as_tuple=False).flatten()
    if valid.numel() < count:
        raise ValueError(f"reservoir has only {valid.numel()} occupied slots")
    if seed is None:
        pick = valid[:count].numpy()
    else:
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(valid.numel(), generator=generator)[:count]
        pick = valid[chosen].sort().values.numpy()
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for module_index, path in enumerate(manifest["modules"]):
        tensors[path] = {}
        for kind in ("p", "g"):
            dim = int(manifest["dims"][path][kind])
            arr = np.memmap(
                reservoir / f"module_{module_index:03d}_{kind}.bf16",
                dtype=np.uint16, mode="r",
                shape=(manifest["ig_k"], manifest["capacity"], dim))
            raw = torch.from_numpy(np.array(arr[0, pick], copy=True))
            tensors[path][kind] = raw.view(torch.bfloat16).to(device)
    return tensors


def exact_gram(target, rows: dict, count: int, device: str,
               block: int = 256) -> torch.Tensor:
    """Exact attribution gram of stored rows via the rank-1 kernel identity.

    Accepts per-module (n, d) tensors (single pass) or (K, n, d) IG stacks;
    IG cross terms are averaged exactly as in the legacy gram stage.
    """
    gram = torch.zeros(count, count, device=device)
    for path in geo67.MODULES:
        weight = target.get_submodule(path).weight.detach().float()
        w2t = (weight * weight).t().contiguous()
        p_all = rows[path]["p"].float()
        g_all = rows[path]["g"].float()
        if p_all.ndim == 2:
            p_all = p_all[None]
            g_all = g_all[None]
        steps = p_all.shape[0]
        for ka in range(steps):
            for kb in range(steps):
                p_rows, gi_rows = p_all[ka], g_all[ka]
                pj_rows, gj_rows = p_all[kb], g_all[kb]
                for i0 in range(0, count, block):
                    pi = p_rows[i0:i0 + block]
                    gi = gi_rows[i0:i0 + block]
                    for j0 in range(0, count, block):
                        pj = pj_rows[j0:j0 + block]
                        gj = gj_rows[j0:j0 + block]
                        pij = pi[:, None, :] * pj[None, :, :]
                        t1 = pij @ w2t
                        gij = gi[:, None, :] * gj[None, :, :]
                        gram[i0:i0 + pi.shape[0], j0:j0 + pj.shape[0]] += \
                            (t1 * gij).sum(-1) / (steps * steps)
                        del pij, t1, gij
    return gram


def gram_metrics(gram: torch.Tensor, estimate: torch.Tensor) -> dict:
    count = gram.shape[0]
    off = ~torch.eye(count, dtype=torch.bool, device=gram.device)
    corr_raw = torch.corrcoef(
        torch.stack([gram[off], estimate[off]]))[0, 1].item()
    corr_normalized = torch.corrcoef(torch.stack([
        geo67.normalize_gram(gram)[off],
        geo67.normalize_gram(estimate)[off]]))[0, 1].item()
    diag_corr = torch.corrcoef(torch.stack(
        [gram.diagonal(), estimate.diagonal()]))[0, 1].item()
    return {"corr_raw": corr_raw, "corr_normalized": corr_normalized,
            "diag_corr": diag_corr,
            "rel_fro": ((estimate - gram).norm() / gram.norm()).item()}


def stage_validate_fresh(args):
    """Fidelity on freshly collected RANDOM corpus rows (legacy-comparable).

    Reservoir rows are stratified one-per-cluster and therefore dominated by
    dissimilar pairs — the hardest population for the estimator. This stage
    reproduces the legacy validation population (random positions from a few
    corpus sequences) under the CURRENT sensor configuration, and also
    reports the mutual agreement of two independently seeded specs to
    separate estimator bias from sampling variance.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run validate_fresh with python, not torchrun")
    device = "cuda"
    spec, scales, dim = load_spec(args.spec_dir, device)
    cap = setup_model(args, device)
    loader = make_loader(args, cap, 0, 1)
    gen = torch.Generator().manual_seed(args.seed)
    idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
    phi, _, gatherer = pass_features(
        args, cap, idx, pos, bi, spec, scales, dim, return_pg=True)
    count = min(args.val_n, phi.shape[0])
    picks = torch.randperm(phi.shape[0], generator=gen)[:count].to(device)
    rows = gatherer.gather(picks)
    gram = exact_gram(cap.target, rows, count, device)
    estimate = phi[picks] @ phi[picks].t()
    result = {"D": dim, "val_n": count, "rows": "fresh_random",
              "sensor": args.sensor, "scalar": args.scalar,
              "ig_k": args.ig_k, "pass_dtype": args.bf16 and "bfloat16"
              or "float32", "spec_dir": str(args.spec_dir),
              **gram_metrics(gram, estimate)}
    if args.second_spec_tag:
        spec2, scales2, dim2 = load_spec(
            args.artifact_root / args.second_spec_tag, device)
        steps = rows[next(iter(rows))]["p"].shape[0] \
            if rows[next(iter(rows))]["p"].ndim == 3 else 1
        phi2 = None
        for k in range(steps):
            gs = {path: (rows[path]["g"][k] if steps > 1 or
                         rows[path]["g"].ndim == 3 else rows[path]["g"])
                  for path in geo67.MODULES}
            ps = {path: (rows[path]["p"][k] if steps > 1 or
                         rows[path]["p"].ndim == 3 else rows[path]["p"])
                  for path in geo67.MODULES}
            part = features_from(spec2, scales2, dim2, gs, ps, device)
            phi2 = part if phi2 is None else phi2 + part
        phi2 /= steps
        estimate2 = phi2 @ phi2.t()
        result["second_spec"] = {
            "spec_dir": str(args.artifact_root / args.second_spec_tag),
            "vs_exact": gram_metrics(gram, estimate2),
            "vs_first_estimate": gram_metrics(estimate, estimate2),
        }
    output = args.spec_dir / f"validate_fresh_D{dim}_{args.sensor}.json"
    output.write_text(json.dumps(result, indent=2))
    geo67.log(f"VALIDATE-FRESH D={dim} sensor={args.sensor}: "
              f"corr_raw={result['corr_raw']:.4f}, "
              f"corr_normalized={result['corr_normalized']:.4f}, "
              f"diag_corr={result['diag_corr']:.4f} -> {output}")


def stage_validate(args):
    """Hard fidelity gate: candidate-spec feature gram vs the exact gram.

    Rows come from the streamed run's detailed reservoir, so the reference
    kernel describes exactly the sensor that produced the production
    decomposition. Used to qualify reduced-D specs before deployment.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run validate with python, not torchrun")
    device = "cuda"
    spec, scales, dim = load_spec(args.spec_dir, device)
    source = args.artifact_root / args.val_tag
    rows = load_reservoir_rows(source, args.val_rank, args.val_n, device,
                               seed=args.seed + 104729)
    target = geo67.load_target(device)
    gram = exact_gram(target, rows, args.val_n, device)
    gs = {path: rows[path]["g"] for path in geo67.MODULES}
    ps = {path: rows[path]["p"] for path in geo67.MODULES}
    phi = features_from(spec, scales, dim, gs, ps, device)
    estimate = phi @ phi.t()
    off = ~torch.eye(args.val_n, dtype=torch.bool, device=device)
    corr_raw = torch.corrcoef(
        torch.stack([gram[off], estimate[off]]))[0, 1].item()
    normalized = geo67.normalize_gram(gram)
    normalized_estimate = geo67.normalize_gram(estimate)
    corr_normalized = torch.corrcoef(
        torch.stack([normalized[off], normalized_estimate[off]]))[0, 1].item()
    relative = ((estimate - gram).norm() / gram.norm()).item()
    result = {"D": dim, "val_n": args.val_n, "val_tag": args.val_tag,
              "spec_dir": str(args.spec_dir), "corr_raw": corr_raw,
              "corr_normalized": corr_normalized, "rel_fro": relative,
              "gate": args.val_gate}
    output = args.spec_dir / f"validate_D{dim}.json"
    output.write_text(json.dumps(result, indent=2))
    geo67.log(f"VALIDATE D={dim}: corr_raw={corr_raw:.4f}, "
              f"corr_normalized={corr_normalized:.4f}, rel_fro={relative:.4f} "
              f"(gate {args.val_gate}) -> {output}")
    if corr_normalized < args.val_gate:
        raise SystemExit("VALIDATION FAILED")


def stage_spec_stats(args):
    """Activation-statistics-weighted feature proposal from reservoir rows.

    q(o, i) ∝ W_oi^2 * E[g_o^2] * E[p_i^2] — the near-optimal proposal for
    the heavy-tailed kernel sum (geo1m.stage_spec), whose statistics source
    (the legacy 16k exact collect) no longer exists. The streamed reservoir
    stores exactly the needed (p, g) rows, so the proposal can be rebuilt
    from the production run itself. Statistics come from rank-0 rows;
    validation should use rank-1 rows to stay disjoint.
    """
    if "RANK" in os.environ:
        raise RuntimeError("run spec_stats with python, not torchrun")
    device = "cuda"
    dim = args.feat_dim or 16384
    source = args.artifact_root / args.val_tag
    rows = load_reservoir_rows(source, 0, args.stats_rows, device,
                               seed=args.seed)
    target = geo67.load_target(device)
    row_weight, mean_p2, mass = {}, {}, {}
    for path in geo67.MODULES:
        weight = target.get_submodule(path).weight.detach().float()
        g2 = rows[path]["g"].float().pow(2).mean(0)
        p2 = rows[path]["p"].float().pow(2).mean(0)
        row_weight[path] = g2 * ((weight * weight) @ p2)
        mean_p2[path] = p2
        mass[path] = row_weight[path].sum().item()
    del rows
    total = sum(mass.values())
    quotas = {path: dim * mass[path] / total for path in geo67.MODULES}
    alloc = {path: math.floor(quotas[path]) for path in geo67.MODULES}
    remainder = dim - sum(alloc.values())
    for path in sorted(geo67.MODULES, key=lambda p: quotas[p] - alloc[p],
                       reverse=True)[:remainder]:
        alloc[path] += 1

    generator = torch.Generator(device=device).manual_seed(args.seed)
    spec, scales = {}, {}
    for path in geo67.MODULES:
        count = alloc[path]
        if count == 0:
            spec[path] = (torch.zeros(0, dtype=torch.int32),
                          torch.zeros(0, dtype=torch.int32))
            scales[path] = torch.zeros(0)
            continue
        weight = target.get_submodule(path).weight.detach().float()
        w2 = weight * weight
        rows_pick = torch.multinomial(row_weight[path], count,
                                      replacement=True, generator=generator)
        col_weight = w2[rows_pick] * mean_p2[path][None, :]
        cols_pick = torch.multinomial(col_weight, 1,
                                      generator=generator).squeeze(1)
        q = ((row_weight[path][rows_pick] / mass[path])
             * (col_weight[torch.arange(count, device=device), cols_pick]
                / col_weight.sum(1).clamp_min(1e-30)))
        w_abs = weight[rows_pick, cols_pick].abs()
        scales[path] = (w_abs / (count * q).clamp_min(1e-30).sqrt()).cpu()
        spec[path] = (rows_pick.int().cpu(), cols_pick.int().cpu())

    args.dir.mkdir(parents=True, exist_ok=True)
    payload = {"spec": spec, "scales": scales, "D": dim, "alloc": alloc,
               "seed": args.seed,
               "proposal": "stats_weighted_reservoir",
               "stats_source": str(source), "stats_rows": args.stats_rows,
               "model": geo1b.model_identity(), "format": 1}
    torch.save(payload, args.dir / "spec.pt")
    geo67.log(f"spec_stats: D={dim} stat-weighted proposal from "
              f"{args.stats_rows} reservoir rows -> {args.dir / 'spec.pt'}")


def stage_ab_master(args):
    """A/B the sensor under fp32 vs bf16 master weights on identical tokens."""
    if "RANK" in os.environ:
        raise RuntimeError("run ab_master with python, not torchrun")
    device = "cuda"
    spec, scales, dim = load_spec(args.spec_dir, device)
    # The mid-stage dtype flip would force a recompile of every linear shape
    # and trip Dynamo's recompile limit; fidelity needs no compilation.
    args.compile = False
    args.master_dtype = "float32"
    cap = setup_model(args, device)
    loader = make_loader(args, cap, 0, 1)
    gen = torch.Generator().manual_seed(args.seed)
    idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
    phi32, _ = pass_features(args, cap, idx, pos, bi, spec, scales, dim)
    cap.target.to(torch.bfloat16)
    phi16, _ = pass_features(args, cap, idx, pos, bi, spec, scales, dim)
    count = phi32.shape[0]
    feature_corr = torch.corrcoef(
        torch.stack([phi32.flatten(), phi16.flatten()]))[0, 1].item()
    gram32 = phi32 @ phi32.t()
    gram16 = phi16 @ phi16.t()
    off = ~torch.eye(count, dtype=torch.bool, device=device)
    gram_corr = torch.corrcoef(
        torch.stack([gram32[off], gram16[off]]))[0, 1].item()
    norm_corr = torch.corrcoef(torch.stack([
        geo67.normalize_gram(gram32)[off],
        geo67.normalize_gram(gram16)[off]]))[0, 1].item()
    result = {"rows": count, "feature_dim": dim,
              "feature_corr": feature_corr, "gram_corr_raw": gram_corr,
              "gram_corr_normalized": norm_corr, "gate": args.val_gate}
    output = args.dir / "ab_master.json"
    args.dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    geo67.log(f"AB-MASTER: feature_corr={feature_corr:.5f}, "
              f"gram corr raw/normalized={gram_corr:.5f}/{norm_corr:.5f} "
              f"-> {output}")
    if norm_corr < args.val_gate:
        raise SystemExit("AB-MASTER FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["spec", "collect", "fit_stream",
                                      "stream", "benchmark", "selftest",
                                      "spec_subset", "validate", "ab_master",
                                      "spec_stats", "validate_fresh"])
    ap.add_argument("--tag", default="run_gim")
    ap.add_argument("--spec_tag", default="full1m")
    ap.add_argument("--pilot_tag", default="stream_pilot")
    ap.add_argument("--stream_model_tag")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--data_path", type=Path, default=geo1b.BIN_PATH)
    ap.add_argument("--synthetic_data", action="store_true",
                    help="throughput benchmark only; do not use for production")
    ap.add_argument("--data_order", choices=["random", "sequential"],
                    default="sequential")
    ap.add_argument("--profile", choices=["baseline", "optimized"])
    ap.add_argument("--sensor", choices=["ig", "gim"], default="gim")
    ap.add_argument("--ig_k", type=int, default=1)
    ap.add_argument("--gim_tau", type=float, default=2.0)
    ap.add_argument("--scalar", choices=["ce", "logp_pred", "logit_pred",
                                         "equal_reward"], default="equal_reward")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--master_dtype", choices=["float32", "bfloat16"],
                    default="float32",
                    help="parameter storage dtype; bfloat16 requires an "
                         "ab_master fidelity pass first")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--fused_attention", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--reuse_fingerprints",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse_spec",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feat_dim", type=int)
    ap.add_argument("--n_positions", type=int, default=1048576)
    ap.add_argument("--pos_per_seq", type=int, default=64)
    ap.add_argument("--sub_per_seq", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=int, default=2048)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pca_iters", type=int, default=4)
    ap.add_argument("--kmeans_iters", type=int, default=30)
    ap.add_argument("--pilot_max_positions", type=int, default=262144)
    ap.add_argument("--reservoir_per_cluster", type=int, default=16,
                    help="global detailed examples retained per cluster")
    ap.add_argument("--checkpoint_batches", type=int, default=128)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--val_n", type=int, default=512,
                    help="reservoir rows for the exact-gram validation")
    ap.add_argument("--val_gate", type=float, default=0.95,
                    help="minimum normalized gram correlation")
    ap.add_argument("--val_tag", default="run1m_stream",
                    help="run whose reservoir supplies validation rows")
    ap.add_argument("--val_rank", type=int, default=1,
                    help="reservoir shard for validation rows (stats use 0)")
    ap.add_argument("--stats_rows", type=int, default=2048,
                    help="reservoir rows for spec_stats statistics")
    ap.add_argument("--second_spec_tag",
                    help="validate_fresh: independently seeded spec whose "
                         "estimate is compared against the first (bias vs "
                         "variance)")
    ap.add_argument("--warmup_batches", type=int, default=2)
    ap.add_argument("--benchmark_batches", type=int, default=8)
    ap.add_argument("--output", type=Path,
                    default=HERE / "out" / "benchmark.json")
    args = ap.parse_args()
    apply_profile(args)
    args.dir = args.artifact_root / args.tag
    args.spec_dir = args.artifact_root / args.spec_tag
    args.stream_model_tag = args.stream_model_tag or args.tag
    if args.stage == "stream":
        args.sub_per_seq = 0
    if args.checkpoint_batches < 1:
        ap.error("--checkpoint_batches must be positive")
    if args.reservoir_per_cluster < 1:
        ap.error("--reservoir_per_cluster must be positive")
    torch.manual_seed(args.seed)
    {"spec": stage_spec, "collect": stage_collect,
     "fit_stream": stage_fit_stream, "stream": stage_stream,
     "benchmark": stage_benchmark, "selftest": stage_selftest,
     "spec_subset": stage_spec_subset, "validate": stage_validate,
     "ab_master": stage_ab_master, "spec_stats": stage_spec_stats,
     "validate_fresh": stage_validate_fresh}[args.stage](args)


if __name__ == "__main__":
    main()
