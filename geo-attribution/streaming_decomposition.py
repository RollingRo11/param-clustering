"""Bounded-memory PCA/k-means fitting and detailed-example reservoirs.

The decomposition is learned once from a bounded pilot feature matrix. The
resulting center, projection, and spherical centroids are then frozen while an
arbitrarily large input stream is assigned online. Only a stratified reservoir
of (pre, gradient) tensors is retained for parameter-bank extraction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collection_runtime import file_sha256, stable_fingerprint


def _pilot_parts(pilot_dir: Path, dim: int) -> tuple[list[dict], int]:
    metas = []
    for path in sorted(pilot_dir.glob("meta_rank*.pt")):
        meta = torch.load(path, weights_only=True, map_location="cpu")
        rank = int(path.stem.removeprefix("meta_rank"))
        n = int(meta["n_local"])
        feat = pilot_dir / f"feat_rank{rank}.f16"
        expected = n * dim * np.dtype(np.float16).itemsize
        if not feat.exists() or feat.stat().st_size != expected:
            raise ValueError(f"invalid pilot feature shard {feat}")
        manifest_path = pilot_dir / f"fingerprint_rank{rank}.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        metas.append({"rank": rank, "n": n, "feat": feat,
                      "fingerprint_id": manifest.get("id")})
    if not metas:
        raise FileNotFoundError(f"no pilot meta_rank*.pt shards in {pilot_dir}")
    return metas, sum(p["n"] for p in metas)


def fit_stream_model(pilot_dir: Path, output_path: Path, *, C: int,
                     embed_dim: int, seed: int, kmeans_iters: int,
                     pca_iters: int, pilot_max_positions: int,
                     device: str = "cuda", log=print) -> dict:
    """Fit a frozen decomposition using only a bounded pilot matrix."""
    spec_path = pilot_dir / "spec.pt"
    if not spec_path.exists():
        raise FileNotFoundError(f"pilot is missing {spec_path}")
    sp = torch.load(spec_path, weights_only=True, map_location="cpu")
    dim = int(sp["D"])
    parts, n = _pilot_parts(pilot_dir, dim)
    if n > pilot_max_positions:
        raise ValueError(
            f"pilot has {n} positions, above --pilot_max_positions "
            f"{pilot_max_positions}; the cap prevents accidental N-scale fitting")
    if C > n:
        raise ValueError(f"C={C} exceeds bounded pilot size N={n}")
    embed_dim = min(embed_dim, dim, n - 1)
    if embed_dim < 1:
        raise ValueError("embed_dim must be positive")

    X16 = torch.empty(n, dim, dtype=torch.float16, device=device)
    off = 0
    for part in parts:
        mm = np.memmap(part["feat"], dtype=np.float16, mode="r",
                       shape=(part["n"], dim))
        for lo in range(0, part["n"], 8192):
            hi = min(lo + 8192, part["n"])
            X16[off + lo:off + hi] = torch.from_numpy(
                np.array(mm[lo:hi], copy=True)).to(device)
        off += part["n"]
    log(f"stream-fit: loaded bounded pilot [{n}, {dim}]")

    mean = torch.zeros(dim, dtype=torch.float64, device=device)
    for lo in range(0, n, 8192):
        mean += X16[lo:lo + 8192].double().sum(0)
    mean = (mean / n).float()
    X = torch.empty(n, dim, dtype=torch.float32, device=device)
    for lo in range(0, n, 8192):
        X[lo:lo + 8192] = X16[lo:lo + 8192].float() - mean
    del X16

    # Randomized right-singular subspace iteration avoids materializing D x D.
    q = min(dim, n, embed_dim + min(64, max(8, embed_dim // 4)))
    gen = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(dim, q, generator=gen, device=device)
    Q = torch.linalg.qr(Q, mode="reduced")[0]
    for it in range(pca_iters):
        Z = X @ Q
        Q = torch.linalg.qr(X.t() @ Z, mode="reduced")[0]
        del Z
        log(f"stream-fit: PCA subspace iteration {it + 1}/{pca_iters}")
    Z = X @ Q
    small = (Z.t() @ Z) / n
    values, vectors = torch.linalg.eigh(0.5 * (small + small.t()))
    order = values.argsort(descending=True)[:embed_dim]
    projector = (Q @ vectors[:, order]).contiguous()
    spectrum = values[order].clamp_min(0)
    del Q, Z, small, values, vectors

    Y = F.normalize(X @ projector, dim=1)
    del X
    perm = torch.randperm(n, generator=gen, device=device)
    centroids = Y[perm[:C]].clone()
    labels = torch.empty(n, dtype=torch.int64, device=device)
    for it in range(kmeans_iters):
        sums = torch.zeros(C, embed_dim, device=device)
        counts = torch.zeros(C, dtype=torch.int64, device=device)
        score = 0.0
        for lo in range(0, n, 16384):
            y = Y[lo:lo + 16384]
            sim = y @ centroids.t()
            best, lab = sim.max(1)
            labels[lo:lo + y.shape[0]] = lab
            sums.index_add_(0, lab, y)
            counts += torch.bincount(lab, minlength=C)
            score += best.sum().item()
        updated = F.normalize(sums, dim=1)
        empty = counts == 0
        if empty.any():
            replacement = Y[torch.randint(
                n, (int(empty.sum()),), generator=gen, device=device)]
            updated[empty] = replacement
        centroids = updated
        if it == 0 or (it + 1) % 5 == 0 or it + 1 == kmeans_iters:
            log(f"stream-fit: k-means {it + 1}/{kmeans_iters}, "
                f"mean cosine={score / n:.5f}, empty={int(empty.sum())}")
    # The loop's labels precede its final centroid update; recompute once so
    # diagnostics and reloaded online assignment describe the saved centroids.
    for lo in range(0, n, 16384):
        y = Y[lo:lo + 16384]
        labels[lo:lo + y.shape[0]] = (y @ centroids.t()).argmax(1)
    sizes = torch.bincount(labels, minlength=C)

    config = {
        "format": "frozen_stream_decomposition_v2",
        "pilot_dir": str(pilot_dir.resolve()),
        "pilot_fingerprints": [p["fingerprint_id"] for p in parts],
        "pilot_positions": n,
        "spec_sha256": file_sha256(spec_path),
        "feature_dim": dim,
        "embed_dim": embed_dim,
        "storage_dtype": "float32",
        "C": C,
        "seed": seed,
        "pca_iters": pca_iters,
        "kmeans_iters": kmeans_iters,
    }
    model_id = stable_fingerprint(config)
    payload = {
        "format": config["format"], "id": model_id, "config": config,
        "mean": mean.cpu(), "projector": projector.cpu(),
        "centroids": centroids.cpu(), "spectrum": spectrum.cpu(),
        "pilot_cluster_sizes": sizes.cpu(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, output_path)
    summary = {"id": model_id, **config,
               "cluster_min": int(sizes.min()),
               "cluster_median": int(sizes.median()),
               "cluster_max": int(sizes.max()),
               "empty_clusters": int((sizes == 0).sum())}
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    log(f"stream-fit complete: model={model_id}, C={C}, d={embed_dim}")
    return summary


def load_stream_model(path: Path, device: str) -> dict:
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if raw.get("format") != "frozen_stream_decomposition_v2":
        raise ValueError(f"unsupported stream model in {path}")
    return {
        **raw,
        "mean": raw["mean"].float().to(device),
        "projector": raw["projector"].float().to(device),
        "centroids": raw["centroids"].float().to(device),
    }


def assign_features(phi: torch.Tensor, model: dict) -> torch.Tensor:
    """Assign one feature batch exactly as pilot fp16 features were fitted."""
    x = phi.clamp(-6e4, 6e4).half().float()
    y = F.normalize((x - model["mean"]) @ model["projector"], dim=1)
    return (y @ model["centroids"].t()).argmax(1)


def local_reservoir_quota(total_per_cluster: int, world: int, rank: int) -> int:
    return total_per_cluster // world + int(rank < total_per_cluster % world)


def reservoir_updates(labels: torch.Tensor, seen: torch.Tensor, quota: int,
                      generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized stratified Algorithm-R updates for one batch.

    Returns source-row and fixed destination-slot arrays. If multiple samples
    replace the same slot inside a batch, only the last (correct) update is
    emitted. ``seen`` is updated in place.
    """
    labels = labels.to("cpu", dtype=torch.int64)
    if quota == 0:
        seen += torch.bincount(labels, minlength=seen.numel())
        z = torch.empty(0, dtype=torch.int64)
        return z, z
    final: dict[int, int] = {}
    for c in labels.unique(sorted=True).tolist():
        source = (labels == c).nonzero(as_tuple=False).flatten()
        old = int(seen[c])
        stream_index = torch.arange(old + 1, old + source.numel() + 1,
                                    dtype=torch.float64)
        draw = torch.rand(source.numel(), generator=generator,
                          dtype=torch.float64)
        slot = torch.floor(draw * stream_index).long()
        initial = stream_index <= quota
        slot[initial] = stream_index[initial].long() - 1
        accept = slot < quota
        for src, dst in zip(source[accept].tolist(), slot[accept].tolist()):
            final[c * quota + dst] = src
        seen[c] += source.numel()
    destinations = torch.tensor(list(final), dtype=torch.int64)
    sources = torch.tensor(list(final.values()), dtype=torch.int64)
    return sources, destinations


class ModuleReservoirWriter:
    """Disk-backed BF16 module tensors with fixed random-access slots."""

    def __init__(self, root: Path, rank: int, modules: list[str], dims: dict,
                 ig_k: int, C: int, quota: int, *, resume: bool):
        self.root = root / f"reservoir_rank{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.modules = modules
        self.dims = dims
        self.ig_k = ig_k
        self.C = C
        self.quota = quota
        self.capacity = C * quota
        self.arrays: dict[tuple[str, str], np.memmap] = {}
        manifest = {"format": "module_reservoir_bf16_v1", "rank": rank,
                    "modules": modules, "dims": dims, "ig_k": ig_k,
                    "C": C, "quota": quota, "capacity": self.capacity}
        manifest_path = self.root / "manifest.json"
        if resume:
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"incompatible reservoir manifest {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if self.capacity == 0:
            return
        for module_index, path in enumerate(modules):
            for kind in ("p", "g"):
                dim = int(dims[path][kind])
                file = self.root / f"module_{module_index:03d}_{kind}.bf16"
                shape = (ig_k, self.capacity, dim)
                mode = "r+" if resume else "w+"
                arr = np.memmap(file, dtype=np.uint16, mode=mode, shape=shape)
                self.arrays[(path, kind)] = arr

    def write(self, pg: dict, sources: torch.Tensor, destinations: torch.Tensor):
        if sources.numel() == 0:
            return
        src = sources.to(next(iter(pg.values()))["p"].device)
        dst = destinations.numpy()
        for path in self.modules:
            for kind in ("p", "g"):
                value = pg[path][kind][:, src].to(torch.bfloat16).cpu().contiguous()
                bits = value.view(torch.uint16).numpy()
                self.arrays[(path, kind)][:, dst] = bits

    def flush(self):
        for arr in self.arrays.values():
            arr.flush()
