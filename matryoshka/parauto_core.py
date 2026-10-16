"""Free rank-one factors, exact-sum projection, and effect scores."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean keep mask of the k largest scores along the last axis."""
    if k >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    index = scores.topk(k, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, index, True)


def cosine_lr(step: int, cfg) -> float:
    if cfg.steps <= 1:
        return cfg.lr
    progress = step / (cfg.steps - 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * progress))


@dataclass
class FreePieces:
    """Columns define C_c = write[:, c] read[:, c]^T; neither side is tied to W."""

    read: torch.Tensor                     # [d_in, components]
    write: torch.Tensor                    # [d_out, components]

    @property
    def count(self):
        return self.read.shape[1]


def reconstruction_error(weight: torch.Tensor, pieces: FreePieces) -> float:
    residual = pieces.write.double() @ pieces.read.double().T - weight.double()
    return float(residual.norm() / weight.double().norm().clamp_min(1e-30))


def projected_pieces(weight: torch.Tensor, pieces: FreePieces, dtype=torch.float64) -> FreePieces:
    """Conditional Euclidean projection onto L R^T = W (not a joint nearest pair).

    If d_out <= d_in, hold L fixed and set
      R <- R + (W - L R^T)^T (L L^T)^-1 L.
    This is the minimum-Frobenius correction to R; its component-space null
    directions survive. For a tall W use the transposed construction on L.
    Both factors receive gradients before this projection. The held factor
    must have full row rank. A singular solve fails rather than relaxing the
    constraint with a ridge/loss or hiding a residual outside the pieces.
    Float64 solves and residuals limit drift when parameters are float32.
    """
    reads, writes, w = pieces.read.to(dtype), pieces.write.to(dtype), weight.to(dtype)
    residual = w - writes @ reads.T
    if w.shape[0] <= w.shape[1]:
        reads = reads + residual.T @ torch.linalg.solve(writes @ writes.T, writes)
        return FreePieces(reads.to(pieces.read.dtype), pieces.write)
    else:
        writes = writes + residual @ torch.linalg.solve(reads @ reads.T, reads)
        return FreePieces(pieces.read, writes.to(pieces.write.dtype))


@torch.no_grad()
def project_piece_sum_(weight: torch.Tensor, pieces: FreePieces, dtype=torch.float64) -> None:
    """Enforce the exact matrix sum in-place, outside the optimizer graph."""
    projected = projected_pieces(weight, pieces) if dtype == torch.float64 else projected_pieces(weight, pieces, dtype=dtype)
    pieces.read.copy_(projected.read)
    pieces.write.copy_(projected.write)
    if not (torch.isfinite(pieces.read).all() and torch.isfinite(pieces.write).all()):
        raise RuntimeError("Non-finite exact-sum projection")


def initialize_pieces(weight: torch.Tensor, components: int, seed: int = 0) -> FreePieces:
    """Random free factors, then project; no duplicated neurons or zero extra pieces."""
    if components < min(weight.shape):
        raise ValueError("components must cover the smaller matrix dimension")
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    # Equal expected factor norms, with product scale comparable to W.
    scale = (weight.norm() / math.sqrt(components)).sqrt()
    read = torch.randn(weight.shape[1], components, generator=generator,
                       device=weight.device, dtype=weight.dtype) * (scale / math.sqrt(weight.shape[1]))
    write = torch.randn(weight.shape[0], components, generator=generator,
                        device=weight.device, dtype=weight.dtype) * (scale / math.sqrt(weight.shape[0]))
    pieces = FreePieces(read, write)
    project_piece_sum_(weight, pieces)
    return pieces


def masked_output(h: torch.Tensor, pieces: FreePieces, keep: torch.Tensor) -> torch.Tensor:
    """Sum selected rank-one outputs; supports binary or fractional masks."""
    return ((h @ pieces.read) * keep.to(h.dtype)) @ pieces.write.T


def free_codes(h: torch.Tensor, g: torch.Tensor, pieces: FreePieces,
               output_basis: torch.Tensor | None = None) -> torch.Tensor:
    """(h r_c)(g l_c), or |h r_c| ||J l_c|| for full output Jacobians.

    Optional exact compression represents g_full = g @ output_basis.T.
    Writers remain full-dimensional, including directions outside this span.
    """
    write = pieces.write if output_basis is None else output_basis.T @ pieces.write
    read = h.to(pieces.read.dtype) @ pieces.read
    sensitivity = g.to(write.dtype) @ write
    if g.dim() == 2:
        return read * sensitivity
    return read.abs() * sensitivity.square().sum(dim=1).add(1e-30).sqrt()


@torch.no_grad()
def piece_singular_values(pieces: FreePieces, rank: int) -> torch.Tensor:
    """[C, k] singular values of each rank-k piece U_c R_c^T (descending), via k x k Gram matrices."""
    U = pieces.write.double().unflatten(1, (-1, rank)).permute(1, 0, 2)    # [C, d_out, k]
    R = pieces.read.double().unflatten(1, (-1, rank)).permute(1, 0, 2)     # [C, d_in, k]
    gu, gr = U.transpose(1, 2) @ U, R.transpose(1, 2) @ R
    lam, vec = torch.linalg.eigh(gu)
    root = vec * lam.clamp_min(0).sqrt().unsqueeze(1)                     # gu = root root^T
    return torch.linalg.eigvalsh(root.transpose(1, 2) @ gr @ root).clamp_min(0).sqrt().flip(-1)
