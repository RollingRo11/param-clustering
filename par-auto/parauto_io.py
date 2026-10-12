"""Checkpoint loading, including read-only compatibility with historical runs."""
from pathlib import Path

import parauto_core as core
import parauto_toy as toy
import torch


def method_directory(root):
    root = Path(root)
    for name in ("method", "joint", "B"):
        if any((root / name / file).exists() for file in ("result.json", "profile.json")):
            return root / name
    return root


def evaluation_payload(result):
    # Historical JSON is an immutable experiment record, not an active arm.
    return result["arms"]["B_lp"] if "arms" in result else result


def load_pieces(root, target, device):
    root = Path(root)
    if (root / "pieces.pt").exists():
        saved = torch.load(root / "pieces.pt", map_location=device, weights_only=True)["pieces"]
    else:
        directory = root if (root / "pieces_layer0.mlp_in.pt").exists() else root / "B_lp"
        saved = [torch.load(directory / f"pieces_{name}.pt", map_location=device, weights_only=True)
                 for name, _ in toy.matrices(target)]
    if len(saved) != len(toy.matrices(target)):
        raise ValueError("Checkpoint matrix count mismatch")
    pieces = [core.FreePieces(item["read"], item["write"]) for item in saved]
    for (name, weight), item in zip(toy.matrices(target), pieces, strict=True):
        if core.reconstruction_error(weight, item) > 1e-5:
            raise ValueError(f"Checkpoint matrix sum mismatch: {name}")
    return pieces
