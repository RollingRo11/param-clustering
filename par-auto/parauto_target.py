"""Frozen residual toy and checkpoint utilities used by the parameter autoencoder."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ToyConfig:
    n_features: int = 100
    d_embed: int = 1000
    d_mlp: int = 25
    n_layers: int = 2


class ToyMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp_in = nn.Linear(cfg.d_embed, cfg.d_mlp, bias=False)
        self.mlp_out = nn.Linear(cfg.d_mlp, cfg.d_embed, bias=False)

    def forward(self, h):
        return self.mlp_out(self.mlp_in(h).relu())


class ResidualToy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = asdict(cfg)
        embedding = F.normalize(torch.randn(cfg.n_features, cfg.d_embed), dim=-1)
        self.register_buffer("W_E", embedding)
        self.register_buffer("W_U", embedding.T.clone())
        self.layers = nn.ModuleList([ToyMLP(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x):
        h = x @ self.W_E
        for layer in self.layers:
            h = h + layer(h)
        return h @ self.W_U


def load_target(path, device):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = ResidualToy(ToyConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False)


def sample(n, features, generator, device, probability=0.01):
    values = 2 * torch.rand(n, features, generator=generator, device=device) - 1
    present = torch.rand(n, features, generator=generator, device=device) < probability
    return values * present


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
    import parauto_tracking as tracking
    tracking.log_result(value, path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()
