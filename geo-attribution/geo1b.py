"""Llama-3.2-1B plumbing for the geo-attribution pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")
sys.path.insert(0, "/workspace/param-decomp")
os.environ.setdefault("HF_HOME", "/dev/shm/hf")
# /dev/shm is noexec on the benchmark host; Inductor loads compiled .so files.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      "/workspace/circuit-decomp/.torchinductor_cache")

import numpy as np
import torch
from torch import nn

import geo67
from collection_runtime_v2 import install_llama_gim

MODEL_ID = "unsloth/Llama-3.2-1B"
SHM_ROOT = Path("/dev/shm/geo1b")
BIN_PATH = SHM_ROOT / "pile_llama_u32.bin"


class LogitsWrapper(nn.Module):
    def __init__(self, hf: nn.Module):
        super().__init__()
        self.hf = hf

    def forward(self, idx):
        return self.hf(input_ids=idx, use_cache=False).logits


def load_target_1b(device: str) -> nn.Module:
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="sdpa")
    hf.config.use_cache = False
    target = LogitsWrapper(hf).to(device)
    geo67.MODULES[:] = [
        f"hf.model.layers.{i}.{sub}"
        for i in range(hf.config.num_hidden_layers)
        for sub in ("self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    for parameter in target.parameters():
        parameter.requires_grad_(True)
    return target


def apply_gim_1b(target: nn.Module, tau: float = 2.0):
    install_llama_gim(target, tau)


def make_loader_1b(batch_size: int, seq_len: int, rank: int, world: int,
                   split: str, seed: int):
    """Yield this rank's rows from the pretokenized Pile stream."""
    toks = np.memmap(BIN_PATH, dtype=np.uint32, mode="r")
    n_rows = len(toks) // seq_len
    local_batch = batch_size // world
    gen = torch.Generator().manual_seed(seed)
    while True:
        rows = torch.randint(n_rows, (batch_size,), generator=gen)
        mine = rows[rank * local_batch:(rank + 1) * local_batch].tolist()
        batch = np.stack([np.asarray(toks[r * seq_len:(r + 1) * seq_len])
                          for r in mine]).astype(np.int64)
        yield torch.from_numpy(batch)


geo67.OUT_ROOT = SHM_ROOT
geo67.load_target = load_target_1b
geo67.apply_gim = apply_gim_1b
geo67.GatedRunner.SOFT_CHUNK = 16
import nano_param_decomp.pile_4L as _p4l  # noqa: E402
_p4l.make_loader = make_loader_1b

if __name__ == "__main__":
    geo67.main()
