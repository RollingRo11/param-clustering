"""Llama-3.2-1B plumbing for the geo-attribution pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PARAM_DECOMP_ROOT = Path(os.environ.get("PARAM_DECOMP_ROOT",
                                        "/workspace/param-decomp"))
if PARAM_DECOMP_ROOT.exists():
    sys.path.insert(0, str(PARAM_DECOMP_ROOT))

# Keep model and compiler caches on persistent, executable storage. Some
# runpods mount /dev/shm noexec, which lets Inductor write kernels but not load
# the generated shared objects.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
# /dev/shm is noexec on the benchmark host; Inductor loads compiled .so files.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(PROJECT_ROOT / ".torchinductor_cache"))

import numpy as np
import torch
from torch import nn

import geo67
from collection_runtime import install_llama_gim

MODEL_ID = os.environ.get("GEO_MODEL_ID", "unsloth/Llama-3.2-1B")
DEFAULT_MODEL_REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
MODEL_REVISION = os.environ.get("GEO_MODEL_REVISION", DEFAULT_MODEL_REVISION)
SHM_ROOT = Path(os.environ.get("GEO_ATTRIBUTION_ARTIFACT_ROOT",
                               "/dev/shm/geo1b"))
BIN_PATH = Path(os.environ.get("GEO_ATTRIBUTION_DATA_PATH",
                               str(SHM_ROOT / "pile_llama_u32.bin")))


class LogitsWrapper(nn.Module):
    def __init__(self, hf: nn.Module):
        super().__init__()
        self.hf = hf

    def forward(self, idx):
        return self.hf(input_ids=idx, use_cache=False).logits


def load_target_1b(device: str) -> nn.Module:
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        attn_implementation="sdpa")
    hf.config.use_cache = False
    target = LogitsWrapper(hf).to(device).eval()
    geo67.MODULES[:] = [
        f"hf.model.layers.{i}.{sub}"
        for i in range(hf.config.num_hidden_layers)
        for sub in ("self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    for parameter in target.parameters():
        parameter.requires_grad_(True)
    return target


def model_identity() -> dict[str, str]:
    """Resolve the immutable Hub revision used by reusable fingerprints."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    revision = getattr(config, "_commit_hash", None) or MODEL_REVISION
    return {"id": MODEL_ID, "revision": revision}


def apply_gim_1b(target: nn.Module, tau: float = 2.0):
    install_llama_gim(target, tau, unpadded_causal=True)


class TokenBatchLoader:
    """Deterministic, resumable batches over a flat uint32 token stream.

    ``sequential`` order is the production setting for billion-token runs: all
    ranks consume disjoint rows from the same global batch and wrap only after
    the complete token file has been traversed. ``random`` preserves the old
    sampling behavior for matched benchmarks and legacy collections.
    """

    def __init__(self, batch_size: int, seq_len: int, rank: int, world: int,
                 seed: int, *, data_path: Path, synthetic: bool,
                 vocab_size: int | None, order: str):
        if batch_size % world:
            raise ValueError(f"batch_size={batch_size} must divide world={world}")
        if order not in {"random", "sequential"}:
            raise ValueError(f"unknown data order: {order}")
        self.batch_size = batch_size
        self.local_batch = batch_size // world
        self.seq_len = seq_len
        self.rank = rank
        self.world = world
        self.order = order
        self.synthetic = synthetic
        self.batch_index = 0
        self.generator: torch.Generator | None = None
        self.tokens = None
        self.n_rows = 0
        self.vocab_size = vocab_size
        if synthetic:
            if vocab_size is None:
                raise ValueError("synthetic loader requires vocab_size")
            self.generator = torch.Generator().manual_seed(seed + 104729 * rank)
            return
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"missing token stream: {data_path}; run prep1b.py or pass "
                "--synthetic_data for a throughput-only benchmark")
        self.tokens = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.n_rows = len(self.tokens) // seq_len
        if self.n_rows == 0:
            raise ValueError(
                f"token stream {data_path} is shorter than seq_len={seq_len}")
        if order == "random":
            # All ranks draw the same global row ids and select disjoint slices.
            self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.synthetic:
            self.batch_index += 1
            return torch.randint(
                self.vocab_size, (self.local_batch, self.seq_len),
                generator=self.generator)
        if self.order == "random":
            rows = torch.randint(self.n_rows, (self.batch_size,),
                                 generator=self.generator)
        else:
            start = (self.batch_index * self.batch_size) % self.n_rows
            rows = (torch.arange(self.batch_size, dtype=torch.int64) + start) \
                .remainder(self.n_rows)
        mine = rows[self.rank * self.local_batch:
                    (self.rank + 1) * self.local_batch].tolist()
        batch = np.stack([
            np.asarray(self.tokens[r * self.seq_len:(r + 1) * self.seq_len])
            for r in mine
        ]).astype(np.int64)
        self.batch_index += 1
        return torch.from_numpy(batch)

    def state_dict(self) -> dict:
        state = {"batch_index": self.batch_index, "order": self.order}
        if self.generator is not None:
            state["generator_state"] = self.generator.get_state()
        return state

    def load_state_dict(self, state: dict):
        if state.get("order", self.order) != self.order:
            raise ValueError("loader checkpoint data order does not match")
        self.batch_index = int(state["batch_index"])
        if self.generator is not None and "generator_state" in state:
            self.generator.set_state(state["generator_state"])


def make_loader_1b(batch_size: int, seq_len: int, rank: int, world: int,
                   split: str, seed: int, *, data_path: Path = BIN_PATH,
                   synthetic: bool = False, vocab_size: int | None = None,
                   order: str = "random"):
    """Return a resumable rank-sharded token batch iterator."""
    del split
    return TokenBatchLoader(
        batch_size, seq_len, rank, world, seed, data_path=data_path,
        synthetic=synthetic, vocab_size=vocab_size, order=order)


geo67.OUT_ROOT = SHM_ROOT
geo67.load_target = load_target_1b
geo67.apply_gim = apply_gim_1b
geo67.GatedRunner.SOFT_CHUNK = 16

# Older pipeline stages import the loader through param-decomp. The fast
# collector calls make_loader_1b directly and therefore no longer requires
# that separate repository merely to benchmark or collect 1B fingerprints.
try:
    import nano_param_decomp.pile_4L as _p4l  # noqa: E402
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("nano_param_decomp"):
        raise
else:
    _p4l.make_loader = make_loader_1b

if __name__ == "__main__":
    geo67.main()
