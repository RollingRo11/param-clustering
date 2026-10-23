"""SimpleStories counterpart of common.py: the same names for the 2-layer SimpleStories target, its token rows and
piece counts. common.py re-exports these when MATRYOSHKA_TARGET=simplestories, so every script that imports common
(matryoshka67.py, its tests, the evals) runs on this target unchanged.

Target: Goodfire's goodfire/spd/runs/gf6rbga0 (the model VPD's ss_llama_simple_mlp-2L decomposes), 7,104 pieces in VPD's
per-matrix counts. Tokens: $SS_DATA/{train,val}.uint16, rows of 513 tokens with VPD's SimpleStories tokenization
(param_decomp.data.tokenize_and_concatenate, lowercased; manifest.json records the recipe and checksums)."""
import hashlib
import json
import os
from pathlib import Path

__all__ = ['DATA', 'TARGET_SHA', 'CONTEXT', 'BATCH', 'MODULES', 'PIECE_COUNTS', 'EXPERIMENT', 'TRAINING_CONFIG',
           'credentials', 'target', 'tokens', 'batch', 'train_updates']

DATA = Path(os.environ.get('SS_DATA', '/workspace/paired/data_ss'))
TARGET_SHA = 'a98273427d063db4ab23f56e14b33a32e674d06d4161a47a89fd676a0988d7af'
CONTEXT, BATCH = 512, 64
MODULES = tuple(f'h.{i}.{m}' for i in range(2) for m in
                ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.c_fc', 'mlp.down_proj'))
PIECE_COUNTS = dict(q_proj=288, k_proj=288, v_proj=384, o_proj=480, c_fc=1152, down_proj=960)   # VPD ss_llama_simple_mlp-2L C
EXPERIMENT = 'matryoshka_ss_mainline'
# The 67M control's training config (output/gim_1h/plan.json there), as the earlier SimpleStories runs inlined it.
TRAINING_CONFIG = dict(steps=8700, batch=32768, lr=0.001, lr_min=1e-05, seed=0, log_interval=100, tau=0.001, p_end=0.5,
                       lp_epsilon=1e-06, normalization_events=256, precision='bf16', projection_precision='fp32')


def credentials():
    os.environ.setdefault('WANDB_ENTITY', 'rohan-kathuria-neu')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def target(device='cuda'):
    import torch
    import yaml
    from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
    path = DATA/'target_model_step_99999.pt'
    assert sha(path) == TARGET_SHA
    cfg = yaml.safe_load((DATA/'target_model_config.yaml').read_text())
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(model_type='LlamaSimpleMLP', **cfg))
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
    return model.to(device).eval().requires_grad_(False)


_verified = set()


def tokens(split='train'):
    import numpy as np
    manifest = json.loads((DATA/'manifest.json').read_text())
    info = manifest[split]
    path = DATA/info['file']
    assert path.stat().st_size == info['rows']*513*2
    if split not in _verified:
        assert sha(path) == info['sha256']; _verified.add(split)
    return np.memmap(path, mode='r', dtype=np.uint16, shape=(info['rows'], 513)), manifest


def train_updates():
    """Full batches in the train split (18,522 = 607M tokens): one epoch, the longest run that never repeats a row."""
    return json.loads((DATA/'manifest.json').read_text())['train']['rows'] // BATCH


def batch(data, step, device='cuda'):
    import numpy as np
    import torch
    rows = np.array(data[step*BATCH:(step+1)*BATCH], dtype=np.int64)
    assert len(rows) == BATCH, f'update {step} runs past the {len(data)} train rows (at most {len(data) // BATCH} updates)'
    return torch.from_numpy(rows).to(device)
