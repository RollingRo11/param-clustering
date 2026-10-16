"""Shared provenance, token order, target, and durable storage for paired runs."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
HERE = Path(__file__).resolve().parent
TARGET_SHA = '650f2eb603fe9fe649671c0c89b86746655bc49c96aa1ad00fbb4c1bf3c6c3b6'
VPD_COMMIT = '74146b555f3eb3cd2bc639c3789fd3b7a099094a'
DATASET = 'danbraunai/pile-uncopyrighted-tok-shuffled'
POSITIONS = 1_000_000_000
CONTEXT, BATCH = 512, 64
STEPS = (POSITIONS + CONTEXT*BATCH-1)//(CONTEXT*BATCH)
MODULES = tuple(f'h.{i}.{m}' for i in range(4) for m in
                ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.c_fc', 'mlp.down_proj'))
R2_PREFIX = 'parauto/vpd_matched_1b_20260910'


def credentials():
    path = ROOT/'secrets.json'
    if path.exists():
        for k, v in json.loads(path.read_text()).items(): os.environ[k] = v
    os.environ.setdefault('WANDB_PROJECT', 'param-autoenc')
    os.environ.setdefault('WANDB_ENTITY', 'rohan-kathuria-neu')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HOME', str(ROOT/'hf'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def save_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def s3():
    import boto3
    credentials()
    return boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT'],
                         aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto')


def upload(path, relative):
    client = s3(); path = Path(path); digest = sha(path)
    key = R2_PREFIX+'/'+relative
    client.upload_file(str(path), 'param-clustering', key, ExtraArgs={'Metadata': {'sha256': digest}})
    info = client.head_object(Bucket='param-clustering', Key=key)
    assert info['ContentLength'] == path.stat().st_size and info['Metadata']['sha256'] == digest
    return dict(key=key, bytes=info['ContentLength'], sha256=digest)


def prepare_target():
    path = ROOT/'data/model_step_99999.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        s3().download_file('param-clustering', 'cofac67/target/model_step_99999.pt', str(path))
    assert sha(path) == TARGET_SHA
    # Verify both write and read credentials before any training starts.
    body = os.urandom(64)
    key = R2_PREFIX+'/storage_probe.bin'
    client = s3(); client.put_object(Bucket='param-clustering', Key=key, Body=body)
    assert client.get_object(Bucket='param-clustering', Key=key)['Body'].read() == body
    save_json({'target_sha256': TARGET_SHA, 'r2_roundtrip_verified': True}, ROOT/'output/preflight.json')
    return path


def target(device='cuda'):
    import torch
    import yaml
    from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig
    path = ROOT/'data/model_step_99999.pt'
    assert sha(path) == TARGET_SHA
    cfg = yaml.safe_load((HERE/'target_config.yaml').read_text())
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**cfg))
    state = torch.load(path, weights_only=True, map_location='cpu')
    state.setdefault('lm_head.weight', state['wte.weight'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def tokens(split='train'):
    import numpy as np
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    info = manifest[split]
    path = ROOT/'data'/info['file']
    assert path.stat().st_size == info['rows']*513*2
    assert sha(path) == info['sha256']
    return np.memmap(path, mode='r', dtype=np.uint16, shape=(info['rows'], 513)), manifest


def batch(data, step, device='cuda'):
    import numpy as np
    import torch
    return torch.from_numpy(np.array(data[step*BATCH:(step+1)*BATCH], dtype=np.int64)).to(device)


def cpu_tree(x):
    import torch
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    if isinstance(x, dict): return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return type(x)(cpu_tree(v) for v in x)
    return x


def checkpoint(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(cpu_tree(value), tmp)
    os.replace(tmp, path)

