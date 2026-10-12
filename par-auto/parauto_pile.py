"""4L-Pile plumbing and randomized full-output Jacobian caches.

The sensor measures the effect of a component at one module/token use on ALL
logits in that context. Independent Rademacher VJPs estimate its squared RMS
effect without materializing a vocabulary-by-activation Jacobian.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'geo-attribution'))
sys.path.insert(0, str(HERE.parent))
import sensor_study67 as target_source
import parauto_core as core
import parauto_structured as structured
from parauto_target import atomic_json, atomic_save, fingerprint
import parauto_tracking as tracking

MODULES = tuple(target_source.MODULES)
TARGET_SHA = '650f2eb603fe9fe649671c0c89b86746655bc49c96aa1ad00fbb4c1bf3c6c3b6'


def load_model(checkpoint, device='cuda'):
    if fingerprint(checkpoint) != TARGET_SHA: raise ValueError('Expected the verified 4-layer Pile target')
    target_source.CKPT = Path(checkpoint)
    model = target_source.load67(device, 'plain').eval().requires_grad_(False)
    assert len(model.h) == 4 and len(MODULES) == 24
    assert model.cfg['n_embd'] == 768 and model.cfg['n_intermediate'] == 3072
    assert sum(model.get_submodule(n).weight.numel() for n in MODULES) == 28311552
    return model


def sample_blocks(path, count, context, start, stop, seed):
    tokens = np.memmap(path, dtype=np.uint16, mode='r')
    if not 0 <= start < stop <= len(tokens): raise ValueError('Token range outside corpus')
    # Disjoint aligned blocks; no duplicate or overlapping cache contexts.
    available = (stop-start) // (context+1)
    if count > available: raise ValueError('Not enough disjoint blocks')
    indices = np.random.default_rng(seed).choice(available, count, replace=False)
    offsets = start + indices * (context+1)
    ids = torch.stack([torch.from_numpy(np.array(tokens[o:o+context+1], dtype=np.int64)) for o in offsets])
    return ids, offsets.tolist()


def capture_probes(model, ids, probes, generator):
    """True VJPs of the unmodified FP32 model, including causal attention.

    A local rank-one intervention has scalar probe effect
    (h_t r_c) (g_{q,t} u_c). E_q[effect^2] is the full-context logit RMS squared.
    It is not an exact Jacobian norm, a next-token loss gradient, or a sum of
    shared parameter interventions over all context positions.
    """
    if probes < 1: raise ValueError('Need at least one Jacobian probe')
    pre, post, hooks = {}, {}, []
    def hook(name):
        def capture(_module, args, output): pre[name], post[name] = args[0], output
        return capture
    try:
        for name in MODULES: hooks.append(model.get_submodule(name).register_forward_hook(hook(name)))
        with torch.enable_grad():
            embed = model.wte(ids).detach().requires_grad_()
            logits = model(ids, embed=embed)
            cot = {name: [] for name in MODULES}
            for q in range(probes):
                direction = torch.randint(0, 2, logits.shape, device=ids.device, generator=generator, dtype=torch.int8)
                direction = direction.to(logits.dtype).mul_(2).sub_(1).div_(math.sqrt(logits.shape[1]*logits.shape[2]))
                grads = torch.autograd.grad(logits, [post[n] for n in MODULES], grad_outputs=direction, retain_graph=q < probes-1)
                for name, g in zip(MODULES, grads): cot[name].append(g.detach())
            entries = {}
            for name in MODULES:
                h = pre[name].detach().reshape(-1, pre[name].shape[-1]).bfloat16()
                g = torch.stack(cot[name], dim=-2).reshape(-1, probes, post[name].shape[-1]).bfloat16()
                if not (torch.isfinite(h).all() and torch.isfinite(g).all()): raise ValueError(f'Nonfinite BF16 cache: {name}')
                value = h.float() @ model.get_submodule(name).weight.detach().T
                net = (g.float() * value[:, None, :]).sum(-1).square().mean(-1).sqrt()
                entries[name] = dict(structured=True, kind='vjp', h=h, events={'cot': g}, shared={},
                                     net_rms=net, output_count=probes)
            return entries, logits.detach()
    finally:
        for h in hooks: h.remove()


def make_cache(model, checkpoint, tokens, output, *, blocks=32, context=512, probes=4,
               train_stop=100000000, seed=0):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists(): raise FileExistsError('Cache already exists; reuse it or choose a fresh directory')
    ids, offsets = sample_blocks(tokens, blocks, context, 0, train_stop, seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed+101)
    buckets = {n: [] for n in MODULES}
    ce_total = 0.
    for i, block in enumerate(ids):
        entries, logits = capture_probes(model, block[None, :-1].to(device), probes, generator)
        ce_total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), block[1:].to(device)))
        for name, e in entries.items():
            buckets[name].append({'h': e['h'].cpu(), 'cot': e['events']['cot'].cpu(), 'net': e['net_rms'].cpu()})
        print(f'CACHE {i+1}/{blocks}', flush=True)
        tracking.log_metrics({'contexts_completed': i+1, 'contexts_total': blocks}, prefix='cache')
        del entries, logits
    files, shapes = {}, {}
    permutation = torch.randperm(blocks*context, generator=torch.Generator().manual_seed(seed+202))
    for index, name in enumerate(MODULES):
        rows = buckets.pop(name)
        entry = dict(structured=True, kind='vjp', h=torch.cat([r['h'] for r in rows])[permutation],
                     events={'cot': torch.cat([r['cot'] for r in rows])[permutation]}, shared={},
                     net_rms=torch.cat([r['net'] for r in rows])[permutation], output_count=probes)
        file = output/f'matrix_{index:02d}.pt'; atomic_save(entry, file)
        files[name] = {'file': file.name, 'sha256': fingerprint(file)}
        shapes[name] = list(model.get_submodule(name).weight.shape)
    atomic_save({'ids': ids, 'offsets': offsets, 'event_permutation': permutation}, output/'inputs.pt')
    manifest = dict(format='parauto_pile_vjp_cache_v1', target_sha256=fingerprint(checkpoint),
                    model_source_sha256=fingerprint(target_source.__file__), modules=list(MODULES), shapes=shapes,
                    token_file=str(Path(tokens).resolve()), token_file_bytes=Path(tokens).stat().st_size,
                    token_file_sha256=fingerprint(tokens), inputs_sha256=fingerprint(output/'inputs.pt'),
                    seed=seed, blocks=blocks, context=context, probes=probes, events=blocks*context,
                    normalization_order='shared seeded permutation of all cached token positions across contexts',
                    train_range=[0, train_stop], offsets=offsets, cache_precision='bf16; net_rms fp32',
                    sensor='independent Rademacher VJPs / sqrt(context*vocabulary); full-context logit RMS sketch per local module/token use',
                    target_forward_precision='fp32', cache_mean_ce=ce_total/blocks, matrices=files)
    atomic_json(manifest, output/'manifest.json')
    return manifest


def cache_manifest(path, checkpoint):
    import json
    path = Path(path); manifest = json.loads((path/'manifest.json').read_text())
    if manifest['target_sha256'] != fingerprint(checkpoint): raise ValueError('Wrong target for cache')
    if manifest['model_source_sha256'] != fingerprint(target_source.__file__): raise ValueError('Target implementation changed since cache collection')
    if manifest['modules'] != list(MODULES): raise ValueError('Cache module scope mismatch')
    if fingerprint(path/'inputs.pt') != manifest['inputs_sha256']: raise ValueError('Cache input records changed')
    return manifest


def load_cache(path, checkpoint, device):
    path = Path(path); manifest = cache_manifest(path, checkpoint)
    entries = []
    for name in MODULES:
        file = path/manifest['matrices'][name]['file']
        if fingerprint(file) != manifest['matrices'][name]['sha256']: raise ValueError(f'Corrupt cache: {name}')
        entry = torch.load(file, map_location=device, weights_only=True)
        if entry['h'].dtype not in (torch.float16, torch.bfloat16) or entry['events']['cot'].dtype != entry['h'].dtype: raise ValueError('Expected 16-bit cached activations')
        entries.append(entry)
    return entries, manifest


@contextmanager
def masked_model(model, pieces, masks):
    originals = []
    try:
        for name, p, mask in zip(MODULES, pieces, masks, strict=True):
            module = model.get_submodule(name); originals.append((module, module.forward))
            def forward(x, p=p, mask=mask, bias=module.bias):
                out = ((x @ p.read) * mask.to(x.dtype)) @ p.write.T
                return out if bias is None else out + bias
            module.forward = forward
        yield
    finally:
        for module, forward in originals: module.forward = forward


@torch.no_grad()
def evaluate(model, pieces, blocks, probes=4, seed=920091, budgets=None):
    total = sum(p.count for p in pieces)
    budgets = sorted(set([0, total] + ([128, 512, 1024, 4096] if budgets is None else list(budgets))))
    budgets = [k for k in budgets if 0 <= k <= total]
    sums = {k: {'kl': 0., 'ce': 0.} for k in budgets}
    device = next(model.parameters()).device; gen = torch.Generator(device=device).manual_seed(seed)
    baseline_ce, tokens = 0., 0
    for block in blocks:
        ids, labels = block[None, :-1].to(device), block[1:].to(device)
        entries, full = capture_probes(model, ids, probes, gen)
        scores = [structured.scores(entries[n], p) for n, p in zip(MODULES, pieces)]
        flat = torch.cat(scores, -1)
        order = flat.argsort(dim=-1, descending=False, stable=True)
        ranks = torch.empty_like(order).scatter_(-1, order, torch.arange(total, device=device).expand_as(order))
        baseline_ce += float(F.cross_entropy(full.reshape(-1, full.shape[-1]), labels, reduction='sum'))
        for keep in budgets:
            masks = [m.unsqueeze(0) for m in (ranks >= total-keep).split([p.count for p in pieces], -1)]
            with masked_model(model, pieces, masks): out = model(ids)
            # FP64 metric reduction avoids cancellation near all-kept equality.
            lp = full.double().log_softmax(-1); lq = out.double().log_softmax(-1)
            kl = (lp.exp()*(lp-lq)).sum(-1).clamp_min(0).sum()
            sums[keep]['kl'] += float(kl)
            sums[keep]['ce'] += float(F.cross_entropy(out.reshape(-1, out.shape[-1]), labels, reduction='sum'))
        tokens += labels.numel()
    return dict(tokens=tokens, target_ce=baseline_ce/tokens, budgets={str(k): {n: v/tokens for n, v in row.items()} for k, row in sums.items()},
                sensor='fresh independent full-context Rademacher VJP sketch; frozen per-token ranking across all 24 matrices',
                mask='binary local module/token masks, least score removed first; actual nonlinear full-model forwards',
                evaluation_precision='fp32 model, bf16 sensor cache, fp64 KL reduction')
