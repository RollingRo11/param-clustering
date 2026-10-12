"""Exact task-loss gradients for the frozen residual toy.

The verified target's labels are x + relu(x), with equal feature importance.
Differentiate each example's mean squared task error, not its distance from
the frozen model's own output (whose derivative would vanish).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import parauto_toy as toy
from parauto_target import fingerprint


SENSOR = "scalar_task_mse"
LOSS = "mean_features((output - (x + relu(x))) ** 2), separately per example"


def verify_label_definition(target_path):
    root = Path(target_path).parent
    provenance = json.loads((root / 'provenance.json').read_text())
    cfg = provenance['target_training']
    coeffs = json.loads((root / 'label_coeffs.json').read_text())
    if not (cfg['label_type'] == 'act_plus_resid' and cfg['loss_type'] == 'readoff'
            and cfg['importance_val'] == 1 and cfg['use_trivial_label_coeffs']
            and cfg['resid_mlp_config']['act_fn_name'] == 'relu'
            and coeffs == [1.] * cfg['resid_mlp_config']['n_features']):
        raise ValueError('Scalar sensor expects the verified x + relu(x) task')
    return {name: fingerprint(root / name) for name in ('provenance.json', 'label_coeffs.json')}


def task_loss(output, x):
    return (output - (x + x.relu())).square().mean(-1)


def cache_events(target, x, chunk=256):
    names_weights = toy.matrices(target)
    store = {name: {'h': [], 'g': []} for name, _ in names_weights}
    outputs, losses = [], []
    for start in range(0, len(x), chunk):
        with torch.enable_grad():
            leaf = x[start:start+chunk].detach().clone().requires_grad_(True)
            output, inputs, module_outputs = toy.forward_with_pieces(target, leaf)
            # Sum over independent examples: no batch-size-dependent scaling.
            per_example = task_loss(output, leaf.detach())
            grads = torch.autograd.grad(per_example.sum(), module_outputs)
        outputs.append(output.detach())
        losses.append(per_example.detach())
        for (name, _), h, g in zip(names_weights, inputs, grads, strict=True):
            store[name]['h'].append(h.detach())
            store[name]['g'].append(g.detach())
    entries = {}
    for name, weight in names_weights:
        h, g = [torch.cat(store[name][key]) for key in ('h', 'g')]
        entries[name] = dict(kind='scalar_loss', structured=True, h=h, events={'g': g},
                             shared={}, output_count=1,
                             net_rms=((h @ weight.T) * g).sum(-1).abs())
    return dict(x=x, output=torch.cat(outputs), task_loss=torch.cat(losses), matrices=entries,
                sensor=SENSOR, loss_definition=LOSS)


def signed_scores(entry, pieces):
    return (entry['h'] @ pieces.read) * (entry['events']['g'] @ pieces.write)
