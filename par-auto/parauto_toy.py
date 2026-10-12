"""Frozen toy forwards, reference Jacobians, and component diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent
for source_directory in (HERE, SOURCE_ROOT):
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

import parauto_core as core  # noqa: E402

EPSILON = 1e-4


def matrices(target) -> list[tuple[str, torch.Tensor]]:
    """The four decomposed matrices in execution order, as (name, W [d_out, d_in])."""
    out = []
    for index, layer in enumerate(target.layers):
        out.append((f"layer{index}.mlp_in", layer.mlp_in.weight.detach()))
        out.append((f"layer{index}.mlp_out", layer.mlp_out.weight.detach()))
    return out


def forward_with_pieces(target, x, pieces=None, keeps=None):
    """Exact per-event sum of kept free rank-one pieces."""
    def apply(which, source, weight):
        if pieces is None or keeps is None or keeps[which] is None:
            return source @ weight.T
        return core.masked_output(source, pieces[which], keeps[which])

    h = x @ target.W_E
    inputs, outputs = [], []
    for index, layer in enumerate(target.layers):
        inputs.append(h)
        pre = apply(2 * index, h, layer.mlp_in.weight)
        outputs.append(pre)
        hidden = pre.relu()
        inputs.append(hidden)
        write = apply(2 * index + 1, hidden, layer.mlp_out.weight)
        outputs.append(write)
        h = h + write
    return h @ target.W_U, inputs, outputs


def cache_events(target, x, chunk=256):
    """Cache true output cotangents, compressed in an exact common row span.

    For a residual-stream write, every downstream Jacobian row belongs to the
    span of W_U columns and subsequent mlp_in rows. This compression retains
    g for arbitrary free writers; ghat = g W would lose required information.
    """
    names = [name for name, _ in matrices(target)]
    weights = [weight for _, weight in matrices(target)]
    bases = []
    for which, weight in enumerate(weights):
        basis = None
        if which % 2 == 1:
            span = [target.W_U]
            span.extend(layer.mlp_in.weight.T for layer in target.layers[which // 2 + 1:])
            span = torch.cat(span, dim=1)
            if span.shape[1] < weight.shape[0]:
                basis = torch.linalg.qr(span, mode="reduced").Q
        bases.append(basis)
    store = {name: {"h": [], "cot": []} for name in names}
    outputs_all = []
    for start in range(0, x.shape[0], chunk):
        with torch.enable_grad():
            leaf = x[start:start + chunk].detach().clone().requires_grad_(True)
            output, inputs, module_outputs = forward_with_pieces(target, leaf)
            differentiated = module_outputs
            jac = [[] for _ in differentiated]
            for j in range(output.shape[1]):
                grads = torch.autograd.grad(output[:, j].sum(), differentiated, retain_graph=True)
                for which, g in enumerate(grads):
                    jac[which].append(g.detach())
        outputs_all.append(output.detach())
        for which, name in enumerate(names):
            g = torch.stack(jac[which], dim=1)                       # [n, P, d_out]
            basis = bases[which]
            store[name]["h"].append(inputs[which].detach())
            store[name]["cot"].append(g if basis is None else g @ basis)
    cache = {}
    for which, name in enumerate(names):
        cache[name] = {"h": torch.cat(store[name]["h"]), "cot": torch.cat(store[name]["cot"]),
                       "side": "g", "output_basis": bases[which]}
    return {"x": x, "output": torch.cat(outputs_all), "matrices": cache}


def piece_codes(cache, pieces, target):
    result = []
    for which, (name, weight) in enumerate(matrices(target)):
        entry = cache["matrices"][name]
        item = pieces[which]
        result.append(torch.cat([core.free_codes(entry["h"][start:start + 128],
                                                entry["cot"][start:start + 128], item,
                                                entry["output_basis"])
                                 for start in range(0, len(entry["h"]), 128)]))
    return result


@torch.no_grad()
def first_order_prediction(cache, pieces, keeps, target, batch=1024):
    """Predicted change -sum_c (1-keep_c) J l_c (r_c . h)."""
    predicted = torch.zeros_like(cache["output"])
    for which, (name, weight) in enumerate(matrices(target)):
        entry = cache["matrices"][name]
        item = pieces[which]
        for start in range(0, predicted.shape[0], batch):
            stop = start + batch
            dropped = core.masked_output(entry["h"][start:stop], item,
                                          1.0 - keeps[which][start:stop])
            if entry["output_basis"] is not None:
                dropped = dropped @ entry["output_basis"]
            predicted[start:stop] -= torch.einsum("npd,nd->np", entry["cot"][start:stop], dropped)
    return predicted


def masked_stat(values, mask, kind):
    """Mean or median over the masked rows; None when the mask is empty (JSON-safe)."""
    if int(mask.sum()) == 0:
        return None
    selected = values[mask]
    return float(selected.mean() if kind == "mean" else selected.median())


@torch.no_grad()
def feature_recovery(target, pieces, k, device):
    """Isolated single-feature events: which pieces fire, and how consistently."""
    features = target.config["n_features"]
    amplitudes = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)
    x = torch.zeros(features * len(amplitudes), features, device=device)
    feature_of = torch.arange(features, device=device).repeat_interleave(len(amplitudes))
    x[torch.arange(x.shape[0], device=device), feature_of] = amplitudes.repeat(features)
    cache = cache_events(target, x)
    a = piece_codes(cache, pieces, target)
    report = {}
    for which, (name, weight) in enumerate(matrices(target)):
        top = a[which].abs().argmax(dim=1).view(features, len(amplitudes))
        mode = top.mode(dim=1).values
        consistency = (top == mode[:, None]).float().mean(dim=1)
        keep = core.topk_mask(a[which].abs(), k)
        active = keep.view(features, len(amplitudes), -1).any(dim=1).sum(dim=1)
        responsive = a[which].abs().amax(dim=1).view(features, len(amplitudes)) > 1e-10
        positive_top = top[:, 2:]
        positive_active = responsive[:, 2:].all(dim=1)
        row = {"features_with_nonzero_code": int(responsive.any(dim=1).sum()),
               "positive_distinct_top_pieces": int(positive_top[:, -1][positive_active].unique().numel()),
               "positive_features_with_consistent_top_piece": int(
                   ((positive_top[:, 0] == positive_top[:, 1]) & positive_active).sum()),
               "distinct_top_pieces_over_features": int(mode[responsive.any(dim=1)].unique().numel()),
               "mean_top_piece_consistency": float(consistency.mean()),
               "features_with_consistent_top_piece": int(((consistency == 1) & responsive.all(dim=1)).sum()),
               "mean_pieces_active_per_feature": float(active.float().mean()),
               "top_piece_per_feature": mode.tolist()}
        if name.endswith("mlp_in"):
            cosine = (F.normalize(target.W_E, dim=1) @ F.normalize(
                pieces[which].read, dim=0)).abs()   # [features, d_in]
            best, index = cosine.max(dim=1)
            row["mean_best_cosine_to_embedding"] = float(best.mean())
            row["features_with_cosine_over_0.9"] = int((best > 0.9).sum())
            row["best_piece_matches_top_piece"] = int((index == mode).sum())
        report[name] = row
    return report

