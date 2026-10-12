"""Inspect every saved rank-one piece by exact single-piece ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import parauto_evaluate as experiment  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_io import load_pieces  # noqa: E402
from parauto_target import (  # noqa: E402
    atomic_json,
    atomic_save,
    fingerprint,
    load_target,  # noqa: E402
)


@torch.no_grad()
def ablated_outputs(target, x, pieces, which, indices):
    """One full-model forward per selected piece, vectorized over pieces.

    All other pieces are kept: subtract just l_c (r_c . h) at its actual
    matrix input, then recompute all downstream nonlinearities exactly.
    """
    reads, writes = pieces.read[:, indices], pieces.write[:, indices]
    h = (x @ target.W_E).unsqueeze(0).expand(len(indices), -1, -1)

    def remove(source):
        strength = torch.einsum("bnd,db->bn", source, reads)
        return strength.unsqueeze(-1) * writes.T.unsqueeze(1)

    for layer_index, layer in enumerate(target.layers):
        pre = h @ layer.mlp_in.weight.T
        if which == 2 * layer_index:
            pre = pre - remove(h)
        hidden = pre.relu()
        output = hidden @ layer.mlp_out.weight.T
        if which == 2 * layer_index + 1:
            output = output - remove(hidden)
        h = h + output
    return h @ target.W_U


def describe_profile(profile, reference_strength=None):
    """profile[f,c] is mean RMS effect for feature f and component c.

    Purity is squared-effect share across equally weighted isolated features.
    A strong selective piece has purity >=90% and at least 10% of that
    feature's total MLP contribution. Report all thresholds explicitly.
    """
    energy = profile.square()
    preferred = profile.argmax(0)
    purity = energy.amax(0) / energy.sum(0).clamp_min(1e-30)
    top = profile.argmax(1)
    feature = torch.arange(len(profile), device=profile.device)
    own = profile[feature, top]
    top_purity = energy[feature, top] / energy[:, top].sum(0).clamp_min(1e-30)
    strength = profile.amax(0)
    active = strength > 1e-8
    if reference_strength is not None:
        active &= strength >= 0.1 * reference_strength[preferred]
        active &= reference_strength[preferred] > 1e-8
    clean = active & (purity >= 0.9)
    matches = []
    for f in range(len(profile)):
        candidates = torch.where(clean & (preferred == f))[0]
        best = int(candidates[profile[f, candidates].argmax()]) if len(candidates) else None
        c = int(top[f])
        other = profile[:, c].clone()
        other[f] = 0
        matches.append({"feature": f, "top_piece": c, "top_piece_purity_for_this_feature": float(top_purity[f]),
                        "top_piece_effect_rms": float(own[f]),
                        "largest_other_feature_effect_rms": float(other.max()),
                        "top_piece_preferred_feature": int(preferred[c]),
                        "strong_selective_piece": best,
                        "selective_piece_purity": float(purity[best]) if best is not None else None,
                        "selective_piece_effect_rms": float(profile[f, best]) if best is not None else None})
    return {"distinct_top_pieces": int(top.unique().numel()),
            "features_whose_top_piece_prefers_them": int((preferred[top] == feature).sum()),
            "features_with_top_piece_purity_over_90pct": int((top_purity >= 0.9).sum()),
            "median_top_piece_purity": float(top_purity.median()),
            "mean_top_piece_purity": float(top_purity.mean()),
            "strong_selective_pieces": int(clean.sum()),
            "features_with_strong_selective_piece": int(preferred[clean].unique().numel()),
            "median_selective_piece_purity": float(purity[clean].median()) if clean.any() else None,
            "per_feature": matches}


import parauto_tracking as tracking

@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    tracking.start_cli(args)
    if fingerprint(args.target) != experiment.TARGET_SHA:
        raise ValueError("Expected verified 10k-step target")
    torch.backends.cuda.matmul.allow_tf32 = False
    target = load_target(args.target, args.device)
    root = Path(args.run)
    saved_result = json.loads((root / "result.json").read_text())
    if saved_result["target_sha256"] != experiment.TARGET_SHA:
        raise ValueError("Decomposition target mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    features = target.config["n_features"]
    amplitudes = torch.tensor([-1., 1.], device=args.device)
    x = (torch.eye(features, device=args.device)[:, None, :] * amplitudes[None, :, None]).flatten(0, 1)
    cache = toy.cache_events(target, x)
    full = cache["output"]
    baseline = x @ target.W_E @ target.W_U
    reference = (full - baseline).square().mean(-1).sqrt().reshape(features, 2).mean(-1)
    pieces = load_pieces(root, target, args.device)
    codes = experiment.all_scores(cache, pieces, target)
    result = {"method": "projected_lp", "target_sha256": experiment.TARGET_SHA,
              "features": features, "signed_probes": [-1, 1],
              "purity_definition": "share of squared mean signed-probe RMS effects belonging to one feature",
              "strong_definition": ">=10% of that feature's mean full MLP contribution RMS",
              "matrices": {}}
    for which, ((name, weight), p) in enumerate(zip(toy.matrices(target), pieces)):
        effects = []
        for start in range(0, p.count, 16):
            indices = torch.arange(start, min(start + 16, p.count), device=args.device)
            masked = ablated_outputs(target, x, p, which, indices)
            effects.append((masked - full).square().mean(-1).sqrt())
        exact = torch.cat(effects).reshape(p.count, features, 2).mean(-1).T
        first_order = codes[which].reshape(features, 2, p.count).mean(1)
        row = {"exact_ablation": describe_profile(exact, reference),
               "first_order": describe_profile(first_order, reference)}
        # This probes a component's raw read coordinate, regardless of the
        # downstream Jacobian or removal-induced nonlinearities.
        reads = (cache["matrices"][name]["h"] @ p.read).abs().reshape(features, 2, p.count).mean(1)
        row["read_profile"] = describe_profile(reads)
        if name.endswith("mlp_out"):
            write_profile = (target.W_U.T @ p.write).abs()
            row["write_output_profile"] = describe_profile(write_profile)
        if which == 0:
            # In the feature subspace the correct selective directions are the
            # embedding's dual basis, not necessarily the raw embedding rows.
            response = target.W_E @ p.read
            dual = target.W_E.T @ torch.linalg.inv(target.W_E @ target.W_E.T)
            projected = dual @ response
            cosine = torch.nn.functional.normalize(dual, dim=0).T @ torch.nn.functional.normalize(projected, dim=0)
            matched = torch.tensor([f["top_piece"] for f in row["exact_ablation"]["per_feature"]], device=args.device)
            alignment = cosine[torch.arange(features, device=args.device), matched].abs()
            row["feature_subspace_dual_alignment"] = {"mean": float(alignment.mean()),
                                                       "median": float(alignment.median()),
                                                       "over_0.99": int((alignment > 0.99).sum())}
        result["matrices"][name] = row
        atomic_save({"exact_ablation_rms": exact.cpu(), "first_order_rms": first_order.cpu(),
                     "read_profile": reads.cpu(), "reference_mlp_rms": reference.cpu()}, output / f"profiles_{name}.pt")
        atomic_json(result, output / "result.json")
        print(json.dumps({"matrix": name, **{key: value for key, value in row["exact_ablation"].items()
                                             if key != "per_feature"},
                          "read_selective_features": row["read_profile"]["features_with_top_piece_purity_over_90pct"],
                          "dual_alignment": row.get("feature_subspace_dual_alignment")}), flush=True)
    print("COMPONENT_INSPECTION_COMPLETE", flush=True)
    return result


if __name__ == "__main__":
    tracking.run_main(main)
