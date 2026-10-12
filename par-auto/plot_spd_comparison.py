"""Reproduce SPD Figure 10 for free read/write pieces.

Reference: spd-paper c6314c9, resid_mlp_interp.py, lines 318-414 and 505-617.
The published plotting code chooses one input piece per feature and layer by
its largest signed neuron contribution; it does not take a pointwise envelope.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_evaluate import TARGET_SHA  # noqa: E402
from parauto_io import load_pieces  # noqa: E402
from parauto_target import (  # noqa: E402
    atomic_json,
    atomic_save,
    fingerprint,
    load_target,  # noqa: E402
)


def contribution_profiles(target, pieces):
    """Return target [L,F,H] and input-piece [L,F,C,H] weight products."""
    reference, candidates = [], []
    for layer, module in enumerate(target.layers):
        inp, out = pieces[2 * layer:2 * layer + 2]
        original_read = target.W_E @ module.mlp_in.weight.T
        original_write = target.W_U.T @ module.mlp_out.weight
        reference.append(original_read * original_write)
        # Equation 12 uses the SUM of all output pieces, not the same-index
        # output piece, a selected subset, or a matched writer.
        reconstructed_output = out.write @ out.read.T
        output_connection = target.W_U.T @ reconstructed_output
        candidates.append(torch.einsum("fc,hc,fh->fch", target.W_E @ inp.read,
                                        inp.write, output_connection))
    return torch.stack(reference), torch.stack(candidates)


def select_like_paper(candidates):
    """Exact max-then-argmax rule in the authors' Figure 10 plotting code."""
    indices = candidates.max(dim=-1).values.argmax(dim=-1)
    layers = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
    features = torch.arange(candidates.shape[1], device=candidates.device)[None, :]
    return candidates[layers, features, indices, :], indices


def metrics(x, y):
    a, b = x.flatten().double(), y.flatten().double()
    return {"points": a.numel(), "pearson_r": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
            "relative_rmse": float((b-a).norm() / a.norm()),
            "slope_through_origin": float((a @ b) / (a @ a)),
            "rmse": float((b-a).square().mean().sqrt())}


def draw(reference, matched, title, stats, output):
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    for layer, (color, label) in enumerate((("blue", "First MLP"), ("red", "Second MLP"))):
        ax.scatter(reference[layer].flatten(), matched[layer].flatten(), alpha=0.3,
                   s=10, color=color, edgecolors="none", label=label)
    low = min(reference.min(), matched.min())
    high = max(reference.max(), matched.max())
    pad = (high-low) * 0.05
    limits = (low-pad, high+pad)
    ax.plot(limits, limits, "k--", alpha=0.2, zorder=0, label="y=x")
    ax.set(xlim=limits, ylim=limits, xlabel="Target Model Neuron Contribution",
           ylabel="Subcomponent Neuron Contribution (Max Subcomponent)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    ax.text(0.05, 0.95, f"Correlation: {stats['pearson_r']:.3f}\nRelative RMSE: {stats['relative_rmse']:.1%}",
            transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


import parauto_tracking as tracking

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    root, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((root / "result.json").read_text())
    assert fingerprint(args.target) == original["target_sha256"] == TARGET_SHA
    torch.backends.cuda.matmul.allow_tf32 = False
    target = load_target(args.target, "cuda")
    pieces = load_pieces(root, target, "cuda")
    errors = {name: core.reconstruction_error(weight, piece) for (name, weight), piece in zip(toy.matrices(target), pieces, strict=True)}
    reference, candidates = contribution_profiles(target, pieces)
    matched, indices = select_like_paper(candidates)
    envelope = candidates.max(dim=2).values
    # Independent sum check: summing all input contributions must recover the
    # full model's weight-product profile, within stored-factor roundoff.
    torch.testing.assert_close(candidates.sum(dim=2), reference, atol=1e-6, rtol=2e-5)
    result = {"method": "projected_lp", "target_sha256": TARGET_SHA,
              "decomposition_steps_per_matrix": original.get('steps_per_matrix', original.get('config', {}).get('steps')),
              "reference_commit": "c6314c9f702b81af593927025aa0ae5aaed4ca4c",
              "source": "https://github.com/goodfire-ai/spd/blob/c6314c9f702b81af593927025aa0ae5aaed4ca4c/spd/experiments/resid_mlp/resid_mlp_interp.py#L505-L617",
              "selection": "one piece per layer and feature: argmax_c max_neuron signed contribution",
              "matrix_relative_errors": errors,
              "paper_plot": metrics(reference, matched),
              "pointwise_max_supplement": metrics(reference, envelope),
              "per_layer": {}}
    for layer in range(reference.shape[0]):
        relative = (matched[layer]-reference[layer]).norm(dim=-1) / reference[layer].norm(dim=-1).clamp_min(1e-30)
        result["per_layer"][str(layer)] = {**metrics(reference[layer], matched[layer]),
            "distinct_selected_pieces": int(indices[layer].unique().numel()),
            "selected_piece_by_feature": indices[layer].tolist(),
            "relative_rmse_by_feature": relative.tolist(),
            "features_below_10pct_error": int((relative < 0.1).sum()),
            "features_below_25pct_error": int((relative < 0.25).sum())}
    atomic_json(result, output / "result.json")
    atomic_save({"target": reference.cpu(), "matched": matched.cpu(),
                 "selected_pieces": indices.cpu(), "pointwise_max": envelope.cpu()}, output / "profiles.pt")
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    draw(reference.cpu().numpy(), matched.cpu().numpy(),
         f"SPD Figure 10 comparison - projected Lp, {original.get('steps_per_matrix', original.get('config', {}).get('steps')):,} steps/matrix\n100 input features, 2 layers, 10k-step target",
         result["paper_plot"], output / "spd_figure10_B")
    draw(reference.cpu().numpy(), envelope.cpu().numpy(),
         "Supplement: pointwise maximum over pieces\nDifferent selection rule from the authors' Figure 10 code",
         result["pointwise_max_supplement"], output / "pointwise_max_B")
    print(json.dumps(result), flush=True)
    print("SPD_FIGURE10_COMPLETE", flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
