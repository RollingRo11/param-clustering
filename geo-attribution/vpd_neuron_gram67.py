"""Compare active VPD directions on both sides of the first MLP.

Reuses the exact checkpoint and activity filter from the read-vector audit.
Run only on an Explorer compute node; no model forward pass is needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vpd_read_gram67 import pair_stats, sha256
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def measure(vectors, indices, output, slug, seed):
    vectors = vectors.double().numpy()[:, indices]
    norms = np.linalg.norm(vectors, axis=0)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Alive components must have finite, nonzero directions")
    raw = vectors.T @ vectors
    unit = vectors / norms
    gram = unit.T @ unit
    if not np.allclose(np.diag(gram), 1, atol=1e-12, rtol=0):
        raise ValueError("Normalized Gram diagonal differs from one")
    if not np.allclose(gram, gram.T, atol=1e-12, rtol=0):
        raise ValueError("Gram is not symmetric")
    values, statistics = pair_stats(gram)
    frame = gram if gram.shape[0] <= unit.shape[0] else unit @ unit.T
    eigenvalues = np.linalg.eigvalsh(frame)
    if eigenvalues[0] < -1e-9 * eigenvalues[-1]:
        raise ValueError("Gram is not positive semidefinite")
    rank = int(np.sum(eigenvalues > max(unit.shape) * np.finfo(float).eps * eigenvalues[-1]))
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(unit.shape)
    random /= np.linalg.norm(random, axis=0)
    random_values, random_statistics = pair_stats(random.T @ random)
    absolute = np.abs(gram).copy()
    np.fill_diagonal(absolute, -1)
    i, j = np.unravel_index(np.argmax(absolute), absolute.shape)
    np.savez_compressed(output / f"{slug}.npz", raw_gram=raw.astype(np.float32),
                        cosine_gram=gram.astype(np.float32), vector_norms=norms,
                        active_component_indices=indices)
    row = {"dimension": unit.shape[0], "n_active": len(indices), "numerical_span_rank": rank,
           "off_diagonal": statistics, "random_unit_vector_control": random_statistics,
           "most_aligned_pair": {"components": indices[[i, j]].tolist(), "cosine": float(gram[i, j])}}
    return row, raw, gram, values, random_values


def plot(rows, output, tokens):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fafafa", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    colors = plt.get_cmap("RdBu_r").copy()
    colors.set_bad("#cccccc")
    for col, (row, raw, gram, values, random_values, sample) in enumerate(rows):
        shown = gram[np.ix_(sample, sample)].copy()
        np.fill_diagonal(shown, np.nan)
        im = axes[0, col].imshow(shown, cmap=colors, vmin=-0.25, vmax=0.25, interpolation="nearest")
        axes[0, col].set_title(f"{row['label']}\n{row['n_active']:,} active directions in {row['dimension']:,}D",
                               loc="left", fontweight="bold", fontsize=12)
        axes[0, col].set_xlabel("256 sampled components · checkpoint order")
        axes[0, col].set_ylabel("Same components")
        ax = axes[1, col]
        bins = np.linspace(0, 1, 101)
        ax.hist(np.abs(random_values), bins=bins, density=True, histtype="step", color="#999999",
                linewidth=1.5, label=f"Random directions in {row['dimension']:,}D")
        ax.hist(np.abs(values), bins=bins, density=True, histtype="step", color="#126477",
                linewidth=1.8, label="VPD · every active pair")
        ax.set(xlim=(0, 1), ylim=(1e-5, 250), yscale="log", xlabel="Absolute cosine similarity",
               ylabel="Probability density · log scale")
        s = row["off_diagonal"]
        caption = (f"RMS: {s['rms_cosine']:.4f}  (random: {row['random_unit_vector_control']['rms_cosine']:.4f})\n"
                   f"99th percentile: {s['absolute_cosine_quantiles']['0.99']:.3f}\n"
                   f"Maximum: {s['absolute_cosine_quantiles']['1.0']:.3f}")
        ax.text(0.97, 0.96, caption, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95, "pad": 6})
        ax.legend(loc="lower left", fontsize=9, frameon=False)
        ax.grid(axis="y", alpha=0.12)
    fig.colorbar(im, ax=axes[0, :], shrink=0.9, extend="both", label="Cosine similarity · diagonal hidden")
    fig.suptitle("VPD inside the first MLP: input space versus neuron space", fontsize=20, fontweight="bold")
    fig.supxlabel(f"Alive = mean clamped CI > 1e-6 over {tokens:,} tokens. First two columns use identical components and samples.", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"vpd_neuron_comparison.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11), layout="constrained")
    for i, (row, raw, gram, *_) in enumerate(rows[1:]):
        limit = float(np.max(np.abs(raw)))
        im = axes[i, 0].imshow(raw, cmap="RdBu_r", interpolation="nearest", rasterized=True,
                              norm=SymLogNorm(linthresh=limit * 0.01, vmin=-limit, vmax=limit))
        axes[i, 0].set_title(row["label"] + "\nRaw " + row["formula"], loc="left", fontweight="bold")
        fig.colorbar(im, ax=axes[i, 0], shrink=0.8, label="Dot product · symmetric log color scale")
        im = axes[i, 1].imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        axes[i, 1].set_title("Normalized Gram matrix", loc="left", fontweight="bold")
        fig.colorbar(im, ax=axes[i, 1], shrink=0.8, label="Cosine similarity")
        for ax in axes[i]:
            ax.set_xlabel("Active component index · checkpoint order")
            ax.set_ylabel("Active component index")
    fig.suptitle("Full VPD Gram matrices in 3,072-dimensional neuron space", fontsize=18, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"vpd_neuron_full_grams.{ext}", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    previous = json.loads((args.previous / "report.json").read_text())
    checkpoint = Path(previous["checkpoint"])
    activity_path = args.previous / "component_activity.json"
    assert sha256(checkpoint) == previous["checkpoint_sha256"], "Checkpoint changed"
    assert sha256(activity_path) == previous["activity_filter"]["source_sha256"], "Activity data changed"
    activity = json.loads(activity_path.read_text())
    corpus = activity["corpora"][previous["activity_filter"]["corpus"]]
    means = dict(zip(activity["modules"], np.split(np.asarray(corpus["mean_clamped_ci"]),
                                                  np.cumsum(activity["split"])[:-1])))
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rows = []
    incoming_ids = np.flatnonzero(means["h.0.mlp.c_fc"] > 1e-6)
    reference = next(row for row in previous["matrices"] if row["module"] == "h.0.mlp.c_fc")
    sample_ids = np.asarray(reference["displayed_component_indices"])
    for module, side, label, formula in (
        ("h.0.mlp.c_fc", "V", "Input matrix · reads from residual", r"$G=V^\top V$"),
        ("h.0.mlp.c_fc", "U", "Input matrix · writes to neurons", r"$G=UU^\top$"),
        ("h.0.mlp.down_proj", "V", "Output matrix · reads from neurons", r"$G=V^\top V$"),
    ):
        indices = np.flatnonzero(means[module] > 1e-6)
        key = "_components." + module.replace(".", "-") + "." + side
        vectors = sd[key] if side == "V" else sd[key].T
        slug = module.replace(".", "_") + "_" + side
        measured = measure(vectors, indices, args.output, slug, seed=117)
        row, raw, gram, values, random_values = measured
        row.update(module=module, side=side, label=label, formula=formula, key=key)
        if module == "h.0.mlp.c_fc":
            assert np.array_equal(indices, incoming_ids)
            sample = np.searchsorted(indices, sample_ids)
            assert np.array_equal(indices[sample], sample_ids)
        else:
            sample = np.sort(np.random.default_rng(17).choice(len(indices), 256, replace=False))
        row["displayed_component_indices"] = indices[sample].tolist()
        if module == "h.0.mlp.c_fc" and side == "V":
            np.testing.assert_allclose(row["off_diagonal"]["rms_cosine"], reference["off_diagonal"]["rms_cosine"], atol=1e-12, rtol=0)
        rows.append((*measured, sample))
        print(json.dumps(row), flush=True)
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": previous["checkpoint_sha256"],
              "activity_filter": previous["activity_filter"], "source_sha256": sha256(__file__),
              "previous_report_sha256": sha256(args.previous / "report.json"),
              "matrices": [row[0] for row in rows],
              "notes": ["Each component of a linear layer is an outer product of a read and a write vector.",
                        "Input-matrix write vectors and output-matrix read vectors both inhabit the MLP neuron space.",
                        "The two matrices have separate VPD component dictionaries; their component IDs are not paired.",
                        "All statistics use every alive pair. Calculated in float64; Gram arrays saved as float32."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(rows, args.output, corpus["n_tokens"])
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
