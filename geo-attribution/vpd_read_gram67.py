"""Inspect VPD read-vector geometry without model execution or refitting."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import SymLogNorm


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def pair_stats(gram):
    values = gram[np.triu_indices(len(gram), 1)]
    absolute = np.abs(values)
    return values, {
        "unique_pairs": len(values),
        "mean_signed_cosine": float(values.mean()),
        "mean_absolute_cosine": float(absolute.mean()),
        "rms_cosine": float(np.sqrt(np.mean(values ** 2))),
        "absolute_cosine_quantiles": {
            str(q): float(np.quantile(absolute, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        "fraction_absolute_cosine_above": {
            str(t): float(np.mean(absolute > t)) for t in (0.1, 0.25, 0.5, 0.9)
        },
    }


def inspect(sd, target, module, output, seed, mean_ci, threshold):
    key = "_components." + module.replace(".", "-")
    read = sd[key + ".V"].double().numpy()
    write = sd[key + ".U"].double().numpy()
    weight_shape = tuple(target[module + ".weight"].shape)
    if read.ndim != 2 or write.ndim != 2 or read.shape[1] != write.shape[0]:
        raise ValueError("Unexpected VPD factor shapes")
    if (write.shape[1], read.shape[0]) != weight_shape:
        raise ValueError("V must index input coordinates, U output coordinates")
    if not np.isfinite(read).all() or not np.isfinite(write).all():
        raise ValueError("Nonfinite checkpoint factors")
    norms = np.linalg.norm(read, axis=0)
    if mean_ci.shape != norms.shape or not np.isfinite(mean_ci).all():
        raise ValueError("Activity statistics must match checkpoint component indices")
    alive = mean_ci > threshold
    indices = np.flatnonzero(alive & (norms > 0))
    if len(indices) < 2:
        raise ValueError("Need at least two nonzero read vectors")
    unit = read[:, indices] / norms[indices]
    raw = read[:, indices].T @ read[:, indices]
    cosine = unit.T @ unit
    diagonal_error = float(np.max(np.abs(np.diag(cosine) - 1)))
    symmetry_error = float(np.max(np.abs(cosine - cosine.T)))
    if diagonal_error > 1e-10 or symmetry_error > 1e-10:
        raise RuntimeError("Normalized Gram sanity check failed")
    values, stats = pair_stats(cosine)
    rng = np.random.default_rng(seed)
    random_read = rng.standard_normal(unit.shape)
    random_read /= np.linalg.norm(random_read, axis=0)
    random_values, random_stats = pair_stats(random_read.T @ random_read)
    # Use the smaller frame operator to determine span dimension, not a C-by-C eigensolve.
    frame = unit @ unit.T if len(indices) > read.shape[0] else cosine
    eigenvalues = np.linalg.eigvalsh(frame)
    rank_tolerance = max(unit.shape) * np.finfo(np.float64).eps * eigenvalues[-1]
    rank = int(np.sum(eigenvalues > rank_tolerance))
    if eigenvalues[0] < -1e-9 * eigenvalues[-1]:
        raise RuntimeError("Gram matrix should be positive semidefinite")
    absolute = np.abs(cosine).copy()
    np.fill_diagonal(absolute, -1)
    i, j = np.unravel_index(np.argmax(absolute), absolute.shape)
    strongest = {"components": [int(indices[i]), int(indices[j])],
                 "cosine": float(cosine[i, j]),
                 "read_norms": [float(norms[indices[i]]), float(norms[indices[j]])],
                 "parameter_piece_frobenius_norms": [
                     float(norms[indices[k]] * np.linalg.norm(write[indices[k]])) for k in (i, j)
                 ]}
    sample = np.sort(rng.choice(len(indices), min(256, len(indices)), replace=False))
    slug = module.replace(".", "_")
    np.savez_compressed(output / f"{slug}_grams.npz", raw_gram=raw.astype(np.float32),
                        cosine_gram=cosine.astype(np.float32), read_norms=norms[indices],
                        active_component_indices=indices, mean_clamped_ci=mean_ci[indices],
                        displayed_component_indices=indices[sample])
    row = {"module": module, "read_key": key + ".V", "write_key": key + ".U",
           "target_weight_shape": weight_shape, "V_shape": read.shape, "U_shape": write.shape,
           "input_dimension": read.shape[0], "stored_read_vectors": read.shape[1],
           "active_read_vectors": len(indices), "excluded_inactive_vectors": int(np.sum(~alive)),
           "active_zero_read_vectors": int(np.sum(alive & (norms == 0))),
           "activity_threshold": threshold,
           "active_counts_by_threshold": {str(t): int(np.sum(mean_ci > t)) for t in (1e-6, 1e-5, 1e-4, 1e-3)},
           "read_norm_quantiles": {str(q): float(np.quantile(norms[indices], q)) for q in (0, 0.01, 0.5, 0.99, 1)},
           "nonzero_reads_with_zero_writer": int(np.sum(np.linalg.norm(write[indices], axis=1) == 0)),
           "numerical_span_rank": rank, "rank_tolerance": float(rank_tolerance),
           "gram_diagonal_error": diagonal_error, "gram_symmetry_error": symmetry_error,
           "off_diagonal": stats, "random_unit_vector_control": random_stats,
           "welch_rms_lower_bound": math.sqrt(max(0, (len(indices) - read.shape[0]) /
                                                  (read.shape[0] * (len(indices) - 1)))),
           "most_aligned_pair": strongest,
           "displayed_component_indices": indices[sample].tolist()}
    print(json.dumps({k: row[k] for k in ("module", "V_shape", "active_read_vectors", "numerical_span_rank", "off_diagonal")}), flush=True)
    return row, raw, cosine, values, random_values, sample


def plot(rows, output, n_tokens, threshold):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fafafa", "axes.facecolor": "white"})
    fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 5.0 * len(rows)), layout="constrained")
    axes = np.asarray(axes).reshape(len(rows), 2)
    colors = plt.get_cmap("RdBu_r").copy()
    colors.set_bad("#dddddd")
    for row_index, (report, raw, cosine, values, random_values, sample) in enumerate(rows):
        matrix = cosine[np.ix_(sample, sample)].copy()
        np.fill_diagonal(matrix, np.nan)
        ax = axes[row_index, 0]
        picture = ax.imshow(matrix, vmin=-0.5, vmax=0.5, cmap=colors, interpolation="nearest", rasterized=True)
        label = "First MLP · input matrix" if "mlp" in report["module"] else "First attention block · query matrix"
        ax.set_title(f"{label}\n{report['active_read_vectors']:,} active reads in {report['input_dimension']:,} dimensions", loc="left", fontweight="bold")
        sample_label = f"{len(sample)} sampled active vectors" if len(sample) < len(cosine) else f"All {len(sample)} active vectors"
        ax.set_xlabel(sample_label + " · original index order")
        ax.set_ylabel("Same read vectors")
        fig.colorbar(picture, ax=ax, shrink=0.78, extend="both", label="Cosine similarity (diagonal hidden)")
        ax = axes[row_index, 1]
        bins = np.linspace(0, 1, 101)
        ax.hist(np.abs(random_values), bins=bins, density=True, histtype="step", linewidth=1.5,
                color="#9b9b9b", label="Random unit vectors, same dimensions")
        ax.hist(np.abs(values), bins=bins, density=True, histtype="step", linewidth=1.8,
                color="#126477", label="VPD · all pairs of active components")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_ylim(1e-5, 200)
        ax.set_xlabel(r"Absolute cosine similarity $|v_i^\top v_j|/(\|v_i\|\|v_j\|)$")
        ax.set_ylabel("Probability density · logarithmic scale")
        ax.set_title("Orthogonal pairs would all be at zero", loc="left", fontweight="bold")
        stats = report["off_diagonal"]
        text = (f"All {stats['unique_pairs']:,} distinct pairs\n"
                f"RMS cosine: {stats['rms_cosine']:.3f}   (random: {report['random_unit_vector_control']['rms_cosine']:.3f})\n"
                f"99th percentile |cos|: {stats['absolute_cosine_quantiles']['0.99']:.3f}\n"
                f"Maximum |cos|: {stats['absolute_cosine_quantiles']['1.0']:.3f}\n"
                f"Pairs with |cos| > 0.5: {100 * stats['fraction_absolute_cosine_above']['0.5']:.2f}%")
        ax.text(0.97, 0.93, text, transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95, "pad": 8})
        ax.legend(loc="lower left", fontsize=9, frameon=False)
        ax.grid(axis="y", alpha=0.12)
    fig.suptitle("Active VPD read vectors: are their directions orthogonal?", fontsize=20, fontweight="bold")
    activity_label = f"Pile-4L · alive = mean clamped causal importance > {threshold:g} over {n_tokens:,} corpus tokens"
    fig.supxlabel(activity_label, fontsize=10)
    fig.savefig(output / "vpd_read_orthogonality.png", dpi=180)
    fig.savefig(output / "vpd_read_orthogonality.pdf", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 5.6 * len(rows)), layout="constrained")
    axes = np.asarray(axes).reshape(len(rows), 2)
    for index, (report, raw, cosine, *_) in enumerate(rows):
        vmax = float(np.abs(raw).max())
        image = axes[index, 0].imshow(raw, cmap="RdBu_r", interpolation="nearest", rasterized=True,
                                     norm=SymLogNorm(linthresh=max(vmax * 0.01, 1e-20), vmin=-vmax, vmax=vmax))
        axes[index, 0].set_title(report["module"] + r" · raw $G=V^\top V$", loc="left")
        fig.colorbar(image, ax=axes[index, 0], shrink=0.8, label="Dot product · symmetric log color scale")
        image = axes[index, 1].imshow(cosine, cmap="RdBu_r", vmin=-1, vmax=1,
                                     interpolation="nearest", rasterized=True)
        axes[index, 1].set_title(r"Normalized $\widehat G_{ij}=G_{ij}/\sqrt{G_{ii}G_{jj}}$", loc="left")
        fig.colorbar(image, ax=axes[index, 1], shrink=0.8, label="Cosine similarity · linear color scale")
        for ax in axes[index]:
            ax.set_xlabel("Active read-vector index · checkpoint order")
            ax.set_ylabel("Active read-vector index")
    fig.suptitle("Full Gram matrices · active VPD read vectors only", fontsize=20, fontweight="bold")
    fig.supxlabel(activity_label, fontsize=10)
    fig.savefig(output / "vpd_read_full_grams.png", dpi=180)
    fig.savefig(output / "vpd_read_full_grams.pdf", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--corpus", default="lab_1m")
    parser.add_argument("--threshold", type=float, default=1e-6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    target = torch.load(args.target, map_location="cpu", weights_only=True)
    activity = json.loads(args.activity.read_text())
    if Path(activity["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Activity must be measured on the same checkpoint")
    corpus = activity["corpora"][args.corpus]
    means = np.split(np.asarray(corpus["mean_clamped_ci"]), np.cumsum(activity["split"])[:-1])
    mean_by_module = dict(zip(activity["modules"], means))
    rows = [inspect(sd, target, module, args.output, seed=17 + index,
                    mean_ci=mean_by_module[module], threshold=args.threshold)
            for index, module in enumerate(("h.0.mlp.c_fc", "h.0.attn.q_proj"))]
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "target": str(args.target), "target_sha256": sha256(args.target),
              "source_sha256": sha256(__file__), "matrices": [row[0] for row in rows],
              "activity_filter": {"source": str(args.activity), "source_sha256": sha256(args.activity),
                                  "corpus": args.corpus, "corpus_path": corpus["path"],
                                  "corpus_sha256": sha256(corpus["path"]), "n_tokens": corpus["n_tokens"],
                                  "threshold": args.threshold, "definition": activity["definition"]},
              "notes": ["Columns of V are reads: a masked linear layer computes ((x @ V) * mask) @ U.",
                        "Computed in float64; saved Gram arrays are float32, with original indices and read norms.",
                        "Only empirically alive components are included; this is a corpus-wide set, not a single-token active set.",
                        "The overview samples at most 256 active directions uniformly; statistics use every active pair.",
                        "Small typical pairwise cosines alone do not establish mutual orthogonality or causal independence."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(rows, args.output, corpus["n_tokens"], args.threshold)
    print(f"Saved figures and Gram matrices to {args.output}", flush=True)


if __name__ == "__main__":
    main()
