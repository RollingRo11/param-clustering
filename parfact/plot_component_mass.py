#!/usr/bin/env python3
#SBATCH --job-name=component-mass
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-component-mass-%j.out
"""Component weight mass, u-simplexed vs non-u-simplexed factorization.

For every component C_c of a run, its mass is ||C_c||_F^2 summed over the
decomposed matrices (the share of the model's squared parameter norm that
component carries). This plots the distribution of those masses and the
heaviest components, for both runs, and dumps the numbers as JSON.

Reads the `components` tensors saved inside factorization.pt, so no model
checkpoint or GPU is needed (if a run predates that field, pass --ckpt and the
components are rebuilt from V through the atom basis).

/cofact is a Beam DurableDisk mount -- it exists only inside Beam containers,
NOT on the Explorer cluster. To run this on Explorer, stage the two
factorization.pt files onto /projects first (see slurm_component_mass.sh) and
point --usimplex/--baseline at the staged copies.

    python plot_component_mass.py                 # both runs, default paths
    sbatch slurm_component_mass.sh                # staged copies, under Slurm

Outputs land in out/component_mass/.
"""
import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USIMPLEX = "/cofact/runs/B_layer_K1200_C600_v2_idiv_usimplex_ssimplex"
BASELINE = "/cofact/runs/B_layer_K1200_C600"

# blue / orange: the repo's figure palette, and the safe categorical pair
COLORS = {"usimplex": "#2a78d6", "baseline": "#e0862c"}
GRID = "#e8e8e8"


def log_bins(pos, n: int):
    """n log-spaced bin edges over the positive masses, or None if degenerate."""
    if pos.size == 0 or pos.min() == pos.max():
        return None
    lo, hi = float(pos.min()), float(pos.max())
    return torch.logspace(torch.log10(torch.tensor(lo)).item(),
                          torch.log10(torch.tensor(hi)).item(), n).numpy()


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_components(run_dir: Path, ckpt: Path | None):
    """name -> [C, *w.shape] component tensors, plus the background C0 if any."""
    fact = torch.load(run_dir / "factorization.pt", weights_only=False,
                      map_location="cpu")
    if "components" in fact:
        comps = {n: t.float() for n, t in fact["components"].items()}
    else:
        if ckpt is None:
            raise SystemExit(f"{run_dir}/factorization.pt has no saved "
                             "components; rerun with --ckpt to rebuild them")
        from induction_model import InductionModel
        from atoms import AtomBasis
        model = InductionModel()
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        model.eval()
        matrices = sorted(set(fact["atom_matrix"]), key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
        comps = {n: t.float() for n, t in basis.components(fact["V"].float()).items()}
    c0 = {n: t.float() for n, t in fact.get("C0", {}).items()} or None
    return comps, c0


def masses(comps: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Total mass per component [C], and the per-matrix breakdown."""
    per_matrix = {n: t.flatten(1).pow(2).sum(1) for n, t in comps.items()}
    total = torch.stack(list(per_matrix.values())).sum(0)
    return total, per_matrix


def summarize(m: torch.Tensor) -> dict:
    md = m.double()
    return {
        "n_components": int(md.numel()),
        "mean": md.mean().item(),
        "median": md.median().item(),
        "std": md.std(unbiased=True).item(),
        "max": md.max().item(),
        "min": md.min().item(),
        "sum": md.sum().item(),
        "argmax": int(md.argmax()),
        "argmin": int(md.argmin()),
        "n_zero": int((md == 0).sum()),
        # how concentrated the mass is: components needed for 50% / 90% of it
        "n_for_50pct": n_for(md, 0.50),
        "n_for_90pct": n_for(md, 0.90),
    }


def n_for(md: torch.Tensor, frac: float) -> int:
    cum = md.sort(descending=True).values.cumsum(0)
    return int((cum < frac * cum[-1]).sum().item()) + 1


def plot_hist(m: torch.Tensor, tag: str, label: str, out_dir: Path) -> Path:
    """Linear and log-x views of the mass distribution."""
    v = m.numpy()
    pos = v[v > 0]
    color = COLORS[tag]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
    fig.patch.set_facecolor("white")

    axes[0].hist(v, bins=60, color=color, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel(r"component mass  $||C_c||_F^2$")
    axes[0].set_ylabel("components")
    axes[0].set_title("linear bins", fontsize=10, color="#555555")

    bins = log_bins(pos, 60)
    if bins is not None:
        axes[1].hist(pos, bins=bins, color=color, edgecolor="white",
                     linewidth=0.5)
        axes[1].set_xscale("log")
    axes[1].set_xlabel(r"component mass  $||C_c||_F^2$ (log)")
    axes[1].set_ylabel("components")
    n_drop = int((v <= 0).sum())
    axes[1].set_title("log bins" + (f" ({n_drop} zero-mass omitted)"
                                    if n_drop else ""),
                      fontsize=10, color="#555555")
    for ax in axes:
        style(ax)
    fig.suptitle(f"Component mass distribution — {label} "
                 f"(C={v.size})", y=0.99)
    fig.tight_layout()
    path = out_dir / f"mass_hist_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_top(m: torch.Tensor, tag: str, label: str, out_dir: Path,
             top_k: int) -> Path:
    order = m.argsort(descending=True)[:top_k]
    vals = m[order].numpy()
    ids = [int(i) for i in order]
    share = 100 * vals / m.double().sum().item()
    fig, ax = plt.subplots(figsize=(7.6, 0.24 * top_k + 1.8), dpi=150)
    fig.patch.set_facecolor("white")
    y = range(len(ids))
    ax.barh(list(y), vals, color=COLORS[tag], height=0.72)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"c{i}" for i in ids], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"component mass  $||C_c||_F^2$")
    ax.set_title(f"Top {top_k} components by mass — {label}")
    pad = 0.012 * vals.max() if vals.size else 0
    for yi, (v, s) in enumerate(zip(vals, share)):
        ax.text(v + pad, yi, f"{s:.1f}%", va="center", fontsize=6.5,
                color="#666666")
    ax.set_xlim(0, max(float(vals.max()) * 1.12, 1e-12) if vals.size else 1)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.75)
    fig.tight_layout()
    path = out_dir / f"top{top_k}_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_compare(runs: dict, out_dir: Path, title: str) -> Path:
    """Both runs on one pair of axes: sorted mass profile and log-x histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    for tag, r in runs.items():
        m = r["mass"]
        srt = m.sort(descending=True).values.numpy()
        axes[0].plot(range(1, srt.size + 1), srt, lw=2, color=COLORS[tag],
                     label=r["label"])
        v = m.numpy()
        bins = log_bins(v[v > 0], 50)
        if bins is not None:
            axes[1].hist(v[v > 0], bins=bins, color=COLORS[tag], alpha=0.55,
                         label=r["label"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("component rank")
    axes[0].set_ylabel(r"mass  $||C_c||_F^2$ (log)")
    axes[0].set_title("sorted mass profile", fontsize=10, color="#555555")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"mass  $||C_c||_F^2$ (log)")
    axes[1].set_ylabel("components")
    axes[1].set_title("mass distribution", fontsize=10, color="#555555")
    for ax in axes:
        style(ax)
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    path = out_dir / "mass_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usimplex", type=Path, default=Path(USIMPLEX))
    ap.add_argument("--baseline", type=Path, default=Path(BASELINE))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out/component_mass")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--usimplex_label", default="u-simplexed (v2 idiv, U+S simplex)")
    ap.add_argument("--baseline_label", default="no u-simplex (v1 B_layer)")
    ap.add_argument("--title", default="Component mass: u-simplex vs no u-simplex",
                    help="suptitle for the two-run comparison figure")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="model checkpoint, only needed for old runs whose "
                         "factorization.pt has no saved components")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    targets = {"usimplex": (args.usimplex, args.usimplex_label),
               "baseline": (args.baseline, args.baseline_label)}
    runs, blob = {}, {}
    for tag, (run_dir, label) in targets.items():
        print(f"\n=== {label} ===\n{run_dir}")
        comps, c0 = load_components(run_dir, args.ckpt)
        mass, per_matrix = masses(comps)
        stats = summarize(mass)

        # theta = sum_c C_c (+ C0), so the total weight mass is recoverable
        theta_mass = sum(
            (comps[n].sum(0) + (c0[n] if c0 and n in c0 else 0)).pow(2).sum().item()
            for n in comps)
        stats["theta_mass"] = theta_mass
        stats["component_mass_frac_of_theta"] = stats["sum"] / theta_mass
        if c0:
            stats["C0_background_mass"] = sum(t.pow(2).sum().item()
                                              for t in c0.values())
            stats["C0_mass_frac_of_theta"] = stats["C0_background_mass"] / theta_mass

        print(f"matrices: {', '.join(sorted(comps))}")
        print(f"  components {stats['n_components']}   "
              f"mean {stats['mean']:.4e}   median {stats['median']:.4e}")
        print(f"  std {stats['std']:.4e}   max {stats['max']:.4e} "
              f"(c{stats['argmax']})   min {stats['min']:.4e} "
              f"(c{stats['argmin']})")
        print(f"  total component mass {stats['sum']:.4e} "
              f"({100 * stats['component_mass_frac_of_theta']:.1f}% of "
              f"||theta||^2 = {theta_mass:.4e})")
        if c0:
            print(f"  background C0 mass {stats['C0_background_mass']:.4e} "
                  f"({100 * stats['C0_mass_frac_of_theta']:.1f}% of ||theta||^2)")
        print(f"  concentration: {stats['n_for_50pct']} components hold 50% "
              f"of the mass, {stats['n_for_90pct']} hold 90%; "
              f"{stats['n_zero']} are exactly zero")

        h = plot_hist(mass, tag, label, args.out)
        b = plot_top(mass, tag, label, args.out, args.top_k)
        print(f"  wrote {h}\n  wrote {b}")

        runs[tag] = {"mass": mass, "label": label}
        order = mass.argsort(descending=True)[:args.top_k]
        blob[tag] = {
            "run_dir": str(run_dir),
            "label": label,
            "matrices": sorted(comps),
            "stats": stats,
            "mass": [float(v) for v in mass],
            "mass_per_matrix": {n: [float(v) for v in t]
                                for n, t in per_matrix.items()},
            "top": [{"component": int(i), "mass": float(mass[i]),
                     "share_of_total": float(mass[i] / mass.sum())}
                    for i in order],
        }

    c = plot_compare(runs, args.out, args.title)
    print(f"\nwrote {c}")
    j = args.out / "component_masses.json"
    j.write_text(json.dumps(blob, indent=1))
    print(f"wrote {j}")


if __name__ == "__main__":
    main()
