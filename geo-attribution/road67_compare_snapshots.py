"""Overlay KLKeep curves from two ROAD evaluation snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads(path.read_text())
    key = str(stats["clusters"][0])
    result = stats["klkeep"][key]
    keep = np.asarray(result["keep_grid"], dtype=float)
    return 100 * keep / float(key), np.asarray(result["kl_mean"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument("--earlier-label", required=True)
    parser.add_argument("--later-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    for path, label, color in (
            (args.earlier, args.earlier_label, "#777777"),
            (args.later, args.later_label, "#b64914")):
        x, y = curve(path)
        valid = x > 0
        ax.plot(x[valid], np.maximum(y[valid], 1e-5), linewidth=2.5,
                label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(.02, 100)
    ax.set_ylim(1e-4, 100)
    ax.grid(True, which="major", alpha=.2)
    ax.set_xlabel("components retained (percent, log scale)")
    ax.set_ylabel("KL to full model (nats)")
    ax.set_title("ROAD KLKeep during ablation training")
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
