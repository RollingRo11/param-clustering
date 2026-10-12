"""Dependency-free KLKeep and mass plots, with optional rsvg PNG rendering."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path


COLORS = {1024: "#17847b", 2048: "#496fb0", 4096: "#b45220"}


def polyline(points, color, width=3):
    values = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def convert(svg: Path) -> None:
    executable = shutil.which("rsvg-convert")
    if executable:
        subprocess.run([executable, "-o", str(svg.with_suffix(".png")), str(svg)],
                       check=True)


def plot_kl(result: dict, output: Path) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 60, 80
    pw, ph = width - left - right, height - top - bottom
    cs = [int(value) for value in result["klkeep"]]
    xmax = max(cs)
    def x(value): return left + math.log2(max(value, 1)) / math.log2(xmax) * pw
    def y(value):
        value = min(100, max(1e-5, value))
        return top + (2 - math.log10(value)) / 7 * ph
    body = [f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="34" text-anchor="middle" font-size="24">KLKeep on held-out Pile predictions</text>',
            f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>']
    for exponent in range(-5, 3):
        yy = y(10 ** exponent)
        body += [f'<line x1="{left}" y1="{yy}" x2="{left+pw}" y2="{yy}" stroke="#ddd"/>',
                 f'<text x="{left-12}" y="{yy+5}" text-anchor="end" font-size="13">10^{exponent}</text>']
    for threshold, dash in ((.1, "7,5"), (.01, "2,4")):
        yy = y(threshold)
        body.append(f'<line x1="{left}" y1="{yy}" x2="{left+pw}" y2="{yy}" stroke="#777" stroke-dasharray="{dash}"/>')
    for ctext, row in result["klkeep"].items():
        c = int(ctext); color = COLORS.get(c, "#555")
        points = [(x(k), y(v)) for k, v in zip(row["keep_grid"], row["kl_mean"])
                  if k > 0]
        body.append(polyline(points, color))
        body.append(f'<text x="{left+18}" y="{top+25+24*cs.index(c)}" fill="{color}" font-size="16">C={c}</text>')
    body += [f'<text x="{left+pw/2}" y="{height-22}" text-anchor="middle" font-size="17">components retained (log2 scale)</text>',
             f'<text x="22" y="{top+ph/2}" text-anchor="middle" font-size="17" transform="rotate(-90 22 {top+ph/2})">KL to full model (nats)</text>']
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">{"".join(body)}</svg>')
    convert(output)


def plot_kl_vpd(result: dict, output: Path) -> None:
    """Plot decomposition resolutions and VPD on a shared retained-percent axis."""
    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 60, 80
    pw, ph = width - left - right, height - top - bottom
    vpd = result["vpd_klkeep"]
    positive = [100 * k / int(c) for c, row in result["klkeep"].items()
                for k in row["keep_grid"] if k > 0]
    positive.extend(vpd["keep_percent"][1:])
    xmin = min(positive)

    def x(value):
        return left + (math.log10(value) - math.log10(xmin)) / (
            2 - math.log10(xmin)) * pw

    def y(value):
        value = min(100, max(1e-5, value))
        return top + (2 - math.log10(value)) / 7 * ph

    method = result.get("method", "ParFact")
    body = [f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="34" text-anchor="middle" font-size="24">KLKeep: {method} vs VPD</text>',
            f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>']
    for exponent in range(-5, 3):
        yy = y(10 ** exponent)
        body += [f'<line x1="{left}" y1="{yy}" x2="{left+pw}" y2="{yy}" stroke="#ddd"/>',
                 f'<text x="{left-12}" y="{yy+5}" text-anchor="end" font-size="13">10^{exponent}</text>']
    for value in (.1, 1, 10, 100):
        xx = x(value)
        body += [f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top+ph}" stroke="#eee"/>',
                 f'<text x="{xx}" y="{top+ph+24}" text-anchor="middle" font-size="13">{value:g}%</text>']
    cs = [int(value) for value in result["klkeep"]]
    for ctext, row in result["klkeep"].items():
        c = int(ctext)
        color = COLORS.get(c, "#555")
        points = [(x(100 * k / c), y(value))
                  for k, value in zip(row["keep_grid"], row["kl_mean"])
                  if k > 0]
        body.append(polyline(points, color))
        body.append(f'<text x="{left+18}" y="{top+25+24*cs.index(c)}" fill="{color}" font-size="16">{method} C={c}</text>')
    vpd_points = [(x(percent), y(value))
                  for percent, value in zip(vpd["keep_percent"], vpd["kl_mean"])
                  if percent > 0]
    vpd_line = polyline(vpd_points, "#7b3fa1", 4).replace('/>', ' stroke-dasharray="9,5"/>')
    body.append(vpd_line)
    body.append(f'<text x="{left+18}" y="{top+25+24*len(cs)}" fill="#7b3fa1" font-size="16">VPD ({vpd["total_components"]:,} subcomponents)</text>')
    body += [f'<text x="{left+pw/2}" y="{height-22}" text-anchor="middle" font-size="17">components retained (percent, log scale)</text>',
             f'<text x="22" y="{top+ph/2}" text-anchor="middle" font-size="17" transform="rotate(-90 22 {top+ph/2})">KL to full model (nats)</text>']
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">{"".join(body)}</svg>')
    convert(output)


def plot_mass(result: dict, output: Path) -> None:
    width, height = 1200, 590
    left, top, bottom, gap = 80, 92, 75, 100
    panel = (width - 2 * left - gap) / 2
    ph = height - top - bottom
    body = [f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="34" text-anchor="middle" font-size="24">Component weight-mass distribution</text>']
    for panel_index, title in enumerate(("Ranked component mass", "Cumulative concentration")):
        x0 = left + panel_index * (panel + gap)
        body += [f'<rect x="{x0}" y="{top}" width="{panel}" height="{ph}" fill="none" stroke="#444"/>',
                 f'<text x="{x0+panel/2}" y="{top-18}" text-anchor="middle" font-size="19">{title}</text>']
    for ctext, row in result["weight_mass"].items():
        c = int(ctext); color = COLORS.get(c, "#555")
        values = row["component_shares_of_total_sorted"]
        # Thin to at most 600 SVG points without changing endpoints.
        stride = max(1, len(values) // 600)
        indices = list(range(0, len(values), stride))
        if indices[-1] != len(values)-1: indices.append(len(values)-1)
        def xrank(i): return left + math.log2(i + 1) / math.log2(len(values)) * panel
        def ymass(v):
            percent = max(1e-4, 100 * v)
            return top + (2 - math.log10(percent)) / 6 * ph
        body.append(polyline([(xrank(i), ymass(values[i])) for i in indices], color, 2.5))
        cumulative, total = 0.0, sum(values)
        points = []
        for i, value in enumerate(values):
            cumulative += value
            if i % stride == 0 or i == len(values)-1:
                xx = left + panel + gap + (i + 1) / len(values) * panel
                yy = top + (1 - cumulative / total) * ph
                points.append((xx, yy))
        body.append(polyline(points, color, 2.5))
        body.append(f'<text x="{left+15}" y="{top+24+22*list(result["weight_mass"]).index(ctext)}" fill="{color}" font-size="15">C={c}</text>')
    x0 = left + panel + gap
    body.append(f'<line x1="{x0}" y1="{top+ph}" x2="{x0+panel}" y2="{top}" stroke="#999" stroke-dasharray="7,5"/>')
    body += [f'<text x="{left+panel/2}" y="{height-22}" text-anchor="middle" font-size="16">component rank (log2)</text>',
             f'<text x="{x0+panel/2}" y="{height-22}" text-anchor="middle" font-size="16">heaviest components retained (%)</text>',
             f'<text x="22" y="{top+ph/2}" text-anchor="middle" font-size="16" transform="rotate(-90 22 {top+ph/2})">weight mass (%)</text>']
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">{"".join(body)}</svg>')
    convert(output)


def plot(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_kl(result, out_dir / "klkeep.svg")
    plot_mass(result, out_dir / "weight_mass.svg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    raw = json.loads(args.stats.read_text())
    plot(raw, args.out_dir or args.stats.parent)
