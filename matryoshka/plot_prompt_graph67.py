"""Draw prompt attribution graphs (prompt_graph67.py) in the VPD paper's style, VPD and ours side by side, plus the
sweep of the importance-minimality coefficient for every procedure: nodes against the label's probability with the
circuit's pieces at full strength and every other piece off.

Graph panels: one column per prompt token, one row per weight type (Embed; per layer Attn QKV, Attn Out, MLP Up, MLP
Down; Output = the label at the last position). A node is a (position, piece) with optimized CI > 0; nodes of one row and
column are spread side by side. Edges are the paper's attr(c' -> c) = (d a_c / d a_c') a_c' g_c' (direct effects), red
positive, blue negative, opacity and width by |attr| relative to the panel's largest. Edges into the Output row are only
those into the label. Usage: plot_prompt_graph67.py GRAPH_JSON SETTING OUTPUT_PNG"""
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
import numpy as np

INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fbfaf7'
RED, BLUE = '#d1495b', '#3d6fb6'
KIND = (('qkv', 'Attn QKV'), ('o', 'Attn Out'), ('up', 'MLP Up'), ('down', 'MLP Down'))
COLORS = dict(vpd='#008300', ours='#eb6834')
PROCEDURES = (('adversarial', 'o', '-'), ('stochastic', 's', '--'), ('ci', '^', ':'))
PROCEDURE_NAMES = dict(adversarial='with the PGD adversary (the paper\'s figure)', stochastic='random masks only',
                       ci='pieces outside the circuit off (CI-only)')
NAMES = dict(vpd='VPD, published (13.1B tokens)', ours='ours, 13B run (13.1B tokens)')


def row_of(layer, n_layers=4):
    """Row index: 0 = Embed, 1 + 4 * block + kind, top = Output."""
    l = layer.lower()
    if 'unembed' in l or 'output' in l or 'lm_head' in l: return 1 + 4 * n_layers
    if 'embed' in l or l == 'wte': return 0
    m = re.match(r'(?:h\.)?(\d+)\.(attn|mlp)\.(\w+)', l)
    assert m, layer
    block, part, w = int(m.group(1)), m.group(2), m.group(3)
    if part == 'attn': kind = 0 if w.startswith(('q', 'k', 'v')) else 1
    else: kind = 2 if w in ('up', 'c_fc', 'fc', 'gate', 'in') else 3
    return 1 + 4 * block + kind


def draw(ax, g, tokens, label_id, title, n_layers=4):
    n_rows, n_pos = 2 + 4 * n_layers, len(tokens)
    last = n_pos - 1
    keep = lambda key: key[0] < n_rows - 1 or (key[1] == last and key[2] == label_id)
    nodes = {(row_of(n['layer']), n['position'], n['piece']) for n in g['nodes']}
    edges = []
    for e in g['edges']:
        s = (row_of(e['source'][0]), e['source'][1], e['source'][2]); t = (row_of(e['target'][0]), e['target'][1], e['target'][2])
        if not (keep(s) and keep(t)): continue
        nodes.update((s, t)); edges.append((s, t, e['strength_x_source_ci']))
    nodes = {k for k in nodes if keep(k)}
    groups = {}
    for r, p, c in sorted(nodes): groups.setdefault((r, p), []).append(c)
    col_w, gap = 1.0, 0.075
    xy = {}
    for (r, p), cs in groups.items():
        width = min(col_w * 0.92, gap * (len(cs) - 1))
        xs = np.linspace(-width / 2, width / 2, len(cs)) if len(cs) > 1 else [0.]
        for c, x in zip(cs, xs): xy[(r, p, c)] = (p * col_w * 1.5 + x, r)
    top = max((abs(w) for *_, w in edges), default=1.) or 1.
    for s, t, w in sorted(edges, key=lambda e: abs(e[2])):
        a = abs(w) / top
        if a < 0.002: continue
        (x0, y0), (x1, y1) = xy[s], xy[t]
        verts = [(x0, y0), (x0, y0 + 0.55 * (y1 - y0)), (x1, y1 - 0.55 * (y1 - y0)), (x1, y1)]
        path = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(PathPatch(path, fc='none', ec=RED if w > 0 else BLUE, lw=0.25 + 1.6 * a ** 0.5, alpha=min(0.9, 0.06 + 0.9 * a ** 0.5)))
    for (r, p, c), (x, y) in xy.items():
        ax.add_patch(FancyBboxPatch((x - 0.022, y - 0.16), 0.044, 0.32, boxstyle='round,pad=0.004,rounding_size=0.012',
                                    fc='white', ec='#8a8a86', lw=0.6, zorder=3))
    for y in np.arange(0.5, n_rows - 1, 4): ax.axhline(y, color=GRID, lw=0.6, zorder=0)        # block boundaries
    labels = ['Embed'] + [name for _ in range(n_layers) for _, name in KIND] + ['Output']
    ax.set_yticks(range(n_rows), labels, fontsize=7.5, color=MUTED, family='monospace')
    ax.set_xticks([p * col_w * 1.5 for p in range(n_pos)], [t.replace(' ', '·') for t in tokens], fontsize=9, family='monospace')
    ax.set_xlim(-0.75, (n_pos - 1) * col_w * 1.5 + 0.75); ax.set_ylim(-0.7, n_rows - 0.3)
    for side in ('top', 'right', 'left', 'bottom'): ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_facecolor(SURFACE)
    n_piece_nodes = sum(1 for r, _, _ in xy if 0 < r < n_rows - 1)
    ax.set_title(f'{title}\n{n_piece_nodes} pieces in the graph', color=INK, fontsize=10, loc='left')
    return n_piece_nodes


def main():
    data, setting, output = json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], Path(sys.argv[3])
    fig = plt.figure(figsize=(16, 9.2), dpi=170, facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.25)
    counts = {}
    for i, method in enumerate(('vpd', 'ours')):
        ax = fig.add_subplot(grid[0, i])
        counts[method] = draw(ax, data['methods'][method]['graphs'][setting], data['tokens'], data['label_id'], NAMES[method])
    ax = fig.add_subplot(grid[0, 2])
    for method in ('vpd', 'ours'):
        for s, marker, ls in PROCEDURES:
            runs = sorted((r for r in data['methods'][method]['runs'] if r['setting'] == s), key=lambda r: r['nodes'])
            ax.plot([r['nodes'] for r in runs], [r['label_prob_binary'] for r in runs], color=COLORS[method], ls='none', marker=marker,
                    ms=5, markeredgewidth=0, alpha=0.85)                             # points only: one per coefficient
            for r in runs:
                if r['imp_min_coeff'] == data['settings'][s]['paper']:
                    ax.plot([r['nodes']], [r['label_prob_binary']], marker=marker, ms=9, mfc='none', mec=COLORS[method], mew=1.5, ls='none')
    intact = data['methods']['vpd']['runs'][0]['label_prob_intact']
    ax.axhline(intact, color=MUTED, lw=1, ls=':'); ax.text(0.01, intact, ' intact model', fontsize=7.5, color=MUTED, va='bottom', transform=ax.get_yaxis_transform())
    ax.set_xscale('log'); ax.set_ylim(0, 1.02)
    ax.set_xlabel('pieces in the circuit', color=INK, fontsize=9)
    ax.set_ylabel(f'P({data["label"].strip()!r}) with only the circuit (full strength, rest off)', color=INK, fontsize=9)
    ax.set_title('sparsity sweep, every procedure (circled: the paper\'s settings)', color=INK, fontsize=10, loc='left')
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=COLORS[m], lw=1.8, label=NAMES[m]) for m in ('vpd', 'ours')] + [
        *[Line2D([], [], color=MUTED, ls='none', marker=marker, ms=5, label=PROCEDURE_NAMES[s]) for s, marker, ls in PROCEDURES],
        Line2D([], [], color=RED, lw=1.5, label='positive attribution'), Line2D([], [], color=BLUE, lw=1.5, label='negative attribution')]
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.01, 0.975))
    fig.suptitle(f'"{data["prompt"]}" -> "{data["label"]}": the paper\'s circuit optimization ({setting}) on VPD and on ours',
                 color=INK, fontsize=11.5, x=0.01, y=1.0, ha='left')
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.06, right=0.99)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, counts, flush=True)


if __name__ == '__main__': main()
