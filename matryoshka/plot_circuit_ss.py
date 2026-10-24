"""Attribution graphs from circuit_ss.py, VPD (left) vs ours (right) at each method's active set (VPD CI > 0, ours above
floor). Edge value = the paper's attr(c' -> c) x source CI; the TOP_EDGES largest |attr| per panel are drawn (blue positive,
red negative, width and opacity by |attr| / panel max). Usage: plot_circuit_ss.py CIRCUIT.json OUT.png [OURS_LABEL]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, BG = '#0b0b0b', '#52514e', '#fcfcfb'
POS, NEG, NODE = '#2a78d6', '#e34948', '#8a8983'
ROWS = ['embed'] + [f'{l}.{m}' for l in range(2) for m in ('attn.q', 'attn.k', 'attn.v', 'attn.o', 'mlp.up', 'mlp.down')] + ['output']
TOP_EDGES = 120


def panel(ax, g, tokens, label_id, title):
    keep = lambda n: n[0] != 'output' or (n[1] == len(tokens) - 1 and n[2] == label_id)
    edges_all = [e for e in g['edges'] if keep(e['target'])]
    edges = sorted(edges_all, key=lambda e: -abs(e['strength_x_source_ci']))[:TOP_EDGES]
    weight = defaultdict(float)
    for e in edges_all:
        weight[tuple(e['source'])] += abs(e['strength_x_source_ci']); weight[tuple(e['target'])] += abs(e['strength_x_source_ci'])
    nodes = {(n['layer'], n['position'], n['piece']) for n in g['nodes']} | {tuple(e['source']) for e in edges} | {tuple(e['target']) for e in edges}
    nodes = {n for n in nodes if n[0] in ROWS and keep(n)}
    cells = defaultdict(list)
    for n in nodes: cells[(n[0], n[1])].append(n)
    xy = {}
    for (layer, pos), members in cells.items():
        members.sort(key=lambda n: -weight[n])
        for i, n in enumerate(members):
            off = 0 if len(members) == 1 else (i / (len(members) - 1) - 0.5) * min(0.8, 0.06 * len(members))
            xy[n] = (pos + off, ROWS.index(layer))
    top = max([abs(e['strength_x_source_ci']) for e in edges] + [1e-9])
    for e in sorted(edges, key=lambda e: abs(e['strength_x_source_ci'])):
        s, t, v = tuple(e['source']), tuple(e['target']), e['strength_x_source_ci']
        if s not in xy or t not in xy: continue
        r = abs(v) / top
        ax.plot([xy[s][0], xy[t][0]], [xy[s][1], xy[t][1]], color=POS if v > 0 else NEG, lw=0.3 + 2.7 * r, alpha=0.15 + 0.75 * r, zorder=2)
    ax.scatter([p[0] for p in xy.values()], [p[1] for p in xy.values()], s=7, color=NODE, zorder=3, linewidths=0)
    for n in sorted(xy, key=lambda n: -weight[n])[:6]:
        if n[0] in ('embed', 'output'): continue
        ax.annotate(f'{n[0]}:{n[2]}', xy[n], fontsize=5.5, color=INK, xytext=(3, 3), textcoords='offset points', zorder=4)
    if ('output', len(tokens) - 1, label_id) in xy:
        ax.scatter(*xy[('output', len(tokens) - 1, label_id)], s=40, facecolor='none', edgecolor=INK, lw=1, zorder=4)
    ax.set_yticks(range(len(ROWS)), ROWS, fontsize=7); ax.set_xticks(range(len(tokens)), [repr(t)[1:-1] for t in tokens], fontsize=8)
    ax.set_xlim(-0.6, len(tokens) - 0.4); ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_facecolor(BG); ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values(): side.set_visible(False)
    for y in range(len(ROWS)): ax.axhline(y, color='#eeede8', lw=0.5, zorder=0)
    ax.set_title(title, fontsize=8.5, color=INK, loc='left')


def main():
    d = json.loads(Path(sys.argv[1]).read_text()); out = sys.argv[2]
    ours_label = sys.argv[3] if len(sys.argv) > 3 else 'Ours (recipe B, above-floor set)'
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150); fig.patch.set_facecolor(BG)
    for ax, (name, label) in zip(axes, [('vpd', 'Goodfire VPD (s-9e774a82, CI > 0 set)'), ('ours', ours_label)]):
        m = d['methods'][name]; g = m['graph']
        stats = (f"{len(g['nodes'])} active pieces ({m['active_mean']:.0f} per position), {len(g['edges'])} edges\n"
                 f"P(label): intact {m['p_intact']:.2f}, circuit only {m['p_circuit_only']:.2f}, "
                 f"4-step attack {m['p_pgd4']:.2f}, 320-step attack {m['p_pgd320']:.2f}")
        panel(ax, g, d['tokens'], d['label_id'], f'{label}\n{stats}')
    fig.suptitle(f"\"{d['prompt']}\" -> \"{d['label']}\"   |   circuit = each method's active set   |   top {TOP_EDGES} edges by |attr|",
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(out); print('SAVED', out)


if __name__ == '__main__': main()
