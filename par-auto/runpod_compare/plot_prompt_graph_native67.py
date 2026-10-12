"""The princess graph with each decomposition's own circuit finder, in the VPD paper's style (draw() of
plot_prompt_graph67.py): left, published VPD through the paper's procedure (prompt_graph67.py, the adversarial setting at
the paper's coefficient); right, our 13B run through MAttr (prompt_graph_mattr67.py, top-k at the description length).
Titles give P(label) and the KL to the intact next-token distribution with only the circuit's pieces on (rest off).
Usage: plot_prompt_graph_native67.py PROMPT_GRAPH_JSON MATTR_GRAPH_JSON OUTPUT_PNG"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_prompt_graph67 import BLUE, INK, RED, SURFACE, draw


def main():
    paper, ours, output = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    assert paper['token_ids'] == ours['token_ids'] and paper['label_id'] == ours['label_id']
    setting = 'adversarial'
    run = next(r for r in paper['methods']['vpd']['runs'] if r['setting'] == setting and r['imp_min_coeff'] == paper['settings'][setting]['paper'])
    label = paper['label'].strip()
    fig, axes = plt.subplots(1, 2, figsize=(13, 9.2), dpi=170, facecolor=SURFACE, gridspec_kw=dict(wspace=0.22))
    counts = dict(
        vpd=draw(axes[0], paper['methods']['vpd']['graphs'][setting], paper['tokens'], paper['label_id'],
                 f'VPD, published: the paper\'s procedure (PGD adversary, sparsity {run["imp_min_coeff"]:g})\n'
                 f'only the circuit on: P({label!r}) {run["label_prob_binary"]:.2f}, KL {run["kl_binary"]:.1f}'),
        ours=draw(axes[1], ours['graph'], ours['tokens'], ours['label_id'],
                  f'ours, 13B run: MAttr, fewest pieces with KL <= {ours["threshold"]:g}\n'
                  f'only the circuit on: P({label!r}) {ours["p_label_circuit"]:.2f}, KL {ours["kl_circuit"]:.2f}'))
    handles = [Line2D([], [], color=RED, lw=1.5, label='positive attribution'), Line2D([], [], color=BLUE, lw=1.5, label='negative attribution')]
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.01, 0.965))
    fig.suptitle(f'"{paper["prompt"]}" -> "{paper["label"]}": each decomposition with its own circuit finder '
                 f'(intact P({label!r}) {run["label_prob_intact"]:.2f})', color=INK, fontsize=11.5, x=0.01, y=1.0, ha='left')
    fig.subplots_adjust(top=0.87, bottom=0.06, left=0.07, right=0.99)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, counts, flush=True)


if __name__ == '__main__': main()
