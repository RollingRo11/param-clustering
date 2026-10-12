"""German removal by editing (german_edit67.py): one piece (ours from the finished 13B run; VPD's h.3.attn.v_proj:513, the
LessWrong post's pick; VPD's h.3.mlp.c_fc:722, picked by our rule) against LoRA rank 1 and 4, at German budgets of 2 to
2,048 tokens, 9 seeds each, every arm at its own config chosen per budget by the post's rule. Left: German CE increase on
held-out Europarl German against German training tokens (median over seeds, band = middle 50% of seeds; dotted: German at chance).
Middle: English damage, CE increase on held-out Pile (dotted: the 0.1-nat selection threshold, applied on a dev split).
Right: CE increase on every eval set at 2,048 German tokens (median over seeds).
Usage: plot_german67.py RESULT_DIR OUTPUT_PNG"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

# (label, file, color, line style); validated palette slots: orange = ours (as the 13B run elsewhere), green/yellow = VPD,
# blue/violet dashed (hatched bars) = LoRA.
ARMS = [('ours, 1 piece (13B run, h.3.mlp.c_fc:2593)', 'piece_ours13b_2593.json', '#eb6834', '-'),
        ("VPD, 1 piece (post's h.3.attn.v_proj:513)", 'piece_vpd_513.json', '#008300', '-'),
        ('VPD, 1 piece (our rule, h.3.mlp.c_fc:722)', 'piece_vpd_722.json', '#eda100', '-'),
        ('LoRA rank 1 (all 24 matrices)', 'lora_r1.json', '#2a78d6', '--'),
        ('LoRA rank 4 (all 24 matrices)', 'lora_r4.json', '#4a3aa7', '--')]
SETS = [('de_eval', 'German'), ('en_eval', 'Europarl\nEnglish'), ('pile_eval', 'Pile'), ('fr_eval', 'French'),
        ('es_eval', 'Spanish'), ('it_eval', 'Italian'), ('code_eval', 'code')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)


def main():
    folder, output = Path(sys.argv[1]), Path(sys.argv[2])
    arms = [(name, json.loads((folder/f).read_text()), color, ls) for name, f, color, ls in ARMS]
    budgets = sorted({int(b) for _, d, _, _ in arms for b in d['selected']})
    chance = arms[0][1]['ceiling'] - arms[0][1]['base_ce']['de_eval']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=170, facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, key, title, ylabel in ((axes[0], 'de_eval', 'German damage', 'German CE increase (nats)'),
                                   (axes[1], 'pile_eval', 'English damage', 'Pile CE increase (nats)')):
        for name, d, color, ls in arms:
            values = [[s['eval'][key] for s in d['selected'][str(b)]['seeds']] for b in budgets]
            ax.fill_between(budgets, [np.percentile(v, 25) for v in values], [np.percentile(v, 75) for v in values], color=color, alpha=0.12, lw=0)
            ax.plot(budgets, [np.median(v) for v in values], color=color, ls=ls, lw=2.2, marker='o', ms=3.6, markeredgewidth=0)
            for b, v in zip(budgets, values):
                if not d['selected'][str(b)]['eligible']:
                    ax.plot([b], [np.median(v)], ls='none', marker='x', ms=7, color=color)
        if key == 'de_eval':
            ax.axhline(chance, color=MUTED, lw=1, ls=':'); ax.text(budgets[-1], chance, 'German at chance ', ha='right', va='bottom', fontsize=7.5, color=MUTED)
            ax.set_ylim(0, 15)
        else:
            ax.axhline(0.1, color=MUTED, lw=1, ls=':'); ax.text(budgets[-1], 0.1, '0.1 nats ', ha='right', va='bottom', fontsize=7.5, color=MUTED)
            ax.set_yscale('log'); ax.set_ylim(1e-3, 10)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(FixedLocator(budgets)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('German training tokens', color=INK, fontsize=9); ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc='left'); style(ax)
    ax = axes[2]
    top = max(budgets); width = 0.8 / len(arms)
    for i, (name, d, color, ls) in enumerate(arms):
        med = [np.median([s['eval'][k] for s in d['selected'][str(top)]['seeds']]) for k, _ in SETS]
        ax.bar(np.arange(len(SETS)) + (i - (len(arms) - 1) / 2) * width, med, width * 0.92, color=color,
               hatch='//' if ls == '--' else None, edgecolor=SURFACE, lw=0)
    ax.set_yscale('log'); ax.set_ylim(1e-3, 20); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xticks(np.arange(len(SETS)), [label for _, label in SETS], fontsize=8)
    ax.set_ylabel('CE increase (nats, median over seeds)', color=INK, fontsize=9)
    ax.set_title(f'every eval set at {top:,} German tokens', color=INK, fontsize=10, loc='left'); style(ax)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, label=n) for n, _, c, ls in arms]
    handles.append(Line2D([], [], ls='none', marker='x', ms=7, color=MUTED, label='no config met the dev English limit (least English damage shown)'))
    fig.suptitle('German removal: editing one decomposition piece vs LoRA (67M target; 9 seeds per budget)', color=INK, fontsize=11,
                 x=0.01, y=1.0, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, d, _, _ in arms:
        print(name)
        for b in budgets:
            sel = d['selected'][str(b)]
            v = lambda k: np.median([s['eval'][k] for s in sel['seeds']])
            print(f'  {b:5d} cfg {sel["config"]} ok={sel["eligible"]} de {v("de_eval"):.2f} pile {v("pile_eval"):.3f} '
                  f'en {v("en_eval"):.3f} fr {v("fr_eval"):.2f} es {v("es_eval"):.2f} it {v("it_eval"):.2f} code {v("code_eval"):.3f}'
                  + (f' alpha {np.median([s["alpha"] for s in sel["seeds"]]):.2f}' if 'alpha' in sel['seeds'][0] else ''))


if __name__ == '__main__': main()
