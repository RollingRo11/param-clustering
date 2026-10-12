"""Figure for cancellation_probe.py results (VPD published, VPD paired, par-auto 1B)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
runs = {'VPD published (13B tok)': 'vpd_published', 'VPD paired (1B tok)': 'vpd_paired', 'par-auto (1B tok)': 'parauto_1b'}
color = {'VPD published (13B tok)': '#c0392b', 'VPD paired (1B tok)': '#e67e22', 'par-auto (1B tok)': '#2471a3'}
data = {k: json.loads((src/f'{v}.json').read_text()) for k, v in runs.items() if (src/f'{v}.json').exists()}
for k, v in runs.items():                                   # 100%-kept control from a separate job
    if k in data and (src/f'{v}_keep1.json').exists(): data[k]['keeps'].update(json.loads((src/f'{v}_keep1.json').read_text())['keeps'])
names = ['0', '1', '2', '3']
subsets = [n for n in next(iter(data.values()))['keeps']['0.01']['kl']]

def worse_with_more(kl):
    """Of all (S, S + one more layer) steps, the share where masking the extra layer LOWERED KL."""
    steps = down = 0
    for s in subsets:
        have = set(s.split('+'))
        for l in names:
            if l in have: continue
            bigger = '+'.join(sorted(have | {l}))
            steps += 1; down += kl[bigger] < kl[s]
    return down/steps

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
width = .26
bars = [('0+1+2', 'layers 0-2 masked,\nreal layer 3'), ('0+1+2+3', 'layers 0-2 masked,\nmasked layer 3'), ('3', 'only layer 3\nmasked')]
for i, (label, d) in enumerate(data.items()):
    kl = d['keeps']['0.01']['kl']
    ax[0].bar([x+(i-1)*width for x in range(3)], [kl[s] for s, _ in bars], width, color=color[label], label=label)
    keeps = sorted([k for k in d['keeps'] if float(k) < 1], key=float)
    xs = [float(k)*100 for k in keeps]
    ax[1].plot(xs, [worse_with_more(d['keeps'][k]['kl']) for k in keeps], 'o-', color=color[label], label=label)
    ax[2].plot([d['keeps'][k]['kl']['0+1+2+3'] for k in keeps],
               [sum(d['keeps'][k]['kept_act_err_all_layers'][l]['median'] for l in ('1', '2', '3'))/3 for k in keeps], 'o-', color=color[label], label=label)
ax[0].set_xticks(range(3), [b for _, b in bars]); ax[0].set_ylabel('KL to real model'); ax[0].set_title('Same 1% mask: does masked layer 3 undo earlier damage?')
ax[0].legend(fontsize=8)
ax[1].set_xscale('log'); ax[1].set_xlabel('% components kept'); ax[1].set_ylim(-.02, 1)
ax[1].set_ylabel('share of cases'); ax[1].set_title('Masking one more layer LOWERS KL')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('KL, all layers masked'); ax[2].set_ylabel('relative error (median, layers 1-3)')
ax[2].set_title('Kept components: activation vs real model')
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); dst.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dst, dpi=150)
for label, d in data.items():
    for k in sorted(d['keeps'], key=float):
        r = d['keeps'][k]
        print(label, k, 'KL', {s: round(v, 2) for s, v in r['kl'].items() if '+' not in s or s == '0+1+2+3'},
              'down', round(worse_with_more(r['kl']), 2),
              'err all/sum', round(r['final_error_norms']['all_layers']['median'], 3), round(r['final_error_norms']['sum_of_single_norms']['median'], 3),
              'kept act err', [round(r['kept_act_err_all_layers'][l]['median'], 3) for l in names])
