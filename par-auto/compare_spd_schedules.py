"""Matched native-sensor KLKeep at 5k, 10k, and 50k steps; solid curves only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch
import parauto_core as core
import parauto_toy as toy
from parauto_target import load_target, sample, fingerprint, atomic_json, atomic_save
from parauto_evaluate import TARGET_SHA
from evaluate_klkeep import removal_ranks, keep_masks, evaluate_curve, summarize


def draw(result, root):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    palettes = {'method': ('#63a8d3', '#2171b5', '#08306b'),
                'spd': ('#f4b35e', '#e87517', '#a63603')}
    for ax, (metric, title) in zip(axes, (('kl_target','KL after softmax (T=1)'), ('mse_target','Raw-output MSE'))):
        for family, name in (('method','Par-auto'),('spd','SPD')):
            for steps, color in zip((5000, 10000, 50000), palettes[family]):
                curve=result['methods'][f'{family}_{steps}']
                ax.plot(curve['kept'][:-1],curve['metrics'][metric]['mean'][:-1],
                        label=f'{name} {steps//1000}k',color=color,linestyle='-',linewidth=2)
        ax.set(title=title,ylabel='Mean error',xlabel='Components retained (all four matrices)',xlim=(1,1600))
        ax.set_xscale('log'); ax.set_xticks([1,4,16,64,256,1600]); ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yscale('symlog',linthresh=1e-10); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
        ax.legend(fontsize=9,ncol=2,loc='lower left')
    fig.suptitle('Native-sensor KLKeep · 4,096 shared held-out inputs · 400 components/matrix\nFrozen 10k-step target · lower is better')
    for ext in ('png','svg'): fig.savefig(root/f'klkeep.{ext}',dpi=180)
    plt.close(fig)


def report(result, root):
    lines=['# Par-auto versus SPD: matched short schedules','',
           'Par-auto uses projected Adam, cosine LR 1e-3 to 1e-5, and p annealing from 1 to 0.5 over the first 80% of each run. '
           'SPD uses the pinned unmodified authors’ trainer and its published constant LR 1e-3, without warmup. Each run starts independently. '
           'Both use 400 components per matrix and batch 2,048. The frozen target was trained for 10,000 steps.','',
           '![Matched KLKeep](klkeep.png)','',
           'All 1,601 budgets use identical saved held-out inputs and true nonlinear masked forwards. '
           'Par-auto ranks by output-effect RMS; SPD ranks by learned causal importance. Each per-input global ranking is frozen before cumulative least-first removal. '
           'Kept components have binary masks, including for SPD. Raw scores and per-input metrics for the new SPD runs are saved alongside the report.','',
           'KL means KL(softmax(target output) || softmax(masked output)) at temperature 1; these are regression outputs, so raw MSE is also shown. '
           'Primary curves include each decomposition’s all-kept reconstruction error. Separate own-all-kept distortion is retained in JSON. '
           'Zero-kept points are omitted only from the logarithmic plot.','']
    for metric,title in (('kl_target','Mean KL'),('mse_target','Mean output MSE')):
        lines += [title,'','| Retained | Par-auto 5k | SPD 5k | Par-auto 10k | SPD 10k | Par-auto 50k | SPD 50k |','|---:|---:|---:|---:|---:|---:|---:|']
        for kept in (4,8,16,32,64,100,400,1600):
            vals=[result['methods'][f'{family}_{steps}']['selected_budgets'][str(kept)][metric] for steps in (5000,10000,50000) for family in ('method','spd')]
            lines.append(f'| {kept} | '+' | '.join(f'{v:.3g}' for v in vals)+' |')
        lines.append('')
    lines += ['| Run | Fit seconds | Peak allocated tensors (GiB) |','|---|---:|---:|']
    for steps in (5000,10000,50000):
        for family in ('method','spd'):
            meta=result['training'][f'{family}_{steps}']
            fit=meta.get('phases',{}).get('fit',{})
            seconds=fit.get('wall_seconds',meta.get('fit_seconds'))
            memory=fit.get('peak_torch_allocated_bytes',meta.get('peak_fit_allocated_bytes'))
            lines.append(f"| {'Par-auto' if family=='method' else 'SPD'} {steps//1000}k | {seconds:.2f} | {memory/2**30:.3f} |")
    lines += ['', 'One training seed per condition; runtime measurements span separate V100 allocations. '
              'The SPD and historical 50k timers include final checkpoint/logging work; standalone par-auto short-fit timing excludes final saves. '
              'Par-auto samples a fixed 16,384-example training cache; SPD generates fresh inputs from the same distribution. '
              'The evaluation set is identical. This tests cumulative prefixes of each ranking, not arbitrary subsets.','']
    (root/'report.md').write_text('\n'.join(lines))


import parauto_tracking as tracking

@torch.no_grad()
def main():
    from benchmark_spd import author_target, load_author_model
    from spd.configs import Config
    from spd.models.component_utils import calc_causal_importances
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--spd-runs',type=Path,required=True)
    parser.add_argument('--method-runs',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--spd-reference-run',type=Path,required=True)
    args=parser.parse_args()
    tracking.start_cli(args)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    reference=json.loads((args.reference/'result.json').read_text())
    short=json.loads((args.method_runs/'comparison/result.json').read_text())
    assert fingerprint(args.target)==TARGET_SHA==reference['target_sha256']==short['target_sha256']
    for key in ('events','seed','temperature'): assert reference[key]==short[key]
    target=load_target(args.target,'cuda')
    x=sample(reference['events'],100,torch.Generator(device='cuda').manual_seed(reference['seed']),'cuda')
    saved=torch.load(args.reference/'par_auto_events.pt',map_location='cpu',weights_only=True)
    torch.testing.assert_close(saved['x'],x.cpu(),atol=0,rtol=0)
    meaningful=saved['meaningful']; del saved
    root=args.spd_runs/'comparison'; root.mkdir(exist_ok=True)
    result={k:reference[k] for k in ('events','meaningful_events','seed','target_sha256','target_steps','temperature','tie_break','kl_definition','metric_caveat')}
    result['format']='matched_spd_schedules_v1'
    result['sensors']={'method':'output-effect RMS','spd':'native learned causal importance on original target activations'}
    result['ranking']='Global per-input ranking, frozen before cumulative removal; binary masks'
    result['methods']={'method_50000':reference['methods']['par_auto'],'spd_50000':reference['methods']['spd']}
    result['checkpoints']={f'{name}_50000':reference['checkpoints'][old] for name,old in (('method','par_auto'),('spd','spd'))}
    result['checkpoint_sha256']={f'{name}_50000':reference['checkpoint_sha256'][old] for name,old in (('method','par_auto'),('spd','spd'))}
    result['training']={'method_50000':json.loads((Path(reference['checkpoints']['par_auto']).parent/'result.json').read_text()),
                        'spd_50000':json.loads((args.spd_reference_run/'result.json').read_text())}
    for steps in (5000,10000):
        key=f'method_{steps}'; old=f'projected_{steps}'
        result['methods'][key]=short['methods'][old]
        result['checkpoints'][key]=short['checkpoints'][old]
        result['checkpoint_sha256'][key]=short['checkpoint_sha256'][old]
        result['training'][key]=short['training'][old]
        run=args.spd_runs/str(steps)
        meta=json.loads((run/'result.json').read_text())
        assert meta['steps_per_matrix']==steps and meta['components_per_matrix']==400 and meta['target_sha256']==TARGET_SHA
        cfg=Config(**json.loads((run/'executed_config.json').read_text()))
        assert cfg.lr_schedule=='constant' and cfg.lr==.001 and cfg.lr_warmup_pct==0
        author=load_author_model(author_target(args.target),cfg,run/f'model_{steps}.pth')
        torch.testing.assert_close(author.model(x),target(x),atol=1e-6,rtol=1e-5)
        components={k.replace('-','.'):v for k,v in author.components.items()}
        gates={k.replace('-','.'):v for k,v in author.gates.items()}
        names=[f'layers.{i}.{kind}' for i in range(2) for kind in ('mlp_in','mlp_out')]
        parts=[core.FreePieces(components[n].A.detach(),components[n].B.T.detach()) for n in names]
        _,pre=author.forward_with_pre_forward_cache_hooks(x,module_names=names)
        ci,_=calc_causal_importances(pre,{n:components[n].A for n in names},gates)
        scores=[ci[n] for n in names]; ranks=removal_ranks(scores)
        for removed in (0,800,1600):
            masks=[m[0].float() for m in keep_masks(ranks[:32],torch.tensor([removed],device='cuda'),[400]*4)]
            expected=author.forward_with_components(x[:32],components=components,masks=dict(zip(names,masks,strict=True)))
            actual=toy.forward_with_pieces(target,x[:32],parts,masks)[0]
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
        print(f'EVALUATE SPD {steps}',flush=True)
        values,_=evaluate_curve(target,x,parts,ranks,batch=128,level_batch=8,temperature=1.)
        key=f'spd_{steps}'
        result['methods'][key]=summarize(values,meaningful,[400]*4,scores)
        result['training'][key]=meta
        result['checkpoints'][key]=str(run/f'model_{steps}.pth')
        result['checkpoint_sha256'][key]=fingerprint(run/f'model_{steps}.pth')
        result.setdefault('authors_forward_parity',{})[key]=True
        atomic_save({'x':x.cpu(),'meaningful':meaningful,'removal_ranks':ranks.cpu(),'scores':[s.cpu() for s in scores],**values},root/f'{key}_events.pt')
        atomic_json(result,root/'result.json')
        del author,components,gates,parts,pre,ci,scores,ranks,values
    draw(result,root); report(result,root)
    print('MATCHED_SPD_SCHEDULES_COMPLETE',flush=True)


if __name__=='__main__': tracking.run_main(main)
