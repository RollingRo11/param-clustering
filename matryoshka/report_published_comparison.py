"""Compare published VPD against finished par-auto, explicitly unequal budgets."""
import json
import shutil
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, HERE, credentials, save_json, upload, sha


def main():
    credentials(); out=ROOT/'output/published_vpd_vs_finished_parauto'
    curves={name:json.loads((out/name/'klkeep.json').read_text()) for name in ('parauto','vpd')}
    for key in ('data_sha256','target_sha256','positions','precision'):
        assert curves['parauto'][key] == curves['vpd'][key], key
    assert abs(curves['parauto']['baseline_ce']-curves['vpd']['baseline_ce']) < 1e-6
    assert curves['parauto']['training_positions'] == 1000000000
    assert curves['vpd']['training_steps'] == 400000
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for name,color,label in [('parauto','#176b46','Par-auto · 1B positions / 30,518 updates'),
                             ('vpd','#9250a1','Published VPD · 13.1B positions / 400k steps')]:
        rows=[r for r in curves[name]['curve'] if r['kept_components']>0]
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],label=label,color=color,marker='o')
            ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_xlabel('Components kept per position, across all 24 matrices')
    axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('Published VPD vs finished par-auto · native-sensor KLKeep\nSame 65,536 held-out positions; different training budgets')
    figure=out/'klkeep_published_vs_finished.png'; fig.savefig(figure,dpi=180); plt.close(fig)
    save_json(curves,out/'comparison.json')
    shutil.copyfile(ROOT/'checkpoints/published_vs_finished/inputs_ready.json',out/'inputs_ready.json')
    scripts = [HERE/name for name in ('evaluate_checkpoint.py','stage_published_comparison.py',
               'report_published_comparison.py','slurm_stage_published.sh','slurm_eval_published.sh',
               'common.py','train_pair.py','vpd_published.yaml','target_config.yaml')]
    save_json({p.name:sha(p) for p in scripts},out/'source_manifest.json')
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for path in scripts: archive.add(path,arcname=path.name,recursive=False)
    save_json(upload(out/'source.tar.gz','evaluations/published_vs_finished/source.tar.gz'),out/'source_r2.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',
                   name='published-vpd-vs-finished-parauto-klkeep',job_type='evaluation',dir=str(out),
                   config=dict(matched_training_budget=False,positions=65536))
    run.log({'eval/klkeep':wandb.Image(str(figure))})
    save_json(dict(id=run.id,url=run.url),out/'wandb.json'); run.finish()
    receipts=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and 'wandb' not in path.parts and path.suffix in ('.json','.png','.md'):
            receipts.append(upload(path,'evaluations/published_vs_finished/'+str(path.relative_to(out))))
    save_json(receipts,out/'r2_receipts.json')


if __name__ == '__main__': main()
