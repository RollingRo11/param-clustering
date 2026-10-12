"""Finite local ablation score vs GIM, on 8,192 held-out tokens, with VPD as a reference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, credentials, save_json, upload

def main():
    credentials(); out=ROOT/'output/finite_5000'
    curves={name:json.loads(path.read_text()) for name,path in [
        ('finite',out/'eval/klkeep.json'),('gim',out/'control_gim/klkeep.json'),('vpd',out/'vpd/klkeep.json')]}
    for key in ('training_steps','training_positions','training_data_sha256','data_sha256','target_sha256','positions','precision','gpu'):
        assert len({c[key] for c in curves.values()})==1,key
    assert curves['finite']['training_steps']==5000 and curves['finite']['positions']==8192
    configs=[curves[n]['checkpoint_provenance']['training_config'] for n in ('finite','gim')]
    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'batch'}
    train=json.loads((out/'result.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for name,label,color in [('gim','Par-auto · GIM score','#176b46'),('finite','Par-auto · finite local ablation','#2563eb'),('vpd','VPD · reference','#9250a1')]:
        rows=[r for r in curves[name]['curve'] if r['kept_components']>0]
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],marker='o',markersize=4,color=color,label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('KL(target || masked), nats / position ↓'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    axes[0].set_xlabel('Components kept per position across all 24 matrices'); axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('Finite local ablation score vs GIM · 5,000 steps each')
    fig.supxlabel('Finite run: loss on 1,024 sampled tokens per step (GIM: all 32,768); same LR, p, data, seed, components. Each ranked by its own score.\n'
                  f'8,192 held-out tokens, all evaluated on one GPU. Finite training: {train["seconds"]/max(1,train["steps_this_job"]):.3f} s/step. VPD: 38,912 components vs 43,008.',fontsize=8)
    fig.savefig(out/'klkeep_finite_logy.png',dpi=180); plt.close(fig)
    save_json(curves,out/'comparison.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='finite-ablation-20260910',
        name='finite-vs-gim-klkeep-5000',job_type='evaluation',dir=str(out),
        config=dict(training_steps=5000,changed_settings=['score','batch'],eval_positions=8192))
    run.log({'eval/klkeep_logy':wandb.Image(str(out/'klkeep_finite_logy.png'))})
    save_json(dict(id=run.id,url=run.url),out/'comparison_wandb.json'); run.finish()
    for path in (out/'comparison.json',out/'klkeep_finite_logy.png',out/'comparison_wandb.json'):
        upload(path,'experiments/finite_5000/'+path.name)
    print('FINITE_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__': main()
