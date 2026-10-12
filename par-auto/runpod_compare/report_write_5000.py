"""Write-size score vs GIM (only the score differs), with VPD as a reference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, credentials, save_json, upload

def main():
    credentials(); out=ROOT/'output/write_5000'
    curves={name:json.loads(path.read_text()) for name,path in [
        ('write',out/'eval/klkeep.json'),('gim',out/'control_gim/klkeep.json'),('vpd',out/'vpd/klkeep.json')]}
    for key in ('training_steps','training_positions','training_data_sha256','data_sha256','target_sha256','positions','precision','gpu'):
        assert len({c[key] for c in curves.values()})==1,key
    assert curves['write']['training_steps']==5000
    configs=[curves[n]['checkpoint_provenance']['training_config'] for n in ('write','gim')]
    assert configs[0]==configs[1]
    train=json.loads((out/'result.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for name,label,color in [('gim','Par-auto · GIM score','#176b46'),('write','Par-auto · write-size score','#2563eb'),('vpd','VPD · reference','#9250a1')]:
        rows=[r for r in curves[name]['curve'] if r['kept_components']>0]
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],marker='o',markersize=4,color=color,label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('KL(target || masked), nats / position ↓'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    axes[0].set_xlabel('Components kept per position across all 24 matrices'); axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('Write-size score vs GIM · 5,000 steps / 163.8M training positions each')
    fig.supxlabel('Par-auto runs differ only in the training score (same LR, p, data, seed, components); each ranked by its own score.\n'
                  f'Same 65,536 held-out positions, all evaluated on one GPU. Write-size training: {train["seconds"]/train["completed"]:.3f} s/step. VPD: 38,912 components vs 43,008.',fontsize=8)
    fig.savefig(out/'klkeep_write_logy.png',dpi=180); plt.close(fig)
    save_json(curves,out/'comparison.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='write-norm-20260910',
        name='write-vs-gim-klkeep-5000',job_type='evaluation',dir=str(out),
        config=dict(training_steps=5000,changed_settings=['score']))
    run.log({'eval/klkeep_logy':wandb.Image(str(out/'klkeep_write_logy.png'))})
    save_json(dict(id=run.id,url=run.url),out/'comparison_wandb.json'); run.finish()
    for path in (out/'comparison.json',out/'klkeep_write_logy.png',out/'comparison_wandb.json'):
        upload(path,'experiments/write_5000/'+path.name)
    print('WRITE_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__': main()
