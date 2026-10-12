"""Controlled stronger sparsity comparison, with VPD as a reference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, credentials, save_json, upload

def main():
    credentials(); out=ROOT/'output/gim_p010_5000'
    old=ROOT/'output/evaluations/gim_normal_vpd_5000'
    curves={name:json.loads(path.read_text()) for name,path in [
        ('gim_p010',out/'eval/klkeep.json'),('gim_p025',out/'control_p025/klkeep.json'),('gim_p05',out/'control_p05/klkeep.json'),('vpd',out/'vpd/klkeep.json')]}
    for key in ('training_steps','training_positions','training_data_sha256','data_sha256','target_sha256','positions','precision','gpu'):
        assert len({c[key] for c in curves.values()})==1,key
    assert curves['gim_p010']['training_steps']==5000
    configs=[curves[n]['checkpoint_provenance']['training_config'] for n in ('gim_p010','gim_p05')]
    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'p_end'}
    assert configs[0]['p_end']==.1 and configs[1]['p_end']==.5
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for name,label,color in [('gim_p05','GIM · target p=0.5','#176b46'),('gim_p025','GIM · target p=0.25','#dc6b18'),('gim_p010','GIM · target p=0.1','#c43d4d'),('vpd','VPD · reference','#9250a1')]:
        rows=[r for r in curves[name]['curve'] if r['kept_components']>0]
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],marker='o',markersize=4,color=color,label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position ↓'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    axes[0].set_xlabel('Components kept per position across all 24 matrices'); axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('Stronger sparsity · 5,000 steps / 163.8M training positions each')
    fig.supxlabel('GIM schedules: p≈0.641 / 0.461 / 0.354 at step 5,000. Same LR, data, seed and components; p=0.1 trained on H200, older runs on H100.\nSame 65,536 held-out positions; all evaluated on H200. VPD uses its native CI sensor and 38,912 components vs GIM’s 43,008.',fontsize=8)
    fig.savefig(out/'klkeep_sparsity.png',dpi=180)
    for ax in axes: ax.set_yscale('log')
    fig.savefig(out/'klkeep_sparsity_logy.png',dpi=180); plt.close(fig)
    save_json(curves,out/'comparison.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gim-sparsity-20260910',
        name='gim-p010-p025-p05-klkeep-5000',job_type='evaluation',dir=str(out),
        config=dict(training_steps=5000,changed_settings=['p_end']))
    run.log({'eval/klkeep':wandb.Image(str(out/'klkeep_sparsity.png')),
             'eval/klkeep_logy':wandb.Image(str(out/'klkeep_sparsity_logy.png'))})
    save_json(dict(id=run.id,url=run.url),out/'comparison_wandb.json'); run.finish()
    for path in (out/'comparison.json',out/'klkeep_sparsity.png',out/'klkeep_sparsity_logy.png',out/'comparison_wandb.json'):
        upload(path,'experiments/gim_p010_5000/'+path.name)
    print('GIM_SPARSITY_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__': main()
