"""A 2x2 sensor/train comparison separates retraining from evaluation ranking."""
import json
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT,HERE,save_json,credentials,upload,sha


def main():
    credentials(); out=ROOT/'output/ig5_retraining'
    plan=json.loads((out/'plan.json').read_text()); curves={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    styles=[('gradient','gradient','#176b46'),('ig5','ig','#2364d2'),
            ('gradient','ig','#d68b15'),('ig5','gradient','#9250a1')]
    for training,sensor,color in styles:
        name=f'{training}_train_{sensor}_eval'; curve=json.loads((out/'eval'/name/'klkeep.json').read_text())
        assert curve['training_steps']==plan['steps']; curves[name]=curve
        rows=[r for r in curve['curve'] if r['kept_components']>0]
        label=f'{training} training / {sensor} ranking'
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],label=label,color=color,marker='o')
            ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position'); ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    for key in ('training_data_sha256','data_sha256','target_sha256','positions','training_positions','precision'):
        assert len({v[key] for v in curves.values()})==1,key
    axes[0].set_xlabel('Components kept per position across all 24 matrices'); axes[1].set_xlabel('Fraction kept')
    fig.suptitle(f'IG5 retraining vs ordinary gradients · {plan["steps"]:,} fresh updates each')
    fig.savefig(out/'klkeep_ig5_retraining.png',dpi=180)
    for ax in axes: ax.set_yscale('log')
    fig.savefig(out/'klkeep_ig5_retraining_logy.png',dpi=180); plt.close(fig)
    save_json(dict(plan=plan,curves=curves),out/'comparison.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='ig5-retraining-20260910',
                   name='ig5-retraining-klkeep-comparison',job_type='evaluation',dir=str(out))
    run.log({'eval/klkeep':wandb.Image(str(out/'klkeep_ig5_retraining.png')),
             'eval/klkeep_logy':wandb.Image(str(out/'klkeep_ig5_retraining_logy.png'))})
    save_json(dict(id=run.id,url=run.url),out/'wandb.json'); run.finish()
    for path in (out/'comparison.json',out/'klkeep_ig5_retraining.png',out/'klkeep_ig5_retraining_logy.png'):
        upload(path,'experiments/ig5_retraining/'+path.name)


if __name__=='__main__': main()
