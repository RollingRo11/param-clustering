"""Three native-sensor KLKeep curves at the same 1,000 training updates."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, credentials, save_json, upload

def main():
    credentials(); out=ROOT/'output/evaluations/gim_normal_vpd_1000'
    old=ROOT/'output/evaluations/gim_vpd_1000'
    curves={name:json.loads(path.read_text()) for name,path in [
        ('normal',out/'normal/klkeep.json'),('gim',old/'parauto/klkeep.json'),('vpd',old/'vpd/klkeep.json')]}
    for key in ('training_steps','training_positions','training_data_sha256','data_sha256','target_sha256','positions','precision','gpu'):
        assert len({c[key] for c in curves.values()})==1,key
    assert curves['normal']['training_steps']==1000
    assert curves['normal']['total_components']==curves['gim']['total_components']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for name,label,color in [('normal','Par-auto · ordinary gradient','#2563eb'),('gim','Par-auto · GIM','#176b46'),('vpd','VPD · learned CI','#9250a1')]:
        rows=[r for r in curves[name]['curve'] if r['kept_components']>0]
        for ax,key in zip(axes,('kept_components','keep_fraction')):
            ax.plot([r[key] for r in rows],[r['kl'] for r in rows],marker='o',markersize=4,color=color,label=label)
    for ax in axes:
        ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position ↓')
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    axes[0].set_xlabel('Components kept per position across all 24 matrices')
    axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('Normal par-auto vs GIM vs VPD · 1,000 steps / 32.8M training positions each')
    fig.supxlabel('Same 65,536 held-out positions and H100 evaluation. LR/p schedules differ.\nPar-auto variants: 43,008 components; VPD: 38,912.',fontsize=9)
    fig.savefig(out/'klkeep_1000.png',dpi=180)
    for ax in axes: ax.set_yscale('log')
    fig.savefig(out/'klkeep_1000_logy.png',dpi=180); plt.close(fig)
    save_json(curves,out/'comparison.json')
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gim-1h-20260910',
        name='normal-gim-vpd-1000-klkeep',job_type='evaluation',dir=str(out),
        config=dict(training_steps=1000,training_positions=32768000,schedule_matched=False))
    run.log({'eval/klkeep':wandb.Image(str(out/'klkeep_1000.png')),
             'eval/klkeep_logy':wandb.Image(str(out/'klkeep_1000_logy.png'))})
    save_json(dict(id=run.id,url=run.url),out/'wandb.json'); run.finish()
    receipts=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and 'wandb' not in path.parts and path.suffix in ('.json','.png'):
            receipts.append(upload(path,'evaluations/gim_normal_vpd_1000/'+str(path.relative_to(out))))
    save_json(receipts,out/'r2_receipts.json')
    upload(out/'r2_receipts.json','evaluations/gim_normal_vpd_1000/r2_receipts.json')
    print('NORMAL_GIM_VPD_1000_COMPLETE',flush=True)

if __name__=='__main__': main()
