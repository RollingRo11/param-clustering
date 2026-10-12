"""No-training exact canonical-basis control and signed-tail diagnostic on 67M."""
import os
from pathlib import Path
import sys
ROOT=Path(os.environ['PAIR_ROOT'])
sys.path.insert(0,str(ROOT/'code/gaussian67_diagnostic_20260914'))
from gaussian67_diagnose import *


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed(); model=target(); data,manifest=tokens('val'); weights=[model.get_submodule(n).weight for n in MODULES]
    trained={}
    for name in ['gim','gaussian']:
        path=ROOT/SPECS[name][2]
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert saved['target_sha256']==TARGET_SHA and saved['completed']==5000
        trained[name]=[core.FreePieces(p['read'].cuda(),p['write'].cuda()) for p in saved['factors']]
        del saved
    row=[core.FreePieces(w.T.contiguous(),torch.eye(w.shape[0],device='cuda')) for w in weights]
    col=[core.FreePieces(torch.eye(w.shape[1],device='cuda'),w.contiguous()) for w in weights]
    neuron=[col[m] if n.endswith('down_proj') else row[m] for m,n in enumerate(MODULES)]
    candidates={**trained,'row_basis':row,'column_basis':col,'mlp_neuron_basis':neuron}
    out=ROOT/'output/gaussian67_diagnostic_20260914/basis_control.json'
    result=dict(target_sha256=TARGET_SHA,data_sha256=manifest['val']['sha256'],start_sequence=384,sequences=16,
                job_id=os.environ['SLURM_JOB_ID'],source_sha256=sha(__file__),gpu=torch.cuda.get_device_name(),
                note='Same 16 diagnostic contexts; no training. Independent residual probe stream, K=64, sums source positions within each sequence before squaring.',
                methods={n:dict(components=sum(p.count for p in ps),batches=[]) for n,ps in candidates.items()})
    began=time.perf_counter()
    for first in range(384,400,2):
        rows=torch.tensor(np.array(data[first:first+2],dtype=np.int64),device='cuda');ids=rows[:,:-1]
        rank_cache=capture_fisher(model,rows,k=64,seed=910_000+first,microbatch=2)
        pred_cache=capture_fisher(model,rows,k=64,seed=1_910_000+first,microbatch=2)
        with torch.no_grad(),trace(model,require_grad=False) as (pre,post,_):
            with torch.autocast('cuda',dtype=torch.bfloat16): baseline=model(ids)[0]
            inputs=[pre[n].detach().float() for n in MODULES];outputs=[post[n].detach().float() for n in MODULES]
        for name,pieces in candidates.items():
            _,scores=scores_both(rank_cache,pieces,ids.shape)
            with torch.no_grad():
                combined=torch.cat(scores,-1);order=order_of(scores);counts=[p.count for p in pieces]
                b=dict(first_sequence=first,budgets={},lp_per_matrix={str(p):[seq_mean((s+1e-6).pow(p).sub(1e-6**p).sum(-1)) for s in scores] for p in [.5,.641,1.]})
                for budget in [512,4096]:
                    masks=mask_of(order,counts,budget)
                    d=dict(actual=kl(forward(model,ids,pieces,masks),baseline).tolist(),
                           diag_proxy=.5*torch.zeros(2,device='cuda'))
                    zsum=torch.zeros(2,512,64,device='cuda'); within_seq=torch.zeros(2,device='cuda');within_position=torch.zeros(2,device='cuda')
                    for m,(piece,mask,score,e) in enumerate(zip(pieces,masks,scores,pred_cache)):
                        # FP32 local error against intact BF16 matrix output; includes numerical reconstruction error.
                        delta=((inputs[m] @ piece.read)*mask) @ piece.write.T-outputs[m]
                        g=e['events']['cot'].float().reshape(2,512,64,-1)
                        z=torch.einsum('btkd,btd->btk',g,delta)
                        zsum+=z
                        within_position+=.5*z.square().mean((1,2))
                        within_seq+=.5*z.sum(1).square().mean(1)/512
                        d['diag_proxy']+=.5*((1-mask)*score.square()).sum(-1).mean(-1)
                    d['diag_proxy']=d['diag_proxy'].tolist()
                    d['old_same_position_proxy']=(.5*zsum.square().mean((1,2))).tolist()
                    d['correct_sequence_proxy']=(.5*zsum.sum(1).square().mean(1)/512).tolist()
                    d['within_matrix_sequence_proxy']=within_seq.tolist()
                    d['within_matrix_position_proxy']=within_position.tolist()
                    d['diagonal_note']='Uses ranking probes; residual proxies use independent probes. Proxy estimates are noisy at K=64.'
                    b['budgets'][str(budget)]=d
                result['methods'][name]['batches'].append(b)
                del combined,order,scores,masks
            result['seconds']=time.perf_counter()-began;save_json(result,out)
            print(json.dumps(dict(method=name,first_sequence=first,seconds=round(result['seconds'],1))),flush=True)
        del rank_cache,pred_cache,baseline,inputs,outputs
    result['complete']=True;save_json(result,out)
    receipts=[upload(out,'experiments/gaussian67_diagnostic_20260914/basis_control.json'),upload(Path(__file__),'experiments/gaussian67_diagnostic_20260914/gaussian67_basis_control.py')]
    save_json(receipts,out.with_name('basis_r2_receipts.json'))
    print('GAUSSIAN67_BASIS_CONTROL_DONE',flush=True)

if __name__=='__main__':main()
