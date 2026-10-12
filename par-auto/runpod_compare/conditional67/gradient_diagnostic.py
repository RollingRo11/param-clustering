"""Read-only diagnostic of production gate derivatives and FP32 loss cancellation."""
from dataclasses import replace
import json
import os
import torch
import engine as e
from common import ROOT, credentials, target, tokens, batch, save_json


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    model=target()
    pieces=[e.core.initialize_pieces(model.get_submodule(n).weight,c,seed=0)
            for n,c in zip(e.names(model),[1024,1024,1024,1024,3072,3584]*4,strict=True)]
    data,_=tokens(); ids=batch(data,0)[:1,:-1]
    cfg=e.Config(precision='fp32')
    lp,support=e.dense_reference(model,ids,pieces,cfg,780123,512)
    gate=support.float().requires_grad_()
    logits=e.run(model,ids,pieces,gate,cfg)
    value=e.kl_per_sequence(logits,lp).sum()
    grad,=torch.autograd.grad(value,gate)
    records=[];offset=0
    for kind in range(6):
        c=pieces[kind].count;block=grad[...,offset:offset+c]
        for count in (4,64):
            direction=torch.zeros_like(gate);local=torch.zeros_like(block).flatten()
            idx=block.abs().flatten().topk(count).indices
            local[idx]=block.flatten()[idx].sign()
            direction[...,offset:offset+c]=local.reshape_as(block)
            predicted=float((grad*direction).sum())
            for eps in (.1,.03,.01,.003,.001):
                with torch.no_grad():
                    plus=e.run(model,ids,pieces,gate.detach()+eps*direction,cfg)
                    minus=e.run(model,ids,pieces,gate.detach()-eps*direction,cfg)
                    naive=float((e.kl_per_sequence(plus,lp).sum()-e.kl_per_sequence(minus,lp).sum())/(2*eps))
                    # Cancel the teacher entropy algebraically and accumulate
                    # the actual change in log probabilities in FP64.
                    stable=float((lp.double().exp()*(minus.double().log_softmax(-1)-plus.double().log_softmax(-1))).sum()/(ids.numel()*2*eps))
                    tangent=float(((plus.double()-minus.double())*(logits.detach().double().softmax(-1)-lp.double().exp())).sum()/(ids.numel()*2*eps))
                row=dict(kind=e.KINDS[kind],count=count,eps=eps,predicted=predicted,naive=naive,stable=stable,tangent=tangent,
                         stable_relative=abs(stable-predicted)/max(abs(stable),abs(predicted),1e-12))
                records.append(row);print(json.dumps(row),flush=True)
        offset+=c
    save_json(records,ROOT/'output/conditional67_gradient_diagnostic_20260916.json')


if __name__=='__main__':main()
