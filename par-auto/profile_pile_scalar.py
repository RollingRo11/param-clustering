"""Measure all-24-matrix scalar training at realistic shapes and batch sizes."""
import argparse
import gc
import time
from pathlib import Path
import torch
import parauto_core as core
import parauto_method as method
import parauto_pile as pile
import parauto_pile_scalar as scalar
import parauto_structured as structured
import parauto_tracking as tracking
from parauto_target import atomic_json, fingerprint


def profile(entries, weights, initial, batch):
    if batch > min(len(e['h']) for e in entries):
        raise ValueError('Cache must contain at least the requested batch size')
    cfg=method.TrainConfig(steps=61,batch=batch,log_interval=10,precision='bf16',projection_precision='fp32')
    ticks=[]
    def log(row):
        if row['matrix_index']==0:
            torch.cuda.synchronize();ticks.append((row['step'],time.perf_counter()))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    fitted,history=method.fit(entries,weights,initial,cfg,log=log)
    torch.cuda.synchronize()
    seconds=(ticks[-1][1]-ticks[1][1])/(ticks[-1][0]-ticks[1][0])
    return dict(batch=batch,seconds_per_update=seconds,positions_per_second=batch/seconds,
                fit_hours_100m=100000000/batch*seconds/3600,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                matrix_sum_max_error=max(core.reconstruction_error(w,p) for w,p in zip(weights,fitted)),
                overflow_skips=history[0][-1].get('overflow_skipped_updates',0),
                measured_steps=ticks[-1][0]-ticks[1][0],timing='steady steps including per-10-step diagnostics; no checkpoint I/O')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--tokens',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batches',type=int,nargs='+',default=[128,2048,8192])
    parser.add_argument('--cache-blocks',type=int,default=16)
    args=parser.parse_args();tracking.start_cli(args)
    if args.cache_blocks < 2 or max(args.batches) > args.cache_blocks*512:
        raise ValueError('Need at least two cache blocks and enough positions for the largest batch')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0)
    model=pile.load_model(args.target)
    weights=[model.get_submodule(n).weight for n in pile.MODULES]
    vpd=dict(q_proj=512,k_proj=512,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)
    counts=[max(1024,vpd[n.split('.')[-1]]) for n in pile.MODULES]
    rank_checks={}
    for n,w in zip(pile.MODULES,weights):
        if n.endswith(('q_proj','k_proj')):
            s=torch.linalg.svdvals(w.double())
            rank_checks[n]=dict(rank=int((s>max(w.shape)*torch.finfo(torch.float64).eps*s[0]).sum()),
                                best_rank512_relative_error=float(s[512:].norm()/s.norm()))
    blocks,_=pile.sample_blocks(args.tokens,args.cache_blocks,512,0,100000000,0)
    buckets=[[] for _ in weights];cache_times=[]
    for i,block in enumerate(blocks):
        ids,labels=block[None,:-1].cuda(),block[None,1:].cuda()
        torch.cuda.synchronize();start=time.perf_counter()
        entries,_=scalar.capture(model,ids,labels)
        torch.cuda.synchronize();cache_times.append(time.perf_counter()-start)
        for rows,e in zip(buckets,entries):rows.append(e)
    entries=[]
    for rows in buckets:
        entries.append(dict(rows[0],h=torch.cat([e['h'] for e in rows]),
                            events={'g':torch.cat([e['events']['g'] for e in rows])},
                            net_rms=torch.cat([e['net_rms'] for e in rows])))
    del buckets,rows
    initial=[core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    cache_s=sum(cache_times[1:])/len(cache_times[1:])/512
    result=dict(target_sha256=fingerprint(args.target),gpu=torch.cuda.get_device_name(),source_sha256=fingerprint(__file__),
                component_counts=dict(zip(pile.MODULES,counts)),total_components=sum(counts),vpd_total_components=38912,
                budget_difference='Q/K use 1024 instead of VPD 512 to preserve exactness and overcompleteness; all other matrices match VPD',
                rank_checks=rank_checks,cache_positions=len(entries[0]['h']),cache_bytes=structured.cache_bytes(dict(enumerate(entries))),
                fresh_ce_cache_seconds_per_position=cache_s,fresh_ce_cache_hours_100m=cache_s*1e8/3600,
                sensor='derivative of context-summed next-token CE; early local uses include effects on later losses',profiles=[])
    atomic_json(result,args.output/'result.json')
    for batch in args.batches:
        try:row=profile(entries,weights,initial,batch)
        except torch.cuda.OutOfMemoryError:
            row=dict(batch=batch,status='out_of_memory');gc.collect();torch.cuda.empty_cache()
        result['profiles'].append(row);atomic_json(result,args.output/'result.json')
        print(row,flush=True)
        if row.get('status') == 'out_of_memory':
            break


if __name__=='__main__':tracking.run_main(main)
