"""20-step Fisher training profile per K; writes only a timing JSON."""
import json, sys, time
import numpy as np
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import core, method
from fisher_attribution import capture_fisher


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(0)
    data, _ = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial = [core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    cfg = method.TrainConfig(**json.loads((ROOT/'output/gim_1h/plan.json').read_text())['training_config'])
    index = torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    result = {}
    for k in [int(x) for x in sys.argv[1:]]:
        provider = lambda step: capture_fisher(model, batch(data, step), k=k, seed=step)
        reference = [method.slice_entry(e, index) for e in provider(0)]
        times, losses = [], []
        def tick(step):
            torch.cuda.synchronize(); times.append(time.perf_counter()); return False
        torch.cuda.reset_peak_memory_stats()
        method.fit(reference, weights, initial, cfg, batch_provider=provider, step_callback=tick, stop_after=20,
                   log=lambda row: losses.append(row['total_loss']) if row['matrix_index']==0 else None)
        assert np.isfinite(losses).all()
        result[k] = dict(seconds_per_step=(times[-1]-times[4])/(len(times)-5), peak_gb=torch.cuda.max_memory_allocated()/1e9, first_loss=losses[0])
        print(json.dumps({k: result[k]}), flush=True)
    save_json(dict(gpu=torch.cuda.get_device_name(), by_k=result), ROOT/'output/smoke/fisher_profile.json')
    print('FISHER_SMOKE_PASSED', flush=True)


if __name__ == '__main__': main()
