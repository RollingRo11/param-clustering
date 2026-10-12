"""20-step write-size profile; writes nothing except a timing JSON."""
import json, time
import numpy as np
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import core, method
from write_attribution import capture_inputs


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(0)
    data, _ = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial = [core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    control = json.loads((ROOT/'output/gim_1h/plan.json').read_text())
    cfg = method.TrainConfig(**control['training_config'])
    provider = lambda step: capture_inputs(model, batch(data, step))
    index = torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    reference = [method.slice_entry(e,index) for e in provider(0)]
    times = []
    def tick(step):
        torch.cuda.synchronize(); times.append(time.perf_counter()); return False
    losses = []
    method.fit(reference, weights, initial, cfg, batch_provider=provider, step_callback=tick, stop_after=20,
               log=lambda row: losses.append(row['total_loss']) if row['matrix_index']==0 else None)
    per_step = (times[-1]-times[4])/(len(times)-5)
    row = dict(seconds_per_step=per_step, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               first_losses=losses[:3], gpu=torch.cuda.get_device_name())
    assert np.isfinite(losses).all()
    save_json(row, ROOT/'output/smoke/write_profile.json'); print(json.dumps(row), flush=True)
    print('WRITE_SMOKE_PASSED', flush=True)


if __name__ == '__main__': main()
