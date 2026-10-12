"""20-step finite-ablation profile plus one eval batch; writes only a timing JSON."""
import json, time
from dataclasses import asdict
import numpy as np
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import core, method
from finite_ablation import capture_finite


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.manual_seed(0)
    data, _ = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial = [core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    cfg = method.TrainConfig(**json.loads((ROOT/'output/gim_1h/plan.json').read_text())['training_config'])
    cfg.batch = 1024
    sampler = lambda seed: torch.Generator(device='cuda').manual_seed(seed)
    provider = lambda step: capture_finite(model, batch(data, step), 16, sampler(1_000_003+step))
    reference = capture_finite(model, batch(data, 0), 16, sampler(123))
    times, losses = [], []
    def tick(step):
        torch.cuda.synchronize(); times.append(time.perf_counter()); return False
    torch.cuda.reset_peak_memory_stats()
    method.fit(reference, weights, initial, cfg, batch_provider=provider, step_callback=tick, stop_after=20,
               log=lambda row: losses.append(row['total_loss']) if row['matrix_index']==0 else None)
    per_step = (times[-1]-times[4])/(len(times)-5)     # first steps include compilation
    peak = torch.cuda.max_memory_allocated()
    # One KLKeep scoring batch (2 held-out sequences, every token scored).
    val, _ = tokens('val'); rows = torch.tensor(np.array(val[:2], dtype=np.int64), device='cuda')
    for repeat in range(2):
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            cache = capture_finite(model, rows)
            scores = [method.scores_of(e, p) for e, p in zip(cache, initial)]
        torch.cuda.synchronize(); eval_batch = time.perf_counter()-start
    assert all(torch.isfinite(s).all() for s in scores) and np.isfinite(losses).all()
    row = dict(seconds_per_step=per_step, peak_allocated_bytes=peak, eval_scoring_seconds_per_2_sequences=eval_batch,
               first_losses=losses[:3], gpu=torch.cuda.get_device_name(), config=asdict(cfg))
    save_json(row, ROOT/'output/smoke/finite_profile.json'); print(json.dumps(row), flush=True)
    print('FINITE_SMOKE_PASSED', flush=True)


if __name__ == '__main__': main()
