"""Kernel-level profile of one q, k and c_fc score forward+backward."""
import torch
from torch.profiler import profile, ProfilerActivity
from torch._dynamo.utils import counters
from common import MODULES, target, tokens, batch
from train_pair import core, method
import finite_ablation as fa


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    data, _ = tokens(); model = target()
    entries = fa.capture_finite(model, batch(data, 0), 16, torch.Generator(device='cuda').manual_seed(5))
    for name, count in (('h.1.attn.q_proj', 1024), ('h.1.attn.k_proj', 1024), ('h.1.mlp.c_fc', 3072)):
        entry = entries[MODULES.index(name)]
        pieces = core.initialize_pieces(model.get_submodule(name).weight, count, 0)
        pieces.read.requires_grad_(); pieces.write.requires_grad_()
        def step():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = method.scores_of(entry, pieces).add(1e-6).pow(.64).sum()
            loss.backward()
        for _ in range(3): step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        print('=====', name, flush=True)
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=18, max_name_column_width=70), flush=True)
    print({k: dict(v) for k, v in counters.items() if k in ('stats', 'inductor', 'graph_break', 'recompiles')}, flush=True)
    print('DIAG_DONE', flush=True)


if __name__ == '__main__': main()
