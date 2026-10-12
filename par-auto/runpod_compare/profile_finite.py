"""Time forward+backward of each finite-ablation score type at 1,024 tokens."""
import json, time
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import core, method
import finite_ablation as fa


def timed(fn, repeats=3):
    fn(); torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(repeats): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-start)/repeats


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    data, _ = tokens(); model = target()
    counts = dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)
    gen = torch.Generator(device='cuda').manual_seed(5)
    rows = batch(data, 0)
    result = dict(capture=timed(lambda: fa.capture_finite(model, rows, 16, gen)))
    entries = fa.capture_finite(model, rows, 16, gen)
    for chunk in (64, 256):
        fa.TOKEN_CHUNK = chunk
        for name in ('h.1.attn.q_proj', 'h.1.attn.k_proj', 'h.1.attn.v_proj', 'h.1.attn.o_proj', 'h.1.mlp.c_fc', 'h.1.mlp.down_proj'):
            if chunk == 256 and not name.endswith(('q_proj', 'k_proj')): continue
            entry = entries[MODULES.index(name)]
            pieces = core.initialize_pieces(model.get_submodule(name).weight, counts[name.split('.')[-1]], 0)
            pieces.read.requires_grad_(); pieces.write.requires_grad_()
            def step():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = method.scores_of(entry, pieces).add(1e-6).pow(.64).sum()
                loss.backward()
            def forward():
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    method.scores_of(entry, pieces)
            torch.cuda.reset_peak_memory_stats()
            result[f'{name}/chunk{chunk}'] = dict(forward=timed(forward), forward_backward=timed(step),
                                                  peak_gb=torch.cuda.max_memory_allocated()/1e9)
            print(name, chunk, json.dumps(result[f'{name}/chunk{chunk}']), flush=True)
    save_json(result, ROOT/'output/smoke/finite_parts.json')
    print('FINITE_PROFILE_DONE', flush=True)


if __name__ == '__main__': main()
