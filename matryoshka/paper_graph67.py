"""The VPD paper's own princess graph (the blog's graphs/princess-full.json: 150 subcomponents with their optimized CI)
scored on the published VPD checkpoint, and its edges recomputed with the code prompt_graph67.py uses, so it can be drawn
next to our graphs.

Scores at the last position, with VPD's delta left out unless said otherwise: only the graph's pieces at mask 1, every
other piece off (P(label), KL to the intact next-token distribution); the same with the graph's CI values as the mask;
the graph's pieces on and every other piece at uniform random levels with the delta at a random level (VPD's stochastic
mask, mean over DRAWS); and the app's eval PGD on the alive pieces (4 steps of 1.0, as prompt_graph67.py). Also: how many
of the graph's pieces are alive (CI > 0) under the checkpoint's own CI function, which checks the index mapping.
Usage: paper_graph67.py --vpd CKPT --graph princess-full.json --output OUT.json
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from common import MODULES, credentials, save_json, target
from prompt_graph67 import graph, load_vpd

KIND = dict(q='attn.q_proj', k='attn.k_proj', v='attn.v_proj', o='attn.o_proj', up='mlp.c_fc', down='mlp.down_proj')
DRAWS = 32


def module(layer):
    """'3.attn.v' -> 'h.3.attn.v_proj' (the blog's layer names to the target's)."""
    block, _, kind = layer.split('.')
    return f'h.{block}.{KIND[kind]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vpd', type=Path, required=True); ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.app.backend.optim_cis import AdvPGDConfig, CELossConfig, run_adv_pgd
    from param_decomp.metrics.pgd_utils import interpolate_pgd_mask
    from param_decomp.models.components import make_mask_infos
    from param_decomp.topology import TransformerTopology
    from param_decomp.topology.gradient_connectivity import get_sources_by_target
    from param_decomp.utils.component_utils import calc_stochastic_component_mask_info
    from param_decomp.routing import AllLayersRouter
    from param_decomp.utils.distributed_utils import init_distributed
    from tokenizers import Tokenizer
    init_distributed()
    paper = json.loads(args.graph.read_text())
    tok = Tokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    ids = tok.encode(''.join(paper['tokens'])).ids
    assert [tok.decode([i]) for i in ids] == paper['tokens'], (ids, paper['tokens'])
    label = int(next(k for k in paper['node_ci_vals'] if k.startswith('output:')).split(':')[2])
    tokens, T, pos = torch.tensor([ids], device='cuda'), len(ids), len(ids) - 1
    model = target()
    cm, cfg, meta = load_vpd(model, args.vpd)
    C = {n: cm.components[n].V.shape[1] for n in MODULES}
    ci = {n: torch.zeros(1, T, C[n], device='cuda') for n in MODULES}
    nodes = [(k, v) for k, v in paper['node_ci_vals'].items() if not k.startswith(('embed:', 'output:'))]
    for k, v in nodes:
        layer, t, c = k.split(':'); ci[module(layer)][0, int(t), int(c)] = float(v)
    with torch.no_grad():
        target_out = cm(tokens)
        ci0 = cm.calc_causal_importances(pre_weight_acts=cm(tokens, cache_type='input').cache, sampling=cfg.sampling, detach_inputs=False)
    alive = {n: ci0.lower_leaky[n] > 0 for n in MODULES}
    alive_nodes = sum(bool(alive[module(k.split(':')[0])][0, int(k.split(':')[1]), int(k.split(':')[2])]) for k, _ in nodes)
    intact = F.log_softmax(target_out[0, pos].float(), -1)

    def score(logits):
        logp = F.log_softmax(logits[0, pos].float(), -1)
        return dict(p_label=float(logp[label].exp()), kl=float((intact.exp() * (intact - logp)).sum()),
                    top5=[[int(i), float(p)] for p, i in zip(*logp.exp().topk(5))])
    binary = {n: (v > 0).float() for n, v in ci.items()}
    with torch.no_grad():
        out = dict(binary=score(cm(tokens, mask_infos=make_mask_infos(binary))), ci_mask=score(cm(tokens, mask_infos=make_mask_infos(ci))))
        torch.manual_seed(0)
        draws = [score(cm(tokens, mask_infos=calc_stochastic_component_mask_info(binary, cfg.sampling, cm.calc_weight_deltas(), AllLayersRouter())))
                 for _ in range(DRAWS)]
    out['random_levels'] = dict(p_label=sum(d['p_label'] for d in draws) / DRAWS, kl=sum(d['kl'] for d in draws) / DRAWS)
    torch.manual_seed(1)
    loss = CELossConfig(coeff=1.0, position=pos, label_token=label)
    adv = run_adv_pgd(model=cm, tokens=tokens, ci=ci, alive_masks=alive, adv_config=AdvPGDConfig(n_steps=4, step_size=1.0, init='random'),
                      target_out=target_out, loss_config=loss)
    with torch.no_grad():
        out['eval_pgd'] = score(cm(tokens, mask_infos=make_mask_infos(interpolate_pgd_mask(ci, adv))))
    topology = TransformerTopology(model)
    g = graph(cm, topology, get_sources_by_target(cm, topology, 'cuda', cfg.sampling), tokens, SimpleNamespace(lower_leaky=ci))
    result = dict(prompt=''.join(paper['tokens']), tokens=paper['tokens'], token_ids=ids, label_id=label, p_label_intact=float(intact[label].exp()),
                  pieces=len(nodes), pieces_alive_under_checkpoint_ci=alive_nodes, vpd=meta, scores=out, graph=g,
                  source='https://static.goodfire.ai/vpd-blog-post/data/graphs/princess-full.json')
    save_json(result, args.output)
    print(json.dumps(dict(pieces=len(nodes), alive=alive_nodes, **{k: {kk: v[kk] for kk in ('p_label', 'kl')} for k, v in out.items()})), flush=True)


if __name__ == '__main__': main()
