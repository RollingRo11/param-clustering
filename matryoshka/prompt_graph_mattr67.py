"""The princess prompt's graph from our MAttr circuit, built like VPD's paper graph (prompt_graph67.py) so the two can be
drawn side by side: VPD's pieces through the paper's procedure, ours through MAttr.

MAttr exactly as prompt_mattr67.py (its prompt 0, seed 0, the same steps and budgets; the scores are recomputed because
that run saved only the curves): the circuit is the top-k (position, piece) pairs of the learned ranking, k the
description length at --threshold (fewest pieces with KL to the intact next-token distribution <= threshold, rest off).
Its nodes and edges come from VPD's compute_edges_from_ci (prompt_graph67.graph) with the circuit as a 0/1 CI, our pieces
wrapped as a VPD ComponentModel. Also saved: the circuit's p(label) and KL with only its pieces on.
Usage: prompt_graph_mattr67.py --ours CKPT --vpd CKPT --output OUT.json [--threshold 0.5]
"""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))
sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import MODULES, credentials, save_json, target
from prompt_graph67 import graph, load_ours, load_vpd
from prompt_mattr67 import GRID, K_MAX, STEPS, Masked, curve, description_length, hard_topk, kl_last, load_factors, log, mattr, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--vpd', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.topology import TransformerTopology
    from param_decomp.topology.gradient_connectivity import get_sources_by_target
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    p = prompts()[0][0]
    assert p['source'] == 'paper', p
    ids = torch.tensor([p['ids']], device='cuda'); T = ids.shape[1]
    model = target()
    factors, meta = load_factors('parauto', args.ours, model)
    net = Masked(model, factors)
    with torch.no_grad():
        intact_logp = net.logits(ids)[0, -1].log_softmax(-1)
    label = int(intact_logp.argmax())
    init = net.gradient_attribution(ids, label).abs().clamp_min(1e-12).log()
    learned, history = mattr(net, ids, init, intact_logp, STEPS, 0, K_MAX, 'off')
    points = curve(net, ids, learned, intact_logp, label, T, GRID, 'off', None)
    k = description_length(points, args.threshold)
    mask = hard_topk(learned.flatten(), torch.tensor([k], device='cuda')).view(1, T, -1)
    with torch.no_grad():
        logits = net.logits(ids, mask)
    kl, p_label = float(kl_last(intact_logp, logits)[0]), float(logits[0, -1].softmax(-1)[label])
    log(k=k, kl=kl, p_label=p_label, label=label, p_label_intact=float(intact_logp[label].exp()))
    vpd_model, ours_model = target(), target()
    _, cfg, _ = load_vpd(vpd_model, args.vpd)
    cm, _, ometa = load_ours(ours_model, args.ours, cfg)
    ci = SimpleNamespace(lower_leaky={n: m.contiguous() for n, m in zip(MODULES, mask.split(net.counts, -1))})
    topology = TransformerTopology(ours_model)
    g = graph(cm, topology, get_sources_by_target(cm, topology, 'cuda', cfg.sampling), ids, ci)
    out = dict(prompt=p['text'], tokens=p['tokens'], token_ids=p['ids'], label_id=label, threshold=args.threshold, k=k,
               kl_circuit=kl, p_label_circuit=p_label, p_label_intact=float(intact_logp[label].exp()), history=history,
               curve=points, ours={**meta, **ometa}, graph=g,
               procedure='prompt_mattr67 MAttr (prompt 0, seed 0, rest off); circuit = top-k at the description length; edges by '
                         'compute_edges_from_ci with the circuit as a 0/1 CI')
    save_json(out, args.output)
    log(done=str(args.output), nodes=len(g['nodes']), edges=len(g['edges']))


if __name__ == '__main__': main()
