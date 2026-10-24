"""Prompt circuits on SimpleStories, Goodfire's VPD (s-9e774a82) against one of our checkpoints, through VPD's own app code
(param_decomp.app.backend.compute_edges_from_ci, as the 20 Oct SS port's native_circuit.py / pg67.py).
Circuit = each method's own notion of active at every prompt position: VPD's CI > 0 set (CI as its mask value, as the app);
ours the pieces whose static score (|h.r_c| ||L^T u_c||) is above their learned floor (MATRYOSHKA_FLOORS runs; mask 1),
or with --ramp w the leaky hard-clipped ramp of the score around the floor (mask in [0, 1], active where > 0).
For each: P(label) intact, with only the circuit (every other piece off), and under two attacks that may switch any piece
outside the circuit partly back on (mask keep + (1 - keep) r, sign-PGD on -log P(label): the app's 4 steps of 1.0 and our
eval's 320 steps of 0.1, one r per piece per position); and the attribution graph (sec 5.1 edges) at the circuit.
Usage: circuit_ss.py --ours MODEL.pth --prompt "lena felt joy swell in" --label her --output OUT.json"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MATRYOSHKA_TARGET', 'simplestories')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy_ci  # noqa: F401  Feb-2026 CI function input path, needed for s-9e774a82
import torch
import torch.nn.functional as F

from common import MODULES, credentials, save_json, target

VPD_RUN = 'wandb:goodfire/spd/runs/s-9e774a82'
ATTACKS = dict(pgd4=dict(steps=4, step=1.0), pgd320=dict(steps=320, step=0.1))


def log(**fields): print(json.dumps(fields), flush=True)


def load_ours(model, saved, cfg):
    """Our pieces as a VPD ComponentModel (rank-1 components; its CI function is never used)."""
    from param_decomp.configs import LayerwiseCiConfig
    from param_decomp.models.batch_and_loss_fns import make_run_batch
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.utils.module_utils import ModulePathInfo
    factors = saved['factors']
    cm = ComponentModel(model, run_batch=make_run_batch(cfg.output_extract),
                        module_path_info=[ModulePathInfo(module_path=n, C=f['read'].shape[1]) for n, f in zip(MODULES, factors)],
                        ci_config=LayerwiseCiConfig(fn_type='mlp', hidden_dims=[1]), sigmoid_type='leaky_hard').cuda()
    with torch.no_grad():
        for n, f in zip(MODULES, factors):
            cm.components[n].V.copy_(f['read'].float()); cm.components[n].U.copy_(f['write'].float().T)
    return cm.eval().requires_grad_(False)


def above_floor(model, tokens, saved, ramp=0.):
    """Per matrix [1, seq, C] mask: 0/1 static score above the piece's learned floor; ramp = w > 0: the leaky hard-clipped
    ramp 0.5 + (score - floor) / (2 w floor) clipped to [0, 1] (MATRYOSHKA_FLOOR_RAMP's forward), active where it is > 0."""
    pre, hooks = {}, []
    for n in MODULES:
        hooks.append(model.get_submodule(n).register_forward_hook(lambda _m, a, _o, n=n: pre.__setitem__(n, a[0].detach().float())))
    try:
        with torch.no_grad(): model(tokens)
    finally:
        for h in hooks: h.remove()
    floors = saved['score_floor'].cuda().split([f['read'].shape[1] for f in saved['factors']])
    out = {}
    for n, f, m, fl in zip(MODULES, saved['factors'], saved['static_metric'], floors):
        score = (pre[n][0] @ f['read'].cuda().float()).abs() * (m['L'].cuda().float().T @ f['write'].cuda().float()).norm(dim=0)
        out[n] = ((0.5 + (score - fl) / (2 * ramp * fl)).clamp(0, 1) if ramp else (score > fl).float())[None]
    return out


def label_prob(cm, tokens, masks, label):
    from param_decomp.models.components import make_mask_infos
    return F.softmax(cm(tokens, mask_infos=make_mask_infos(masks))[0, -1].float(), -1)[label]


def attack(cm, tokens, keep, label, steps, step, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    r = {n: torch.rand(keep[n].shape, device='cuda', generator=g).requires_grad_(True) for n in MODULES}
    for _ in range(steps):
        p = label_prob(cm, tokens, {n: keep[n] + (1 - keep[n]) * r[n] for n in MODULES}, label)
        grads = torch.autograd.grad(-torch.log(p), [r[n] for n in MODULES])
        with torch.no_grad():
            for n, gr in zip(MODULES, grads): r[n].add_(step * gr.sign()).clamp_(0, 1)
    with torch.no_grad():
        return float(label_prob(cm, tokens, {n: keep[n] + (1 - keep[n]) * r[n] for n in MODULES}, label))


def graph(cm, topology, sources, tokens, keep):
    from param_decomp.app.backend.compute import compute_edges_from_ci
    with torch.no_grad():
        pre = cm(tokens, cache_type='input').cache
        logits = cm(tokens)
    r = compute_edges_from_ci(model=cm, topology=topology, tokens=tokens, ci_lower_leaky=keep, pre_weight_acts=pre,
                              sources_by_target=sources, target_out_probs=F.softmax(logits, -1), target_out_logits=logits,
                              output_prob_threshold=0.01, device='cuda')
    ci_of = {}
    for n in MODULES:
        canon = topology.target_to_canon(n)
        for t, c in (keep[n][0] > 0).nonzero().tolist(): ci_of[(canon, t, c)] = float(keep[n][0, t, c])
    node = lambda x: [x.layer, x.seq_pos, x.component_idx]
    edges = [dict(source=node(e.source), target=node(e.target), strength=e.strength,
                  strength_x_source_ci=e.strength * ci_of.get((e.source.layer, e.source.seq_pos, e.source.component_idx), 1.),
                  cross_position=e.is_cross_seq) for e in r.edges]
    top = r.target_out_probs[-1].topk(5)
    return dict(nodes=[dict(layer=l, position=t, piece=c, ci=v) for (l, t, c), v in sorted(ci_of.items())], edges=edges,
                output_top5=[[int(i), float(p)] for p, i in zip(top.values, top.indices)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--prompt', required=True); ap.add_argument('--label', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ramp', type=float, default=0., help='gate ours by the leaky hard-clipped ramp of this half-width x floor')
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
    from param_decomp.topology import TransformerTopology
    from param_decomp.topology.gradient_connectivity import get_sources_by_target
    from param_decomp.utils.distributed_utils import init_distributed
    from transformers import AutoTokenizer
    init_distributed()
    tok = AutoTokenizer.from_pretrained('SimpleStories/test-SimpleStories-gpt2-1.25M')
    ids = tok(args.prompt.lower(), add_special_tokens=False)['input_ids']; label_ids = tok(args.label.lower(), add_special_tokens=False)['input_ids']
    assert len(label_ids) == 1, label_ids
    tokens, label = torch.tensor([ids], device='cuda'), label_ids[0]
    result = dict(prompt=args.prompt, label=args.label, tokens=[tok.decode([i]) for i in ids], token_ids=ids, label_id=label,
                  attacks=ATTACKS, ours=str(args.ours), ramp=args.ramp, methods={})
    ri = ParamDecompRunInfo.from_path(VPD_RUN)
    vpd, cfg = ComponentModel.from_run_info(ri).cuda().eval().requires_grad_(False), ri.config
    with torch.no_grad():
        ci = vpd.calc_causal_importances(pre_weight_acts=vpd(tokens, cache_type='input').cache, sampling=cfg.sampling, detach_inputs=False)
    vpd_keep = {n: ci.lower_leaky[n].float() for n in MODULES}
    saved = torch.load(args.ours, map_location='cpu', weights_only=False)
    ours_model = target(); ours = load_ours(ours_model, saved, cfg)
    ours_keep = above_floor(ours_model, tokens, saved, args.ramp)
    for name, cm, keep in (('vpd', vpd, vpd_keep), ('ours', ours, ours_keep)):
        with torch.no_grad():
            intact = float(F.softmax(cm(tokens)[0, -1].float(), -1)[label])
            circuit = float(label_prob(cm, tokens, keep, label))
        per_pos = [int(sum(float((keep[n][0, t] > 0).sum()) for n in MODULES)) for t in range(len(ids))]
        per_matrix = {n: float((keep[n][0] > 0).float().sum(-1).mean()) for n in MODULES}
        entry = dict(p_intact=intact, p_circuit_only=circuit, active_per_position=per_pos, active_mean=sum(per_pos) / len(per_pos),
                     active_per_matrix=per_matrix, **{f'p_{a}': attack(cm, tokens, keep, label, **c) for a, c in ATTACKS.items()})
        log(method=name, **{k: v for k, v in entry.items() if k != 'active_per_matrix'})
        topology = TransformerTopology(cm.target_model)
        entry['graph'] = graph(cm, topology, get_sources_by_target(cm, topology, 'cuda', cfg.sampling), tokens, keep)
        result['methods'][name] = entry
        save_json(result, args.output)
    log(done=str(args.output))


if __name__ == '__main__': main()
