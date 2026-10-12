"""Prompt-level attribution graphs (the VPD paper's circuits section, e.g. "The princess lost" -> " her") for published VPD
and for our decomposition, both through VPD's own code:

  nodes   param_decomp.app.backend.optim_cis.optimize_ci_values: per-prompt causal importances re-optimized with the
          pieces frozen; a node is a (position, piece) with CI > 0 after the optimization.
  edges   param_decomp.app.backend.compute.compute_edges_from_ci: (d a_c / d a_c') * a_c' between piece activations
          (a_c = x . read_c), direct effects only, across positions through attention; the paper's figure also weights
          by the source's CI (both are saved).

Our pieces enter VPD's ComponentModel as rank-1 components (V = read, U = write^T; the weight delta W - V U is ~1e-8).
VPD's starting CI comes from its gate; only pieces with CI > 0 there can enter the graph (alive). Ours has no gate: at
each position the alive set is our top-n pieces by the Fisher K=8 score, n = VPD's alive count at that position, all
starting at CI = 1, so both optimizations search equally many pieces and the adversary can turn on equally many.

Paper settings (footnotes of the princess graphs): CE on the label at the last position; "adversarial": coefficient 0.5
for CE with stochastic masks + 0.5 for CE with 4 steps of PGD, importance minimality 0.09, p = 0.3, 2,000 steps, lr 1
(the PGD step size; 150 subcomponents for VPD); "stochastic": the same without PGD (14 for VPD); "ci" (CI-only pruning): coefficient 1.0 for
CE with the CI as the mask, importance minimality 1.0 (6 for VPD). Not given in the paper and taken from VPD's
app: CI learning rate 0.01 (hard-coded, CI_LR), PGD random init, AdamW without weight decay, cosine lr with 1% warmup, beta 0.
A sweep of the importance-minimality coefficient gives node count against the label probability for each setting.
Every result also gets the app's eval PGD (4 steps, step 1.0) on the final circuit, and the binary circuit: its pieces
at mask 1, every other piece 0 (VPD's delta left out), label probability and KL(intact || circuit) at the last position.
Usage: prompt_graph67.py --ours CKPT --vpd CKPT --prompt "The princess lost" --label " her" --output OUT.json
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import torch
import torch.nn.functional as F

from common import MODULES, TARGET_SHA, credentials, target, save_json, sha

# mask: VPD's mask during the optimization (stochastic: ci + (1 - ci) U(0,1) for every piece; ci: the CI itself, so pieces
# outside the circuit are off); loss: CE coefficient; paper: the paper's importance-minimality coefficient for that graph
# type (the graph is drawn there); sweep: the coefficients run.
CI_LR = 1e-2       # the app's CI learning rate, hard-coded (app/backend/routers/graphs.py:777); the paper's "lr 1" is PGD's step
SETTINGS = dict(adversarial=dict(pgd=True, mask='stochastic', loss=0.5, paper=0.09, sweep=(0.02, 0.05, 0.09, 0.2, 0.5)),
                stochastic=dict(pgd=False, mask='stochastic', loss=1.0, paper=0.09, sweep=(0.02, 0.05, 0.09, 0.2, 0.5)),
                ci=dict(pgd=False, mask='ci', loss=1.0, paper=1.0, sweep=(0.1, 0.3, 1.0, 3.0)))


def log(**fields): print(json.dumps(fields), flush=True)


def load_vpd(model, checkpoint):
    from train_pair import VPD
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
    trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
    trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
    return trainer.model, trainer.cfg, dict(completed=saved.get('completed'), checkpoint_sha256=sha(checkpoint))


def load_ours(model, checkpoint, cfg):
    """Our pieces as a VPD ComponentModel (rank-1 components, no gate: calc_causal_importances is replaced per prompt)."""
    from param_decomp.configs import LayerwiseCiConfig
    from param_decomp.models.batch_and_loss_fns import make_run_batch
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.utils.module_utils import ModulePathInfo
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    factors = saved['factors']
    cm = ComponentModel(model, run_batch=make_run_batch(cfg.output_extract),
                        module_path_info=[ModulePathInfo(module_path=n, C=f['read'].shape[1]) for n, f in zip(MODULES, factors)],
                        ci_config=LayerwiseCiConfig(fn_type='mlp', hidden_dims=[1]), sigmoid_type='leaky_hard').cuda()
    with torch.no_grad():
        for n, f in zip(MODULES, factors):
            cm.components[n].V.copy_(f['read'].float()); cm.components[n].U.copy_(f['write'].float().T)
    cm.eval().requires_grad_(False)
    delta = max(float(d.abs().max()) for d in cm.calc_weight_deltas().values())
    return cm, factors, dict(completed=saved.get('completed'), checkpoint_sha256=sha(checkpoint), max_weight_delta=delta)


def fisher_scores(model, tokens, factors, k=8, seed=0):
    """Our Fisher K=k score at every prompt position (as fisher_attribution.capture_fisher, which assumes 512-token rows):
    a_c = |h . r_c| * RMS over k next tokens sampled from the model of (g_k . u_c), g_k the gradient of the summed CE at
    the matrix output. [seq, total pieces]"""
    pre, post, hooks = {}, {}, []
    def keep(n):
        def hook(_m, a, o):
            pre[n], post[n] = a[0], o                                           # returns None: the output is unchanged
        return hook
    for n in MODULES: hooks.append(model.get_submodule(n).register_forward_hook(keep(n)))
    hooks.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
    try:
        with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens)[0][0]                                            # [seq, vocab]
        g = torch.Generator(device='cuda').manual_seed(seed)
        samples = torch.multinomial(logits.detach().float().softmax(-1), k, replacement=True, generator=g)
        sq = [0.] * len(MODULES)
        for j in range(k):
            loss = F.cross_entropy(logits.float(), samples[:, j], reduction='sum')
            grads = torch.autograd.grad(loss, [post[n] for n in MODULES], retain_graph=j < k - 1)
            for i, (gr, f) in enumerate(zip(grads, factors)):
                sq[i] = sq[i] + (gr[0].float() @ f['write'].cuda().float()) ** 2
    finally:
        for h in hooks: h.remove()
    return torch.cat([(pre[n][0].detach().float() @ f['read'].cuda().float()).abs() * (q / k).sqrt()
                      for n, f, q in zip(MODULES, factors, sq)], -1)


def our_initial_ci(cm, model, factors, tokens, label, alive_counts):
    """Replace cm.calc_causal_importances: at the prompt, our top-n pieces per position (Fisher K=8 score) start at CI 1,
    the rest at 0 (not alive); any other input (the app's connectivity probe) gets CI 1 everywhere."""
    from param_decomp.models.component_model import CIOutputs
    scores = fisher_scores(model, tokens, factors)
    counts = [f['read'].shape[1] for f in factors]
    pre = torch.zeros_like(scores)
    for pos, n in enumerate(alive_counts):
        pre[pos, scores[pos].topk(n).indices] = 1.
    init = dict(zip(MODULES, (p[None] for p in pre.split(counts, -1))))

    def calc(pre_weight_acts, sampling=None, detach_inputs=False, **_):
        shape = next(iter(pre_weight_acts.values())).shape[:-1]
        if tuple(shape) == (1, tokens.shape[1]): p = init
        else: p = {n: torch.ones(*shape, c, device='cuda') for n, c in zip(MODULES, counts)}
        return CIOutputs(lower_leaky={k: cm.lower_leaky_fn(v) for k, v in p.items()},
                         upper_leaky={k: cm.upper_leaky_fn(v) for k, v in p.items()}, pre_sigmoid=p)
    cm.calc_causal_importances = calc
    return scores


def run_setting(cm, cfg, tokens, label, setting, coeff, steps, seed=0):
    from param_decomp.app.backend.optim_cis import (AdvPGDConfig, CELossConfig, OptimCIConfig, optimize_ci_values, run_adv_pgd)
    from param_decomp.configs import ImportanceMinimalityLossConfig
    from param_decomp.metrics.pgd_utils import interpolate_pgd_mask
    from param_decomp.models.components import make_mask_infos
    pos = tokens.shape[1] - 1
    loss = CELossConfig(coeff=SETTINGS[setting]['loss'], position=pos, label_token=label)
    config = OptimCIConfig(seed=seed, lr=CI_LR, steps=steps, weight_decay=0.0, lr_schedule='cosine', lr_exponential_halflife=None,
                           lr_warmup_pct=0.01, log_freq=max(1, steps // 4),
                           imp_min_config=ImportanceMinimalityLossConfig(coeff=coeff, pnorm=0.3, beta=0.0),
                           loss_config=loss, sampling=cfg.sampling, ce_kl_rounding_threshold=0.5, mask_type=SETTINGS[setting]['mask'],
                           adv_pgd=AdvPGDConfig(n_steps=4, step_size=1.0, init='random') if SETTINGS[setting]['pgd'] else None)
    torch.manual_seed(seed)
    started = time.perf_counter()
    result = optimize_ci_values(model=cm, tokens=tokens, config=config, device='cuda')
    ci = result.params.create_ci_outputs(cm, 'cuda')
    nodes = {n: int((ci.lower_leaky[n] > 0).sum()) for n in MODULES}
    with torch.no_grad():
        target_out = cm(tokens)
        p_intact = float(F.softmax(target_out[0, pos].float(), -1)[label])
        p_ci = float(F.softmax(cm(tokens, mask_infos=make_mask_infos(ci.lower_leaky))[0, pos].float(), -1)[label])
        binary = cm(tokens, mask_infos=make_mask_infos({n: (v > 0).float() for n, v in ci.lower_leaky.items()}))[0, pos].float()
        logp_t, logp_b = F.log_softmax(target_out[0, pos].float(), -1), F.log_softmax(binary, -1)
        p_binary, kl_binary = float(logp_b[label].exp()), float((logp_t.exp() * (logp_t - logp_b)).sum())
    torch.manual_seed(seed + 1)                                     # the app's eval PGD on the final circuit
    adv = run_adv_pgd(model=cm, tokens=tokens, ci=ci.lower_leaky, alive_masks=result.params.alive_info.alive_masks,
                      adv_config=AdvPGDConfig(n_steps=4, step_size=1.0, init='random'), target_out=target_out, loss_config=loss)
    with torch.no_grad():
        p_eval_pgd = float(F.softmax(cm(tokens, mask_infos=make_mask_infos(interpolate_pgd_mask(ci.lower_leaky, adv)))[0, pos].float(), -1)[label])
    out = dict(setting=setting, imp_min_coeff=coeff, steps=steps, ci_lr=CI_LR, nodes=sum(nodes.values()), nodes_per_module=nodes,
               nodes_per_position=[int(sum((ci.lower_leaky[n][0, t] > 0).sum() for n in MODULES)) for t in range(tokens.shape[1])],
               label_prob_intact=p_intact, label_prob_ci_mask=p_ci, label_prob_eval_pgd=p_eval_pgd,
               label_prob_binary=p_binary, kl_binary=kl_binary,
               metrics=result.metrics.model_dump(), seconds=round(time.perf_counter() - started, 1))
    return out, ci


def graph(cm, topology, sources, tokens, ci):
    """Nodes and edges at the circuit (VPD's compute_edges_from_ci with the optimized CI)."""
    from param_decomp.app.backend.compute import compute_edges_from_ci
    with torch.no_grad():
        pre = cm(tokens, cache_type='input').cache
        logits = cm(tokens)
    r = compute_edges_from_ci(model=cm, topology=topology, tokens=tokens, ci_lower_leaky=ci.lower_leaky, pre_weight_acts=pre,
                              sources_by_target=sources, target_out_probs=F.softmax(logits, -1), target_out_logits=logits,
                              output_prob_threshold=0.01, device='cuda')
    ci_of = {}
    for n in MODULES:
        canon = topology.target_to_canon(n)
        for t, c in (ci.lower_leaky[n][0] > 0).nonzero().tolist(): ci_of[(canon, t, c)] = float(ci.lower_leaky[n][0, t, c])
    node = lambda x: [x.layer, x.seq_pos, x.component_idx]
    edges = [dict(source=node(e.source), target=node(e.target), strength=e.strength,
                  strength_x_source_ci=e.strength * ci_of.get((e.source.layer, e.source.seq_pos, e.source.component_idx), 1.),
                  cross_position=e.is_cross_seq) for e in r.edges]
    out_probs = r.target_out_probs[-1]
    top = out_probs.topk(5)
    return dict(nodes=[dict(layer=l, position=t, piece=c, ci=v) for (l, t, c), v in sorted(ci_of.items())],
                edges=edges, output_top5=[[int(i), float(p)] for p, i in zip(top.values, top.indices)],
                ci_masked_output_top5=[[int(i), float(p)] for p, i in zip(*reversed(r.ci_masked_out_probs[-1].topk(5)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--vpd', type=Path, required=True)
    ap.add_argument('--prompt', default='The princess lost'); ap.add_argument('--label', default=' her')
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--settings', nargs='+', default=list(SETTINGS), choices=list(SETTINGS))
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    from param_decomp.topology import TransformerTopology
    from param_decomp.topology.gradient_connectivity import get_sources_by_target
    from tokenizers import Tokenizer
    init_distributed()
    tok = Tokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    ids, label_ids = tok.encode(args.prompt).ids, tok.encode(args.label).ids
    assert len(label_ids) == 1, label_ids
    tokens, label = torch.tensor([ids], device='cuda'), label_ids[0]
    result = dict(prompt=args.prompt, label=args.label, tokens=[tok.decode([i]) for i in ids], token_ids=ids, label_id=label,
                  settings=SETTINGS, methods={})
    model = target()
    vpd, cfg, vmeta = load_vpd(model, args.vpd)
    with torch.no_grad():
        ci0 = vpd.calc_causal_importances(pre_weight_acts=vpd(tokens, cache_type='input').cache, sampling=cfg.sampling, detach_inputs=False)
    alive_counts = [int(sum((ci0.lower_leaky[n][0, t] > 0).sum() for n in MODULES)) for t in range(len(ids))]
    log(tokens=result['tokens'], vpd_alive_per_position=alive_counts)
    ours_model = target()                                          # a separate copy: each ComponentModel hooks its own target
    ours, factors, ometa = load_ours(ours_model, args.ours, cfg)
    our_initial_ci(ours, ours_model, factors, tokens, label, alive_counts)
    for name, cm, meta, base in (('vpd', vpd, vmeta, model), ('ours', ours, ometa, ours_model)):
        topology = TransformerTopology(base)
        sources = get_sources_by_target(cm, topology, 'cuda', cfg.sampling)
        entry = dict(**meta, alive_per_position=alive_counts, runs=[], graphs={})
        for setting in args.settings:
            for coeff in SETTINGS[setting]['sweep']:
                run, ci = run_setting(cm, cfg, tokens, label, setting, coeff, args.steps)
                entry['runs'].append(run)
                log(method=name, **{k: v for k, v in run.items() if k not in ('metrics', 'nodes_per_module')})
                if coeff == SETTINGS[setting]['paper']:
                    entry['graphs'][setting] = graph(cm, topology, sources, tokens, ci)
                    log(method=name, setting=setting, graph_nodes=len(entry['graphs'][setting]['nodes']),
                        graph_edges=len(entry['graphs'][setting]['edges']))
        result['methods'][name] = entry
        save_json(result, args.output)
    log(done=str(args.output))


if __name__ == '__main__': main()
