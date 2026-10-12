"""Matched final-MLP pilot: exact full-logit effects versus per-token CE gradient."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import parauto_core as core
import parauto_method as method
import parauto_pile as pile
import parauto_pile_last as last
import parauto_structured as structured
import parauto_tracking as tracking
from parauto_target import atomic_json, atomic_save, fingerprint
from train_pile import cpu_tree


def to_device(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list): return [to_device(v, device) for v in value]
    return value


def component_counts(args):
    return [args.components_up or args.components, args.components_down or args.components]


@torch.no_grad()
def validate_real(model, ids, labels, r):
    exact, tail = last.capture(model, ids, labels, r, torch.float32)
    rounded, _ = last.capture(model, ids, labels, r, torch.bfloat16)
    result = {}
    for i, name in enumerate(last.MODULES):
        weight = model.get_submodule(name).weight
        gen = torch.Generator(device=weight.device).manual_seed(51+i)
        p = core.FreePieces(torch.randn(weight.shape[1], 4, device=weight.device, generator=gen)*.01,
                            torch.randn(weight.shape[0], 4, device=weight.device, generator=gen)*.01)
        observed = structured.scores(method.slice_entry(exact['output'][i], slice(0, 1)), p)[0]
        original_z = model.get_submodule(last.MODULES[0])(tail['h'][:1])
        references = []
        for c in range(4):
            def forward(delta):
                if i == 0:
                    mlp = model.get_submodule(last.MODULES[1])(F.gelu(original_z + delta, approximate='tanh'))
                else:
                    mlp = model.get_submodule(last.MODULES[1])(F.gelu(original_z, approximate='tanh')) + delta
                return model.lm_head(model.ln_f(tail['prefix'][:1] + mlp))
            direction = (exact['output'][i]['h'][0] @ p.read[:, c]) * p.write[:, c]
            _, jvp = torch.autograd.functional.jvp(forward, torch.zeros_like(direction), direction)
            references.append(jvp.square().mean().sqrt()/ids.shape[1]**.5)
        ref = torch.stack(references)
        torch.testing.assert_close(observed, ref, atol=1e-7, rtol=2e-4)
        rounded_scores = structured.scores(method.slice_entry(rounded['output'][i], slice(0, 1)), p)[0]
        result[name] = {'direct_full_vocabulary_jvp_relative_error': float((observed-ref).norm()/ref.norm()),
                        'bf16_cache_relative_error': float((rounded_scores-observed).norm()/observed.norm())}
    return result


def cache(args, model):
    started = time.perf_counter()
    if (args.cache/'manifest.json').exists(): raise FileExistsError('Cache already exists')
    args.cache.mkdir(parents=True, exist_ok=True)
    r, gram_error = last.head_factor(model, args.context)
    blocks, offsets = pile.sample_blocks(args.tokens, args.blocks, args.context, 0, args.train_stop, args.seed)
    buckets = {sensor: [[], []] for sensor in ('output', 'scalar')}
    baseline_ce = 0.
    for i, block in enumerate(blocks):
        ids, labels = block[None, :-1].cuda(), block[None, 1:].cuda()
        entries, tail = last.capture(model, ids, labels, r)
        baseline_ce += float(F.cross_entropy(tail['logits'], labels.flatten()))
        if i == 0: validation = validate_real(model, ids, labels, r)
        for sensor, matrices in entries.items():
            for j, entry in enumerate(matrices): buckets[sensor][j].append(cpu_tree(entry))
        tracking.log_metrics({'contexts_completed': i+1, 'contexts_total': args.blocks}, prefix='cache')
        print(json.dumps({'cache_contexts': i+1}), flush=True)
        del entries, tail
    order = torch.randperm(args.blocks*args.context, generator=torch.Generator().manual_seed(args.seed+202))
    for sensor, matrices in buckets.items():
        joined = []
        for rows in matrices:
            entry = dict(rows[0])
            entry['h'] = torch.cat([v['h'] for v in rows])[order]
            entry['net_rms'] = torch.cat([v['net_rms'] for v in rows])[order]
            entry['events'] = {key: torch.cat([v['events'][key] for v in rows])[order] for key in entry['events']}
            joined.append(entry)
        atomic_save(joined, args.cache/f'{sensor}.pt')
    atomic_save(dict(ids=blocks, offsets=offsets, permutation=order), args.cache/'inputs.pt')
    torch.cuda.synchronize()
    manifest = dict(target_sha256=fingerprint(args.target), token_file_sha256=fingerprint(args.tokens),
                    model_source_sha256=fingerprint(pile.target_source.__file__), sensor_source_sha256=fingerprint(last.__file__),
                    modules=list(last.MODULES), config=json.loads(json.dumps(vars(args), default=str)), blocks=args.blocks, context=args.context,
                    events=args.blocks*args.context, train_stop=args.train_stop, offsets=offsets,
                    cache_sha256={s: fingerprint(args.cache/f'{s}.pt') for s in buckets},
                    full_rank_head_gram_error=gram_error, validation=validation,
                    baseline_ce=baseline_ce/args.blocks, cache_seconds=time.perf_counter()-started,
                    output_sensor='exact full-context full-vocabulary logit RMS derivative per local last-MLP use',
                    scalar_sensor='exact per-token ground-truth next-token cross-entropy derivative',
                    precision='BF16 h/GELU derivatives/scalar gradients; FP32 fixed RMSNorm metadata and full-rank head factor')
    atomic_json(manifest, args.cache/'manifest.json')
    atomic_json(manifest, args.output/'result.json')


def load_manifest(args):
    manifest = json.loads((args.cache/'manifest.json').read_text())
    assert manifest['target_sha256'] == fingerprint(args.target)
    assert manifest['model_source_sha256'] == fingerprint(pile.target_source.__file__)
    assert manifest['sensor_source_sha256'] == fingerprint(last.__file__)
    return manifest


def train(args, model):
    manifest = load_manifest(args)
    assert fingerprint(args.cache/f'{args.sensor}.pt') == manifest['cache_sha256'][args.sensor]
    entries = to_device(torch.load(args.cache/f'{args.sensor}.pt', weights_only=True), 'cuda')
    weights = [model.get_submodule(n).weight for n in last.MODULES]
    counts = component_counts(args)
    cfg = method.TrainConfig(steps=args.steps, batch=args.batch, seed=args.seed,
                             precision='bf16', projection_precision='fp32',
                             log_interval=10 if args.phase == 'profile' else 100)
    tracking.configure({'training_config': asdict(cfg)}, matrix_names=last.MODULES)
    saved = None
    if args.resume:
        saved = torch.load(args.output/'checkpoint.pt', map_location='cpu', weights_only=True)
        assert saved['cache_sha256'] == manifest['cache_sha256'][args.sensor]
        assert [p['read'].shape[1] for p in saved['training']['pieces']] == counts
        initial = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['training']['pieces']]
    else:
        if (args.output/'checkpoint.pt').exists(): raise FileExistsError('Use --resume')
        initial = []
        for n, w, count in zip(last.MODULES, weights, counts):
            p, info = method.prepare_initial(w, count, cfg)
            tracking.log_initialization(n, info); initial.append(p)
    next_step = saved['training']['next_step'] if saved else 0
    first_step = next_step
    def checkpoint(state):
        nonlocal next_step
        next_step = state['next_step']
        atomic_save(cpu_tree(dict(training=state, cache_sha256=manifest['cache_sha256'][args.sensor])), args.output/'checkpoint.pt')
        tracking.log_metrics({'step': next_step}, prefix='checkpoint')
    ticks = []
    def log(row):
        if row['matrix_index'] == 0:
            torch.cuda.synchronize(); ticks.append((row['step'], time.perf_counter()))
        print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    pieces, histories = method.fit(entries, weights, initial, cfg, log=log,
        resume=saved['training'] if saved else None, checkpoint=checkpoint, checkpoint_interval=500,
        stop_after=40 if args.phase == 'profile' else args.max_steps)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    atomic_save({'pieces': cpu_tree([{'read': p.read, 'write': p.write} for p in pieces])}, args.output/'pieces.pt')
    atomic_json(dict(zip(last.MODULES, histories)), args.output/'training.json')
    seconds_per_step = (ticks[-1][1]-ticks[1][1])/(ticks[-1][0]-ticks[1][0]) if len(ticks)>2 else None
    result = dict(sensor=args.sensor, config=json.loads(json.dumps(vars(args), default=str)), training_config=asdict(cfg), target_sha256=manifest['target_sha256'],
                  cache_sha256=manifest['cache_sha256'][args.sensor], source_sha256=fingerprint(__file__),
                  modules=list(last.MODULES), components_per_matrix=counts[0] if len(set(counts)) == 1 else dict(zip(last.MODULES, counts)),
                  component_counts=counts, steps_per_matrix=next_step,
                  complete=next_step == args.steps, invocation_seconds=elapsed, invocation_steps=next_step-first_step,
                  steady_seconds_per_step=seconds_per_step, estimated_5k_seconds=seconds_per_step*5000 if seconds_per_step else None,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), cache_bytes=structured.cache_bytes(dict(enumerate(entries))),
                  gpu=torch.cuda.get_device_name(), optimizer='projected_adam_no_rejection_v1',
                  optimizer_updates=next_step-histories[0][-1].get('overflow_skipped_updates', 0),
                  matrix_sum_errors={n: core.reconstruction_error(w, p) for n,w,p in zip(last.MODULES, weights, pieces)})
    atomic_json(result, args.output/'result.json')
    print(json.dumps(result, default=str), flush=True)


@torch.no_grad()
def evaluate(args, model):
    manifest = load_manifest(args)
    assert fingerprint(args.tokens) == manifest['token_file_sha256']
    root = args.runs
    counts = component_counts(args)
    runs = {s: json.loads((root/s/'result.json').read_text()) for s in args.training_sensors}
    assert all(v['training_config'] == next(iter(runs.values()))['training_config'] for v in runs.values())
    assert all(v['complete'] and v.get('component_counts', [v['components_per_matrix']]*2) == counts for v in runs.values())
    parts = {s: [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in
                 torch.load(root/s/'pieces.pt', map_location='cpu', weights_only=True)['pieces']] for s in runs}
    ids, offsets = pile.sample_blocks(args.tokens, args.eval_blocks, manifest['context'], manifest['train_stop'],
                                      args.tokens.stat().st_size//2, args.seed+920091)
    total = sum(counts)
    kept = list(range(total, -1, -1)) if args.all_levels else sorted(set([0, total, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536]), reverse=True)
    keys = tuple(k for k in ('output_output', 'scalar_scalar', 'scalar_output', 'output_scalar') if k.split('_')[0] in runs)
    sums = {k: {m: torch.zeros(len(kept), device='cuda', dtype=torch.float64) for m in ('kl_target', 'ce', 'mse_target')} for k in keys}
    r, _ = last.head_factor(model, manifest['context'])
    tokens, baseline_ce = 0, 0.
    for bi, block in enumerate(ids):
        entries, tail = last.capture(model, block[None, :-1].cuda(), block[None, 1:].cuda(), r, torch.float32)
        lp = tail['logits'].double().log_softmax(-1); prob = lp.exp()
        baseline_ce += float(F.cross_entropy(tail['logits'], tail['labels'], reduction='sum'))
        for key in keys:
            trained, sensor = key.split('_'); pieces = parts[trained]
            scores = [torch.cat([structured.scores(method.slice_entry(e, slice(a,a+16)), p)
                                for a in range(0, len(e['h']),16)]) for e,p in zip(entries[sensor],pieces)]
            scores = torch.cat(scores,-1)
            order = scores.argsort(dim=-1, stable=True)
            ranks = torch.empty_like(order).scatter_(-1, order, torch.arange(total, device='cuda').expand_as(order))
            for j,k in enumerate(kept):
                masks = (ranks >= total-k).split(counts, -1)
                out = last.tail_forward(model, tail, pieces, masks)
                if k == total: torch.testing.assert_close(out, tail['logits'], atol=2e-4, rtol=2e-4)
                kl = (prob*(lp-out.double().log_softmax(-1))).sum(-1).clamp_min(0)
                sums[key]['kl_target'][j] += kl.sum()
                sums[key]['ce'][j] += F.cross_entropy(out, tail['labels'], reduction='sum').double()
                sums[key]['mse_target'][j] += (out.double()-tail['logits'].double()).square().mean(-1).sum()
            print(json.dumps({'eval_context':bi+1,'curve':key,'levels':len(kept)}),flush=True)
            tracking.log_metrics({'contexts_completed':bi+1,'curves_completed_in_context':keys.index(key)+1},prefix='evaluation')
        tokens += len(tail['h'])
    result = dict(methods={key: {'kept': kept, 'metrics': {m: {'mean': (v/tokens).cpu().tolist()} for m,v in metrics.items()}}
                          for key,metrics in sums.items()}, training=runs, tokens=tokens, offsets=offsets, target_ce=baseline_ce/tokens,
                  target_sha256=manifest['target_sha256'], checkpoint_sha256={s:fingerprint(root/s/'pieces.pt') for s in runs},
                  curve='frozen per-token global ranking across the two final-MLP matrices; binary masks; least score removed first',
                  evaluation='FP32 sensors/factors/nonlinear final MLP, RMSNorm and all vocabulary logits; FP64 KL; cached prefix unchanged exactly',
                  all_removal_levels=args.all_levels, component_counts=counts, ranking_uses_labels={'output':False,'scalar':True})
    for key in keys:
        if result['methods'][key]['metrics']['kl_target']['mean'][0] > 1e-6: raise RuntimeError('All-kept KL failed')
    atomic_json(result,args.output/'result.json')
    draw_report(result,args.output)


def draw_report(result, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(('output_output','scalar_scalar','scalar_output','output_scalar'),('#2171b5','#e87517','#26966d','#9a5bc5')))
    fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for key,curve in result['methods'].items():
        color = colors[key]
        label = key.replace('_',' training / ')+' ranking'
        axes[0].plot(curve['kept'][:-1],curve['metrics']['kl_target']['mean'][:-1],color=color,label=label)
        axes[1].plot(curve['kept'][:-1],[v-result['target_ce'] for v in curve['metrics']['ce']['mean'][:-1]],color=color,label=label)
    for ax,ylabel in zip(axes,('KL to frozen LM','Next-token CE increase')):
        ax.set(xscale='log',xlabel='Components retained across final MLP (2 matrices)',ylabel=ylabel)
        ax.set_yscale('symlog',linthresh=1e-6);ax.grid(alpha=.2);ax.legend(fontsize=8)
    counts = result.get('component_counts', [1024, 1024])
    steps = next(iter(result['training'].values()))['steps_per_matrix']
    fig.suptitle(f"4-layer Pile: final-MLP ablatability\n{steps:,} steps · {counts[0]:,} up / {counts[1]:,} down pieces · {result['tokens']:,} held-out tokens · no probes")
    for suffix in ('png','svg'):fig.savefig(root/f'klkeep.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Final-layer MLP: exact output effects versus scalar CE','', '![KLKeep](klkeep.png)','',
           f"Training sensors: {', '.join(result['training'])}. Each run uses {steps:,} scheduled steps, "
           f"{counts[0]:,} up and {counts[1]:,} down components. "
           'BF16 cached activations and score matmuls; FP32 parameters, Adam, projection/backward and reduced norms. '
           'The full-rank vocabulary metric includes the final RMSNorm derivative, with no probes or truncation. '
           'The scalar sensor differentiates per-token ground-truth CE. All curves use FP32 sensors on held-out contexts.', '',
           f"{result['tokens']:,} held-out prediction positions. Baseline CE: {result['target_ce']:.6f}. "
           f"Every removal level evaluated: {result['all_removal_levels']}. Rankings are frozen globally across the two matrices per token. "
           'The complete nonlinear final MLP, RMSNorm and vocabulary head are recomputed for each mask; the unaffected prefix is cached.', '',
           '| Kept | '+' | '.join(k.replace('_',' / ')+' KL' for k in result['methods'])+' |',
           '|---:|'+'---:|'*len(result['methods'])]
    for k in sorted(set([16,32,64,128,256,512,1024,2048,sum(counts)])):
        vals=[c['metrics']['kl_target']['mean'][c['kept'].index(k)] for c in result['methods'].values()]
        lines.append(f'| {k} | '+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines+=['','One seed and a small fixed cache: this is a feasibility and sensor pilot, not a full-model decomposition or a 1B-position run. '
            'Scalar rankings use held-out next-token labels. Exact refers to derivative norms at cached activations, up to floating-point precision; '
            'KLKeep measures actual finite nonlinear ablations.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('cache','profile','train','evaluate'))
    for key in ('target','tokens','cache','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--runs',type=Path)
    parser.add_argument('--sensor',choices=('output','scalar'),default='output')
    parser.add_argument('--blocks',type=int,default=32)
    parser.add_argument('--context',type=int,default=512)
    parser.add_argument('--train-stop',type=int,default=100000000)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--components',type=int,default=1024)
    parser.add_argument('--components-up',type=int)
    parser.add_argument('--components-down',type=int)
    parser.add_argument('--training-sensors',nargs='+',choices=('output','scalar'),default=['output','scalar'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--eval-blocks',type=int,default=2)
    parser.add_argument('--all-levels',action='store_true')
    args=parser.parse_args()
    tracking.start(f'pile-last-{args.phase}-{args.sensor}' if args.phase in ('profile','train') else f'pile-last-{args.phase}',
                   args.output,vars(args),resume=args.resume)
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    model=pile.load_model(args.target)
    if args.phase=='cache':cache(args,model)
    elif args.phase in ('profile','train'):train(args,model)
    else:evaluate(args,model)


if __name__=='__main__':tracking.run_main(main)
