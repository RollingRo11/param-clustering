"""One GPU per method, identical saved batches and exact 1B-position budget.

VPD's model, CI network, losses, PGD updates, warmup, schedules, and optimizers
come directly from the pinned vpd-paper release. This harness supplies matching
data, explicit timing, complete resume state, and common checkpoint/eval cadence.
"""
import argparse
from dataclasses import asdict
import json
import os
import random
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from common import (ROOT, HERE, STEPS, POSITIONS, CONTEXT, BATCH, MODULES, VPD_COMMIT, TARGET_SHA,
                    credentials, target, tokens, batch, sha, save_json, checkpoint, cpu_tree)

sys.path.insert(0, str(HERE.parent/'par-auto'))
import parauto_core as core
import parauto_method as method
import parauto_tracking as tracking


def bounded_recon_kl(pred, target):
    """Upstream KL, summed over position chunks with activation recomputation."""
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    p, t = pred.flatten(0, -2), target.flatten(0, -2)
    value = pred.new_zeros((), dtype=torch.float32)
    for first in range(0, len(p), 2048):
        value = value + activation_checkpoint(lambda a, b: recon_loss_kl(a, b)[0],
                                             p[first:first+2048], t[first:first+2048], use_reentrant=False)
    return value, len(p)


class VPD:
    def __init__(self, model, evaluation=False):
        from param_decomp.configs import Config, PersistentPGDReconLossConfig
        from param_decomp.models.component_model import ComponentModel
        from param_decomp.models.batch_and_loss_fns import make_run_batch, recon_loss_kl
        from param_decomp.utils.module_utils import expand_module_patterns
        from param_decomp.run_param_decomp import run_faithfulness_warmup
        from param_decomp.persistent_pgd import PersistentPGDState
        raw = yaml.safe_load((HERE/'vpd_published.yaml').read_text())
        raw.update(steps=STEPS, batch_size=BATCH, wandb_project='param-autoenc')
        self.cfg = Config.model_validate(raw)
        self.model = ComponentModel(model, run_batch=make_run_batch(self.cfg.output_extract),
                                    module_path_info=expand_module_patterns(model, self.cfg.all_module_info),
                                    ci_config=self.cfg.ci_config, sigmoid_type=self.cfg.sigmoid_type).cuda()
        # Preserve the published full-batch objective on one GPU. Recompute
        # CI transformer blocks during backward instead of retaining their
        # internal activations; module/state-dict names are unchanged.
        from param_decomp.models.components import TransformerBlock
        from torch.utils.checkpoint import checkpoint as activation_checkpoint
        for block in self.model.ci_fn.modules():
            if isinstance(block, TransformerBlock):
                original = block.forward
                block.forward = lambda x, fn=original: activation_checkpoint(fn, x, use_reentrant=False)
        self.params = [p for name in self.model.target_module_paths for p in self.model.components[name].parameters()]
        self.ci_params = list(self.model.ci_fn.parameters())
        if evaluation: return
        self.optimizers = [torch.optim.AdamW(params, lr=cfg.lr_schedule.start_val,
                                            betas=cfg.betas, weight_decay=cfg.weight_decay)
                           for params, cfg in ((self.params, self.cfg.components_optimizer),
                                               (self.ci_params, self.cfg.ci_fn_optimizer))]
        self.recon = bounded_recon_kl
        self.ppgd = {cfg: PersistentPGDState(self.model.module_to_c, (BATCH, CONTEXT), 'cuda',
                                            self.cfg.use_delta_component, cfg, self.recon)
                     for cfg in self.cfg.loss_metric_configs if isinstance(cfg, PersistentPGDReconLossConfig)}
        self.warmup = lambda: run_faithfulness_warmup(self.model, self.params, self.cfg)

    def step(self, rows, step):
        from param_decomp.losses import compute_losses
        from param_decomp.utils.general_utils import get_scheduled_value
        for opt, cfg in zip(self.optimizers, (self.cfg.components_optimizer, self.cfg.ci_fn_optimizer)):
            opt.zero_grad()
            for group in opt.param_groups:
                group['lr'] = get_scheduled_value(step, STEPS, cfg.lr_schedule)
        for cfg, state in self.ppgd.items():
            state.update_lr(step, STEPS)
            if len(rows) != next(iter(state.sources.values())).shape[0]:
                # Exact final partial batch: preserve the first slots' sources,
                # moment estimates and step counter; only discard unused slots.
                state.sources = {k: v[:len(rows)].detach().clone().requires_grad_() for k, v in state.sources.items()}
                state.optimizer._m = {k: v[:len(rows)].clone() for k, v in state.optimizer._m.items()}
                state.optimizer._v = {k: v[:len(rows)].clone() for k, v in state.optimizer._v.items()}
        deltas = self.model.calc_weight_deltas()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model(rows[:, :-1], cache_type='input')
            ci = self.model.calc_causal_importances(out.cache, detach_inputs=False, sampling=self.cfg.sampling)
            for cfg, state in self.ppgd.items():
                state.warmup(self.model, rows[:, :-1], out.output, ci.lower_leaky, deltas)
            losses = compute_losses(loss_metric_configs=self.cfg.loss_metric_configs, model=self.model,
                                    batch=rows[:, :-1], ci=ci, target_out=out.output, weight_deltas=deltas,
                                    current_frac_of_training=step/STEPS, sampling=self.cfg.sampling,
                                    use_delta_component=self.cfg.use_delta_component,
                                    n_mask_samples=self.cfg.n_mask_samples, ppgd_states=self.ppgd,
                                    reconstruction_loss=self.recon)
        total = sum(cfg.coeff*value for cfg, value in losses.items())
        grads = {cfg: state.get_grads(losses[cfg], retain_graph=True) for cfg, state in self.ppgd.items()}
        total.backward()
        for cfg, state in self.ppgd.items(): state.step(grads[cfg])
        for params, cfg, opt in zip((self.params, self.ci_params),
                                    (self.cfg.components_optimizer, self.cfg.ci_fn_optimizer), self.optimizers):
            if cfg.grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            opt.step()
        metrics = {cfg.classname: float(v.detach()) for cfg, v in losses.items()}
        metrics['total_loss'] = float(total.detach())
        if not np.isfinite(list(metrics.values())).all(): raise RuntimeError('Nonfinite VPD loss')
        return metrics

    def state(self):
        return dict(model=self.model.state_dict(), optimizers=[o.state_dict() for o in self.optimizers],
                    ppgd=[dict(sources=s.sources, optimizer=vars(s.optimizer)) for s in self.ppgd.values()])

    def load(self, state):
        self.model.load_state_dict(state['model'])
        for opt, saved in zip(self.optimizers, state['optimizers']): opt.load_state_dict(saved)
        for p, saved in zip(self.ppgd.values(), state['ppgd']):
            p.sources = {k: v.cuda().requires_grad_() for k, v in saved['sources'].items()}
            for k, v in saved['optimizer'].items():
                setattr(p.optimizer, k, {n: t.cuda() for n, t in v.items()} if isinstance(v, dict) else v)


def capture(model, rows, microbatch=4, output_transform=None):
    """True scalar CE derivatives through the same upstream BF16 target."""
    entries = None
    for first in range(0, len(rows), microbatch):
        block = rows[first:first+microbatch]
        pre, post, hooks = {}, {}, []
        def retain(name):
            def hook(_m, inputs, out):
                pre[name], post[name] = inputs[0], out
                # Retain the original output BEFORE an optional attribution
                # backward rule; its VJP must include that rule in the graph.
                if output_transform is not None: return output_transform(name, out)
            return hook
        for name in MODULES: hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
        hooks.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
        try:
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(block[:, :-1])[0]
                loss = F.cross_entropy(logits.flatten(0, 1), block[:, 1:].flatten(), reduction='sum')
            grads = torch.autograd.grad(loss, [post[n] for n in MODULES])
            if entries is None:
                entries = [dict(structured=True, kind='scalar_loss', shared={},
                                h=torch.empty(len(rows)*CONTEXT, pre[n].shape[-1], dtype=torch.bfloat16, device='cuda'),
                                events={'g': torch.empty(len(rows)*CONTEXT, g.shape[-1], dtype=torch.bfloat16, device='cuda')},
                                net_rms=torch.zeros(len(rows)*CONTEXT, device='cuda')) for n, g in zip(MODULES, grads)]
            lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
            for n, g, e in zip(MODULES, grads, entries):
                e['h'][lo:hi] = pre[n].detach().flatten(0, 1).bfloat16()
                e['events']['g'][lo:hi] = g.detach().flatten(0, 1).bfloat16()
        finally:
            for hook in hooks: hook.remove()
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['vpd', 'parauto'])
    parser.add_argument('--pilot-steps', type=int, default=0)
    parser.add_argument('--data-split', choices=['train', 'val'], default='train')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    args = parser.parse_args()
    if args.data_split != 'train' and not 0 < args.pilot_steps <= 32:
        raise ValueError('Validation data is only allowed for a discarded, <=32-step pilot')
    credentials()
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    data, manifest = tokens(args.data_split)
    data_sha = manifest[args.data_split]['sha256']
    out = ROOT/'output'/('pilot' if args.pilot_steps else 'train')/args.method
    out.mkdir(parents=True, exist_ok=True)
    path = out/'checkpoint.pt'
    if path.exists() and not args.resume: raise FileExistsError('Checkpoint exists; use --resume')
    previous = json.loads((out/'wandb.json').read_text()) if args.resume else None
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu',
                     name=f'paired-1b-{args.method}'+('-pilot' if args.pilot_steps else ''),
                     group='paired-h200-1b-20260910', id=previous['id'] if previous else None,
                     resume='must' if previous else None, dir=str(out),
                     config=dict(method=args.method, target_sha256=TARGET_SHA, vpd_commit=VPD_COMMIT,
                                 steps=STEPS, positions=POSITIONS, batch_sequences=BATCH, context=CONTEXT,
                                 dataset_manifest=manifest, precision='bf16 autocast; fp32 parameters/Adam',
                                 gpu=torch.cuda.get_device_name(), pilot_steps=args.pilot_steps,
                                 physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 source_manifest_sha256=sha(ROOT/'output/source_manifest.json') if (ROOT/'output/source_manifest.json').exists() else None))
    save_json(dict(id=run.id, url=run.url), out/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*', step_metric='update')
    model = target()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)
    resume = torch.load(path, map_location='cpu', weights_only=False) if args.resume else None
    if resume and resume['data_sha256'] != data_sha: raise ValueError('Resume data mismatch')
    completed = resume['completed'] if resume else 0
    first_step = completed
    train_seconds = 0.
    save_seconds = 0.
    def save(state, completed, factors=None):
        nonlocal save_seconds
        started = time.perf_counter()
        checkpoint(dict(method=args.method, completed=completed, data_sha256=data_sha,
                        target_sha256=TARGET_SHA, state=state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), path)
        if factors is not None and (completed in (1000, 5000, 10000, 20000, 30000, STEPS)):
            checkpoint(dict(step=completed, factors=factors, target_sha256=TARGET_SHA), out/f'model_{completed}.pth')
        save_seconds += time.perf_counter()-started
        save_json(dict(completed=completed, positions=min(POSITIONS, completed*BATCH*CONTEXT),
                       checkpoint=path.name, modified_ns=path.stat().st_mtime_ns), out/'checkpoint_ready.json')
    invocation = time.perf_counter()
    if args.method == 'vpd':
        trainer = VPD(model)
        run.config.update({'vpd_config': trainer.cfg.model_dump(mode='json')})
        run.config.update({'memory_execution': 'CI-block activation checkpointing; upstream KL in 2048-position checkpointed chunks'})
        if resume:
            trainer.load(resume['state'])
            torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state(resume['cuda_rng'])
        else: trainer.warmup()
        setup_seconds = time.perf_counter()-invocation
        torch.cuda.reset_peak_memory_stats()
        end = min(STEPS, completed+args.pilot_steps) if args.pilot_steps else STEPS
        for step in range(completed, end):
            torch.cuda.synchronize(); tick = time.perf_counter()
            rows = batch(data, step)
            metrics = trainer.step(rows, step)
            torch.cuda.synchronize(); train_seconds += time.perf_counter()-tick
            completed = step+1
            if completed % 20 == 0 or completed == end:
                metrics.update(seconds_per_update=train_seconds/(completed-first_step),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               positions=min(POSITIONS, completed*BATCH*CONTEXT))
                run.log({'update': completed, **{'train/'+k: v for k, v in metrics.items()}})
                print(json.dumps(dict(completed=completed, **metrics)), flush=True)
            if completed % args.checkpoint_interval == 0 or completed == end or stop:
                save(trainer.state(), completed, trainer.model.state_dict())
            if stop: break
        factors = trainer.model.state_dict()
    else:
        weights = [model.get_submodule(n).weight for n in MODULES]
        counts = [dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)[n.split('.')[-1]] for n in MODULES]
        cfg = method.TrainConfig(steps=STEPS, batch=BATCH*CONTEXT, precision='bf16', projection_precision='fp32', log_interval=100)
        run.config.update({'training_config': asdict(cfg), 'component_counts': dict(zip(MODULES, counts))})
        init = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, counts)]
        if resume: init = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in resume['state']['pieces']]
        first = capture(model, batch(data, 0))
        ix = torch.randperm(len(first[0]['h']), device='cuda')[:256]
        reference = [method.slice_entry(e, ix) for e in first]
        pending = [first] if not resume else []
        del first
        def provide(step): return pending.pop() if step == 0 and pending else capture(model, batch(data, step))
        setup_seconds = time.perf_counter()-invocation
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        def finish(completed_step):
            nonlocal completed, tick, train_seconds
            torch.cuda.synchronize()
            train_seconds += time.perf_counter()-tick
            completed = completed_step
            if completed % 20 == 0:
                metrics = dict(seconds_per_update=train_seconds/(completed-first_step),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               positions=min(POSITIONS, completed*BATCH*CONTEXT))
                run.log({'update': completed, **{'train/'+k: v for k, v in metrics.items()}})
                print(json.dumps(dict(completed=completed, **metrics)), flush=True)
            tick = time.perf_counter()
            return stop
        def save_ours(state):
            nonlocal tick
            save(state, state['next_step'], state['pieces'])
            tick = time.perf_counter()
        def log(row):
            if row['matrix_index'] == 0:
                run.log({'update': row['step']+1, 'train/total_loss': row['total_loss'],
                         'train/matrix0_sum_error': row['reconstruction_relative_error']})
            run.log({'update': row['step']+1,
                     f'matrices/{MODULES[row["matrix_index"]]}/loss': row['loss'],
                     f'matrices/{MODULES[row["matrix_index"]]}/sum_error': row['reconstruction_relative_error']})
        fitted, histories = method.fit(reference, weights, init, cfg, batch_provider=provide,
                                       step_callback=finish, checkpoint=save_ours,
                                       checkpoint_interval=args.checkpoint_interval, allow_partial_batch=True,
                                       resume=resume['state'] if resume else None,
                                       stop_after=args.pilot_steps or None, log=log)
        factors = [{'read': p.read, 'write': p.write} for p in fitted]
    checkpoint(dict(completed=completed, factors=factors, method=args.method, target_sha256=TARGET_SHA), out/'final.pth')
    result = dict(method=args.method, completed=completed, positions=min(POSITIONS, completed*BATCH*CONTEXT),
                  complete=completed == STEPS, pilot=bool(args.pilot_steps), setup_seconds=setup_seconds,
                  train_seconds=train_seconds, checkpoint_seconds=save_seconds,
                  seconds_per_update=train_seconds/(completed-first_step),
                  invocation_seconds=time.perf_counter()-invocation, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu=torch.cuda.get_device_name(), target_sha256=TARGET_SHA, data_sha256=data_sha)
    save_json(result, out/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
