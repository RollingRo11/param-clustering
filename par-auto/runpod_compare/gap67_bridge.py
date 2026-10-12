"""Convert published VPD into GIM's exact-sum, identical-capacity factor family.

No factor training. Appended residual SVD pieces plus exact projection are
evaluated with mapped original CI supports and common GIM loss normalizers.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/gap67_20260915'))
import gap67 as audit
import torch
import numpy as np
from common import target, tokens, save_json, sha, credentials, MODULES, upload, TARGET_SHA
from train_pair import core, method
from gim_attribution import capture_gim
from fisher_attribution import capture_fisher

OUT = ROOT/'output/gap67_bridge_20260915'


@torch.no_grad()
def convert(model, old, desired):
    result, records = [], []
    for name, p, target_piece in zip(MODULES, old, desired, strict=True):
        weight = model.get_submodule(name).weight
        extra = target_piece.count-p.count
        assert extra >= 0
        read, write = p.read.clone(), p.write.clone()
        residual = weight.double()-write.double() @ read.double().T
        if extra:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            assert extra <= len(s)
            root = s[:extra].sqrt()
            write = torch.cat([write, (u[:, :extra]*root).float()], -1)
            read = torch.cat([read, (vh[:extra].T*root).float()], -1)
        augmented = core.FreePieces(read, write)
        before = core.reconstruction_error(weight, augmented)
        exact = core.projected_pieces(weight, augmented)
        after = core.reconstruction_error(weight, exact)
        assert after < 1e-5
        assert exact.count == target_piece.count
        result.append(exact)
        records.append(dict(module=name, original_count=p.count, converted_count=exact.count,
                            original_relative_remainder=float(residual.norm()/weight.double().norm()),
                            after_padding_relative_remainder=before, exact_sum_error=after,
                            original_read_relative_change=float((exact.read[:, :p.count]-p.read).norm()/p.read.norm()),
                            original_write_relative_change=float((exact.write[:, :p.count]-p.write).norm()/p.write.norm())))
    return result, records


def remap(keep, original, converted):
    return torch.cat([torch.nn.functional.pad(m, (0, q.count-p.count))
                      for m,p,q in zip(keep.split([p.count for p in original], -1), original, converted, strict=True)], -1)


def variable_support(scores, counts):
    order = scores.argsort(dim=-1, descending=True, stable=True)
    rank_mask = torch.arange(scores.shape[-1], device='cuda')[None] < counts[:, None]
    return torch.zeros_like(rank_mask).scatter_(-1, order, rank_mask)


@torch.no_grad()
def objective(cache, pieces, initial_scores):
    result = {}
    cfg = method.TrainConfig(precision='bf16')
    scored = [method.training_scores(e, p, cfg) for e,p in zip(cache, pieces, strict=True)]
    for power in [.5, .641, 1.]:
        result[str(power)] = sum(float(method.lp_concentration(s, power, 1e-6).mean()/
                                      method.lp_concentration(ref, power, 1e-6).mean())
                                for s,ref in zip(scored, initial_scores, strict=True))
    count = sum(float((s/(s+.001)).sum(-1).mean()) for s in scored)
    ref = sum(float((s.abs()/(s.abs()+.001)).sum(-1).mean()) for s in initial_scores)
    result['soft_count'] = count
    result['normalized_count'] = count*len(MODULES)/ref
    result['lp_plus_count_p641'] = result['0.641']+result['normalized_count']
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    OUT.mkdir(parents=True, exist_ok=True)
    model = target(); data, manifest = tokens('val')
    old, trainer, oldmeta = audit.shared.load(model, 'vpd_published')
    ours, _, oursmeta = audit.shared.load(model, 'gim_5000')
    bridge, conversion = convert(model, old, ours)
    save_json(conversion, OUT/'conversion.json')
    torch.save(dict(method='parauto', completed=0, target_sha256=TARGET_SHA,
                    factors=[dict(read=p.read.cpu(), write=p.write.cpu()) for p in bridge],
                    provenance=dict(kind='VPD converted to exact GIM capacity; no training', source=oldmeta)), OUT/'bridge_model.pth')
    # This checkpoint's reference scores were bitwise matched to normal GIM's
    # initialization in the prior count validation. mmap avoids loading Adam state.
    cp = ROOT/'output/gim_count67_20260914/checkpoints/005000/checkpoint.pt'
    saved = torch.load(cp, map_location='cpu', mmap=True, weights_only=False)
    initial_scores = [v.cuda() for v in saved['state']['initial_scores']]
    del saved
    plan = dict(source_sha256=sha(__file__), target_sha256=TARGET_SHA, data_sha256=manifest['val']['sha256'],
                reference_checkpoint=str(cp), original=oldmeta, ours=oursmeta,
                source_components=sum(p.count for p in old), converted_components=sum(p.count for p in bridge),
                first_sequence=424, sequences=8, conversion=conversion,
                note='No training; same rank-one capacity and exact-sum family as GIM. GIM loss uses common fixed initial reference scores.')
    save_json(plan, OUT/'plan.json')
    records = []
    for first in range(424, 432):
        dest = OUT/f'{first}.json'
        if dest.exists():
            records.append(json.loads(dest.read_text()))
            continue
        rows = torch.tensor(np.array(data[first:first+1], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        base, inputs = audit.clean_capture(model, ids)
        logp = base.log_softmax(-1)
        gc = capture_gim(model, rows, microbatch=1)
        fc = capture_fisher(model, rows, k=128, seed=6_100_000+first, microbatch=1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            cached = trainer.model(ids, cache_type='input')
            ci = trainer.model.calc_causal_importances(cached.cache, sampling='continuous').lower_leaky
            native = torch.cat([ci[n].float()[0] for n in MODULES], -1)
            positive = native > 0
            order = native.argsort(dim=-1, descending=True, stable=True)
            fixed = audit.shared.support(order, 512)
            del cached, ci, native
        counts = positive.sum(-1)
        record = dict(first_sequence=first, native_count_mean=float(counts.float().mean()),
                      native_count_min=int(counts.min()), native_count_max=int(counts.max()), methods={})
        all_masks = {}
        for label, pieces in [('vpd_original', old), ('vpd_exact', bridge), ('gim', ours)]:
            entry = dict(objective=objective(gc, pieces, initial_scores), variants={})
            scores, _ = audit.rank_and_census(fc, gc, pieces)
            variants = {'fisher_native_budget': variable_support(scores, counts),
                        'fisher512': audit.shared.support(scores.argsort(dim=-1, descending=True, stable=True), 512)}
            if label == 'vpd_original':
                variants.update(native_positive=positive, native512=fixed)
            elif label == 'vpd_exact':
                variants.update(native_positive=remap(positive, old, bridge), native512=remap(fixed, old, bridge))
            else:
                variants['native_allocation'] = audit.allocated_support(scores, pieces, positive.split([p.count for p in old], -1))
            for name, keep in variants.items():
                entry['variants'][name] = audit.diagnose(model, ids, pieces, keep, logp, inputs)
                entry['variants'][name]['count_mean'] = float(keep.sum(-1).float().mean())
                all_masks[label+'_'+name] = keep.nonzero().cpu().numpy().astype(np.uint16)
            if label == 'vpd_exact':
                inherited = variants['native_positive']
                best, history = audit.endpoint_search(model, ids, pieces, logp, inherited, 32)
                entry['variants']['native_refined'] = dict(**audit.diagnose(model, ids, pieces, best, logp, inputs), search=history,
                                                         count_mean=float(best.sum(-1).float().mean()))
                all_masks[label+'_native_refined'] = best.nonzero().cpu().numpy().astype(np.uint16)
            with torch.no_grad():
                entry['all_kept_kl'] = float(audit.loss(audit.run(model, ids, pieces, torch.ones_like(scores)), logp))
            record['methods'][label] = entry
        sp = OUT/f'{first}_supports.npz'
        np.savez_compressed(sp, **all_masks)
        record['supports_sha256'] = sha(sp)
        save_json(record, dest); records.append(record)
        print(json.dumps(dict(sequence=first, native_count=record['native_count_mean'],
                              kl={m:{n:v['kl'] for n,v in e['variants'].items()} for m,e in record['methods'].items()})), flush=True)
    from gap67_bridge_report import finish
    finish(OUT, plan, records)


if __name__ == '__main__':
    main()
