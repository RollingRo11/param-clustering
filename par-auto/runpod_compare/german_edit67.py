"""German removal by editing, as in "Exploration: fine-tuning with parameter decomposition" (LessWrong, ieoWstubDQWLrMnhH),
which used Goodfire's published VPD of this 67M target. Arms, all on the same data, objective and selection rule:

  piece   one decomposition piece (ours, or a VPD subcomponent) gets one trainable scalar alpha, init 1:
          W' = W + (alpha - 1) u_c r_c^T on the target model (r_c read, u_c write; VPD: V[:, c], U[c, :]). The post's
          single-mask fine-tune of h.3.attn.v_proj:513 is this arm with VPD's piece (applied to the target's weights; the
          post edits the reconstructed model).
  lora    rank-r adapters B A^T (B = 0 at init, A ~ N(0, 1/d_in), scale lora_alpha / r) on all 24 linears.

Objective (the post's): maximise German CE capped at chance, ln(50,277) = 10.83 nats, with a KL penalty to the original
model on English: loss = relu(ceiling - CE_de) + lambda * KL(p_original || p_edited) on a Europarl English block, both
per-token means. Adam; stop when the EMA (0.99) of the loss has not improved by 1e-4 for 300 steps, or at 3,000 steps.
German budgets 2 ... 2,048 tokens; seed s trains on its own German text (2,048-token slice s of the German train pool;
budget b takes its first b tokens, in pieces of at most 512 tokens). Per budget, every (lr, lambda) config is trained on seeds
0-2 and the config with the largest median dev German CE increase among those with median dev English damage (CE increase
on held-out Pile) < 0.1 nats is chosen (the post's rule, which picks the lr per budget; its exact threshold and lambda grid
are not given); seeds 3-8 then run at that config only. Reported on separate
eval blocks: CE increase on Europarl German, English, French, Spanish, Italian, held-out Pile, and code.

Piece selection (rank): each piece's share of tokens where it is among the token's top 128 pieces (ours: Fisher K=8
score; VPD: causal importance), German minus English; the single piece is, of the top 16, the one with the largest
German share minus its largest share on English, French, Spanish or Italian (a label-free stand-in for the post's pick of
the one German-only autointerp label). VPD's mean-CI German-minus-English ranking (the post's) is reported too.

Usage:
  german_edit67.py data --output DATA.pt
  german_edit67.py rank --method parauto|vpd --checkpoint CKPT --data DATA.pt --output RANK.json
  german_edit67.py edit --arm piece --method parauto|vpd --checkpoint CKPT --piece h.3.attn.v_proj:513 --data DATA.pt --output OUT.json
  german_edit67.py edit --arm lora --rank 1 --data DATA.pt --output OUT.json
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch
import torch.nn.functional as F

from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha

ROW = 513                                   # tokens per block: 512 inputs, 512 next-token targets
EUROPARL, CODEPARROT, TOKENIZER = 'Helsinki-NLP/europarl', 'codeparrot/codeparrot-clean-valid', 'EleutherAI/gpt-neox-20b'
SEEDS, SEED_TOKENS = 9, 2048
BLOCKS = dict(rank=4, dev=2, eval=8)
BUDGETS = (2, 4, 8, 16, 32, 64, 128, 512, 2048)
PILE_DEV, PILE_EVAL = range(2000, 2004), range(2004, 2012)      # rows of the paired val split
TOP, CANDIDATES = 128, 16
EVAL_SETS = ('de', 'en', 'fr', 'es', 'it', 'pile', 'code')


def log(**fields): print(json.dumps(fields), flush=True)


# ---------------------------------------------------------------------------------------------------------- data
def europarl(tokenizer, config, languages, count):
    """Aligned Europarl sentences, each followed by EOS, until every language has `count` tokens."""
    from datasets import load_dataset
    out = {l: [] for l in languages}
    for example in load_dataset(EUROPARL, config, split='train', streaming=True):
        for l in languages: out[l] += tokenizer.encode(example['translation'][l]).ids + [0]
        if all(len(out[l]) >= count for l in languages): return out
    raise ValueError(f'{config}: ran out of text')


def as_blocks(ids, count):
    need = count * ROW
    assert len(ids) >= need, (len(ids), need)
    return torch.tensor(ids[:need], dtype=torch.long).view(count, ROW)


def make_data(args):
    from datasets import load_dataset
    from tokenizers import Tokenizer
    credentials()
    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    assert tokenizer.token_to_id('<|endoftext|>') == 0
    train_blocks = SEEDS * SEED_TOKENS // 512                           # German: 4 blocks of 512 targets per seed
    n_de = BLOCKS['rank'] + train_blocks + BLOCKS['dev'] + BLOCKS['eval']
    n_en = BLOCKS['rank'] + 4 + BLOCKS['eval']                           # English: rank, 4 preserve blocks, eval
    pair = europarl(tokenizer, 'de-en', ('de', 'en'), max(n_de, n_en) * ROW)
    de, en = as_blocks(pair['de'], n_de), as_blocks(pair['en'], n_en)
    data, i = {}, 0
    for name, n in (('rank', BLOCKS['rank']), ('train', train_blocks), ('dev', BLOCKS['dev']), ('eval', BLOCKS['eval'])):
        data[f'de_{name}'] = de[i:i+n]; i += n
    data['en_rank'], data['en_preserve'], data['en_eval'] = en[:4], en[4:8], en[8:8+BLOCKS['eval']]
    for language in ('fr', 'es', 'it'):
        other = europarl(tokenizer, f'en-{language}', ('en', language), (BLOCKS['rank'] + BLOCKS['eval']) * ROW)
        b = as_blocks(other[language], BLOCKS['rank'] + BLOCKS['eval'])
        data[f'{language}_rank'], data[f'{language}_eval'] = b[:BLOCKS['rank']], b[BLOCKS['rank']:]
    code = []
    for example in load_dataset(CODEPARROT, split='train', streaming=True):
        code += tokenizer.encode(example['content']).ids + [0]
        if len(code) >= BLOCKS['eval'] * ROW: break
    data['code_eval'] = as_blocks(code, BLOCKS['eval'])
    val, manifest = tokens('val')
    data['pile_dev'] = torch.from_numpy(np.array(val[PILE_DEV.start:PILE_DEV.stop], dtype=np.int64))
    data['pile_eval'] = torch.from_numpy(np.array(val[PILE_EVAL.start:PILE_EVAL.stop], dtype=np.int64))
    data['meta'] = dict(tokenizer=TOKENIZER, europarl=EUROPARL, codeparrot=CODEPARROT, val_sha256=manifest['val']['sha256'],
                        pile_dev_rows=[PILE_DEV.start, PILE_DEV.stop], pile_eval_rows=[PILE_EVAL.start, PILE_EVAL.stop],
                        shapes={k: list(v.shape) for k, v in data.items() if torch.is_tensor(v)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    log(wrote=str(args.output), sha256=sha(args.output), **data['meta']['shapes'])


# ------------------------------------------------------------------------------------------------------ pieces
def load_pieces(method, checkpoint, model):
    """(read [d_in, C], write [d_out, C]) per matrix, plus a scorer(rows) -> [tokens, total pieces] and the VPD trainer."""
    if method == 'vpd':
        from param_decomp.utils.distributed_utils import init_distributed
        init_distributed()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'), checkpoint_sha256=sha(checkpoint))
    if method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        factors = [(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().T.float()) for n in MODULES]

        def scores(rows):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(rows[:, :-1], cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                return torch.cat([ci[n].float() for n in MODULES], -1).clamp(0., 1.).flatten(0, 1)
    else:
        from train_pair import core, method as pmethod
        from fisher_attribution import capture_fisher
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        factors = [(p.read.float(), p.write.float()) for p in pieces]

        def scores(rows):
            cache = capture_fisher(model, rows, k=8, seed=0)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                return torch.cat([pmethod.scores_of(e, p).float() for e, p in zip(cache, pieces)], -1)
    del saved
    return factors, scores, meta


def piece_names(factors):
    return [f'{n}:{c}' for n, (read, _) in zip(MODULES, factors) for c in range(read.shape[1])]


def rank(args):
    credentials()
    data = torch.load(args.data, weights_only=False)
    model = target()
    factors, scores, meta = load_pieces(args.method, args.checkpoint, model)
    names = piece_names(factors)
    share, mean = {}, {}
    for language in ('de', 'en', 'fr', 'es', 'it'):
        hits, total, count = 0., 0., 0
        for row in data[f'{language}_rank'].cuda().split(2):
            s = scores(row)                                                     # [tokens, pieces]
            top = s.topk(TOP, dim=-1).indices
            hits = hits + torch.zeros_like(s).scatter_(-1, top, 1.).sum(0)
            total = total + s.sum(0); count += s.shape[0]
        share[language], mean[language] = (hits / count).cpu(), (total / count).cpu()
    contrast = share['de'] - share['en']
    top16 = contrast.topk(CANDIDATES).indices.tolist()
    others = torch.stack([share[l] for l in ('en', 'fr', 'es', 'it')]).amax(0)
    specific = share['de'] - others
    pick = max(top16, key=lambda i: float(specific[i]))
    row = lambda i: dict(piece=names[i], **{f'share_{l}': round(float(share[l][i]), 4) for l in share},
                         german_minus_english=round(float(contrast[i]), 4), german_minus_max_other=round(float(specific[i]), 4))
    result = dict(method=args.method, checkpoint=str(args.checkpoint), **meta, top=TOP, rank_tokens={l: int(data[f'{l}_rank'].numel()) for l in share},
                  top16=[row(i) for i in top16], selected=row(pick))
    if args.method == 'vpd':                                                    # the post's ranking: mean CI, German - English
        ci_contrast = mean['de'] - mean['en']
        order = ci_contrast.argsort(descending=True).tolist()
        result['post_ranking_top16'] = [dict(piece=names[i], mean_ci_de=round(float(mean['de'][i]), 4),
                                             mean_ci_en=round(float(mean['en'][i]), 4)) for i in order[:CANDIDATES]]
        if 'h.3.attn.v_proj:513' in names:
            j = names.index('h.3.attn.v_proj:513')
            result['post_piece'] = dict(**row(j), post_ranking_position=order.index(j) + 1, our_ranking_position=contrast.argsort(descending=True).tolist().index(j) + 1)
    save_json(result, args.output)
    log(selected=result['selected'], top16=[r['piece'] for r in result['top16']], post_piece=result.get('post_piece'))


# -------------------------------------------------------------------------------------------------------- edits
class PieceEdit:
    """W' = W + (alpha - 1) u r^T on one matrix, as a forward hook (exact: the matrix is linear in W)."""

    def __init__(self, model, module, read, write):
        self.model, self.read, self.write = model, read.cuda().float(), write.cuda().float()
        self.alpha = torch.nn.Parameter(torch.ones((), device='cuda'))
        self.hook = model.get_submodule(module).register_forward_hook(
            lambda _m, a, out: out + ((a[0].float() @ self.read)[..., None] * self.write * (self.alpha - 1)).to(out.dtype))

    def parameters(self): return [self.alpha]

    def reset(self):
        with torch.no_grad(): self.alpha.fill_(1.)

    def state(self): return dict(alpha=float(self.alpha))


class Lora:
    def __init__(self, model, rank, lora_alpha, seed):
        self.scale, self.rank, self.seed, self.mods, self.hooks = lora_alpha / rank, rank, seed, [], []
        for name in MODULES:
            linear = model.get_submodule(name)
            d_out, d_in = linear.weight.shape
            a = torch.nn.Parameter(torch.zeros(rank, d_in, device='cuda')); b = torch.nn.Parameter(torch.zeros(d_out, rank, device='cuda'))
            self.mods.append((a, b))
            self.hooks.append(linear.register_forward_hook(
                lambda _m, inp, out, a=a, b=b: out + (((inp[0].float() @ a.T) @ b.T) * self.scale).to(out.dtype)))
        self.reset()

    def parameters(self): return [t for ab in self.mods for t in ab]

    def reset(self):
        g = torch.Generator(device='cuda').manual_seed(self.seed)
        with torch.no_grad():
            for a, b in self.mods:
                a.copy_(torch.randn(a.shape, device='cuda', generator=g) / math.sqrt(a.shape[1])); b.zero_()

    def state(self): return dict(b_norm=float(torch.sqrt(sum((b.detach() ** 2).sum() for _, b in self.mods))))


def logits_of(model, rows):
    with torch.autocast('cuda', dtype=torch.bfloat16): return model(rows[:, :-1])[0].float()


def ce(logits, rows): return F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten())


def german_rows(data, seed, budget):
    """Seed's first `budget` German tokens as rows of at most 513 tokens (budget 2,048 = 4 rows of 512)."""
    pool = data['de_train'][4*seed:4*seed+4]
    if budget <= ROW: return pool[:1, :budget].cuda()
    return pool[:, :512][: budget // 512].cuda()


def edit(args):
    credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    data = torch.load(args.data, weights_only=False)
    model = target()
    vocab = model.lm_head.weight.shape[0] if hasattr(model, 'lm_head') else 50277
    ceiling = math.log(vocab)
    meta = {}
    if args.arm == 'piece':
        factors, _, meta = load_pieces(args.method, args.checkpoint, model)
        module, index = args.piece.rsplit(':', 1)
        read, write = factors[MODULES.index(module)]
        editor = PieceEdit(model, module, read[:, int(index)], write[:, int(index)])
        meta.update(piece=args.piece, piece_norm=float(read[:, int(index)].norm() * write[:, int(index)].norm()))
        del factors
        configs = [dict(lr=lr, kl_lambda=l) for lr in args.lrs or (0.01, 0.03, 0.1) for l in args.lambdas]
    else:
        editor = Lora(model, args.rank, args.lora_alpha, seed=0)
        meta.update(rank=args.rank, lora_alpha=args.lora_alpha, modules=list(MODULES),
                    parameters=sum(t.numel() for t in editor.parameters()))
        configs = [dict(lr=lr, kl_lambda=l) for lr in args.lrs or (0.001, 0.003, 0.01) for l in args.lambdas]
    for t in editor.parameters(): t.requires_grad_(True)

    with torch.no_grad():                                       # the original model's outputs: edits start at the identity
        base_logp_en = [F.log_softmax(logits_of(model, r[None].cuda()), -1) for r in data['en_preserve']]
        base_ce = {}
        for name in ('de_dev', 'pile_dev') + tuple(f'{s}_eval' for s in EVAL_SETS):
            base_ce[name] = float(np.mean([float(ce(logits_of(model, r[None].cuda()), r[None].cuda())) for r in data[name]]))

    def evaluate(names):
        with torch.no_grad():
            return {n: float(np.mean([float(ce(logits_of(model, r[None].cuda()), r[None].cuda())) for r in data[n]])) - base_ce[n] for n in names}

    out_path = Path(args.output)
    done = json.loads(out_path.read_text()) if out_path.exists() else dict(runs=[])
    key = lambda budget, seed, cfg: (budget, seed, cfg['lr'], cfg['kl_lambda'])
    runs = {(r['budget'], r['seed'], r['lr'], r['kl_lambda']): r for r in done['runs']}
    started = time.perf_counter()

    def train(budget, seed, cfg):
        if key(budget, seed, cfg) in runs: return runs[key(budget, seed, cfg)]
        de = german_rows(data, seed, budget)
        editor.reset()
        opt = torch.optim.Adam(editor.parameters(), lr=cfg['lr'])
        ema, best, since, t0 = None, float('inf'), 0, time.perf_counter()
        for step in range(args.max_steps):
            j = step % len(base_logp_en)
            en = data['en_preserve'][j][None].cuda()
            de_ce = ce(logits_of(model, de), de)
            logp = F.log_softmax(logits_of(model, en), -1)
            kl = (base_logp_en[j].exp() * (base_logp_en[j] - logp)).sum(-1).mean()
            loss = F.relu(ceiling - de_ce) + cfg['kl_lambda'] * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            value = loss.item()
            if not math.isfinite(value): break
            ema = value if ema is None else 0.99 * ema + 0.01 * value
            if ema < best - 1e-4: best, since = ema, 0
            else: since += 1
            if since >= args.patience: break
        run = dict(budget=budget, seed=seed, **cfg, steps=step + 1, seconds=round(time.perf_counter() - t0, 1),
                   train_de_ce=de_ce.item(), train_en_kl=kl.item(), finite=math.isfinite(value), **editor.state())
        run['dev'] = evaluate(('de_dev', 'pile_dev'))
        run['eval'] = evaluate(tuple(f'{s}_eval' for s in EVAL_SETS))
        runs[key(budget, seed, cfg)] = run; done['runs'].append(run)
        log(**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in run.items() if k not in ('dev', 'eval')},
            dev_de=round(run['dev']['de_dev'], 3), dev_pile=round(run['dev']['pile_dev'], 4))
        return run

    for budget in args.budgets:
        # Stage 1: every config on the first select_seeds seeds; the post's rule on the median over those seeds.
        trial = {key(budget, 0, cfg): [train(budget, seed, cfg) for seed in range(args.select_seeds)] for cfg in configs}
        chosen, eligible = choose(configs, [trial[key(budget, 0, cfg)] for cfg in configs], args.max_dev_en_damage)
        # Stage 2: the remaining seeds at the chosen config only.
        final = [train(budget, seed, chosen) for seed in range(args.seeds)]
        done.setdefault('selected', {})[str(budget)] = dict(
            config=chosen, eligible=eligible,
            seeds=[dict(seed=r['seed'], dev=r['dev'], eval=r['eval'], steps=r['steps'],
                        **{k: r[k] for k in ('alpha', 'b_norm') if k in r}) for r in final])
        done.update(arm=args.arm, method=args.method, checkpoint=str(args.checkpoint) if args.checkpoint else None, **meta,
                    ceiling=ceiling, base_ce=base_ce, configs=configs, max_dev_en_damage=args.max_dev_en_damage,
                    protocol=dict(budgets=list(args.budgets), seeds=args.seeds, select_seeds=args.select_seeds, max_steps=args.max_steps,
                                  patience=args.patience, ema=0.99, english_preserve_blocks=len(base_logp_en), data=str(args.data),
                                  data_sha256=sha(args.data)))
        save_json(done, out_path)
        log(budget=budget, chosen=chosen, eligible=eligible,
            eval_de=[round(r['eval']['de_eval'], 2) for r in final], eval_pile=[round(r['eval']['pile_eval'], 3) for r in final])
    log(done=str(out_path), runs=len(done['runs']), minutes=round((time.perf_counter() - started) / 60, 1))


def choose(configs, trials, threshold):
    """The post's rule per budget: among configs whose median dev English (Pile) damage over the trial seeds is below the
    threshold, the one with the largest median dev German damage. If none qualifies, the least English damage (flagged)."""
    med = lambda rs, k: float(np.median([r['dev'][k] if r['finite'] else float('nan') for r in rs]))
    scored = [(cfg, med(rs, 'de_dev'), med(rs, 'pile_dev')) for cfg, rs in zip(configs, trials)]
    ok = [x for x in scored if math.isfinite(x[1]) and math.isfinite(x[2]) and x[2] < threshold]
    if ok: return max(ok, key=lambda x: x[1])[0], True
    return min((x for x in scored if math.isfinite(x[2])), key=lambda x: x[2])[0], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['data', 'rank', 'edit'])
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--data', type=Path)
    ap.add_argument('--method', choices=['parauto', 'vpd']); ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--arm', choices=['piece', 'lora']); ap.add_argument('--piece')
    ap.add_argument('--rank', type=int, default=1); ap.add_argument('--lora-alpha', type=float, default=16.)
    ap.add_argument('--budgets', type=int, nargs='+', default=list(BUDGETS)); ap.add_argument('--seeds', type=int, default=SEEDS)
    ap.add_argument('--select-seeds', type=int, default=3)
    ap.add_argument('--lrs', type=float, nargs='+'); ap.add_argument('--lambdas', type=float, nargs='+', default=[1., 10., 100., 1000.])
    ap.add_argument('--max-steps', type=int, default=3000); ap.add_argument('--patience', type=int, default=300)
    ap.add_argument('--max-dev-en-damage', type=float, default=0.1)
    args = ap.parse_args()
    dict(data=make_data, rank=rank, edit=edit)[args.phase](args)


if __name__ == '__main__': main()
