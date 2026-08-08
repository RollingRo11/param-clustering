"""VPD Section-6 emoticon redirect on the C=4096 streamed decomposition.

Find components whose attribution fingerprints fire when PREDICTING emoticon
tokens, then rewrite their write-side action toward one target emoticon's
unembedding (norm-matched rank-1, read side untouched) — the Llama analog of
VPD's "rewrite U toward the ':o' unembedding" edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import geo67
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args
from german67 import ENGLISH

# Sentence banks from emote1b.py (that file is a script; importing it runs
# the legacy pipeline, so the lists are inlined here).
EMO_RANK = [
    "Thanks so much for your help! :)",
    "See you tomorrow :)",
    "That was such a fun party :D",
    "I can't believe we won :D",
    "Sorry to hear that :(",
    "I missed the bus again :(",
    "Good luck on your exam :)",
    "This pizza is amazing :D",
    "My cat knocked over the plant :(",
    "Happy birthday!! :D",
    "No worries at all :)",
    "That movie made me cry :(",
    "Great job on the presentation :)",
    "I got the job! :D",
    "It's raining on my day off :(",
    "Welcome to the team :)",
    "You're the best :P",
    "Nice to meet you :)",
    "Can't wait for the weekend :D",
    "My phone battery died again :(",
]
EMO_EVAL = [
    "Thanks for coming to my show :)",
    "We finally finished the project :D",
    "The bakery was out of croissants :(",
    "Have a safe flight :)",
    "I aced the interview :D",
    "My favorite mug broke this morning :(",
    "Say hi to your family for me :)",
    "Snow day, no school :D",
    "The concert got cancelled :(",
    "Enjoy your vacation :)",
    "Our team won the hackathon :D",
    "I left my umbrella on the train :(",
]

TARGETS = [" :o", ":o", " :O", ":O", "o"]
EMO_STRINGS = [" :)", ":)", " :(", ":(", " :D", ":D", " :P", ":P",
               " :o", ":o", " :O", ":O"]


def chunk(sentences, tokenizer, seq_len, device):
    ids = []
    for sentence in sentences:
        ids += tokenizer.encode(sentence, add_special_tokens=False)
        ids += tokenizer.encode("\n\n", add_special_tokens=False)
    count = len(ids) // seq_len
    if count < 1:
        raise ValueError(f"corpus too small for seq_len={seq_len}")
    return torch.tensor(ids[:count * seq_len],
                        dtype=torch.long).view(count, seq_len).to(device)


def single_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--rank_temperature", type=float, default=0.05)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument("--gains", type=float, nargs="+",
                        default=[1.0, 2.0, 4.0])
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.artifact_root / args.tag
    device = "cuda"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    emo_ids = sorted({tid for s in EMO_STRINGS
                      if (tid := single_token(tokenizer, s)) is not None})
    target_id = next(tid for t in TARGETS
                     if (tid := single_token(tokenizer, t)) is not None)
    target_str = tokenizer.decode([target_id])
    log(f"emoticon token ids {emo_ids}; target {target_str!r} = {target_id}")
    emo_t = torch.tensor(emo_ids, device=device)

    rank_emo = chunk(EMO_RANK, tokenizer, args.seq_len, device)
    eval_emo = chunk(EMO_EVAL, tokenizer, args.seq_len, device)
    rank_ctrl = chunk(ENGLISH[:12], tokenizer, args.seq_len, device)
    eval_ctrl = chunk(ENGLISH[12:], tokenizer, args.seq_len, device)

    def predicting_positions(idx):
        mask = torch.zeros_like(idx, dtype=torch.bool)
        mask[:, :-1] = torch.isin(idx[:, 1:], emo_t)
        mask[:, :4] = False
        return mask

    # ---- rank components on the frozen fingerprint posterior ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    meta = {key: bank[key] for key in
            ("format", "C", "modules", "sensor", "gim_tau", "scalar")
            if key in bank}
    cfg = ranking_args(meta)
    cap = setup_model(cfg, device)
    spec, scales, dim = load_spec(run_dir, device)
    stream_model = load_stream_model(run_dir / "stream_model.pt", device)

    def posterior_for(idx, mask):
        batch_i, pos_i = mask.nonzero(as_tuple=True)
        phi, _ = pass_features(cfg, cap, idx, pos_i[None], batch_i[None],
                               spec, scales, dim, return_pg=False)
        x = phi.clamp(-6e4, 6e4).half().float()
        y = F.normalize((x - stream_model["mean"]) @
                        stream_model["projector"], dim=1)
        sims = y @ stream_model["centroids"].t()
        return torch.softmax(sims / args.rank_temperature, dim=1).mean(0)

    emo_mask = predicting_positions(rank_emo)
    log(f"rank: {int(emo_mask.sum())} emoticon-predicting positions")
    post_emo = posterior_for(rank_emo, emo_mask)
    ctrl_mask = torch.zeros_like(rank_ctrl, dtype=torch.bool)
    ctrl_mask[:, 4:-2] = True
    post_ctrl = posterior_for(rank_ctrl, ctrl_mask)
    contrast = post_emo - post_ctrl
    order = contrast.argsort(descending=True)[:args.candidate_k]
    candidates = order.tolist()
    log(f"top emoticon-contrast components: {candidates} "
        f"(activity {[round(float(post_emo[c]), 4) for c in candidates]})")
    del cap, stream_model, spec, scales
    torch.cuda.empty_cache()

    # ---- redirect edit on the plain target ----
    target = geo1b.load_target_1b(device)
    unembed = target.get_submodule("hf.model.embed_tokens").weight
    direction = (unembed[target_id].float()
                 / unembed[target_id].float().norm()).to(device)

    write_side = [p for p in meta["modules"]
                  if p.endswith(("o_proj", "down_proj"))]
    shares = {}
    for path in write_side:
        sidx = bank["sidx"][path].to(device)
        swgt = bank["swgt"][path].to(device)
        shares[path] = (sidx, swgt)
    log(f"{len(write_side)} write-side matrices")

    def component_share(path, comps):
        sidx, swgt = shares[path]
        return (torch.isin(sidx, comps) * swgt).sum(0).float()

    saved = {}

    def redirect(comps, gain, mode):
        """mode 'replace': swap owned slice for the target write (VPD S6);
        mode 'additive': keep the slice, add the write (clean injection)."""
        comps_t = torch.tensor(comps, device=device)
        for path in write_side:
            linear = target.get_submodule(path)
            if path not in saved:
                saved[path] = linear.weight.detach().clone()
            share = component_share(path, comps_t)
            owned = (share * linear.weight.detach()).float()
            norm = owned.norm()
            if norm < 1e-8:
                continue
            _, _, v = torch.svd_lowrank(owned, q=4)
            new = gain * norm * torch.outer(direction, v[:, 0])
            delta = new - owned if mode == "replace" else new
            linear.weight.data += delta.to(linear.weight.dtype)

    def restore():
        for path, weight in saved.items():
            target.get_submodule(path).weight.data.copy_(weight)
        saved.clear()

    def metrics():
        out = {}
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            logits_e = target(eval_emo)
            logits_c = target(eval_ctrl)
        mask = predicting_positions(eval_emo)
        probs = F.softmax(logits_e[mask].float(), -1)
        out["p_target"] = probs[:, target_id].mean().item()
        out["p_emoset"] = probs[:, emo_t].sum(-1).mean().item()
        out["argmax_target_rate"] = (
            probs.argmax(-1) == target_id).float().mean().item()
        ce = F.cross_entropy(
            logits_c[:, :-1].float().flatten(0, 1),
            eval_ctrl[:, 1:].flatten())
        out["ctrl_ce"] = ce.item()
        return out

    def generate(prompt):
        ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=device)
        for _ in range(args.max_new_tokens):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                logits = target(ids[:, -512:])
            ids = torch.cat(
                [ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokenizer.decode(ids[0].tolist())

    prompts = ["Thanks again for dinner last night!",
               "I just got back from the best vacation ever.",
               "The meeting went really well today. See you tomorrow"]
    base = {"metrics": metrics(), "gens": [generate(p) for p in prompts]}
    log(f"BASE: {json.dumps(base['metrics'])}")
    for line in base["gens"]:
        log("base gen: " + repr(line[-60:]))

    results = {"target": {"id": target_id, "text": target_str},
               "emo_ids": emo_ids, "candidates": candidates,
               "base": base, "arms": {}}
    for k in args.k_values:
        comps = candidates[:k]
        for mode in ("additive", "replace"):
            for gain in args.gains:
                redirect(comps, gain, mode)
                arm = {"metrics": metrics(),
                       "gens": [generate(p) for p in prompts]}
                restore()
                name = f"{mode}_k{k}_g{gain:g}"
                results["arms"][name] = arm
                m = arm["metrics"]
                log(f"{name}: p_target={m['p_target']:.3f} "
                    f"p_emoset={m['p_emoset']:.3f} "
                    f"argmax_target={m['argmax_target_rate']:.2f} "
                    f"ctrl_ce={m['ctrl_ce']:.3f}")
                for line in arm["gens"]:
                    log(f"{name} gen: " + repr(line[-60:]))

    output = run_dir / "emote4096.json"
    output.write_text(json.dumps(results, indent=2))
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
