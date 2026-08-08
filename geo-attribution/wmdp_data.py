"""Protocol data for the hazardous-capability unlearning experiment.

The German experiment had an easy retain set: French is not German. Here the
retain set is the whole point. Hazardous virology and ordinary virology share
almost all of their vocabulary, so a method that removes the first by damaging
the second has not localised anything — it has just broken biology. So the
collateral panel is deliberately adversarial:

  bio_retain   benign biology (cais/wmdp-corpora bio-retain-corpus, PubMed)
  cyber        the neighbouring hazardous domain, to see if the edit is
               bio-specific or just a general "technical text" lesion
  pile         ordinary English
  mmlu         general knowledge, scored as accuracy elsewhere

The forget side is the WMDP-bio benchmark split into DISJOINT question sets:
the edit is fit on train, selected on dev, and reported on eval. Removal that
does not transfer to questions the edit never saw is memorisation, not
unlearning, and the split is what makes the difference visible.

Nothing hazardous is generated: the benchmark is a public research artifact and
every measurement is a cross-entropy or an argmax over four letters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b
from german_vpd_1b import log, blocks

BIO_RETAIN = ("cais/wmdp-corpora", "bio-retain-corpus")
CYBER_RETAIN = ("cais/wmdp-corpora", "cyber-retain-corpus")
LETTERS = ["A", "B", "C", "D"]
SUBJECT = {"wmdp-bio": "biology", "wmdp-chem": "chemistry",
           "wmdp-cyber": "computer security"}


def mcq_prompt(q, choices, subject):
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"The following are multiple choice questions (with answers) "
            f"about {subject}.\n\n{q.strip()}\n{body}\nAnswer:")


MMLU_BIO = ("virology", "college_biology", "high_school_biology",
            "medical_genetics", "anatomy", "college_medicine",
            "professional_medicine", "clinical_knowledge")


def mmlu_bio_blocks(tokenizer, n_blocks, seq_len, seed):
    """Benign biology in the IDENTICAL multiple-choice template.

    Without this control the ranking's top hit is a list-item-marker component:
    the hazardous corpus is A./B./C./D. formatted and the controls were prose,
    so the contrast finds the format instead of the content. MMLU even has a
    'virology' subject, so this cancels topic as well as layout.
    """
    from datasets import load_dataset
    d = load_dataset("cais/mmlu", "all", split="test")
    d = d.filter(lambda r: r["subject"] in MMLU_BIO).shuffle(seed=seed)
    toks: list[int] = []
    need = n_blocks * seq_len
    for r in d:
        p = mcq_prompt(r["question"], r["choices"],
                       r["subject"].replace("_", " "))
        toks.extend(tokenizer.encode(p, add_special_tokens=False))
        if len(toks) >= need + 1:
            break
    return blocks(toks, n_blocks, seq_len)


def corpus_blocks(tokenizer, repo, config, n_blocks, seq_len, skip=0):
    """Concatenate documents until there are enough tokens for n_blocks."""
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train")
    need = n_blocks * seq_len
    toks: list[int] = []
    for i in range(skip, len(ds)):
        toks.extend(tokenizer.encode(ds[i]["text"], add_special_tokens=False))
        if len(toks) >= need + 1:
            break
    if len(toks) < need:
        raise ValueError(f"{config}: only {len(toks)} tokens, need {need}")
    return blocks(toks, n_blocks, seq_len)


def question_split(task, seq_len, seed, n_train, n_dev):
    """Disjoint train / dev / eval question sets for one WMDP task."""
    from datasets import load_dataset
    d = load_dataset("cais/wmdp", task, split="test").shuffle(seed=seed)
    items = [(mcq_prompt(r["question"], r["choices"], SUBJECT[task]),
              int(r["answer"])) for r in d]
    return {"train": items[:n_train],
            "dev": items[n_train:n_train + n_dev],
            "eval": items[n_train + n_dev:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=geo1b.SHM_ROOT / "run1b_streamC4096")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_dev", type=int, default=150)
    ap.add_argument("--retain_blocks", type=int, default=24)
    ap.add_argument("--out", default="wmdp_data.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    S = args.seq_len
    data: dict[str, object] = {
        "format": "wmdp_unlearn_protocol_v1",
        "seq_len": S, "seed": args.seed,
        "sources": {"mcq": "cais/wmdp:{bio,chem,cyber}/test",
                    "bio_retain": "/".join(BIO_RETAIN),
                    "cyber_retain": "/".join(CYBER_RETAIN),
                    "pile": str(geo1b.BIN_PATH)},
    }

    # ---- forget: the benchmark, split three ways ----
    for task in ("wmdp-bio", "wmdp-chem", "wmdp-cyber"):
        sp = question_split(task, S, args.seed, args.n_train, args.n_dev)
        data[f"mcq_{task}"] = sp
        log(f"{task}: {len(sp['train'])} train / {len(sp['dev'])} dev / "
            f"{len(sp['eval'])} eval questions")

    # Hazardous-domain TEXT for the attribution ranking: the train questions
    # concatenated. Ranking must never see dev or eval.
    haz = []
    for p, _ in data["mcq_wmdp-bio"]["train"]:
        haz.extend(tok.encode(p, add_special_tokens=False))
    n_haz = min(args.retain_blocks, len(haz) // S)
    data["bio_hazard_rank"] = blocks(haz, n_haz, S)
    log(f"hazardous-bio ranking text: {n_haz} blocks ({n_haz * S} tokens)")

    # ---- retain: benign biology, the adversarial control ----
    nb = args.retain_blocks
    data["mmlu_bio_rank"] = mmlu_bio_blocks(tok, n_haz, S, args.seed)
    log(f"benign-bio MCQ control (format-matched): {n_haz} blocks")
    bio = corpus_blocks(tok, *BIO_RETAIN, 2 * nb, S)
    data["bio_retain_rank"] = bio[:nb]          # matched to bio_hazard_rank
    data["bio_retain_eval"] = bio[nb:2 * nb]
    log(f"benign-bio retain: {nb} rank + {nb} eval blocks")

    cyb = corpus_blocks(tok, *CYBER_RETAIN, nb, S)
    data["cyber_retain_eval"] = cyb

    pile = np.memmap(geo1b.BIN_PATH, dtype=np.uint32, mode="r")
    need = nb * S
    data["pile_eval"] = torch.from_numpy(
        np.array(pile[-need:], dtype=np.int64, copy=True)).view(nb, S)
    data["pile_rank"] = torch.from_numpy(
        np.array(pile[-2 * need:-need], dtype=np.int64, copy=True)).view(nb, S)

    path = args.run_dir / args.out
    torch.save(data, path)
    log(f"wrote {path}")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else
                          list(v.shape) if torch.is_tensor(v) else str(type(v)))
                      for k, v in data.items() if k != "sources"}, indent=1))


if __name__ == "__main__":
    main()
