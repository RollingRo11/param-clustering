"""German-erasure gate edit on the geo-attribution 67M partition decomposition.

Replicates the trained-decomposition German-erasure probe (gate space): rank
components by German-vs-English attribution-share contrast on a RANKING split,
zero the top-k in the gated forward, and measure per-token CE deltas on a
held-out EVAL split of both languages, with random-k controls.

  python3.12 german67.py --banks_tag part8 --gate_thresh 0.02
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")
sys.path.insert(0, "/workspace/param-decomp")

import torch
import torch.nn.functional as F

from geo67 import OUT_ROOT, GatedRunner, load_target, log

GERMAN = [
    "Die Bundesregierung hat heute neue Maßnahmen zur Bekämpfung der Inflation angekündigt, die vor allem kleine und mittlere Unternehmen entlasten sollen.",
    "Der Wetterbericht sagt für das Wochenende starken Regen und Gewitter voraus, besonders im Süden des Landes.",
    "In der Altstadt von Heidelberg gibt es viele kleine Cafés, in denen man gemütlich sitzen und den Blick auf das Schloss genießen kann.",
    "Die Forscher der Universität München haben eine neue Methode entwickelt, um die Ausbreitung von Viren in Innenräumen zu messen.",
    "Mein Bruder arbeitet seit drei Jahren als Ingenieur bei einem großen Automobilhersteller in Stuttgart.",
    "Die deutsche Nationalmannschaft hat das Spiel gegen Frankreich mit zwei zu eins gewonnen, obwohl sie in der ersten Halbzeit schlecht spielte.",
    "Am Montag beginnt die Schule wieder, und die Kinder müssen früh aufstehen, um pünktlich zum Unterricht zu kommen.",
    "Der Roman erzählt die Geschichte einer Familie, die während des Krieges aus ihrer Heimat fliehen musste.",
    "Die Preise für Lebensmittel sind im vergangenen Jahr deutlich gestiegen, was viele Haushalte vor Probleme stellt.",
    "Im Herbst färben sich die Blätter der Bäume rot und gelb, und die Tage werden merklich kürzer.",
    "Die neue U-Bahn-Linie verbindet den Hauptbahnhof mit dem Flughafen und verkürzt die Fahrzeit erheblich.",
    "Viele junge Menschen ziehen in die großen Städte, weil sie dort bessere Arbeitsmöglichkeiten finden.",
    "Das Museum zeigt eine Ausstellung über die Geschichte der Industrialisierung im Ruhrgebiet.",
    "Die Ärztin erklärte dem Patienten, dass er sich mehr bewegen und gesünder essen sollte.",
    "Wegen des Streiks der Lokführer fallen heute zahlreiche Züge im Fernverkehr aus.",
    "Die Katze meiner Nachbarin sitzt jeden Morgen auf der Fensterbank und beobachtet die Vögel im Garten.",
    "Der Bundestag debattierte gestern über die geplante Reform des Gesundheitssystems.",
    "Im Sommer fahren wir gerne an die Ostsee, um dort zu schwimmen und am Strand zu liegen.",
    "Die Firma hat angekündigt, dass sie im nächsten Jahr hundert neue Arbeitsplätze schaffen wird.",
    "Nach dem Abitur möchte meine Tochter Medizin studieren, am liebsten in Berlin oder Hamburg.",
    "Der Handwerker konnte den Wasserschaden in der Küche erst nach mehreren Stunden reparieren.",
    "Die Polizei sucht Zeugen für einen Unfall, der sich am Freitagabend auf der Autobahn ereignete.",
    "In den bayerischen Alpen kann man auch im Frühjahr noch gut Ski fahren, wenn genug Schnee liegt.",
    "Das Orchester spielte Werke von Beethoven und Brahms vor ausverkauftem Haus.",
]
ENGLISH = [
    "The federal government announced new measures today to combat inflation, aimed primarily at relieving small and medium-sized businesses.",
    "The weather forecast predicts heavy rain and thunderstorms for the weekend, especially in the south of the country.",
    "In the old town there are many small cafés where you can sit comfortably and enjoy the view of the castle.",
    "Researchers at the university have developed a new method to measure the spread of viruses in indoor spaces.",
    "My brother has been working as an engineer at a large car manufacturer for three years.",
    "The national team won the match against France two to one, although they played poorly in the first half.",
    "School starts again on Monday, and the children have to get up early to be on time for class.",
    "The novel tells the story of a family that had to flee their homeland during the war.",
    "Food prices have risen significantly over the past year, which is causing problems for many households.",
    "In autumn the leaves of the trees turn red and yellow, and the days become noticeably shorter.",
    "The new subway line connects the main station with the airport and shortens the journey considerably.",
    "Many young people move to the big cities because they find better job opportunities there.",
    "The museum is showing an exhibition on the history of industrialization in the region.",
    "The doctor explained to the patient that he should exercise more and eat healthier.",
    "Because of the train drivers' strike, numerous long-distance trains are cancelled today.",
    "My neighbor's cat sits on the windowsill every morning and watches the birds in the garden.",
    "Parliament debated the planned reform of the health care system yesterday.",
    "In summer we like to go to the coast to swim and lie on the beach.",
    "The company has announced that it will create a hundred new jobs next year.",
    "After graduation my daughter wants to study medicine, preferably in a big city.",
    "The repairman was only able to fix the water damage in the kitchen after several hours.",
    "The police are looking for witnesses to an accident that occurred on the highway on Friday evening.",
    "In the mountains you can still ski well in spring if there is enough snow.",
    "The orchestra played works by Beethoven and Brahms to a sold-out house.",
]


def chunks_from(sents, tok, seq_len, device):
    ids = []
    for s in sents:
        ids.extend(tok.encode(s).ids)
        ids.extend(tok.encode("\n\n").ids)
    n = len(ids) // seq_len
    return torch.tensor(ids[: n * seq_len]).view(n, seq_len).to(device)


def ce_per_tok(logits, idx):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                           idx[:, 1:].flatten()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--banks_tag", default="part8")
    ap.add_argument("--gate_thresh", type=float, default=0.02)
    ap.add_argument("--ig_k", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=192)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--space", choices=["gate", "weight"], default="gate")
    args = ap.parse_args()
    d = OUT_ROOT / args.tag

    device = "cuda"
    target = load_target(device)
    bk = torch.load(d / f"banks_{args.banks_tag}.pt", weights_only=True,
                    map_location="cpu")
    run = GatedRunner(target, bk, device)
    C = bk["C"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(OUT_ROOT / "target_local" / "tokenizer.json"))

    half = len(GERMAN) // 2
    rank_de = chunks_from(GERMAN[:half], tok, args.seq_len, device)
    rank_en = chunks_from(ENGLISH[:half], tok, args.seq_len, device)
    eval_de = chunks_from(GERMAN[half:], tok, args.seq_len, device)
    eval_en = chunks_from(ENGLISH[half:], tok, args.seq_len, device)
    log(f"chunks: rank de/en {rank_de.shape[0]}/{rank_en.shape[0]}, "
        f"eval de/en {eval_de.shape[0]}/{eval_en.shape[0]}")

    def mean_share(idx):
        attr, _ = run.attribution(idx, args.ig_k)
        share = attr / attr.sum(-1, keepdim=True).clamp_min(1e-30)
        return share[:, 2:-2].mean((0, 1))

    contrast = mean_share(rank_de) - mean_share(rank_en)
    order = contrast.argsort(descending=True)
    log("top-8 german-contrast comps: " + str(order[:8].tolist()))

    def gated_ce(idx, comps=None, alpha=0.0):
        attr, lt = run.attribution(idx, args.ig_k)
        amax = attr.amax(-1, keepdim=True).clamp_min(1e-30)
        gates = (attr / amax > args.gate_thresh).float()
        if comps is not None:
            gates[..., comps] *= alpha       # 0 = ablate, <0 = invert (VPD-style)
        with torch.no_grad():
            lg = run.gated_pass(idx, gates)
        return ce_per_tok(lg, idx), ce_per_tok(lt, idx)

    def weight_ce(idx, comps=None, alpha=0.0):
        """Permanent weight surgery: scale the selected components' OWNED SHARE
        of each entry by alpha (share is 0/1 for the hard partition, fractional
        for softpart), run the PLAIN target (no gates), restore."""
        saved = {}
        if comps is not None:
            comps_t = comps if torch.is_tensor(comps) else torch.tensor(
                comps, device=device)
            for p in bk["modules"]:
                lin = target.get_submodule(p)
                share = run.component_share(p, comps_t)
                saved[p] = lin.weight.data.clone()
                lin.weight.data *= (1.0 - (1.0 - alpha) * share)
        with torch.no_grad():
            lt, _ = run.target_pass(idx)
        for p, w in saved.items():
            target.get_submodule(p).weight.data.copy_(w)
        return ce_per_tok(lt, idx)

    base_de, tgt_de = gated_ce(eval_de)
    base_en, tgt_en = gated_ce(eval_en)
    res = {"target_ce": {"de": tgt_de, "en": tgt_en},
           "gated_base_ce": {"de": base_de, "en": base_en},
           "contrast_top16": order[:16].tolist(), "edits": {}}
    log(f"target CE de/en {tgt_de:.3f}/{tgt_en:.3f}; "
        f"gated base de/en {base_de:.3f}/{base_en:.3f}")
    if args.space == "weight":
        base_de, base_en = weight_ce(eval_de), weight_ce(eval_en)
        res["weight_base_ce"] = {"de": base_de, "en": base_en}
        log(f"weight-space mode: unedited target CE de/en {base_de:.3f}/{base_en:.3f}")

    def edit_ce(idx, comps, alpha):
        return (weight_ce(idx, comps, alpha) if args.space == "weight"
                else gated_ce(idx, comps, alpha)[0])

    for k in [4, 8, 16, 32]:
        for alpha in args.alphas:
            ce_de = edit_ce(eval_de, order[:k], alpha)
            ce_en = edit_ce(eval_en, order[:k], alpha)
            rnd_de, rnd_en = [], []
            for s in range(3):
                g = torch.Generator().manual_seed(s)
                rnd = torch.randperm(C, generator=g)[:k].to(device)
                rnd_de.append(edit_ce(eval_de, rnd, alpha))
                rnd_en.append(edit_ce(eval_en, rnd, alpha))
            res["edits"][f"k{k}_a{alpha}"] = {
                "dce_de": ce_de - base_de, "dce_en": ce_en - base_en,
                "rnd_dce_de": sum(rnd_de) / 3 - base_de,
                "rnd_dce_en": sum(rnd_en) / 3 - base_en}
            r = res["edits"][f"k{k}_a{alpha}"]
            log(f"k={k} alpha={alpha}: dCE de {r['dce_de']:+.3f} "
                f"en {r['dce_en']:+.3f} | random de {r['rnd_dce_de']:+.3f} "
                f"en {r['rnd_dce_en']:+.3f}")
    suf = "_weight" if args.space == "weight" else ""
    (d / f"german_{args.banks_tag}{suf}.json").write_text(json.dumps(res, indent=1))
    log("GERMAN " + json.dumps(res["edits"], indent=1))


if __name__ == "__main__":
    main()
