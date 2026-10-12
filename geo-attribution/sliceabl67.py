"""Individual parameters as the basis: two measurements on the 67M target.

A. `entry`  Is first-order attribution accurate for single-parameter ablation?
            For a few tokens, zero individual weight entries one at a time
            (top by |attribution| and random) and compare the measured change
            in the gold log-prob with the first-order estimate.  Then remove
            the top-k entries of one unit's row together, k = 1 ... whole row,
            and compare the measured change with the sum of attributions: the
            set size at which linearity breaks.
B. `slice`  Are per-entry cluster slices ablatable?  From the per-entry bank
            (each entry's 8 strongest clusters by attribution mass), a slice
            is the set of a unit's entries that one cluster owns.  Zero a
            slice, measure KL on the cluster's own tokens vs all others, and
            compare with the whole unit and with the unit minus the slice.

    STREAM67_RUN=... python -u sliceabl67.py entry|slice ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stream67 as S
from stream67 import MODULES, SENSORS
import unitabl67 as U


def log(m):
    print(f"[sliceabl] {m}", flush=True)


def spearman(x, y):
    return U.spearman(x, y)


# ------------------------------------------------------------------ A -----

def phase_entry(args, dev):
    cfg = SENSORS[args.sensor]
    net = U.Net(dev, cfg["mode"])
    lab = torch.load(U.lab_path(args), map_location="cpu", weights_only=False)
    ids, labels = lab["ids"].long(), lab["labels"]
    N, T = ids.shape
    g = torch.Generator().manual_seed(args.seed + 7)
    cand = ((labels >= 0) & (torch.arange(T)[None] >= 32)).nonzero()
    pick = cand[torch.randperm(cand.shape[0], generator=g)[:args.n_tokens]].tolist()
    units = U.build_units()
    mlp_units = [u for u in units if u[0] == "mlp"]
    flat_p = []                      # (module, n_entries) in a fixed order
    for p in MODULES:
        flat_p.append((p, net.W0[p].numel()))
    n_ent = sum(n for _, n in flat_p)
    offs = np.cumsum([0] + [n for _, n in flat_p])

    def locate(j):
        i = int(np.searchsorted(offs, j, side="right") - 1)
        p, n = flat_p[i]
        e = j - offs[i]
        d_in = net.W0[p].shape[1]
        return p, e // d_in, e % d_in

    @torch.no_grad()
    def gold_lp(b, t):
        lp = F.log_softmax(net.model(ids[b:b + 1].to(dev))[0, t].float(), -1)
        return lp

    per_tok, sets = [], []
    t0 = time.perf_counter()
    for j, (b, t) in enumerate(pick):
        AW = U.token_attr(net, cfg, ids[b].to(dev), t)
        a = torch.cat([AW[p].reshape(-1) for p in MODULES]).float()     # first-order dlogp of removal = -a
        base = gold_lp(b, t)
        gold = int(ids[b, t + 1])
        base_g = float(base[gold])
        # entries: top by |a|, random among the top 1%, uniform random
        order = a.abs().argsort(descending=True)
        top = order[:args.n_top]
        top1 = order[:max(args.n_top, n_ent // 100)]
        r1 = top1[torch.randperm(top1.numel(), generator=g)[:args.n_top]]
        r2 = torch.randperm(n_ent, generator=g)[:2 * args.n_top].to(dev)
        sel = torch.cat([top, r1, r2]).unique()
        meas, pred, kls = [], [], []
        with torch.no_grad():
            for jj in sel.tolist():
                p, o, i = locate(jj)
                w = net.Wt[p][o, i].item()
                net.Wt[p][o, i] = 0.0
                lp = gold_lp(b, t)
                net.Wt[p][o, i] = w
                meas.append(float(lp[gold]) - base_g)
                pred.append(-float(a[jj]))
                kls.append(U.kl_of(lp, base))
        meas_t, pred_t = torch.tensor(meas), torch.tensor(pred)
        nz = (pred_t != 0) & (meas_t != 0)
        row = {"token": [b, t], "n_entries": len(sel),
               "spearman_abs": spearman(pred_t.abs(), meas_t.abs()),
               "pearson": float(((meas_t - meas_t.mean()) * (pred_t - pred_t.mean())).mean()
                                / (meas_t.std() * pred_t.std()).clamp_min(1e-12)),
               "sign_agreement": float((torch.sign(meas_t[nz]) == torch.sign(pred_t[nz])).float().mean()),
               "top_spearman_abs": spearman(pred_t[:args.n_top].abs(), meas_t[:args.n_top].abs()),
               "median_ratio_top": float((meas_t[:args.n_top] / pred_t[:args.n_top].clamp(min=1e-12) if False else
                                          (meas_t[:args.n_top] / torch.where(pred_t[:args.n_top].abs() > 1e-12, pred_t[:args.n_top], torch.full_like(pred_t[:args.n_top], 1e-12)))).median()),
               "max_abs_dlogp_single_entry": float(meas_t.abs().max()),
               "max_kl_single_entry": float(max(kls)),
               "pairs": [(round(pv, 6), round(mv, 6)) for pv, mv in zip(pred[:args.n_top], meas[:args.n_top])][:128]}
        per_tok.append(row)
        # set removal within one unit's row: does the sum of attributions predict the joint effect?
        rows_a = {}
        for u in [mlp_units[i] for i in torch.randperm(len(mlp_units), generator=g)[:args.n_units].tolist()]:
            p, _, lo, hi = u[3][0]                       # c_fc row
            a_row = AW[p][lo].float()                    # [d_in]
            ordr = a_row.abs().argsort(descending=True)
            entry = {"unit": [u[0], u[1], u[2]], "sizes": [], "measured": [], "predicted": []}
            with torch.no_grad():
                for k in [1, 4, 16, 64, 256, a_row.numel()]:
                    idx = ordr[:k]
                    saved = net.Wt[p][lo, idx].clone()
                    net.Wt[p][lo, idx] = 0.0
                    lp = gold_lp(b, t)
                    net.Wt[p][lo, idx] = saved
                    entry["sizes"].append(int(k))
                    entry["measured"].append(round(float(lp[gold]) - base_g, 6))
                    entry["predicted"].append(round(-float(a_row[idx].sum()), 6))
            sets.append(entry)
        net.restore()
        log(f"token {j + 1}/{len(pick)}: {len(sel)} entries, Spearman(|pred|,|meas|) {row['spearman_abs']:.2f}, "
            f"sign {row['sign_agreement']:.2f}, max |dlogp| single entry {row['max_abs_dlogp_single_entry']:.4f} "
            f"({time.perf_counter() - t0:.0f}s)")
    # summary of the set test: ratio measured/predicted by size
    by_size = {}
    for e in sets:
        for k, m, pr in zip(e["sizes"], e["measured"], e["predicted"]):
            by_size.setdefault(k, []).append((m, pr))
    set_summary = {str(k): {"n": len(v),
                            "median_abs_ratio": float(np.median([abs(m) / max(abs(pr), 1e-9) for m, pr in v])),
                            "sign_agreement": float(np.mean([np.sign(m) == np.sign(pr) for m, pr in v if m != 0 and pr != 0])) if any(m != 0 and pr != 0 for m, pr in v) else None,
                            "spearman": spearman(torch.tensor([abs(pr) for _, pr in v]), torch.tensor([abs(m) for m, _ in v])) if len(v) > 2 else None}
                   for k, v in sorted(by_size.items())}
    out = {"format": "sliceabl_entry_v1", "n_tokens": len(pick), "n_entries_total": n_ent,
           "per_token": per_tok,
           "mean": {k: float(np.mean([r[k] for r in per_tok])) for k in
                    ("spearman_abs", "pearson", "sign_agreement", "top_spearman_abs", "max_abs_dlogp_single_entry", "max_kl_single_entry")},
           "set_removal": set_summary, "set_rows": sets}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}: " + ", ".join(f"{k} {v:.3f}" for k, v in out["mean"].items()))
    log("set removal (measured/predicted by size): " + ", ".join(f"k={k}: ratio {v['median_abs_ratio']:.2f} sign {v['sign_agreement']}" for k, v in set_summary.items()))


# ------------------------------------------------------------------ B -----

def phase_slice(args, dev):
    cfg = SENSORS[args.sensor]
    net = U.Net(dev, cfg["mode"])
    U.resolve_device_defaults(args, dev)
    if args.precision == "fp16":
        net.half()
    lab = torch.load(U.lab_path(args), map_location="cpu", weights_only=False)
    ids = lab["ids"].long()[:args.sel_seqs].to(dev)
    labels = lab["labels"][:args.sel_seqs].to(dev)
    N, T = ids.shape
    bk = torch.load(S.RUN / f"bank_C{args.C}_{args.sensor}{args.bank_tag}.pt",
                    map_location="cpu", weights_only=False)
    C = int(bk["C"])
    sidx = {p: bk["sidx"][p].to(dev) for p in MODULES}          # int16 [S, d_out, d_in]
    swgt = {p: bk["swgt"][p].to(dev).float() for p in MODULES}
    log(f"bank {bk.get('own_mode')} S={sidx[MODULES[0]].shape[0]}; corpus {N}x{T}")
    units = U.build_units()
    look = U.unit_lookup(units, "head", net.W0, dev)
    base_lp = torch.empty(N, T, net.model.cfg["vocab_size"], device=dev)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            base_lp[c0:c0 + args.chunk] = F.log_softmax(net.model(ids[c0:c0 + args.chunk]).float(), -1)

    @torch.no_grad()
    def kl_map():
        out = torch.empty(N, T, device=dev)
        for c0 in range(0, N, args.chunk):
            lp = F.log_softmax(net.model(ids[c0:c0 + args.chunk]).float(), -1)
            b = base_lp[c0:c0 + args.chunk]
            out[c0:c0 + args.chunk] = (b.exp() * (b - lp)).sum(-1)
            del lp
        return out

    def unit_owner_mass(u):
        """[C] summed share of the unit's entries per cluster, and entry counts"""
        mass = torch.zeros(C, device=dev)
        cnt = torch.zeros(C, device=dev)
        for p, axis, lo, hi in u[3]:
            si = sidx[p][:, lo:hi, :] if axis == 0 else sidx[p][:, :, lo:hi]
            sw = swgt[p][:, lo:hi, :] if axis == 0 else swgt[p][:, :, lo:hi]
            mass.index_add_(0, si.reshape(-1).long(), sw.reshape(-1))
            cnt.index_add_(0, si.reshape(-1).long(), (sw.reshape(-1) > 0).float())
        return mass, cnt

    @torch.no_grad()
    def apply_slice(u, c, keep_slice):
        """zero the unit's entries owned by c (keep_slice=False) or the rest of the unit (True)"""
        net.restore()
        for p, axis, lo, hi in u[3]:
            si = sidx[p][:, lo:hi, :] if axis == 0 else sidx[p][:, :, lo:hi]
            sw = swgt[p][:, lo:hi, :] if axis == 0 else swgt[p][:, :, lo:hi]
            owned = ((si == c) & (sw > 0)).any(0)
            m = owned if keep_slice else ~owned
            if axis == 0:
                net.Wt[p][lo:hi, :] *= m.to(net.Wt[p].dtype)
            else:
                net.Wt[p][:, lo:hi] *= m.to(net.Wt[p].dtype)

    g = torch.Generator().manual_seed(args.seed + 11)
    mlp = [i for i, u in enumerate(units) if u[0] == "mlp"]
    heads = [i for i, u in enumerate(units) if u[0] == "head"]
    chosen_units = [mlp[i] for i in torch.randperm(len(mlp), generator=g)[:args.n_units].tolist()] + heads
    valid = labels >= 0
    lab_flat = labels
    rows = []
    t0 = time.perf_counter()
    for n_u, ui in enumerate(chosen_units):
        u = units[ui]
        mass, cnt = unit_owner_mass(u)
        n_ent_unit = float(cnt.sum() / max(1, sidx[MODULES[0]].shape[0]))  # approx entries per owner slot
        n_entries_unit = sum((hi - lo) * (net.W0[p].shape[1] if axis == 0 else net.W0[p].shape[0]) for p, axis, lo, hi in u[3])
        top = mass.argsort(descending=True)[:args.top_clusters].tolist()
        elig = (cnt >= args.min_entries).nonzero().flatten()
        rnd = elig[torch.randperm(elig.numel(), generator=torch.Generator().manual_seed(args.seed + ui))[:args.rand_clusters].tolist()].tolist() if elig.numel() else []
        # whole unit
        keep = torch.ones(len(units), dtype=torch.bool, device=dev)
        keep[ui] = False
        net.apply_unit_mask(keep, look)
        Ku = kl_map()
        for c in list(dict.fromkeys(top + rnd)):
            own = lab_flat == c
            other = valid & ~own
            if own.sum() < args.min_own_tokens:
                continue
            apply_slice(u, c, keep_slice=False)
            Ks = kl_map()
            apply_slice(u, c, keep_slice=True)
            Kr = kl_map()
            rows.append({"unit": [u[0], u[1], u[2]], "cluster": int(c), "rank_in_unit": top.index(c) if c in top else None,
                         "n_entries": int(cnt[c]), "frac_of_unit": float(cnt[c] / n_entries_unit),
                         "share_mass": float(mass[c]), "own_tokens": int(own.sum()),
                         "kl_own_slice": float(Ks[own].mean()), "kl_other_slice": float(Ks[other].mean()),
                         "kl_own_rest": float(Kr[own].mean()), "kl_other_rest": float(Kr[other].mean()),
                         "kl_own_unit": float(Ku[own].mean()), "kl_other_unit": float(Ku[other].mean())})
        net.restore()
        if (n_u + 1) % 8 == 0:
            log(f"{n_u + 1}/{len(chosen_units)} units, {len(rows)} slices ({time.perf_counter() - t0:.0f}s)")
    net.restore()
    r = rows
    def med(key):
        return float(np.median([x[key] for x in r])) if r else None
    summ = {"n_slices": len(r), "n_units": len(chosen_units),
            "median_frac_of_unit": med("frac_of_unit"),
            "selectivity_slice": float(np.median([x["kl_own_slice"] / max(x["kl_other_slice"], 1e-9) for x in r])) if r else None,
            "selectivity_unit": float(np.median([x["kl_own_unit"] / max(x["kl_other_unit"], 1e-9) for x in r])) if r else None,
            "slice_share_of_unit_own_effect": float(np.median([x["kl_own_slice"] / max(x["kl_own_unit"], 1e-9) for x in r])) if r else None,
            "additivity_own": float(np.median([(x["kl_own_slice"] + x["kl_own_rest"]) / max(x["kl_own_unit"], 1e-9) for x in r])) if r else None,
            "median_kl_own_slice": med("kl_own_slice"), "median_kl_other_slice": med("kl_other_slice"),
            "median_kl_own_unit": med("kl_own_unit")}
    top_rows = [x for x in r if x["rank_in_unit"] == 0]
    if top_rows:
        summ["top_slice_selectivity"] = float(np.median([x["kl_own_slice"] / max(x["kl_other_slice"], 1e-9) for x in top_rows]))
        summ["top_slice_frac_of_unit"] = float(np.median([x["frac_of_unit"] for x in top_rows]))
    out = {"format": "sliceabl_slice_v1", "bank": bk.get("own_mode"), "C": C, "n_seqs": N,
           "summary": summ, "rows": r}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summ.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["entry", "slice"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--sensor", default="ig5")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--lab_tag", default="")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/sliceabl.json")
    ap.add_argument("--n_tokens", type=int, default=8)
    ap.add_argument("--n_top", type=int, default=256)
    ap.add_argument("--n_units", type=int, default=16)
    ap.add_argument("--bank_tag", default="_entry")
    ap.add_argument("--sel_seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16", "auto"], default="auto")
    ap.add_argument("--top_clusters", type=int, default=4)
    ap.add_argument("--rand_clusters", type=int, default=2)
    ap.add_argument("--min_entries", type=int, default=8)
    ap.add_argument("--min_own_tokens", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    {"entry": phase_entry, "slice": phase_slice}[args.phase](args, dev)


if __name__ == "__main__":
    main()
