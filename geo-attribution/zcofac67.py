"""Signed unit-atom co-factorization of the 67M target (cofac, fixed).

Atoms are the model's own units in unitabl67's "fine" grouping: an MLP
neuron (c_fc row + down_proj column), a head's QK rows, one OV head dim
(v row + o column).  15,384 atoms that sum to the weights exactly and each
have an off state (neuron silent, value coordinate zero, score pair inert).

Attribution of an atom at a position is the first-order effect of removing
it on the differentiated scalar.  Removing a neuron sets the down_proj
input coordinate a_j to 0, removing an OV dim sets the o_proj input
coordinate to 0, so those atoms use grad x activation at that ONE
coordinate (not row + column, which double counts).  A QK atom zeroes the
head's q and k outputs, first order = sum over both.  Optional second-order
correction (arXiv:2606.09899, HVP form): -1/2 a^2 H_aa with the Hessian
diagonal estimated by Hutchinson probes (MLP / OV atoms only).  Sign: Z is
the contribution, so removal changes the scalar by about -Z.

Columns are divided by the atom's RMS over events (no static magnitude
profile), rows scaled to unit L2 norm.  No absolute value, no simplex, no
catch-all.

Fit:  Z ~= U D
  D  [C, J]  exactly one owner per atom -> disjoint supports -> rows are
             orthonormal after scaling, so the sparse code is the plain
             projection Z D^T followed by per-event hard thresholding
  U  [N, C]  signed; per event, coefficients kept above the universal
             threshold (median |coef| / 0.6745 * sqrt(2 ln C)); --k
             overrides with a fixed top-k
Alternate (i) code events given D and (ii) assign every atom to the
component whose usage vector across events best predicts the atom's
column (max squared correlation), coefficient = regression slope.  The
ownership step needs only two sums over events, G = U^T Z and |u_c|^2, so
the fit streams: `stream` fuses sensor and fit over the Pile token bin,
refreshing ownership every chunk of events and never storing Z.

Phases
  collect   per-token events (one scalar per position, batched copies),
            Z stored -- small corpora / validation
  fit       offline fit on stored Z shards
  stream    fused sensor + online fit over the token bin, checkpointed
  report    components with atoms, signs, top events in context
  hvpcheck  first-order vs HVP-corrected attribution against measured
            single-atom ablations

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k \
    python -u zcofac67.py stream --lab_tag _100m --n_seq 208800 --C 2048 1024 4096
    python -u zcofac67.py report --lab_tag _100m --lab_source bin --C 2048
    python -u unitabl67.py eval --sensor gim --lab_tag _100m --variant fine_z2048 ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stream67 as S                      # sets CKPT / pile cache, TF32
import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS
import unitabl67 as U

GROUPING = "fine"
TOP_EVENTS = 16                           # per component per sign, kept by fit
N_LAYER, N_HEAD, D_MODEL, D_MLP = 4, 6, 768, 3072
J_PER_LAYER = D_MLP + N_HEAD + D_MODEL    # mlp, qk, ov (build_units order)


def log(m):
    print(f"[zcofac] {m}", flush=True)


def zdir(args) -> Path:
    d = Path(args.z_dir) if args.z_dir else S.RUN.parent / "zcofac"
    d.mkdir(parents=True, exist_ok=True)
    return d


def z_stem(args) -> str:
    return f"Z{args.lab_tag}_{args.sensor}_{args.scalar}"


def z_path(args, shard=None, nshard=None) -> Path:
    if nshard and nshard > 1:
        return zdir(args) / f"{z_stem(args)}_s{shard}of{nshard}.pt"
    return zdir(args) / f"{z_stem(args)}.pt"


def z_files(args) -> list[Path]:
    fs = sorted(zdir(args).glob(f"{z_stem(args)}_s*of*.pt"))
    single = zdir(args) / f"{z_stem(args)}.pt"
    if single.exists():
        fs = [single] + fs
    assert fs, f"no Z files for {z_stem(args)} in {zdir(args)}"
    return fs


def fit_path(args, C) -> Path:
    return zdir(args) / f"fit{args.lab_tag}_{args.sensor}_{args.scalar}_C{C}.pt"


def make_net(args, dev, mode):
    if args.smoke:
        cfg = {"vocab_size": 50277, "n_layer": 4, "n_head": 6, "n_embd": 768,
               "n_intermediate": 3072, "rotary_base": 10000.0,
               "rms_norm_eps": 1e-6, "n_ctx": 512}
        torch.manual_seed(0)
        model = S67.Model67(cfg, mode).float().to(dev).eval()
        return U.Net(dev, mode, model=model)
    return U.Net(dev, mode)


class Corpus:
    """Sequence accessor: corpus[s] -> int64 [T].  pile: the labelled corpus
    tensor; bin: seq-token windows of the token bin from bin_offset."""

    def __init__(self, args):
        self.T = args.seq
        if args.lab_source == "bin":
            self.arr = np.memmap(args.tokbin, dtype=np.uint16, mode="r")
            self.off = args.bin_offset
            avail = (len(self.arr) - self.off) // self.T
            self.n = args.n_seq if args.n_seq > 0 else avail
            assert self.n <= avail, f"bin holds {avail} sequences past the offset"
            self.ids = None
        elif args.smoke:
            g = torch.Generator().manual_seed(1)
            n = args.n_seq if args.n_seq > 0 else 6
            self.ids = torch.randint(0, 50277, (n, self.T), generator=g)
            self.arr = None
        else:
            lab = torch.load(U.lab_path(args, with_clu=False), map_location="cpu",
                             weights_only=False)
            self.ids = lab["ids"].long()
            if args.n_seq > 0:
                self.ids = self.ids[:args.n_seq]
            self.T = self.ids.shape[1]
            self.arr = None

    def __len__(self):
        return self.n if self.ids is None else self.ids.shape[0]

    def __getitem__(self, s):
        if self.ids is not None:
            return self.ids[s]
        a = self.off + s * self.T
        return torch.from_numpy(np.asarray(self.arr[a:a + self.T], dtype=np.int64))

    def rows(self, seq_ids):
        return torch.stack([self[int(s)] for s in seq_ids])


# ----------------------------------------------------------------- sensor ---

def atom_units():
    """Atom index -> global unit id (unitabl67 fine grouping), and a check
    that the per-layer order is [mlp 0..3071, qk 0..5, ov 0..767]."""
    units = U.build_units()
    sub = U.unit_ids_of(units, GROUPING)
    assert sub.numel() == N_LAYER * J_PER_LAYER
    for l in range(N_LAYER):
        blk = [units[int(g)] for g in sub[l * J_PER_LAYER:(l + 1) * J_PER_LAYER]]
        assert all(u[0] == "mlp" and u[1] == l and u[2] == i for i, u in enumerate(blk[:D_MLP]))
        assert all(u[0] == "qk" and u[1] == l and u[2] == i
                   for i, u in enumerate(blk[D_MLP:D_MLP + N_HEAD]))
        assert all(u[0] == "ov" and u[1] == l and u[2] == i
                   for i, u in enumerate(blk[D_MLP + N_HEAD:]))
    return units, sub


def atom_attr(net, idx, reward_fn, hvp_probes=0, gen=None):
    """Per-position atom attribution A [B, T, J] for the scalar reward_fn(logits).

    Hooked activations per layer: down_proj input (neuron outputs), o_proj
    input (head outputs), q_proj and k_proj outputs.  First order = grad x
    activation at the coordinate(s) the atom zeroes; with hvp_probes > 0 the
    MLP / OV atoms get the second-order term -1/2 a^2 H_aa (Hutchinson)."""
    model = net.model
    acts, hs = {}, []

    def mk(name, take_input):
        def hook(mm, inp, out):
            acts[name] = inp[0] if take_input else out
        return hook
    for l in range(N_LAYER):
        hs.append(model.get_submodule(f"h.{l}.mlp.down_proj").register_forward_hook(mk(f"a{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.o_proj").register_forward_hook(mk(f"o{l}", True)))
        hs.append(model.get_submodule(f"h.{l}.attn.q_proj").register_forward_hook(mk(f"q{l}", False)))
        hs.append(model.get_submodule(f"h.{l}.attn.k_proj").register_forward_hook(mk(f"k{l}", False)))
    lg = model(idx)
    for h in hs:
        h.remove()
    reward = reward_fn(lg)
    names = [f"{k}{l}" for l in range(N_LAYER) for k in ("a", "o", "q", "k")]
    tensors = [acts[n] for n in names]
    grads = torch.autograd.grad(reward, tensors, create_graph=hvp_probes > 0)
    B, T = idx.shape
    diag = None
    if hvp_probes > 0:
        diag = [torch.zeros_like(t) for t in tensors]
        for _ in range(hvp_probes):
            zs = [torch.randint(0, 2, t.shape, generator=gen, device=t.device).float() * 2 - 1
                  for t in tensors]
            gz = sum((g * z).sum() for g, z in zip(grads, zs))
            hz = torch.autograd.grad(gz, tensors, retain_graph=True)
            for d, z, h in zip(diag, zs, hz):
                d += z * h
        diag = [d / hvp_probes for d in diag]
    with torch.no_grad():
        out = []
        for l in range(N_LAYER):
            i = 4 * l
            a, o, q, k = tensors[i:i + 4]
            ga, go, gq, gk = grads[i:i + 4]
            mlp = ga.float() * a.float()                              # [B,T,3072]
            ov = go.float() * o.float()                               # [B,T,768]
            if diag is not None:
                mlp = mlp - 0.5 * a.float().pow(2) * diag[i].float()
                ov = ov - 0.5 * o.float().pow(2) * diag[i + 1].float()
            qk = ((gq.float() * q.float()).view(B, T, N_HEAD, -1).sum(-1)
                  + (gk.float() * k.float()).view(B, T, N_HEAD, -1).sum(-1))  # [B,T,6]
            out += [mlp, qk, ov]
        A = torch.cat(out, dim=-1)                                    # [B,T,J]
    del grads, tensors, acts, lg
    return A


def reward_at_positions(lg, ar, posf, scalar, idx):
    """Sum over rows of the scalar at each row's own target position."""
    lgt = lg[ar, posf].float()
    lsm = F.log_softmax(lgt, -1)
    tgt = lgt.argmax(-1) if scalar == "pred" else idx[ar, posf + 1]
    return lsm[ar, tgt].sum(), tgt, lsm[ar, tgt]


def reward_all_positions(lg, idx, lo, scalar):
    """Sum over batch and positions lo..T-2 of the per-position scalar."""
    lgt = lg[:, lo:-1].float()
    lsm = F.log_softmax(lgt, -1)
    tgt = lgt.argmax(-1) if scalar == "pred" else idx[:, lo + 1:]
    lp = lsm.gather(-1, tgt[..., None]).squeeze(-1)
    return lp.sum(), tgt, lp


# ---------------------------------------------------------------- collect ---

def phase_collect(args, dev):
    cfg = SENSORS[args.sensor]
    assert cfg["K"] == 1, "collect uses the plain (GIM) backward; no IG steps"
    corpus = Corpus(args)
    T = corpus.T
    my_seqs = torch.arange(args.shard, len(corpus), args.nshard)   # interleaved
    N_seq = my_seqs.numel()
    net = make_net(args, dev, cfg["mode"])
    units, sub = atom_units()
    J = sub.numel()
    P, SB = args.pos_per_seq, args.seq_per_batch
    avail = torch.arange(args.min_pos, T - 1)            # t with a next token
    assert avail.numel() >= P, "not enough positions per sequence"
    n_ev = N_seq * P
    Z = torch.zeros(n_ev, J, dtype=torch.float16)
    ev = {k: torch.zeros(n_ev, dtype=torch.int32) for k in ("seq", "pos", "pred", "gold")}
    ev["lp"] = torch.zeros(n_ev)
    ev["norm"] = torch.zeros(n_ev)
    gen = torch.Generator().manual_seed(args.seed + 1000 * args.shard)
    hgen = torch.Generator(device=dev).manual_seed(args.seed + 77 + 1000 * args.shard)
    log(f"collect shard {args.shard}/{args.nshard}: {N_seq} seqs x {P} positions = "
        f"{n_ev} events, {J} atoms, sensor {args.sensor} (mode {cfg['mode']}), "
        f"scalar {args.scalar}, hvp probes {args.hvp_probes}, source {args.lab_source}")
    out = z_path(args, args.shard, args.nshard)
    part = out.with_suffix(".part.pt")
    s0 = 0
    if part.exists():
        pt = torch.load(part, map_location="cpu", weights_only=False)
        if pt["n_seq"] == N_seq and pt["P"] == P and pt["J"] == J:
            s0 = int(pt["done"])
            n0 = s0 * P
            Z[:n0] = pt["Z"][:n0]
            for k in ev:
                ev[k][:n0] = pt[k][:n0]
            gen.set_state(pt["gen_state"])
            log(f"resuming from {part.name}: {s0}/{N_seq} sequences done")
        else:
            log(f"ignoring incompatible {part.name}")
    t0 = time.perf_counter()
    for b0 in range(s0, N_seq, SB):
        b1 = min(N_seq, b0 + SB)
        rows = corpus.rows(my_seqs[b0:b1])                             # [sb, T]
        sb = rows.shape[0]
        B = sb * P
        pos = torch.stack([avail[torch.randperm(avail.numel(), generator=gen)[:P]]
                           .sort().values for _ in range(sb)])         # [sb, P]
        idx = rows.to(dev)[:, None, :].expand(sb, P, T).reshape(B, T).contiguous()
        posf = pos.to(dev).reshape(B)
        ar = torch.arange(B, device=dev)
        rec = {}

        def reward_fn(lg):
            r, tgt, lp = reward_at_positions(lg, ar, posf, args.scalar, idx)
            rec["tgt"], rec["lp"] = tgt, lp
            return r
        A = atom_attr(net, idx, reward_fn, args.hvp_probes, hgen)   # [B,T,J]
        a_units = A.sum(1)                                            # over positions
        e0, e1 = b0 * P, b1 * P
        Z[e0:e1] = a_units.half().cpu()
        ev["norm"][e0:e1] = a_units.norm(dim=1).cpu()
        ev["seq"][e0:e1] = my_seqs[b0:b1].repeat_interleave(P).int()
        ev["pos"][e0:e1] = posf.int().cpu()
        ev["pred"][e0:e1] = rec["tgt"].int().cpu()
        ev["gold"][e0:e1] = idx[ar, posf + 1].int().cpu()
        ev["lp"][e0:e1] = rec["lp"].detach().cpu()
        del A, a_units
        if (b0 // SB) % 16 == 0:
            log(f"seq {b1}/{N_seq} ({time.perf_counter() - t0:.0f}s)")
        if b1 % args.save_every < SB and b1 < N_seq:
            torch.save({"Z": Z, **ev, "done": b1, "n_seq": N_seq, "P": P, "J": J,
                        "gen_state": gen.get_state()}, part)
            if args.stop_after and b1 >= args.stop_after:
                log(f"stop_after {args.stop_after}: checkpoint written, exiting")
                return
    torch.save({"Z": Z, **ev, "unit_ids": sub, "sensor": args.sensor,
                "scalar": args.scalar, "lab_tag": args.lab_tag,
                "lab_source": args.lab_source, "bin_offset": args.bin_offset,
                "hvp_probes": args.hvp_probes,
                "shard": args.shard, "nshard": args.nshard, "n_seq": N_seq,
                "T": T, "P": P,
                "unit_type": [units[g][0] for g in sub.tolist()],
                "unit_layer": [units[g][1] for g in sub.tolist()]}, out)
    log(f"wrote {out} ({time.perf_counter() - t0:.0f}s)")
    if part.exists():
        part.unlink()


# ------------------------------------------------------------- fit pieces ---

def build_D(own, p, C):
    """[C, Jl] with one owner per atom, rows unit norm (zero rows for empty)."""
    Jl = own.numel()
    D = torch.zeros(C, Jl, device=own.device)
    D[own, torch.arange(Jl, device=own.device)] = p
    return F.normalize(D, dim=1)


def sparse_code(Zb, D, k):
    """Coefficients of each event on the orthonormal dictionary rows, kept
    per event by the universal threshold (k <= 0) or by fixed top-k."""
    corr = Zb @ D.t()                                    # [n, C]
    C = corr.shape[1]
    if k > 0 and k < C:
        top = corr.abs().topk(k, dim=1).indices
        return torch.zeros_like(corr).scatter_(1, top, corr.gather(1, top))
    if k >= C:
        return corr
    sigma = corr.abs().median(dim=1, keepdim=True).values / 0.6745
    thr = sigma * math.sqrt(2.0 * math.log(C))
    return corr * (corr.abs() > thr)                     # sigma 0 -> keep none


META_KEYS = ("seq", "pos", "pred", "gold", "lp")


class TopEvents:
    """Running top-k events per component, positive and negative coefficients,
    with each kept event's meta (seq, pos, pred, gold, lp)."""

    def __init__(self, C, k, dev):
        self.k, self.C, self.dev = k, C, dev
        self.v = {s: torch.full((k, C), -1.0, device=dev) for s in ("pos", "neg")}
        self.meta = {s: {m: torch.zeros(k, C, device=dev) for m in META_KEYS}
                     for s in ("pos", "neg")}

    def update(self, Ub, meta):
        """Ub [n, C]; meta: dict of [n] tensors (on device)."""
        for s, vals in (("pos", Ub), ("neg", -Ub)):
            kk = min(self.k, vals.shape[0])
            v, i = vals.topk(kk, dim=0)                          # [kk, C]
            allv = torch.cat([self.v[s], v])
            self.v[s], sel = allv.topk(self.k, dim=0)
            for m in META_KEYS:
                cand = torch.cat([self.meta[s][m], meta[m].float()[i]])
                self.meta[s][m] = cand.gather(0, sel)

    def state(self):
        return {"v": {s: t.cpu() for s, t in self.v.items()},
                "meta": {s: {m: t.cpu() for m, t in d.items()} for s, d in self.meta.items()}}

    def load(self, st):
        self.v = {s: t.to(self.dev) for s, t in st["v"].items()}
        self.meta = {s: {m: t.to(self.dev) for m, t in d.items()} for s, d in st["meta"].items()}


class Accum:
    """Sufficient statistics of one sweep (or chunk) for one C."""

    def __init__(self, C, Jl, dev):
        self.G = torch.zeros(C, Jl, device=dev)
        self.un = torch.zeros(C, device=dev)
        self.nz = torch.zeros(C, device=dev)
        self.npos = torch.zeros(C, device=dev)
        self.asum = torch.zeros(C, device=dev)
        self.act_hist = torch.zeros(C + 1, device=dev)
        self.expl = 0.0
        self.n = 0

    def add(self, Ub, Zb):
        self.G += Ub.t() @ Zb
        self.un += Ub.pow(2).sum(0)
        nzb = Ub != 0
        self.nz += nzb.sum(0)
        self.npos += (Ub > 0).sum(0)
        self.asum += Ub.abs().sum(0)
        self.act_hist += torch.bincount(nzb.sum(1), minlength=self.act_hist.numel()).float()
        self.expl += float(Ub.pow(2).sum())
        self.n += Ub.shape[0]

    def decay(self, rho):
        for t in (self.G, self.un, self.nz, self.npos, self.asum, self.act_hist):
            t.mul_(rho)
        self.expl *= rho
        self.n = int(self.n * rho)

    def act_quantiles(self):
        cs = self.act_hist.cumsum(0) / self.act_hist.sum().clamp_min(1)
        return {q: int((cs < q).sum()) for q in (0.5, 0.9)}


def ownership_step(acc, own, p, colsq, tol_reseed=True):
    """Assign every atom to the component whose usage best predicts its column."""
    C, Jl = acc.G.shape
    arj = torch.arange(Jl, device=acc.G.device)
    score = acc.G.pow(2) / acc.un.clamp_min(1e-12)[:, None]
    score[acc.un <= 0] = -1.0
    new_own = score.argmax(0)
    p_new = acc.G[new_own, arj] / acc.un[new_own].clamp_min(1e-12)
    own_expl = float(score[new_own, arj].clamp_min(0).sum())
    sizes = torch.bincount(new_own, minlength=C)
    empty = (sizes == 0).nonzero().flatten()
    if empty.numel() and tol_reseed:
        col_frac = score[new_own, arj].clamp_min(0) / colsq
        cand = (sizes[new_own] > 1).nonzero().flatten()     # keep >= 1 atom
        worst = cand[col_frac[cand].argsort()[:empty.numel()]]
        new_own[worst] = empty[:worst.numel()]
        p_new[worst] = 1.0
    changed = float((new_own != own).float().mean())
    return new_own, p_new, own_expl, changed, int(empty.numel())


def finish_C(args, C, own, p, acc, top, rms, live, dead, sub, units, hist, N,
             src_info, t0):
    """Write fit file + unitabl67 own file + summary JSON for one C."""
    J = rms.numel()
    Jl = live.numel()
    sizes = torch.bincount(own, minlength=C)
    usage_freq = acc.nz / max(acc.n, 1)
    usage_pos_frac = acc.npos / acc.nz.clamp_min(1)
    mean_abs = acc.asum / acc.nz.clamp_min(1)
    own_full = torch.full((J,), -1, dtype=torch.long)
    own_full[live.cpu()] = own.cpu()
    members = [sub[own_full == c] for c in range(C)]
    inert = sub[dead.cpu()]
    p_full = torch.zeros(J)
    p_full[live.cpu()] = p.cpu()
    utype = np.array([units[g][0] for g in sub.tolist()])
    ulayer = np.array([units[g][1] for g in sub.tolist()])
    mixed = sum(1 for c in range(C) if sizes[c] > 1
                and len(set(utype[(own_full == c).numpy()])) > 1)
    multilayer = sum(1 for c in range(C) if sizes[c] > 1
                     and len(set(ulayer[(own_full == c).numpy()])) > 1)
    aq = acc.act_quantiles()
    fit = {"format": "zcofac67_fit_v3", "C": C, "N": N, "J": J, "Jl": Jl,
           "sensor": args.sensor, "scalar": args.scalar, "lab_tag": args.lab_tag,
           "hvp_probes": args.hvp_probes, **src_info,
           "k": args.k, "own": own_full, "p": p_full, "live": live.cpu(),
           "dead": dead.cpu(), "rms": rms.cpu(), "unit_ids": sub,
           "sizes": sizes.cpu(), "usage_freq": usage_freq.cpu(),
           "usage_pos_frac": usage_pos_frac.cpu(), "mean_abs_coef": mean_abs.cpu(),
           "active_hist": acc.act_hist.cpu(), "top_events": top.state(),
           "hist": hist, "seed": args.seed, "proj_dim": args.proj_dim}
    torch.save(fit, fit_path(args, C))
    ownf = {"grouping": GROUPING, "variant": f"z{C}", "C": C,
            "members": members, "pool": [], "inert": inert,
            "unit_ids": sub, "unit_mean_kl": rms.cpu(),
            "group_label": own_full, "coef": p_full,
            "source": "zcofac67", "sensor": args.sensor, "scalar": args.scalar,
            "lab_tag": args.lab_tag, "n_tok": N}
    opath = U.out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{GROUPING}_z{C}.pt"
    torch.save(ownf, opath)
    summ = {"C": C, "N": N, "J": J, "dead": int(dead.sum()),
            "explained_code": round(acc.expl / max(acc.n, 1), 5), "iters": len(hist),
            "active_per_event": {"median": aq[0.5], "p90": aq[0.9]},
            "sizes": U.qdict(sizes[sizes > 0].cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "largest": int(sizes.max()), "empty": int((sizes == 0).sum()),
            "usage_freq": U.qdict(usage_freq.cpu().numpy(), (0.1, 0.5, 0.9, 0.99)),
            "components_mixing_unit_types": mixed,
            "components_spanning_layers": multilayer,
            "seconds": round(time.perf_counter() - t0, 1)}
    log(f"C={C} done: sizes median {int(sizes.median())} max {int(sizes.max())}, "
        f"usage freq median {usage_freq.median():.4f} max {usage_freq.max():.3f}, "
        f"{mixed} comps mix unit types, {multilayer} span layers -> {opath.name}")
    outj = Path(args.out).with_name(Path(args.out).stem + f"_C{C}.json")
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps({**summ, "hist": hist}, indent=1))


# -------------------------------------------------------------------- fit ---

class ZStore:
    """All Z shards as one fp16 [N, J] tensor in CPU RAM plus event meta;
    yields standardised fp32 row batches on the device."""

    def __init__(self, files, dev, RB):
        heads = [torch.load(f, map_location="cpu", weights_only=False, mmap=True)
                 for f in files]
        self.N = sum(int(h["Z"].shape[0]) for h in heads)
        self.J = int(heads[0]["Z"].shape[1])
        self.Z16 = torch.empty(self.N, self.J, dtype=torch.float16)
        self.meta = {k: torch.empty(self.N, dtype=heads[0][k].dtype) for k in META_KEYS}
        i = 0
        for h in heads:
            n = int(h["Z"].shape[0])
            self.Z16[i:i + n] = h["Z"]
            for k in self.meta:
                self.meta[k][i:i + n] = h[k]
            i += n
        self.unit_ids = heads[0]["unit_ids"]
        self.info = {"lab_source": heads[0].get("lab_source", "pile"),
                     "bin_offset": heads[0].get("bin_offset", 0)}
        del heads
        self.dev, self.RB = dev, RB
        self.live = self.scale = None
        log(f"Z store: {self.N} events x {self.J} atoms from {len(files)} file(s), "
            f"{self.Z16.numel() * 2 / 2**30:.1f} GiB fp16 in RAM")

    def raw_batches(self):
        for i in range(0, self.N, self.RB):
            yield i, self.Z16[i:i + self.RB].to(self.dev, non_blocking=True).float()

    def batches(self):
        for i, blk in self.raw_batches():
            yield i, standardize(blk, self.live, self.scale)

    def meta_batch(self, i0, n):
        return {k: self.meta[k][i0:i0 + n].to(self.dev) for k in META_KEYS}


def standardize(blk, live, scale):
    if live is not None:
        blk = blk[:, live]
    blk = blk / scale
    return blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-30)


def rms_stats(sq, N, args):
    rms = (sq / N).sqrt().float()
    med = rms[rms > 0].median()
    dead = rms < args.dead_frac * med
    live = (~dead).nonzero().flatten()
    scale = rms[live].clamp_min(args.floor_frac * med)
    log(f"{int(dead.sum())} dead atoms (rms < {args.dead_frac:g} x median {med:.3e})")
    return rms, dead, live, scale


def kmeans_init(Y, C, args, dev):
    _, own = S67.kmeans(Y, C, args.kmeans_iters, args.seed, spherical=True)
    return own.to(dev), torch.ones(Y.shape[0], device=dev)


def phase_fit(args, dev):
    store = ZStore(z_files(args), dev, args.row_batch)
    N, J = store.N, store.J
    sub = store.unit_ids
    units = U.build_units()
    sq = torch.zeros(J, device=dev, dtype=torch.float64)
    for _, blk in store.raw_batches():
        sq += blk.pow(2).sum(0).double()
    rms, dead, live, scale = rms_stats(sq, N, args)
    Jl = live.numel()
    store.live = live if Jl < J else None
    store.scale = scale
    colsq = torch.zeros(Jl, device=dev, dtype=torch.float64)
    total = 0.0
    for _, blk in store.batches():
        colsq += blk.pow(2).sum(0).double()
        total += float(blk.pow(2).sum())
    colsq = colsq.float().clamp_min(1e-12)
    for C in args.C:
        t0 = time.perf_counter()
        Y = torch.zeros(Jl, args.proj_dim, device=dev)
        for bi, (i0, blk) in enumerate(store.batches()):
            g = torch.Generator(device=dev).manual_seed(args.seed * 7919 + bi)
            R = torch.randn(blk.shape[0], args.proj_dim, generator=g, device=dev)
            Y += blk.t() @ R
        own, p = kmeans_init(Y / math.sqrt(args.proj_dim), C, args, dev)
        del Y
        hist = []
        for it in range(args.iters):
            D = build_D(own, p, C)
            acc = Accum(C, Jl, dev)
            for i0, blk in store.batches():
                acc.add(sparse_code(blk, D, args.k), blk)
            own, p, own_expl, changed, n_empty = ownership_step(acc, own, p, colsq)
            aq = acc.act_quantiles()
            hist.append({"it": it, "explained_code": round(acc.expl / total, 5),
                         "explained_own": round(own_expl / total, 5),
                         "changed": round(changed, 5), "active_median": aq[0.5],
                         "active_p90": aq[0.9], "empty_reseeded": n_empty})
            log(f"C={C} it {it}: explained {acc.expl / total:.4f} (own step "
                f"{own_expl / total:.4f}), owners changed {changed:.4f}, active/event "
                f"median {aq[0.5]} p90 {aq[0.9]}, reseeded {n_empty} "
                f"({time.perf_counter() - t0:.0f}s)")
            if it >= 3 and changed < args.tol:
                break
        D = build_D(own, p, C)
        acc = Accum(C, Jl, dev)
        top = TopEvents(C, TOP_EVENTS, dev)
        for i0, blk in store.batches():
            Ub = sparse_code(blk, D, args.k)
            acc.add(Ub, blk)
            top.update(Ub, store.meta_batch(i0, blk.shape[0]))
        finish_C(args, C, own, p, acc, top, rms, live, dead, sub, units, hist, N,
                 store.info, t0)
        del D, acc
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()


# ----------------------------------------------------------------- stream ---

def phase_stream(args, dev):
    """Fused sensor + online fit over the corpus.  Pilot batches give the
    atom RMS (then the k-means init); afterwards every batch is coded
    against the current dictionaries and its statistics accumulated;
    ownership is refreshed every --chunk_events events with the previous
    statistics decayed by --decay.  Checkpointed every --ckpt_every batches."""
    cfg = SENSORS[args.sensor]
    assert cfg["K"] == 1, "stream uses the plain (GIM) backward; no IG steps"
    if args.lab_source != "bin" and not args.smoke:
        log("warning: stream on the small pile corpus; intended for --lab_source bin")
    corpus = Corpus(args)
    T, lo = corpus.T, args.min_pos
    n_pos = T - 1 - lo
    N_seq = len(corpus)
    SB = args.seq_per_batch
    n_batches = math.ceil(N_seq / SB)
    net = make_net(args, dev, cfg["mode"])
    units, sub = atom_units()
    J = sub.numel()
    Cs = list(args.C)
    ck = zdir(args) / f"stream{args.lab_tag}_{args.sensor}_{args.scalar}_ckpt.pt"
    hgen = torch.Generator(device=dev).manual_seed(args.seed + 77)
    log(f"stream: {N_seq} seqs x {n_pos} positions = {N_seq * n_pos} events in "
        f"{n_batches} batches of {SB} seqs; C={Cs}; sensor {args.sensor}, scalar "
        f"{args.scalar}, hvp probes {args.hvp_probes}; pilot {args.pilot_batches} "
        f"batches, chunk {args.chunk_events} events, decay {args.decay}")

    st = {"b": 0, "phase": "rms", "sq": torch.zeros(J, dtype=torch.float64, device=dev), "n_rms": 0,
          "Y": None, "n_init": 0, "rms": None, "dead": None, "live": None, "scale": None,
          "colsq": None, "own": {}, "p": {}, "acc": {}, "top": {}, "hist": {c: [] for c in Cs},
          "chunk_n": 0, "n_events": 0}
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        log(f"resuming from {ck.name}: batch {st['b']}/{n_batches}, phase {st['phase']}")
        for key in ("sq", "Y", "rms", "dead", "live", "scale", "colsq"):
            if st[key] is not None:
                st[key] = st[key].to(dev)
        for c in Cs:
            if c in st["own"]:
                st["own"][c], st["p"][c] = st["own"][c].to(dev), st["p"][c].to(dev)
                acc = Accum(c, st["scale"].numel(), dev)
                for k, v in st["acc"][c].items():
                    if torch.is_tensor(v):
                        setattr(acc, k, v.to(dev))
                    else:
                        setattr(acc, k, v)
                st["acc"][c] = acc
                top = TopEvents(c, TOP_EVENTS, dev)
                top.load(st["top"][c])
                st["top"][c] = top
        hgen.manual_seed(args.seed + 77 + st["b"])

    def save_ckpt():
        out = {k: v for k, v in st.items() if k not in ("acc", "top")}
        for key in ("sq", "Y", "rms", "dead", "live", "scale", "colsq"):
            if out[key] is not None:
                out[key] = out[key].cpu()
        out["own"] = {c: t.cpu() for c, t in st["own"].items()}
        out["p"] = {c: t.cpu() for c, t in st["p"].items()}
        out["acc"] = {c: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(a).items()}
                      for c, a in st["acc"].items()}
        out["top"] = {c: t.state() for c, t in st["top"].items()}
        tmp = ck.with_suffix(".tmp")
        torch.save(out, tmp)
        tmp.replace(ck)

    t0 = time.perf_counter()
    D = {c: build_D(st["own"][c], st["p"][c], c) for c in Cs if c in st["own"]}
    for b in range(st["b"], n_batches):
        seq_ids = torch.arange(b * SB, min(N_seq, (b + 1) * SB))
        rows = corpus.rows(seq_ids)
        idx = rows.to(dev)
        rec = {}

        def reward_fn(lg):
            r, tgt, lp = reward_all_positions(lg, idx, lo, args.scalar)
            rec["tgt"], rec["lp"] = tgt, lp
            return r
        A = atom_attr(net, idx, reward_fn, args.hvp_probes, hgen)    # [sb,T,J]
        Zraw = A[:, lo:-1].reshape(-1, J)                             # [n, J]
        n = Zraw.shape[0]
        del A
        if st["phase"] == "rms":
            st["sq"] += Zraw.pow(2).sum(0).double()
            st["n_rms"] += n
            if b + 1 >= args.pilot_batches:
                rms, dead, live, scale = rms_stats(st["sq"], st["n_rms"], args)
                st.update(rms=rms, dead=dead, live=live if live.numel() < J else None,
                          scale=scale, phase="init",
                          Y=torch.zeros(live.numel(), args.proj_dim, device=dev),
                          colsq=torch.zeros(live.numel(), device=dev, dtype=torch.float64))
        elif st["phase"] == "init":
            Zb = standardize(Zraw, st["live"], st["scale"])
            g = torch.Generator(device=dev).manual_seed(args.seed * 7919 + b)
            R = torch.randn(n, args.proj_dim, generator=g, device=dev)
            st["Y"] += Zb.t() @ R
            st["colsq"] += Zb.pow(2).sum(0).double()
            st["n_init"] += n
            if st["n_init"] >= args.pilot_batches * n:
                Y = st["Y"] / math.sqrt(args.proj_dim)
                Jl = Y.shape[0]
                for c in Cs:
                    st["own"][c], st["p"][c] = kmeans_init(Y, c, args, dev)
                    st["acc"][c] = Accum(c, Jl, dev)
                    st["top"][c] = TopEvents(c, TOP_EVENTS, dev)
                    D[c] = build_D(st["own"][c], st["p"][c], c)
                st["Y"] = None
                st["phase"] = "fit"
                log(f"init done after {st['n_init']} events: k-means for C={Cs}")
        else:
            Zb = standardize(Zraw, st["live"], st["scale"])
            st["colsq"] += Zb.pow(2).sum(0).double()
            meta = {"seq": seq_ids.to(dev)[:, None].expand(-1, n_pos).reshape(-1),
                    "pos": torch.arange(lo, T - 1, device=dev)[None].expand(len(seq_ids), -1).reshape(-1),
                    "pred": rec["tgt"].reshape(-1), "gold": idx[:, lo + 1:].reshape(-1),
                    "lp": rec["lp"].detach().reshape(-1)}
            for c in Cs:
                Ub = sparse_code(Zb, D[c], args.k)
                st["acc"][c].add(Ub, Zb)
                st["top"][c].update(Ub, meta)
            st["chunk_n"] += n
            st["n_events"] += n
            if st["chunk_n"] >= args.chunk_events or b + 1 == n_batches:
                colsq = (st["colsq"] / max(st["n_events"], 1) * st["acc"][Cs[0]].n
                         ).float().clamp_min(1e-12)
                for c in Cs:
                    acc = st["acc"][c]
                    own, p, own_expl, changed, n_empty = ownership_step(
                        acc, st["own"][c], st["p"][c], colsq)
                    st["own"][c], st["p"][c] = own, p
                    D[c] = build_D(own, p, c)
                    aq = acc.act_quantiles()
                    tot = max(acc.n, 1)
                    st["hist"][c].append({
                        "batch": b + 1, "events_seen": st["n_events"],
                        "explained_code": round(acc.expl / tot, 5),
                        "explained_own": round(own_expl / tot, 5),
                        "changed": round(changed, 5), "active_median": aq[0.5],
                        "active_p90": aq[0.9], "empty_reseeded": n_empty})
                    log(f"C={c} chunk @ {st['n_events']} events: explained "
                        f"{acc.expl / tot:.4f} (own step {own_expl / tot:.4f}), owners "
                        f"changed {changed:.4f}, active/event median {aq[0.5]} p90 "
                        f"{aq[0.9]}, reseeded {n_empty} ({time.perf_counter() - t0:.0f}s)")
                    acc.decay(args.decay)
                st["chunk_n"] = 0
        del Zraw
        st["b"] = b + 1
        if (b + 1) % 50 == 0:
            log(f"batch {b + 1}/{n_batches}, phase {st['phase']}, {st['n_events']} fit "
                f"events ({time.perf_counter() - t0:.0f}s)")
        if (b + 1) % args.ckpt_every == 0 and b + 1 < n_batches:
            save_ckpt()
            if args.stop_after and b + 1 >= args.stop_after:
                log(f"stop_after {args.stop_after} batches: checkpoint written, exiting")
                return
    src_info = {"lab_source": args.lab_source, "bin_offset": args.bin_offset,
                "n_seq": N_seq, "positions_per_seq": n_pos, "stream": True,
                "chunk_events": args.chunk_events, "decay": args.decay}
    for c in Cs:
        finish_C(args, c, st["own"][c], st["p"][c], st["acc"][c], st["top"][c],
                 st["rms"], st["live"] if st["live"] is not None else torch.arange(J, device=dev),
                 st["dead"], sub, units, st["hist"][c], st["n_events"], src_info, t0)
    if ck.exists():
        ck.unlink()
    log(f"stream done: {st['n_events']} events ({time.perf_counter() - t0:.0f}s)")


# ----------------------------------------------------------------- report ---

def phase_report(args, dev):
    corpus = Corpus(args)
    units = U.build_units()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    except Exception as e:                                # noqa: BLE001
        log(f"tokenizer unavailable ({e}); contexts as token ids")
        tok = None

    def ctx(m, j, c, n_before=args.ctx_tokens):
        s, t = int(m["seq"][j, c]), int(m["pos"][j, c])
        row = corpus[s]
        left = row[max(0, t - n_before):t + 1].tolist()
        pred, gold = int(m["pred"][j, c]), int(m["gold"][j, c])
        lp = round(float(m["lp"][j, c]), 3)
        if tok is None:
            return {"left": left, "pred": pred, "gold": gold, "lp": lp}
        return {"left": tok.decode(left), "pred": tok.decode([pred]),
                "gold": tok.decode([gold]), "lp": lp}

    for C in args.C:
        fit = torch.load(fit_path(args, C), map_location="cpu", weights_only=False)
        own, sizes, freq = fit["own"], fit["sizes"], fit["usage_freq"]
        sub = fit["unit_ids"]
        te = fit["top_events"]
        order = freq.argsort(descending=True)
        g = torch.Generator().manual_seed(args.seed)
        picks = order[:args.n_show].tolist()
        rnd = torch.randperm(C, generator=g)[:args.n_show].tolist()
        picks += [c for c in rnd if c not in picks]
        comps = []
        for c in picks:
            atoms = (own == c).nonzero().flatten()
            if atoms.numel() == 0:
                continue
            desc = [f"L{units[int(sub[q])][1]}:{units[int(sub[q])][0]}{units[int(sub[q])][2]}"
                    f"{'+' if fit['p'][q] >= 0 else '-'}" for q in atoms.tolist()]
            ev = {}
            for s in ("pos", "neg"):
                v = te["v"][s][:, c]
                ok = (v > 0).nonzero().flatten()[:args.top_events].tolist()
                ev[s] = [{"coef": round(float(v[j]) * (1 if s == "pos" else -1), 4),
                          **ctx(te["meta"][s], j, c)} for j in ok]
            comps.append({"component": int(c), "n_atoms": int(sizes[c]),
                          "usage_freq": round(float(freq[c]), 5),
                          "pos_frac": round(float(fit["usage_pos_frac"][c]), 3),
                          "atoms": desc[:64],
                          "top_positive": ev["pos"], "top_negative": ev["neg"]})
        out = Path(args.out).with_name(Path(args.out).stem + f"_report_C{C}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"C": C, "components": comps}, indent=1))
        log(f"C={C}: report on {len(comps)} components -> {out}")


# --------------------------------------------------------------- hvpcheck ---

def phase_hvpcheck(args, dev):
    """For n_tokens events: zero a sample of atoms one at a time and compare
    the measured change in the scalar with the first-order and the
    HVP-corrected attribution (both computed with GIM's backward)."""
    cfg = SENSORS[args.sensor]
    corpus = Corpus(args)
    T = corpus.T
    net = make_net(args, dev, cfg["mode"])
    units, sub = atom_units()
    J = sub.numel()
    look = U.unit_lookup(units, GROUPING, net.W0, dev)
    g = torch.Generator().manual_seed(args.seed + 5)
    hgen = torch.Generator(device=dev).manual_seed(args.seed + 99)
    n_all = len(units)
    res = {"format": "zcofac_hvpcheck_v1", "n_tokens": args.n_tokens,
           "n_atoms": args.n_atoms, "hvp_probes": args.hvp_probes, "scalar": args.scalar,
           "per_token": []}
    ar = torch.arange(1, device=dev)
    for i in range(args.n_tokens):
        s = int(torch.randint(0, len(corpus), (1,), generator=g))
        t = int(torch.randint(args.min_pos, T - 1, (1,), generator=g))
        idx = corpus[s][None].to(dev)
        posf = torch.tensor([t], device=dev)

        def reward_fn(lg):
            return reward_at_positions(lg, ar, posf, args.scalar, idx)[0]
        A1 = atom_attr(net, idx, reward_fn, 0, None).sum(1)[0]           # [J]
        probes = [1, 2, 4, 8, 16]
        probes = [pn for pn in probes if pn <= args.hvp_probes] or [args.hvp_probes]
        A2 = {pn: atom_attr(net, idx, reward_fn, pn, hgen).sum(1)[0] for pn in probes}
        with torch.no_grad():
            base = float(reward_fn(net.model(idx)))
        # atoms: top by |first order| plus random
        top = A1.abs().topk(args.n_atoms // 2).indices
        rnd = torch.randperm(J, generator=g)[:args.n_atoms // 2].to(dev)
        atoms = torch.unique(torch.cat([top, rnd]))
        meas = torch.zeros(atoms.numel(), device=dev)
        with torch.no_grad():
            for k, q in enumerate(atoms.tolist()):
                keep = torch.ones(n_all, dtype=torch.bool, device=dev)
                keep[int(sub[q])] = False
                net.apply_unit_mask(keep, look)
                meas[k] = float(reward_fn(net.model(idx))) - base     # measured change
            net.restore()
        pred1 = -A1[atoms]
        rec = {"seq": s, "pos": t, "n_atoms": int(atoms.numel()),
               "first": {"err": float((pred1 - meas).abs().mean()),
                         "corr": float(torch.corrcoef(torch.stack([pred1, meas]))[0, 1]),
                         "spearman": U.spearman(pred1.cpu(), meas.cpu())},
               "meas_abs_mean": float(meas.abs().mean())}
        for pn, a2 in A2.items():
            pred2 = -a2[atoms]
            rec[f"hvp{pn}"] = {"err": float((pred2 - meas).abs().mean()),
                               "corr": float(torch.corrcoef(torch.stack([pred2, meas]))[0, 1]),
                               "spearman": U.spearman(pred2.cpu(), meas.cpu())}
        # by unit type
        types = np.array([units[int(sub[q])][0] for q in atoms.tolist()])
        rec["by_type"] = {}
        for ty in ("mlp", "ov", "qk"):
            m = torch.from_numpy(types == ty).to(dev)
            if m.sum() >= 3:
                rec["by_type"][ty] = {
                    "first_err": float((pred1[m] - meas[m]).abs().mean()),
                    **{f"hvp{pn}_err": float((-a2[atoms][m] - meas[m]).abs().mean())
                       for pn, a2 in A2.items()}}
        res["per_token"].append(rec)
        log(f"token {i}: {atoms.numel()} atoms, first-order err {rec['first']['err']:.3e} "
            f"corr {rec['first']['corr']:.3f} | " + " ".join(
                f"hvp{pn} err {rec[f'hvp{pn}']['err']:.3e} corr {rec[f'hvp{pn}']['corr']:.3f}"
                for pn in A2))

    def mean_of(key, sub_key):
        return round(float(np.mean([r[key][sub_key] for r in res["per_token"]])), 6)
    res["summary"] = {k: {sk: mean_of(k, sk) for sk in ("err", "corr", "spearman")}
                      for k in ["first"] + [f"hvp{pn}" for pn in A2]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log("SUMMARY " + json.dumps(res["summary"]))


# ------------------------------------------------------------------- main ---

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["collect", "fit", "stream", "report", "hvpcheck"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lab_tag", default="_1m")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--out_dir", default="", help="unitabl own-file dir override")
    ap.add_argument("--z_dir", default="", help="Z / fit dir (default RUN/../zcofac)")
    ap.add_argument("--out", default="out/zcofac67.json")
    ap.add_argument("--sensor", default="gim")
    ap.add_argument("--scalar", choices=["pred", "gold"], default="pred")
    ap.add_argument("--hvp_probes", type=int, default=0,
                    help="Hutchinson probes for the second-order term (0 = first order)")
    # corpus
    ap.add_argument("--lab_source", choices=["pile", "bin"], default="pile",
                    help="pile: lab{lab_tag}.pt ids; bin: the Pile token bin")
    ap.add_argument("--tokbin", default=str(S.RUN / "pile_1b_uint16.bin"))
    ap.add_argument("--bin_offset", type=int, default=82_000_000,
                    help="bin: first token (past the 80M stream fit + 1M lab corpus)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n_seq", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--pos_per_seq", type=int, default=64)
    ap.add_argument("--seq_per_batch", type=int, default=8,
                    help="sequences per batch (collect batch = seq_per_batch x pos_per_seq)")
    ap.add_argument("--min_pos", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=128,
                    help="collect: checkpoint the partial Z every this many seqs")
    ap.add_argument("--stop_after", type=int, default=0,
                    help="collect: exit after this many seqs; stream: batches (resume test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="random-init model and random tokens (no checkpoint)")
    # fit / stream
    ap.add_argument("--C", type=int, nargs="+", default=[2048])
    ap.add_argument("--k", type=int, default=0,
                    help="fixed components per event; 0 = universal threshold")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="fit: stop when this fraction of owners changes")
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--row_batch", type=int, default=32768)
    ap.add_argument("--dead_frac", type=float, default=1e-3,
                    help="atoms with rms below this x median are inert")
    ap.add_argument("--floor_frac", type=float, default=1e-2,
                    help="rms floor for standardisation, x median")
    ap.add_argument("--pilot_batches", type=int, default=64,
                    help="stream: batches for the RMS estimate, then again for the init")
    ap.add_argument("--chunk_events", type=int, default=1_000_000,
                    help="stream: events between ownership refreshes")
    ap.add_argument("--decay", type=float, default=0.5,
                    help="stream: weight of the previous statistics at a refresh")
    ap.add_argument("--ckpt_every", type=int, default=500, help="stream: batches")
    # report
    ap.add_argument("--n_show", type=int, default=32)
    ap.add_argument("--top_events", type=int, default=8)
    ap.add_argument("--ctx_tokens", type=int, default=24)
    # hvpcheck
    ap.add_argument("--n_tokens", type=int, default=16)
    ap.add_argument("--n_atoms", type=int, default=256)
    return ap.parse_args(argv)


def main():
    args = parse()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    {"collect": phase_collect, "fit": phase_fit, "stream": phase_stream,
     "report": phase_report, "hvpcheck": phase_hvpcheck}[args.phase](args, dev)


if __name__ == "__main__":
    main()
