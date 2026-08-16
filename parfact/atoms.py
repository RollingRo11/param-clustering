"""Additive parameter atoms and per-event attribution (paper secs 4.1-4.2).

Atom variants, all satisfying "the additive atoms sum to the parameters":

  A: scalar atoms          b_j = theta_j e_j
     attribution           a_ij = theta_j * ds_i/dtheta_j
  B: rank-one SVD atoms    B_q = sigma_q u_q v_q^T
     attribution           a_iq = <grad_W s_i, B_q>_F = sigma_q u_q^T G_i v_q
  C: outer-product singular-vector directions  u_q v_q^T
     attribution           a_iq = <grad_W s_i, u_q v_q^T>_F = u_q^T G_i v_q

Variants B and C share the same additive atoms (the SVD terms sigma_q u_q v_q^T,
which is what components are rebuilt from, keeping sum_c C_c = theta exact);
they differ in the attribution functional: B weights the gradient projection by
the singular value (gradient-times-parameter along the direction), C uses the
bare gradient projection onto the unit outer product u_q v_q^T.

A prediction event is (sequence, position): s_i = log p(y* | x_<t) with
y* = argmax_y p(y | x_<t) (paper sec 4.1).
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from induction_model import InductionModel, gen_batch

# weight matrices decomposed by default: all attention projections.
ATTN_MATRICES = tuple(
    f"layers.{l}.{m}.weight" for l in range(2) for m in ("wq", "wk", "wv", "wo")
)
EMBED_MATRICES = ("embed.weight", "pos.weight", "unembed.weight")


def layer_of(param_name: str) -> str:
    """Depth group of a matrix: 'layer0', 'layer1', or the matrix itself."""
    if param_name.startswith("layers."):
        return f"layer{param_name.split('.')[1]}"
    return param_name


@dataclass
class AtomBasis:
    """Maps per-event weight-matrix gradients to atom attributions and back."""
    variant: str                       # 'A' | 'B' | 'C'
    matrices: list[str]                # param names, in fixed order
    weights: dict[str, torch.Tensor]   # detached weight tensors
    svd: dict[str, tuple] = field(default_factory=dict)  # name -> (U, S, Vh)
    atom_matrix: list[str] = field(default_factory=list)  # source matrix per atom
    atom_layer: list[str] = field(default_factory=list)   # depth group per atom

    @classmethod
    def build(cls, model: InductionModel, matrices: list[str], variant: str):
        assert variant in ("A", "B", "C")
        params = dict(model.named_parameters())
        weights = {n: params[n].detach().clone() for n in matrices}
        self = cls(variant=variant, matrices=list(matrices), weights=weights)
        for name in matrices:
            w = weights[name]
            if variant in ("B", "C"):
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                self.svd[name] = (u, s, vh)
                n_atoms = s.shape[0]
            else:
                n_atoms = w.numel()
            self.atom_matrix += [name] * n_atoms
            self.atom_layer += [layer_of(name)] * n_atoms
        return self

    @property
    def n_atoms(self) -> int:
        return len(self.atom_matrix)

    def attributions(self, grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """grads: name -> [N, *w.shape] per-event gradients. Returns A [N, J]."""
        cols = []
        for name in self.matrices:
            g = grads[name]
            if self.variant == "A":
                cols.append((g * self.weights[name]).flatten(1))
            else:
                u, s, vh = self.svd[name]
                proj = torch.einsum("dq,ndk,qk->nq", u, g, vh)  # u_q^T G v_q
                cols.append(proj * s if self.variant == "B" else proj)
        return torch.cat(cols, dim=1)

    def components(self, V: torch.Tensor) -> dict[str, torch.Tensor]:
        """C_c = sum_j V_jc b_j per matrix. V: [J, C] with rows on the simplex.
        Returns name -> [C, *w.shape]; components sum to the weights exactly."""
        out, j0 = {}, 0
        for name in self.matrices:
            w = self.weights[name]
            if self.variant == "A":
                vc = V[j0: j0 + w.numel()]                 # [numel, C]
                out[name] = (vc.T.reshape(-1, *w.shape) * w)
                j0 += w.numel()
            else:
                u, s, vh = self.svd[name]
                vc = V[j0: j0 + s.shape[0]]                # [r, C]
                # sum_q V_qc sigma_q u_q v_q^T
                out[name] = torch.einsum("qc,q,dq,qk->cdk", vc, s, u, vh)
                j0 += s.shape[0]
        return out


def make_events(model: InductionModel, n_seq: int, positions: str,
                extra_per_seq: int = 2, seed: int = 0):
    """Build prediction events (seq, pos, y=argmax) + metadata.

    positions: 'final' (the induction prediction only), 'mixed' (final plus
    `extra_per_seq` random earlier positions per sequence), or 'all'.
    Returns dict of tensors: seq [E, n_ctx], pos [E], y [E], is_induction [E],
    m_token [E] (ground-truth induction target for that sequence).
    """
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    seq, s_pos, m_tok = gen_batch(n_seq, device, gen)
    n_ctx = seq.shape[1]
    final = n_ctx - 1

    if positions == "final":
        pos = torch.full((n_seq, 1), final, device=device)
    elif positions == "mixed":
        extra = torch.randint(4, final, (n_seq, extra_per_seq), device=device,
                              generator=gen)
        pos = torch.cat([torch.full((n_seq, 1), final, device=device), extra], 1)
    elif positions == "all":
        pos = torch.arange(4, n_ctx, device=device).expand(n_seq, -1)
    else:
        raise ValueError(positions)

    rows = torch.arange(n_seq, device=device).unsqueeze(1).expand_as(pos)
    seq_e, pos_e = seq[rows.flatten()], pos.flatten()
    with torch.no_grad():
        logits = model(seq)                                # [n_seq, n_ctx, V]
        y = logits[rows.flatten(), pos_e].argmax(-1)
    return {"seq": seq_e, "pos": pos_e, "y": y,
            "is_induction": pos_e == final,
            "m_token": m_tok[rows.flatten()],
            "s_pos": s_pos[rows.flatten()]}


def _event_logprob_fn(model: InductionModel, fixed: dict[str, torch.Tensor]):
    def f(wanted: dict[str, torch.Tensor], seq, pos, y):
        logits = functional_call(model, {**fixed, **wanted}, (seq.unsqueeze(0),))
        # one-hot contractions instead of scalar indexing: vmap-compatible
        # (built via comparison, since F.one_hot's scatter_ breaks under vmap)
        pos_oh = (torch.arange(logits.shape[1], device=logits.device) == pos)
        logp = F.log_softmax(pos_oh.to(logits.dtype) @ logits[0], dim=-1)
        y_oh = (torch.arange(logp.shape[0], device=logp.device) == y)
        return logp @ y_oh.to(logp.dtype)
    return f


def collect_grads(model: InductionModel, matrices: list[str], seq, pos, y,
                  chunk: int = 1024) -> dict[str, torch.Tensor]:
    """Per-event gradients of s_i = log p(y_i | x, pos_i) wrt each matrix.

    Returns name -> [N, *w.shape]. One vmapped backward per chunk.
    """
    params = {n: p.detach() for n, p in model.named_parameters()}
    wanted = {n: params[n] for n in matrices}
    fixed = {n: p for n, p in params.items() if n not in matrices}
    gfn = vmap(grad(_event_logprob_fn(model, fixed)), in_dims=(None, 0, 0, 0))
    outs: dict[str, list] = {n: [] for n in matrices}
    for i in range(0, seq.shape[0], chunk):
        sl = slice(i, i + chunk)
        g = gfn(wanted, seq[sl], pos[sl], y[sl])
        for n in matrices:
            outs[n].append(g[n])
    return {n: torch.cat(v) for n, v in outs.items()}


def estimate_fisher(model: InductionModel, basis: AtomBasis, seq, pos,
                    n_samples: int = 4, seed: int = 0,
                    chunk: int = 1024) -> torch.Tensor:
    """Diagonal Fisher in atom space: F_j ~= E_x E_{y~p} [ a_j(x, y)^2 ],
    the typical squared attribution when labels are sampled from the model
    (paper sec 4.3). Uses the same attribution functional as the variant, so
    the normalization is consistent across atom bases."""
    device = seq.device
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        logits = model(seq)
        probs = F.softmax(logits[torch.arange(seq.shape[0], device=device), pos],
                          dim=-1)
    acc = torch.zeros(basis.n_atoms, device=device, dtype=torch.float64)
    for _ in range(n_samples):
        y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        grads = collect_grads(model, basis.matrices, seq, pos, y, chunk=chunk)
        acc += basis.attributions(grads).double().pow(2).mean(0)
    return (acc / n_samples).float()
