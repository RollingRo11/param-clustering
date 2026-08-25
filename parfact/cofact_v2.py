"""Co-FAC v2 co-factorization for the toy induction model.

Same method as ``cofac67.fit`` (the v2 code path written for the 67M VPD
target): M_bar ~= U S V^T with

  * U rows on the simplex (``--u_simplex``; softplus otherwise),
  * S nonnegative via softplus,
  * V rows on a (C+1)-simplex -- the extra column C0 is a residual /
    background group that absorbs atom mass no component claims,
  * I-divergence (generalized KL) objective instead of v1's Frobenius loss.

v1 (``cofact.TriFactorization``) is the plain C-simplex + Frobenius variant.
"""
import torch
import torch.nn.functional as F


class TriFactorizationV2(torch.nn.Module):
    """v2: U-simplex, residual V column, I-divergence objective."""

    def __init__(self, n_events: int, n_atoms: int, k_factors: int,
                 c_groups: int, u_simplex: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Wu = torch.nn.Parameter(
            torch.rand(n_events, k_factors, generator=g) * 0.5 + 0.2)
        self.Ws = torch.nn.Parameter(
            torch.rand(k_factors, c_groups, generator=g) * 0.5 + 0.2)
        # C+1 columns: the last one is the residual/background group C0
        self.Wv = torch.nn.Parameter(
            torch.randn(n_atoms, c_groups + 1, generator=g) * 0.05)
        self.c_groups = c_groups
        self.u_simplex = u_simplex

    @property
    def U(self) -> torch.Tensor:
        return (torch.softmax(self.Wu, dim=1) if self.u_simplex
                else F.softplus(self.Wu))

    @property
    def S(self) -> torch.Tensor:
        return F.softplus(self.Ws)

    @property
    def Vfull(self) -> torch.Tensor:
        return torch.softmax(self.Wv, dim=1)          # [J, C+1]

    @property
    def V(self) -> torch.Tensor:
        return self.Vfull[:, :self.c_groups]

    @property
    def residual(self) -> torch.Tensor:
        return self.Vfull[:, -1]

    def fit(self, M_bar: torch.Tensor, steps: int = 4000, lr: float = 2e-2,
            log_every: int = 250, row_chunk: int = 0) -> dict:
        """row_chunk > 0 computes the (identical) loss in event-row chunks
        with gradient accumulation, so huge atom counts (scalar atoms:
        J ~ 1e5) never materialize the full N x J reconstruction."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mass = M_bar.sum().clamp_min(1e-8)
        N = M_bar.shape[0]
        chunk = row_chunk if 0 < row_chunk < N else N
        hist = []

        def idiv(Mb, Mh):
            return (Mb * ((Mb + 1e-8).log() - (Mh + 1e-8).log())
                    - Mb + Mh).sum() / mass

        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            U_full = self.U
            SV = self.S @ self.V.T
            loss_val = 0.0
            for i0 in range(0, N, chunk):
                Mh = U_full[i0:i0 + chunk] @ SV
                loss_c = idiv(M_bar[i0:i0 + chunk], Mh)
                loss_c.backward(retain_graph=(i0 + chunk < N))
                loss_val += loss_c.item()
            opt.step()
            if step % log_every == 0 or step == steps - 1:
                with torch.no_grad():
                    SVn = self.S @ self.V.T
                    Un = self.U
                    resid = sum(float((M_bar[i0:i0 + chunk]
                                       - Un[i0:i0 + chunk] @ SVn)
                                      .pow(2).sum())
                                for i0 in range(0, N, chunk))
                    rel = (resid ** 0.5) / float(M_bar.norm())
                hist.append(f"step {step} idiv {loss_val:.4e} rel {rel:.4f}")
                print("  " + hist[-1], flush=True)
        with torch.no_grad():
            SVn = self.S @ self.V.T
            Un = self.U
            resid = sum(float((M_bar[i0:i0 + chunk]
                               - Un[i0:i0 + chunk] @ SVn).pow(2).sum())
                        for i0 in range(0, N, chunk))
            total = float((M_bar - M_bar.mean()).pow(2).sum())
            out = {"idiv": loss_val,
                   "rel_err": (resid ** 0.5) / float(M_bar.norm()),
                   "r2_attr_euclid": 1 - resid / total,
                   "mean_residual_membership": self.residual.mean().item(),
                   "log": hist[-3:]}
        return out


def singular_values(basis) -> torch.Tensor:
    """sigma_j per atom, in atom order (variant B/C only)."""
    return torch.cat([basis.svd[n][1] for n in basis.matrices])


def component_mass(V: torch.Tensor, residual: torch.Tensor,
                   sigma: torch.Tensor) -> dict:
    """Per-component weight mass.

    atom_mass_c = sum_j V_jc  -- each atom row softmax-splits one unit across
      the C components + the residual column, so mass is in units of 'atoms'.
    fro2_c      = sum_j V_jc sigma_j^2  -- the component's share of the
      decomposed weights' squared Frobenius norm ('weight mass' proper).
    """
    sig2 = sigma.pow(2)
    atom_mass = V.sum(0)                                    # [C]
    fro2 = (V * sig2[:, None]).sum(0)                       # [C]
    tot_fro2 = sig2.sum()
    resid_atom_mass = residual.sum()
    resid_fro2 = (residual * sig2).sum()

    order = fro2.argsort(descending=True)
    cum = fro2[order].cumsum(0) / tot_fro2
    # participation ratio: effective number of atoms a component draws on
    eff_atoms = atom_mass.pow(2) / V.pow(2).sum(0).clamp_min(1e-12)

    rows = []
    for rank, c in enumerate(order.tolist()):
        rows.append({
            "rank": rank,
            "component": int(c),
            "atom_mass": round(float(atom_mass[c]), 6),
            "atom_mass_frac": round(float(atom_mass[c] / atom_mass.sum()), 6),
            "fro2": round(float(fro2[c]), 6),
            "fro_share": round(float(fro2[c] / tot_fro2), 6),
            "cum_fro_share": round(float(cum[rank]), 6),
            "eff_atoms": round(float(eff_atoms[c]), 3),
        })

    def n_for(frac):
        return int((cum < frac).sum().item()) + 1

    return {
        "J": int(V.shape[0]), "C": int(V.shape[1]),
        "total_fro2_decomposed": float(tot_fro2),
        "residual_atom_mass": round(float(resid_atom_mass), 6),
        "residual_fro_share": round(float(resid_fro2 / tot_fro2), 8),
        "components_for_50pct_mass": n_for(0.5),
        "components_for_90pct_mass": n_for(0.9),
        "components_for_99pct_mass": n_for(0.99),
        "n_components_over_1_atom": int((atom_mass > 1.0).sum()),
        "n_components_over_0p1_atom": int((atom_mass > 0.1).sum()),
        "per_component": rows,
    }


def group_mass(V: torch.Tensor, groups: list[str]) -> dict[str, list[float]]:
    out = {}
    for g in sorted(set(groups)):
        idx = torch.tensor([i for i, gi in enumerate(groups) if gi == g],
                           device=V.device)
        out[g] = V[idx].sum(0).tolist()
    return out
