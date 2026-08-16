"""Attribution normalization + joint soft co-factorization (paper secs 4.3-4.6).

Pipeline:
  A (signed, [N events x J atoms])
    -> pre-normalization ('layer' rms groups or diagonal 'fisher', or 'none')
    -> M = |A~|, row-normalized to an allocation M_bar (rows ~ simplex)
    -> M_bar ~= U S V^T, nonneg, V rows on the simplex (soft atom memberships)
    -> components C_c = sum_j V_jc b_j;  usage z_ic = sum_j V_jc a_ij.
"""
import torch
import torch.nn.functional as F


def normalize_attributions(A: torch.Tensor, mode: str,
                           groups: list[str] | None = None,
                           fisher: torch.Tensor | None = None,
                           eps: float = 1e-8) -> torch.Tensor:
    """Pre-normalize signed attributions across atoms (paper sec 4.3).

    mode 'layer':  A~_ij = A_ij / sqrt(E_{i, j in g}[A^2] + eps), where g is the
                   atom's group (depth group or source matrix).
    mode 'fisher': A~_ij = A_ij / sqrt(F_j + eps), diagonal Fisher per atom.
    mode 'none':   raw A.
    """
    if mode == "none":
        return A
    if mode == "layer":
        assert groups is not None
        A = A.clone()
        for g in set(groups):
            idx = torch.tensor([i for i, gi in enumerate(groups) if gi == g],
                               device=A.device)
            block = A[:, idx]
            A[:, idx] = block / (block.pow(2).mean().sqrt() + eps)
        return A
    if mode == "fisher":
        assert fisher is not None
        return A / (fisher + eps).sqrt()
    raise ValueError(mode)


def allocation_matrix(A_tilde: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """M_bar: attribution magnitudes, each event normalized to an allocation."""
    M = A_tilde.abs()
    return M / (M.sum(dim=1, keepdim=True) + eps)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class TriFactorization(torch.nn.Module):
    """M_bar ~= U S V^T with U in R+^{N x K}, S in R+^{K x C}, V in R+^{J x C}.

    Nonnegativity is structural (softplus); V rows always live on the simplex
    (softmax), giving every atom a soft allocation over parameter groups.
    U rows can optionally be put on the simplex too (paper sec 4.4).
    """

    def __init__(self, n_events: int, n_atoms: int, k_factors: int,
                 c_groups: int, u_simplex: bool = False, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.u_hat = torch.nn.Parameter(
            torch.rand(n_events, k_factors, generator=gen) * 0.5 + 0.2)
        self.s_hat = torch.nn.Parameter(
            torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2)
        self.v_hat = torch.nn.Parameter(
            torch.randn(n_atoms, c_groups, generator=gen) * 0.05)
        self.u_simplex = u_simplex

    @property
    def U(self) -> torch.Tensor:
        u = F.softplus(self.u_hat)
        return u / u.sum(1, keepdim=True) if self.u_simplex else u

    @property
    def S(self) -> torch.Tensor:
        return F.softplus(self.s_hat)

    @property
    def V(self) -> torch.Tensor:
        return F.softmax(self.v_hat, dim=1)

    def forward(self) -> torch.Tensor:
        return self.U @ self.S @ self.V.T

    @torch.no_grad()
    def _rescale_init(self, M_bar: torch.Tensor):
        R = self()
        alpha = (M_bar * R).sum() / R.pow(2).sum().clamp_min(1e-12)
        self.s_hat.copy_(_inv_softplus(F.softplus(self.s_hat) * alpha.clamp_min(1e-6)))

    def fit(self, M_bar: torch.Tensor, steps: int = 3000, lr: float = 2e-2,
            lambda_u: float = 0.0, lambda_v: float = 0.0,
            log_every: int = 500) -> dict:
        """Minimize ||M_bar - U S V^T||_F^2 (+ sparsity regularizers).

        R(V) is row entropy (V rows are simplex); R(U) is entropy if U is on
        the simplex, else mean L1.
        """
        self._rescale_init(M_bar)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        denom = M_bar.pow(2).sum()
        for step in range(steps):
            U, S, V = self.U, self.S, self.V
            recon = U @ S @ V.T
            loss = (M_bar - recon).pow(2).sum() / denom
            if lambda_v:
                loss = loss + lambda_v * -(V * (V + 1e-12).log()).sum(1).mean()
            if lambda_u:
                if self.u_simplex:
                    loss = loss + lambda_u * -(U * (U + 1e-12).log()).sum(1).mean()
                else:
                    loss = loss + lambda_u * U.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % log_every == 0 or step == steps - 1:
                print(f"  fact step {step:5d}  rel err {loss.item():.5f}",
                      flush=True)
        with torch.no_grad():
            recon = self()
            r2 = r2_attr(M_bar, recon)
        return {"rel_err": ((M_bar - recon).pow(2).sum() / denom).item(),
                "r2_attr": r2}


def r2_attr(M_bar: torch.Tensor, recon: torch.Tensor) -> float:
    """R^2_attr = 1 - ||M_bar - USV^T||_F^2 / ||M_bar - mean||_F^2."""
    resid = (M_bar - recon).pow(2).sum()
    total = (M_bar - M_bar.mean()).pow(2).sum()
    return (1 - resid / total).item()


def component_usage(A_signed: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """z_ic = sum_j V_jc a_ij: signed first-order component attribution
    (paper sec 4.6), from the already-collected attribution matrix."""
    return A_signed @ V


def effective_number(P: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """exp(entropy) of rows normalized along `dim`: effective count of active
    factors/groups (polygenicity diagnostic)."""
    p = P / P.sum(dim, keepdim=True).clamp_min(1e-12)
    return (-(p * (p + 1e-12).log()).sum(dim)).exp()


def group_mass(V: torch.Tensor, groups: list[str]) -> dict[str, list[float]]:
    """d_cg = sum_{j in g} V_jc for each atom group g (paper sec 9.1: layer
    mass; also reported per source matrix)."""
    out = {}
    for g in sorted(set(groups)):
        idx = torch.tensor([i for i, gi in enumerate(groups) if gi == g],
                           device=V.device)
        out[g] = V[idx].sum(0).tolist()
    return out
