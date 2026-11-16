"""
Shared helpers for exp_gamma_directions_1d.py and exp_gamma_directions_2d.py.

By Eq. (deff_spectral), d_eff^pde = sum_i (1 + lam*gamma_i)^-2, where gamma_i is the
Rayleigh quotient E[(L du_i)^2] / E[du_i^2] of the operator on the function change
du_i = grad_theta g^T w_i produced by the generalized eigenvector w_i (Eq. rayleigh).

Computation (a generalized SVD of the pair (Psi, Phi_p)):
  Psi_n = Psi / sqrt(N)      output Jacobian on a dense grid              [N, d]
  Phi_n = Phi_p / sqrt(N_r)  PDE-residual Jacobian on the residual grid   [N_r, d]
  Psi_n = U S V^T, keep the k directions with s_i > psi_cut * s_1
  M = Phi_n V_k S_k^-1       residual change per unit RMS output change
  gamma_i = sigma_i(M)^2,    du_i = U_k z_i (unit RMS),   w_i = V_k S_k^-1 z_i
where z_i are the right singular vectors of M. If k > N_r, at least k - N_r directions
get gamma = 0 by counting alone, because the residual is seen at only N_r points.
"""

import os

import numpy as np
import torch
from torch.func import functional_call

from adr.training import train
from adr.utils import set_seed, get_target_params, compute_jacobian


def trained_model(make_model, pde_fn, x_col, x_bc, adam, lbfgs, lr, seed, path, device, retrain=False):
    """Build the network after set_seed(seed), as the paper's scripts do, then train PDE-only
    or load the saved state at `path`."""
    set_seed(seed)
    model = make_model().to(device)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"loaded {path}")
    else:
        train(model, pde_fn, x_col, x_bc, torch.zeros(len(x_bc), 1, device=device),
              adam_epochs=adam, lbfgs_max_iter=lbfgs, lr=lr, lambdas=(1.0, 0.0, 0.0), print_every=0)
        torch.save(model.state_dict(), path)
        print(f"trained and saved {path}")
    model.eval()
    return model


def jacobians(model, pde_fn, layers, x_out, x_res, chunk=128):
    """Psi_n on x_out and Phi_n on x_res for the given Linear layers, computed in chunks of points."""
    params = get_target_params(model, layers)
    ids = {id(p) for p in params}
    names = [n for n, p in model.named_parameters() if id(p) in ids]
    static = {n: p for n, p in model.named_parameters() if id(p) not in ids}

    def pd(*p):
        return {**static, **dict(zip(names, p))}

    def jac(fn, x):
        rows = []
        for i in range(0, len(x), chunk):
            xc = x[i:i + chunk].clone().requires_grad_(True)
            rows.append(compute_jacobian(lambda *p: fn(pd(*p), xc), params))
        return torch.cat(rows)

    Psi = jac(lambda d, xc: functional_call(model, d, xc).view(-1), x_out)
    Phi = jac(lambda d, xc: pde_fn(model, xc, param_dict=d).view(-1), x_res)
    return Psi / len(x_out) ** 0.5, Phi / len(x_res) ** 0.5, sum(p.numel() for p in params)


def gamma_spectrum(Psi_n, Phi_n, psi_cut, gram=False):
    """Return gamma (ascending), du [k, N] (unit RMS), dr [k, N_r] (L du at the residual points),
    ||w_i|| per unit RMS du_i, k, and the number of directions with gamma = 0 by counting.

    gram=True takes U and S of Psi_n from the eigendecomposition of the N x N Gram matrix
    Psi_n Psi_n^T and forms M = (Phi_n Psi_n^T) U_k / S_k^2, which equals Phi_n V_k / S_k
    without the SVD of the wide N x d matrix Psi_n (much faster for large N and d). The
    Gram matrix squares the condition number; with psi_cut = 1e-6 the retained eigenvalues
    are at least 1e-12 of the largest, well above float64 rounding."""
    n_out, n_res = len(Psi_n), len(Phi_n)
    if gram:
        ev, U = torch.linalg.eigh(Psi_n @ Psi_n.T)
        ev, U = ev.flip(0).clamp(min=0), U.flip(1)
        S = ev.sqrt()
        k = int((S > psi_cut * S[0]).sum())
        M = (Phi_n @ Psi_n.T) @ U[:, :k] / S[:k] ** 2
    else:
        U, S, Vh = torch.linalg.svd(Psi_n, full_matrices=False)
        k = int((S > psi_cut * S[0]).sum())
        M = Phi_n @ Vh[:k].T / S[:k]
    _, sig, Bh = torch.linalg.svd(M, full_matrices=True)
    n_count = max(0, k - len(sig))
    if n_count:
        sig = torch.cat([sig, torch.zeros(n_count, dtype=sig.dtype, device=sig.device)])
    o = torch.argsort(sig)
    Z = Bh[o]
    gamma = (sig[o] ** 2).cpu().numpy()
    du = (Z @ U[:, :k].T).cpu().numpy() * n_out ** 0.5
    dr = (Z @ M.T).cpu().numpy() * n_res ** 0.5
    wnorm = (Z / S[:k]).norm(dim=1).cpu().numpy()
    for i in range(k):  # sign convention: largest |du| positive
        s = np.sign(du[i, np.argmax(np.abs(du[i]))])
        du[i] *= s
        dr[i] *= s
    return gamma, du, dr, wnorm, k, n_count


def deff_ntk(Psi_n, Phi_n, lam, rcond):
    """d_eff^pde from the same Jacobians with the formula of adr.deff (NTK engine)."""
    N = len(Psi_n)
    K = torch.cat([Psi_n, lam**0.5 * Phi_n])
    G = K @ K.T
    scale = G.diagonal().mean().clamp(min=1e-30).sqrt()
    P = G @ (torch.linalg.pinv(G / scale, rcond=rcond, hermitian=True) / scale)
    return torch.sum(P[:N, :N] ** 2).item()


def report(gamma, k, n_count, d, n_out, n_res, psi_cut, lam, rcond, Psi_n, Phi_n, same_points):
    print(f"d = {d} parameters, {n_out} output points, {n_res} residual points")
    print(f"output-resolvable directions (psi cut {psi_cut:g}): k = {k}"
          + (f"  ({n_count} with gamma = 0 by counting: k > N_res)" if n_count else ""))
    print(f"d_eff^pde from gamma (lam = {lam:g}): {np.sum(1.0 / (1.0 + lam * gamma) ** 2):.4f}")
    if same_points:
        print(f"d_eff^pde from the NTK formula on the same points (rcond {rcond:g}): "
              f"{deff_ntk(Psi_n, Phi_n, lam, rcond):.4f}")
