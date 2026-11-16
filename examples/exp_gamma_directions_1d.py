"""
Experiment: the gamma directions behind d_eff^pde for the 1D ODEs of Table 1

Trains the Table 1 network (PDE-only, seed 42, 100 collocation points,
Adam 3000 + L-BFGS 5000) for d^n u/dx^n = f on [-1, 1] and computes the Rayleigh
quotients gamma_i and function changes du_i of Section 2.4 (see gamma_utils.py).

The first n directions should be the kernel, polynomials of degree < n, with
gamma ~ 0. The next ones should follow the continuum problem with natural boundary
conditions:  n = 1  Neumann eigenvalues (k pi / 2)^2,
             n = 2  free-free beam, gamma = beta^4 with cos(2 beta) cosh(2 beta) = 1.

Output and residual Jacobians are evaluated on the same dense grid (default 401
points; --residual-grid colloc evaluates the residual at the 100 training points).
Writes a figure: (a) kernel directions, (b) next directions with theory,
(c) |L du_i| on the residual grid.

Run:
    python examples/exp_gamma_directions_1d.py                 # 2nd order, Tanh MLP 3 x 100
    python examples/exp_gamma_directions_1d.py --order 1 --depth 1 --width 50
    python examples/exp_gamma_directions_1d.py --layers 0      # input layer only
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr.pinn import MLP, ResNet
from adr.utils import use_float64
from exp_invariance import ode_residual
from gamma_utils import trained_model, jacobians, gamma_spectrum, report

use_float64()
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "mathtext.fontset": "cm",
    }
)

ACTS = {"Tanh": nn.Tanh, "SiLU": nn.SiLU}
ARCHS = {"MLP": MLP, "ResNet": ResNet}


def theory(order, n):
    """Continuum non-kernel gamma on [-1, 1] with natural BCs (None if not implemented)."""
    if order == 1:
        return [(k * np.pi / 2) ** 2 for k in range(1, n + 1)]
    if order == 2:
        f = lambda b: np.cos(2 * b) * np.cosh(2 * b) - 1.0
        return [brentq(f, (j + 0.5) * np.pi / 2 - 0.4, (j + 0.5) * np.pi / 2 + 0.4) ** 4
                for j in range(1, n + 1)]
    return None


# Figure file names used in the paper, keyed by the run name
PAPER_NAMES = {"ode2_MLP_Tanh_d3_w100_s42_layers0123_dense": "gamma_directions_ode2"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--arch", choices=list(ARCHS), default="MLP")
    ap.add_argument("--act", choices=list(ACTS), default="Tanh")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="Linear-layer indices (default: all)")
    ap.add_argument("--residual-grid", choices=["dense", "colloc"], default="dense")
    ap.add_argument("--n-dense", type=int, default=401)
    ap.add_argument("--psi-cut", type=float, default=1e-6)
    ap.add_argument("--n-show", type=int, default=None, help="directions to plot (default order + 4)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rcond", type=float, default=1e-8)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gamma_directions"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pde_fn = ode_residual(args.order)
    x_col = torch.linspace(-1, 1, 100, device=device).view(-1, 1)
    x_dense = torch.linspace(-1, 1, args.n_dense, device=device).view(-1, 1)
    x_bc = torch.tensor([[-1.0]], device=device)
    make_model = lambda: ARCHS[args.arch](n_in=1, n_out=1, hidden=args.width, depth=args.depth, activation=ACTS[args.act])
    tag = f"ode{args.order}_{args.arch}_{args.act}_d{args.depth}_w{args.width}_s{args.seed}"
    model = trained_model(make_model, pde_fn, x_col, x_bc, 3000, 5000, 5e-3, args.seed,
                          os.path.join(args.out, tag + ".pt"), device, args.retrain)

    n_linear = args.depth + 1
    layers = args.layers if args.layers is not None else list(range(n_linear))
    x_res = x_dense if args.residual_grid == "dense" else x_col
    Psi_n, Phi_n, d = jacobians(model, pde_fn, layers, x_dense, x_res)
    gamma, du, dr, wnorm, k, n_count = gamma_spectrum(Psi_n, Phi_n, args.psi_cut)

    name = f"{tag}_layers{''.join(str(l % n_linear) for l in layers)}_{args.residual_grid}"
    print(f"\n{name}")
    report(gamma, k, n_count, d, len(x_dense), len(x_res), args.psi_cut, args.lam, args.rcond,
           Psi_n, Phi_n, args.residual_grid == "dense")

    xs = x_dense.view(-1).cpu().numpy()
    P = np.stack([xs**j for j in range(args.order)], 1)
    th = theory(args.order, 10)
    n_show = min(args.n_show or args.order + 4, k)
    print(f"\n  i   gamma        ||w||      misfit to degree<{args.order} polynomial   theory")
    for i in range(n_show):
        c, *_ = np.linalg.lstsq(P, du[i], rcond=None)
        mis = np.linalg.norm(du[i] - P @ c) / np.linalg.norm(du[i])
        j = i - args.order
        t = f"{th[j]:.4g}" if th and 0 <= j < len(th) else ""
        print(f"  {i:<3d} {gamma[i]:.4e}   {wnorm[i]:.2e}   {mis:.1e}                          {t}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    cols = plt.cm.tab10.colors
    for i in range(min(args.order, k)):
        ax[0].plot(xs, du[i], color=cols[i], lw=1.8, label=f"$\\gamma_{{{i}}}$ = {gamma[i]:.1e}")
    # Same rounding as the text: one decimal below 100, integers above
    fmt = lambda v: f"{v:.1f}" if v < 100 else f"{v:.0f}"
    for i in range(args.order, n_show):
        j = i - args.order
        lbl = f"$\\gamma_{{{i}}}$ = {fmt(gamma[i])}" + (f" (theory {fmt(th[j])})" if th and j < len(th) else "")
        ax[1].plot(xs, du[i], color=cols[i % 10], lw=1.4, label=lbl)
    xr = x_res.view(-1).cpu().numpy()
    for i in range(n_show):
        ax[2].semilogy(xr, np.maximum(np.abs(dr[i]), 1e-300), lw=1.0, color=cols[i % 10],
                       label=f"$\\gamma_{{{i}}}$ = {gamma[i]:.1e}")
    for a, lbl in zip(ax, ["(a)", "(b)", "(c)"]):
        a.set_xlabel("$x$")
        a.legend(fontsize=9)
        a.text(0.5, -0.18, lbl, transform=a.transAxes, ha="center", va="top", fontsize=12)
    ax[0].set_ylabel(r"$\delta u_i$")
    ax[1].set_ylabel(r"$\delta u_i$")
    ax[2].set_ylabel(r"$|\mathcal{L}\,\delta u_i|$")
    # The paper's configuration is written under its name in the paper
    path = os.path.join(args.out, PAPER_NAMES.get(name, name) + ".pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
