"""
Experiment: the gamma directions behind d_eff^pde for 2D Poisson (Section 6.1)

Trains the Section 6.1 base model (ResNet 3 x 100, Tanh or SiLU, seed 42, 15 x 15
collocation grid by default, --n-col to change it, PDE-only, Adam 10k + L-BFGS 30k) and computes the Rayleigh
quotients gamma_i and function changes du_i of Section 2.4 (see gamma_utils.py).

The kernel of the Laplacian is every harmonic function, so there is no gap after a
fixed number of directions: the small-gamma directions should be (approximately)
harmonic and ordered roughly by harmonic degree. Each direction is labelled by its
harmonic fraction h (share of du_i in the span of Re/Im (x + iy)^k, k <= 25) and its
leading harmonic polynomials.

Output and residual Jacobians are evaluated on the same dense grid (default 41 x 41).
At the 15 x 15 training points (--residual-grid colloc) the number of
output-resolvable directions can exceed 225, and the excess gets gamma = 0 by
counting alone (directions that keep the residual zero only at those points).

Run:
    python examples/exp_gamma_directions_2d.py                     # Tanh, all layers
    python examples/exp_gamma_directions_2d.py --act SiLU
    python examples/exp_gamma_directions_2d.py --layers 3          # output layer only
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adr.pinn import ResNet
from adr.utils import use_float64
from exp_layer_selection import pde_fn
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
K_HARM = 25


def grid(n, device):
    g = torch.linspace(-1, 1, n, device=device)
    xg, yg = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([xg.flatten(), yg.flatten()], 1)


def harmonic_design(xy, K):
    """Re/Im (x + iy)^k, k <= K, each scaled to unit RMS on the grid, and their names."""
    z = xy[:, 0] + 1j * xy[:, 1]
    cols, names = [np.ones(len(z))], ["1"]
    low = {1: ("x", "y"), 2: ("x^2-y^2", "2xy"), 3: ("x^3-3xy^2", "3x^2y-y^3")}
    for k in range(1, K + 1):
        cols += [(z**k).real, (z**k).imag]
        names += list(low.get(k, (f"\\mathrm{{Re}}\\,z^{{{k}}}", f"\\mathrm{{Im}}\\,z^{{{k}}}")))
    A = np.stack(cols, 1)
    return A / np.sqrt((A**2).mean(0)), names


def harmonic_fraction(du, A):
    Q, _ = np.linalg.qr(A)
    return ((du @ Q) ** 2).sum(1) / (du**2).sum(1)


def harmonic_label(du, A, names, n_terms=2, rel=0.3):
    c, *_ = np.linalg.lstsq(A, du / np.sqrt((du**2).mean()), rcond=None)
    o = np.argsort(-np.abs(c))
    terms = [(names[j], c[j]) for j in o[:n_terms] if abs(c[j]) >= rel * abs(c[o[0]])]
    s = ""
    for i, (nm, cj) in enumerate(terms):
        s += ("-" if cj < 0 else "") if i == 0 else (" - " if cj < 0 else " + ")
        s += f"{abs(cj):.2f}\\,({nm})"
    return f"${s}$"


# Figure file names used in the paper, keyed by the run name
PAPER_NAMES = {"poisson_ResNet_Tanh_d3_w100_s42_col20_layers0123_dense": "gamma_directions_poisson"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", choices=list(ACTS), default="Tanh")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="Linear-layer indices (default: all)")
    ap.add_argument("--n-col", type=int, default=15, help="training collocation grid points per side")
    ap.add_argument("--residual-grid", choices=["dense", "colloc"], default="dense")
    ap.add_argument("--n-dense", type=int, default=41, help="dense grid points per side")
    ap.add_argument("--psi-cut", type=float, default=1e-6)
    ap.add_argument("--n-show", type=int, default=16, help="smallest-gamma directions to plot")
    ap.add_argument("--n-nonharm", type=int, default=8,
                    help="first non-harmonic directions (h < 0.5) to plot in an extra row")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rcond", type=float, default=1e-8)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gamma_directions"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_col = grid(args.n_col, device)
    x_dense = grid(args.n_dense, device)
    x_bc = torch.tensor([[-1.0, -1.0]], device=device)
    make_model = lambda: ResNet(n_in=2, n_out=1, hidden=args.width, depth=args.depth, activation=ACTS[args.act])
    tag = f"poisson_ResNet_{args.act}_d{args.depth}_w{args.width}_s{args.seed}" + \
          ("" if args.n_col == 15 else f"_col{args.n_col}")
    model = trained_model(make_model, pde_fn, x_col, x_bc, 10_000, 30_000, 5e-3, args.seed,
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

    xy = x_dense.cpu().numpy()
    A, names = harmonic_design(xy, K_HARM)
    hf = harmonic_fraction(du, A)
    print(f"directions with harmonic fraction > 0.9: {int(np.sum(hf > 0.9))} of {k}")
    n_show = min(args.n_show, k)
    print(f"\n  i   gamma        ||w||      harm.frac   leading harmonic polynomials")
    labels = []
    for i in range(n_show):
        labels.append(harmonic_label(du[i], A, names))
        print(f"  {i:<3d} {gamma[i]:.4e}   {wnorm[i]:.2e}   {hf[i]:.3f}       {labels[-1]}")

    nonharm = [i for i in range(k) if hf[i] < 0.5][:args.n_nonharm]
    print(f"\nfirst non-harmonic directions (h < 0.5):")
    for i in nonharm:
        print(f"  {i:<3d} {gamma[i]:.4e}   {wnorm[i]:.2e}   {hf[i]:.3f}")

    n = args.n_dense
    ncol = 8
    rows = [list(range(i, min(i + ncol, n_show))) for i in range(0, n_show, ncol)]
    if nonharm:
        rows.append(nonharm)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(2.4 * ncol, 3.0 * len(rows)),
                            squeeze=False, constrained_layout=True)
    letters = "abcdefghijklmnopqrstuvwxyz"
    panel = 0
    for r, idx in enumerate(rows):
        for c in range(ncol):
            a = axs[r, c]
            a.set_xticks([])
            a.set_yticks([])
            if c >= len(idx):
                a.axis("off")
                continue
            i = idx[c]
            v = du[i].reshape(n, n).T  # rows y, columns x
            m = np.abs(v).max()
            a.imshow(v, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-m, vmax=m)
            a.set_title(f"$\\gamma_{{{i}}}$ = {gamma[i]:.2g}", fontsize=11)
            a.text(0.5, -0.06, f"({letters[panel]})", transform=a.transAxes, ha="center", va="top", fontsize=12)
            panel += 1
    # The paper's configuration is written under its name in the paper
    path = os.path.join(args.out, PAPER_NAMES.get(name, name) + ".pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
