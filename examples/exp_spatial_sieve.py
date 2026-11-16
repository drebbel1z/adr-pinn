"""
Experiment: The Spatial Sieve and Spectral Hallucination

Trains a PINN on the 2nd-order ODE u'' + (2pi)^2 sin(2pi x) = 0 with strict
Dirichlet BCs u(-1) = u(1) = 1.  The unique analytical solution is
u(x) = sin(2pi x) + 1.

As grid density increases, d_eff_total (joint PDE + BC) collapses from ~3
(under-constrained, free to hallucinate) to ~0 (fully locked).  The discrete
PDE loss reaches machine zero at every density, but the continuous PDE loss and
RMSE expose whether the network is actually physical between collocation points.

Run:
    python examples/exp_spatial_sieve.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import blended_transform_factory

from torch.func import functional_call

from adr.pinn import MLP, ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64

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


# ---------------------------------------------------------------------------
# PDE and exact solution
# ---------------------------------------------------------------------------


def pde_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        return u_xx + (4.0 * np.pi**2) * torch.sin(2.0 * np.pi * xi[0])

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def exact_solution(x):
    return torch.sin(2.0 * np.pi * x) + 1.0


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Spatial Sieve Study -- device: {device}\n")

    x_dense = torch.linspace(-1, 1, 1000, device=device).view(-1, 1)
    u_exact = exact_solution(x_dense)

    x_bc = torch.tensor([[-1.0], [1.0]], device=device)
    y_bc = torch.tensor([[1.0], [1.0]], device=device)

    grid_densities = [6, 10, 14, 18, 22, 26]

    model_classes = [
        {"name": "Feedforward", "cls": MLP},
        {"name": "Skip Connection", "cls": ResNet},
    ]

    results = []

    for mc in model_classes:
        print(f"\n{'#'*60}")
        print(f"  ARCHITECTURE: {mc['name']}")
        print(f"{'#'*60}")

        for n_pts in grid_densities:
            print(f"\n{'='*55}")
            print(f"  Grid points: {n_pts}")
            print(f"{'='*55}")

            set_seed(42)

            x_pde = torch.linspace(-1, 1, n_pts, device=device).view(-1, 1)

            model = mc["cls"](
                n_in=1, n_out=1, hidden=100, depth=4, activation=nn.SiLU
            ).to(device)

            n_linear = 5  # depth=4 -> 4 hidden + 1 output linear layers
            all_layers = list(range(n_linear))

            # Pre-training d_eff_total
            x_pde_t = x_pde.clone().requires_grad_(True)
            d_pre = get_d_eff(
                model, pde_fn, all_layers, x_pde_t, x_bc, mode="total", engine="ntk"
            )
            print(f"  Pre-train  d_eff_total = {d_pre:.4f}")

            # Train with both PDE and BC losses
            train(
                model,
                pde_fn,
                x_pde,
                x_bc,
                y_bc,
                adam_epochs=2000,
                lbfgs_max_iter=5000,
                lambdas=(1.0, 1.0, 0.0),
                print_every=0,
            )

            # Post-training d_eff_total
            x_pde_t = x_pde.clone().requires_grad_(True)
            d_post = get_d_eff(
                model, pde_fn, all_layers, x_pde_t, x_bc, mode="total", engine="ntk"
            )
            print(f"  Post-train d_eff_total = {d_post:.4f}")

            # Post-training d_eff_pde (PDE only, no BC)
            x_pde_t = x_pde.clone().requires_grad_(True)
            d_pde = get_d_eff(
                model, pde_fn, all_layers, x_pde_t, x_bc, mode="pde", engine="ntk"
            )
            print(f"  Post-train d_eff_pde   = {d_pde:.4f}")

            # Discrete PDE loss (on training grid)
            x_pde_eval = x_pde.clone().requires_grad_(True)
            res_disc = pde_fn(model, x_pde_eval)
            disc_loss = torch.mean(res_disc**2).item()

            # Continuous PDE loss (dense grid)
            x_dense_t = x_dense.clone().requires_grad_(True)
            res_cont = pde_fn(model, x_dense_t)
            cont_loss = torch.mean(res_cont**2).item()

            # Continuous RMSE vs exact
            with torch.no_grad():
                u_pred = model(x_dense)
                rmse = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()

            print(f"  Discrete PDE loss  = {disc_loss:.2e}")
            print(f"  Continuous PDE loss = {cont_loss:.2e}")
            print(f"  Continuous RMSE     = {rmse:.2e}")

            results.append(
                {
                    "architecture": mc["name"],
                    "grid_pts": n_pts,
                    "d_eff_pre": round(d_pre, 3),
                    "d_eff_post": round(d_post, 3),
                    "d_eff_pde": round(d_pde, 3),
                    "disc_pde_loss": disc_loss,
                    "cont_pde_loss": cont_loss,
                    "cont_rmse": rmse,
                    "u_pred_np": u_pred.cpu().numpy().flatten(),
                }
            )

    df = pd.DataFrame(results)
    print("\n\n" + "=" * 90)
    print("RESULTS: SPATIAL SIEVE AND SPECTRAL HALLUCINATION")
    print("=" * 90)
    pd.set_option("display.float_format", "{:.2e}".format)
    print(df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figure: 2x3 panels per architecture, one PDF each
    # ------------------------------------------------------------------
    x_np = x_dense.cpu().numpy().flatten()
    u_exact_np = u_exact.cpu().numpy().flatten()
    letters = "abcdef"

    exact_handle = mlines.Line2D([], [], color="k", lw=1.8, label="Exact solution")
    pred_handle = mlines.Line2D(
        [], [], color="tab:red", lw=1.8, linestyle="--", label="PINN"
    )

    file_tags = {"Feedforward": "feedforward", "Skip Connection": "skipconn"}

    for mc in model_classes:
        arch_rows = [r for r in results if r["architecture"] == mc["name"]]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axes = axes.flatten()

        for idx, r in enumerate(arch_rows):
            ax = axes[idx]
            n_pts = r["grid_pts"]
            u_pred_np = r["u_pred_np"]

            ax.plot(x_np, u_exact_np, "k-", lw=1.8, zorder=4)
            ax.fill_between(
                x_np, u_exact_np, u_pred_np, alpha=0.18, color="tab:red", zorder=2
            )
            ax.plot(x_np, u_pred_np, color="tab:red", lw=1.8, linestyle="--", zorder=3)

            # Collocation rug along the bottom of the axes
            x_pde_np = np.linspace(-1, 1, n_pts)
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(
                x_pde_np,
                np.zeros(n_pts),
                "|",
                transform=trans,
                color="steelblue",
                markersize=9,
                markeredgewidth=1.5,
                zorder=5,
                clip_on=False,
            )

            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(x)$")
            ax.set_xlim(-1, 1)
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.tick_params(
                axis="both", direction="in", top=True, right=True, length=5, width=1.0
            )
            ax.grid(True, alpha=0.25)
            ax.text(
                0.5,
                -0.13,
                rf"({letters[idx]}) $N_p = {n_pts}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=13,
            )

            if idx == 0:
                rug_handle = mlines.Line2D(
                    [],
                    [],
                    color="steelblue",
                    marker="|",
                    linestyle="none",
                    markersize=9,
                    markeredgewidth=1.5,
                    label="Collocation pts",
                )
                ax.legend(handles=[exact_handle, pred_handle, rug_handle], fontsize=9)

        tag = file_tags[mc["name"]]
        out_path = os.path.join(os.path.dirname(__file__), f"spatial_sieve_{tag}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved -> {out_path}")
        plt.close(fig)

    return df


if __name__ == "__main__":
    run()
