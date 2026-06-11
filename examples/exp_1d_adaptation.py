"""
Experiment: 1D Boundary Adaptation via Subspace Projection (Figure 1)

Trains several PINNs on the 2nd-order ODE

    u'' + (2pi)^2 sin(2pi x) = 0,   x in [-1, 1]

using different random seeds and the PDE residual alone (no BC loss).
Because the ODE has a 2D null space (u = sin(2pi x) + C1 x + C2), each
seed converges to a distinct member of the solution manifold.

The final trained model (highlighted in red) is then adapted to three BC
scenarios using null-space subspace projection with corrector=False:
  (i)   Symmetric Dirichlet:     u(-1)=0,   u(1)=0
  (ii)  Asymmetric Dirichlet:    u(-1)=10,  u(1)=2
  (iii) Mixed Dirichlet+Neumann: u(-1)=-5,  u'(1)=8

Exact solutions for all three scenarios are shown as solid reference lines.

Output: examples/pinn_1d_sine_comparison.pdf

Run:
    python examples/exp_1d_adaptation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from torch.func import functional_call

from adr.pinn import MLP
from adr.training import train
from adr.adaptation import specify_bcs
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
# Hyperparameters
# ---------------------------------------------------------------------------

SEEDS = [0, 10, 20, 30, 40]  # last seed is the adaptation model

ADAM_EPOCHS = 10_000
LBFGS_MAX_ITER = 30_000
LR = 5e-3

N_PDE = 100
HIDDEN = 100
DEPTH = 1  # 1 hidden layer -> 2 nn.Linear layers total

NUM_INCREMENTS = 100  # predictor steps per adaptation

# ---------------------------------------------------------------------------
# PDE residual:  u'' + (2pi)^2 sin(2pi x) = 0
# ---------------------------------------------------------------------------


def pde_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(
            xi
        )[0]
        return u_xx + (4.0 * np.pi**2) * torch.sin(2.0 * np.pi * xi[0])

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def exact_u(x_np, C1=0.0, C2=0.0):
    """General solution family: sin(2pi x) + C1 x + C2."""
    return np.sin(2.0 * np.pi * x_np) + C1 * x_np + C2


# ---------------------------------------------------------------------------
# BC scenarios and their analytical constants
#
#  General solution:  u(x) = sin(2pi x) + C1 x + C2
#  u(-1) = -C1 + C2,   u(1) = C1 + C2
#  u'(x) = 2pi cos(2pi x) + C1,  u'(1) = 2pi + C1
#
#  (i)   u(-1)=0,  u(1)=0   -> C1=0,      C2=0
#  (ii)  u(-1)=10, u(1)=2   -> C1=-4,     C2=6
#  (iii) u(-1)=-5, u'(1)=8  -> C1=8-2pi,  C2=3-2pi
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label": r"$u(-1)=0,\;u(1)=0$",
        "color": "tab:green",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": 0.0},
            {"coords": [1.0], "type": "dirichlet", "val": 0.0},
        ],
        "C1": 0.0,
        "C2": 0.0,
        "dirichlet_pts": [(-1.0, 0.0), (1.0, 0.0)],
    },
    {
        "label": r"$u(-1)=10,\;u(1)=2$",
        "color": "tab:orange",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": 10.0},
            {"coords": [1.0], "type": "dirichlet", "val": 2.0},
        ],
        "C1": -4.0,
        "C2": 6.0,
        "dirichlet_pts": [(-1.0, 10.0), (1.0, 2.0)],
    },
    {
        "label": r"$u(-1)=-5,\;u'(1)=8$",
        "color": "tab:purple",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": -5.0},
            {"coords": [1.0], "type": "neumann", "val": 8.0},
        ],
        "C1": 8.0 - 2.0 * np.pi,
        "C2": 3.0 - 2.0 * np.pi,
        "dirichlet_pts": [(-1.0, -5.0)],
    },
]

ALL_LAYERS = [0, 1]  # both nn.Linear layers in a depth=1 MLP
LAST_LAYER = [1]  # output layer only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    x_pde = torch.linspace(-1, 1, N_PDE, device=device).view(-1, 1)
    x_bc_dummy = torch.tensor([[-1.0], [1.0]], device=device)
    y_bc_dummy = torch.zeros(2, 1, device=device)

    x_plot = torch.linspace(-1, 1, 500, device=device).view(-1, 1)
    x_np = x_plot.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Phase 1: train multiple base models (PDE-only)
    # ------------------------------------------------------------------
    base_preds = []
    adapt_model = None

    for i, seed in enumerate(SEEDS):
        is_adapt = i == len(SEEDS) - 1
        label = f"Seed {seed}" + (" [ADAPT]" if is_adapt else "")
        print(f"{'='*55}\n  {label}\n{'='*55}")

        set_seed(seed)
        model = MLP(n_in=1, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=nn.Tanh).to(
            device
        )

        train(
            model,
            pde_fn,
            x_pde,
            x_bc_dummy,
            y_bc_dummy,
            adam_epochs=ADAM_EPOCHS,
            lbfgs_max_iter=LBFGS_MAX_ITER,
            lr=LR,
            lambdas=(1.0, 0.0, 0.0),
            print_every=0,
        )

        x_pde_t = x_pde.clone().requires_grad_(True)
        base_res = pde_fn(model, x_pde_t)
        base_pde = torch.mean(base_res**2).item()
        print(f"  Post-train PDE residual (MSE): {base_pde:.2e}")

        with torch.no_grad():
            u = model(x_plot).cpu().numpy().flatten()
        base_preds.append(u)

        if is_adapt:
            adapt_model = model

    # ------------------------------------------------------------------
    # d_eff metrics for the adaptation base model
    # ------------------------------------------------------------------
    print(f"\n{'#'*55}\n  d_eff metrics (adaptation base)\n{'#'*55}")
    for layer_idx in ALL_LAYERS:
        x_pde_t = x_pde.clone().requires_grad_(True)
        d = get_d_eff(
            adapt_model,
            pde_fn,
            [layer_idx],
            x_pde_t,
            x_bc_dummy,
            mode="all",
            engine="ntk",
        )
        print(
            f"  Layer {layer_idx}: d_eff_pde={d['pde']:.3f}  d_eff_bc={d['bc']:.3f}  d_eff_total={d['total']:.3f}"
        )
    x_pde_t = x_pde.clone().requires_grad_(True)
    d_all = get_d_eff(
        adapt_model, pde_fn, ALL_LAYERS, x_pde_t, x_bc_dummy, mode="all", engine="ntk"
    )
    print(
        f"  All layers: d_eff_pde={d_all['pde']:.3f}  d_eff_bc={d_all['bc']:.3f}  d_eff_total={d_all['total']:.3f}"
    )

    # ------------------------------------------------------------------
    # Phase 2: subspace projection -- corrector=False then corrector=True
    # ------------------------------------------------------------------
    def run_adaptation(use_corrector):
        label = "corrector=True" if use_corrector else "corrector=False"
        print(f"\n{'#'*55}\n  {label}\n{'#'*55}")
        preds, pde_res_list, times = [], [], []
        for sc in SCENARIOS:
            print(f"\n{'='*55}\n  Adapting to: {sc['label']}\n{'='*55}")
            t0 = time.perf_counter()
            specify_bcs(
                adapt_model,
                pde_fn,
                sc["bcs"],
                ALL_LAYERS,
                x_pde,
                num_increments=NUM_INCREMENTS,
                max_iter=15,
                tolerance=1e-16,
                use_corrector=use_corrector,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            with torch.no_grad():
                u_ad = adapt_model(x_plot).cpu().numpy().flatten()
            preds.append(u_ad)

            x_pde_t = x_pde.clone().requires_grad_(True)
            res = pde_fn(adapt_model, x_pde_t)
            pde_res = torch.mean(res**2).item()
            pde_res_list.append(pde_res)

            u_exact_np = exact_u(x_np, sc["C1"], sc["C2"])
            rmse = np.sqrt(np.mean((u_ad - u_exact_np) ** 2))
            print(
                f"  PDE residual (MSE): {pde_res:.2e}    RMSE (u): {rmse:.2e}    Time: {elapsed:.2f}s"
            )
        return preds, pde_res_list, times

    preds_no_corr, pde_res_no_corr, times_no_corr = run_adaptation(use_corrector=False)
    preds_with_corr, pde_res_with_corr, times_with_corr = run_adaptation(
        use_corrector=True
    )

    # ------------------------------------------------------------------
    # Phase 3: last-layer adaptation (1 increment -- map is linear, exact)
    # ------------------------------------------------------------------
    print(f"\n{'#'*55}\n  Last-layer adaptation\n{'#'*55}")
    preds_last, pde_res_last, times_last = [], [], []
    for sc in SCENARIOS:
        print(f"\n{'='*55}\n  Adapting to: {sc['label']}\n{'='*55}")
        t0 = time.perf_counter()
        specify_bcs(
            adapt_model,
            pde_fn,
            sc["bcs"],
            LAST_LAYER,
            x_pde,
            num_increments=1,
            max_iter=15,
            tolerance=1e-8,
            use_corrector=False,
        )
        elapsed = time.perf_counter() - t0
        times_last.append(elapsed)
        with torch.no_grad():
            u_ad = adapt_model(x_plot).cpu().numpy().flatten()
        preds_last.append(u_ad)

        x_pde_t = x_pde.clone().requires_grad_(True)
        res = pde_fn(adapt_model, x_pde_t)
        pde_res = torch.mean(res**2).item()
        pde_res_last.append(pde_res)

        u_exact_np = exact_u(x_np, sc["C1"], sc["C2"])
        rmse = np.sqrt(np.mean((u_ad - u_exact_np) ** 2))
        print(
            f"  PDE residual (MSE): {pde_res:.2e}    RMSE (u): {rmse:.2e}    Time: {elapsed:.2f}s"
        )

    # ------------------------------------------------------------------
    # Figure: 2x2 panels
    #   (a) base  |  (b) last-layer
    #   (c) full, no corrector  |  (d) full, with corrector
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    scenario_handles = [
        mlines.Line2D([], [], color=sc["color"], lw=2.0, label=sc["label"])
        for sc in SCENARIOS
    ]
    exact_handle = mlines.Line2D(
        [],
        [],
        color="k",
        lw=1.5,
        linestyle="--",
        alpha=0.9,
        label="Exact solution",
    )
    bc_handle = mlines.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        color="gray",
        markersize=7,
        markeredgecolor="k",
        label="Dirichlet target",
    )

    def plot_adaptation_panel(ax, preds, panel_label, show_legend=False):
        for sc, u_ad in zip(SCENARIOS, preds):
            u_exact = exact_u(x_np, sc["C1"], sc["C2"])
            ax.plot(x_np, u_ad, color=sc["color"], lw=2.0, alpha=0.4, zorder=3)
            ax.plot(
                x_np, u_exact, color="k", lw=1.5, linestyle="--", alpha=0.9, zorder=4
            )
            for xb, yb in sc["dirichlet_pts"]:
                ax.plot(
                    xb,
                    yb,
                    "o",
                    color=sc["color"],
                    markersize=7,
                    markeredgecolor="k",
                    markeredgewidth=0.7,
                    zorder=5,
                )
        if show_legend:
            ax.legend(
                handles=scenario_handles + [exact_handle, bc_handle], fontsize=8.5
            )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x)$")
        ax.set_xlim(-1, 1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.grid(True, alpha=0.25)
        ax.text(
            0.5,
            -0.13,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    # --- Panel (a): base solutions ---
    ax = axes[0, 0]
    for i, u in enumerate(base_preds):
        is_adapt = i == len(SEEDS) - 1
        if is_adapt:
            ax.plot(x_np, u, color="tab:red", lw=2.2, zorder=5)
            ax.plot(
                [-1.0, 1.0],
                [u[0], u[-1]],
                linestyle="none",
                marker="D",
                color="tab:red",
                markersize=7,
                zorder=6,
            )
        else:
            ax.plot(x_np, u, color="steelblue", lw=1.0, alpha=0.7)
    ax.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color="steelblue",
                lw=1.0,
                alpha=0.6,
                label="Other initializations",
            ),
            mlines.Line2D(
                [],
                [],
                color="tab:red",
                lw=2.2,
                marker="D",
                markersize=7,
                label="Adaptation base",
            ),
        ],
        fontsize=9,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_xlim(-1, 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.5, -0.13, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=13
    )

    # --- Panel (b): full projection, no corrector ---
    plot_adaptation_panel(axes[0, 1], preds_no_corr, "(b)")

    # --- Panel (c): last-layer ---
    plot_adaptation_panel(axes[1, 0], preds_last, "(c)")

    # --- Panel (d): full projection, with corrector ---
    plot_adaptation_panel(axes[1, 1], preds_with_corr, "(d)", show_legend=True)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "pinn_1d_sine_comparison.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col = 35
    print("\n" + "=" * 145)
    print(
        f"  {'Scenario':<{col}}  {'PDE MSE (last)':>14}  {'RMSE (last)':>11}  {'t (last)':>8}  {'PDE MSE (no corr)':>17}  {'RMSE (no corr)':>14}  {'t (no corr)':>11}  {'PDE MSE (corr)':>14}  {'RMSE (corr)':>11}  {'t (corr)':>8}"
    )
    print("=" * 145)
    for i, sc in enumerate(SCENARIOS):
        u_exact_np = exact_u(x_np, sc["C1"], sc["C2"])
        rmse_last = np.sqrt(np.mean((preds_last[i] - u_exact_np) ** 2))
        rmse_no = np.sqrt(np.mean((preds_no_corr[i] - u_exact_np) ** 2))
        rmse_yes = np.sqrt(np.mean((preds_with_corr[i] - u_exact_np) ** 2))
        print(
            f"  {sc['label']:<{col}}  {pde_res_last[i]:>14.2e}  {rmse_last:>11.2e}  {times_last[i]:>7.2f}s"
            f"  {pde_res_no_corr[i]:>17.2e}  {rmse_no:>14.2e}  {times_no_corr[i]:>10.2f}s"
            f"  {pde_res_with_corr[i]:>14.2e}  {rmse_yes:>11.2e}  {times_with_corr[i]:>7.2f}s"
        )


if __name__ == "__main__":
    run()
