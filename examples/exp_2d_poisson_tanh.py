"""
Experiment: 2D Poisson Adaptation -- Tanh Network (Figure 3 / Table 6)

Trains a 3-hidden-layer Tanh PINN on the 2D Poisson equation

    u_xx + u_yy = -2*pi^2 * sin(pi*x) * sin(pi*y),   (x,y) in [-1,1]^2

with PDE-only training (u=0 BCs suppressed during base training). The base
model is then adapted to non-homogeneous BCs

    u(x,-1)=0,  u(x,1)=sin(pi*x),  u(-1,y)=0,  u(1,y)=0

using subspace projection targeting individual and combined layers.

Output: examples/pinn_poisson_tanh.pdf

Run:
    python examples/exp_2d_poisson_tanh.py
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
from matplotlib.gridspec import GridSpec

from torch.func import functional_call

from adr.pinn import MLP
from adr.training import train
from adr.adaptation import specify_bcs
from adr.utils import set_seed, use_float64
from adr.deff import get_d_eff

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

N_PDE = 15
HIDDEN = 100
DEPTH = 3  # 4 nn.Linear layers total (indices -4 ... -1)
LR = 5e-3
ADAM_EPOCHS = 10_000
LBFGS_MAX_ITER = 30_000
NUM_INCS = 100

# Adaptation configs: (label, layer_indices)
# All 4 individual layers + last two combined; [-1] and [-2] are also plotted
ADAPT_CONFIGS = [
    ("Layer -1 (output)", [-1]),
    ("Layer -2 (penultimate)", [-2]),
    ("Layer -3", [-3]),
    ("Layer -4 (input)", [-4]),
    ("Layers -1 & -2", [-2, -1]),
    ("Layers -1, -2 & -3", [-3, -2, -1]),
    ("Layers -1, -2, -3 & -4", [-4, -3, -2, -1]),
]

# Which configs feed into the figure panels (b), (c), (d)
PLOT_KEYS = [(-1,), (-2,), (-2, -1)]


# ---------------------------------------------------------------------------
# PDE residual:  u_xx + u_yy + 2*pi^2 * sin(pi*x) * sin(pi*y) = 0
# ---------------------------------------------------------------------------


def pde_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(
            xi
        )[0]
        u_yy = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[1])(
            xi
        )[1]
        f = -2.0 * np.pi**2 * torch.sin(np.pi * xi[0]) * torch.sin(np.pi * xi[1])
        return u_xx + u_yy - f

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def compute_pde_loss(model, x_pde):
    res = pde_fn(model, x_pde)
    return torch.mean(res**2).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Domain setup
    # ------------------------------------------------------------------
    edge = torch.linspace(-1, 1, N_PDE, device=device)
    x_grid, y_grid = torch.meshgrid(edge, edge, indexing="ij")
    x_pde = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

    bc_bottom = torch.stack([edge, torch.full_like(edge, -1)], dim=1)
    bc_top = torch.stack([edge, torch.full_like(edge, 1)], dim=1)
    bc_left = torch.stack([torch.full_like(edge, -1), edge], dim=1)
    bc_right = torch.stack([torch.full_like(edge, 1), edge], dim=1)
    x_bc = torch.cat([bc_bottom, bc_top, bc_left, bc_right], dim=0)
    y_bc = torch.zeros(x_bc.shape[0], 1, device=device)

    # Test grid
    N_test = 50
    xt = torch.linspace(-1, 1, N_test)
    yt = torch.linspace(-1, 1, N_test)
    Xg, Yg = torch.meshgrid(xt, yt, indexing="ij")
    x_test = torch.stack([Xg.flatten(), Yg.flatten()], dim=1).to(device)

    # Exact solution: sin(pi*x)*sin(pi*y) + sin(pi*x)*sinh(pi*(y+1))/sinh(2*pi)
    U_exact = torch.sin(np.pi * Xg) * torch.sin(np.pi * Yg) + torch.sin(
        np.pi * Xg
    ) * torch.sinh(np.pi * (Yg + 1)) / np.sinh(2 * np.pi)
    U_exact_np = U_exact.numpy()

    # New BCs for adaptation
    bcs_new = []
    for val in edge:
        v = val.item()
        bcs_new.append({"type": "dirichlet", "coords": [v, -1.0], "val": 0.0})
        bcs_new.append(
            {"type": "dirichlet", "coords": [v, 1.0], "val": float(np.sin(np.pi * v))}
        )
        if not np.isclose(abs(v), 1.0):
            bcs_new.append({"type": "dirichlet", "coords": [-1.0, v], "val": 0.0})
            bcs_new.append({"type": "dirichlet", "coords": [1.0, v], "val": 0.0})

    # ------------------------------------------------------------------
    # Train base model (PDE only)
    # ------------------------------------------------------------------
    set_seed(42)
    model = MLP(n_in=2, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=nn.Tanh).to(
        device
    )

    print("Training base model (lambda_pde=1, lambda_bc=0)...")
    train(
        model,
        pde_fn,
        x_pde,
        x_bc,
        y_bc,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lr=LR,
        lambdas=(1.0, 0.0, 0.0),
        print_every=1000,
    )

    base_pde = compute_pde_loss(model, x_pde)
    print(f"\nBase PDE residual (MSE): {base_pde:.4e}\n")

    with torch.no_grad():
        preds_base = model(x_test).reshape(N_test, N_test).cpu()

    # ------------------------------------------------------------------
    # Adaptation: all configs
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}\n  Adaptation (all layer configs)\n{'#'*60}")

    table_data = []
    preds_store = {}  # keyed by tuple(layers)

    for label, layers in ADAPT_CONFIGS:
        print(f"\n{'='*55}\n  {label}  (layers={layers})\n{'='*55}")

        # d_eff
        d = get_d_eff(
            model, pde_fn, layers, x_pde, x_bc, mode="all", engine="ntk", rcond=1e-8
        )
        pde_cap = d["pde"]
        bc_cap = d["bc"]
        ratio = bc_cap / pde_cap if pde_cap > 0 else float("inf")
        print(f"  d_eff_pde={pde_cap:.3f}  d_eff_bc={bc_cap:.3f}  ratio={ratio:.3f}")

        t0 = time.perf_counter()
        specify_bcs(
            model,
            pde_fn,
            bcs_new,
            layers,
            x_pde,
            num_increments=NUM_INCS,
            use_corrector=True,
            tolerance=1e-8,
            max_iter=15,
            rcond=1e-8,
        )
        elapsed = time.perf_counter() - t0

        pde_loss = compute_pde_loss(model, x_pde)
        with torch.no_grad():
            preds = model(x_test).reshape(N_test, N_test).cpu()
        rmse = float(torch.sqrt(torch.mean((preds - U_exact) ** 2)).item())
        print(f"  PDE MSE: {pde_loss:.4e}    RMSE: {rmse:.4e}    Time: {elapsed:.2f}s")

        table_data.append(
            {
                "label": label,
                "layers": layers,
                "d_pde": pde_cap,
                "d_bc": bc_cap,
                "ratio": ratio,
                "pde_loss": pde_loss,
                "rmse": rmse,
                "time": elapsed,
            }
        )
        preds_store[tuple(layers)] = preds

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    w = 26
    print("\n\n" + "=" * 95)
    print("TABLE 6 SUMMARY (Tanh)")
    print("=" * 95)
    print(
        f"  {'Candidate Layer':<{w}} {'d_eff_pde':>10} {'d_eff_bc':>10} {'Ratio':>8} {'RMSE':>14} {'PDE MSE':>14} {'Time (s)':>10}"
    )
    print("-" * 107)
    for row in table_data:
        print(
            f"  {row['label']:<{w}} {row['d_pde']:>10.3f} {row['d_bc']:>10.3f} {row['ratio']:>8.3f} "
            f"{row['rmse']:>14.4e} {row['pde_loss']:>14.4e} {row['time']:>9.2f}s"
        )
    print("=" * 107 + "\n")

    # ------------------------------------------------------------------
    # Figure: 2x4 grid
    #   Row 0 (predictions): (a) base | (b) layer -1 | (c) layer -2 | (d) layers -1,-2
    #   Row 1 (errors):      [blank]  | (e) error -1 | (f) error -2 | (g) error -1,-2
    # ------------------------------------------------------------------
    preds_m1 = preds_store[(-1,)]
    preds_m2 = preds_store[(-2,)]
    preds_comb = preds_store[(-2, -1)]

    err_m1 = np.abs(preds_m1.numpy() - U_exact_np)
    err_m2 = np.abs(preds_m2.numpy() - U_exact_np)
    err_comb = np.abs(preds_comb.numpy() - U_exact_np)
    max_err = max(err_m1.max(), err_m2.max(), err_comb.max())

    x_np, y_np = Xg.numpy(), Yg.numpy()

    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[1, 3])

    levels_pred = np.linspace(-1.5, 1.5, 50)
    cmap_pred = "RdBu_r"

    c_a = ax_a.contourf(
        x_np,
        y_np,
        preds_base.numpy(),
        levels=levels_pred,
        cmap=cmap_pred,
        extend="both",
    )
    c_b = ax_b.contourf(
        x_np, y_np, preds_m1.numpy(), levels=levels_pred, cmap=cmap_pred, extend="both"
    )
    c_c = ax_c.contourf(
        x_np, y_np, preds_m2.numpy(), levels=levels_pred, cmap=cmap_pred, extend="both"
    )
    c_d = ax_d.contourf(
        x_np,
        y_np,
        preds_comb.numpy(),
        levels=levels_pred,
        cmap=cmap_pred,
        extend="both",
    )

    cbar_pred = fig.colorbar(
        c_d, ax=[ax_a, ax_b, ax_c, ax_d], location="right", aspect=20, pad=0.02
    )
    cbar_pred.set_label(r"$u(x,y)$", size=14, labelpad=10)
    cbar_pred.outline.set_linewidth(1.2)

    levels_err = np.linspace(0, max_err, 50)
    cmap_err = "magma_r"

    c_e = ax_e.contourf(
        x_np, y_np, err_m1, levels=levels_err, cmap=cmap_err, extend="max"
    )
    c_f = ax_f.contourf(
        x_np, y_np, err_m2, levels=levels_err, cmap=cmap_err, extend="max"
    )
    c_g = ax_g.contourf(
        x_np, y_np, err_comb, levels=levels_err, cmap=cmap_err, extend="max"
    )

    cbar_err = fig.colorbar(
        c_g, ax=[ax_e, ax_f, ax_g], location="right", aspect=20, pad=0.02
    )
    cbar_err.set_label(
        r"$|u_{\mathrm{pred}} - u_{\mathrm{exact}}|$", size=14, labelpad=10
    )
    cbar_err.outline.set_linewidth(1.2)

    panel_info = [
        (ax_a, "(a)"),
        (ax_b, "(b)"),
        (ax_c, "(c)"),
        (ax_d, "(d)"),
        (ax_e, "(e)"),
        (ax_f, "(f)"),
        (ax_g, "(g)"),
    ]
    for ax, lbl in panel_info:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.tick_params(
            axis="both", direction="in", top=True, right=True, length=6, width=1.0
        )
        ax.text(
            0.5, -0.18, lbl, transform=ax.transAxes, ha="center", va="top", fontsize=12
        )

    out_path = os.path.join(os.path.dirname(__file__), "pinn_poisson_tanh.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
