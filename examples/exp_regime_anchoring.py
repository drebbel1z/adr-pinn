"""
Experiment: Regime Anchoring -- Warm-Start Base and Physics Null Space (Figure 2)

Trains a PINN on u'' + (2pi)^2 sin(2pi x) = 0 with a small BC loss
(lambda_pde=1, lambda_bc=1e-3) anchoring the base model near u=1 at both
boundaries.  The right panel draws random vectors from the physics null
space and plots the corresponding solution family, illustrating that the
model retains ample freedom to satisfy arbitrary BCs while preserving the
learned physics.

Output: examples/pinn_1d_sine_comparison_kinda_close_bc.pdf

Run:
    python examples/exp_regime_anchoring.py
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
from adr.utils import set_seed, use_float64, get_target_params, build_param_dict

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

ADAM_EPOCHS = 10_000
LBFGS_MAX_ITER = 37_500
LR = 5e-3

N_PDE = 100
HIDDEN = 100
DEPTH = 1
ALL_LAYERS = [0, 1]
LAST_LAYER = [1]

BC_ANCHOR = 1.0  # warm-start target: u=1 at both boundaries
NUM_SAMPLES = 20  # null-space curves to draw
SAMPLE_SCALE = 1  # magnitude of random perturbations
NUM_INCREMENTS = 100  # predictor steps per adaptation

# BC scenarios (same as exp_1d_adaptation.py)
SCENARIOS = [
    {
        "label": r"$u(-1)=0,\;u(1)=0$",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": 0.0},
            {"coords": [1.0], "type": "dirichlet", "val": 0.0},
        ],
        "C1": 0.0,
        "C2": 0.0,
    },
    {
        "label": r"$u(-1)=10,\;u(1)=2$",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": 10.0},
            {"coords": [1.0], "type": "dirichlet", "val": 2.0},
        ],
        "C1": -4.0,
        "C2": 6.0,
    },
    {
        "label": r"$u(-1)=-5,\;u'(1)=8$",
        "bcs": [
            {"coords": [-1.0], "type": "dirichlet", "val": -5.0},
            {"coords": [1.0], "type": "neumann", "val": 8.0},
        ],
        "C1": 8.0 - 2.0 * np.pi,
        "C2": 3.0 - 2.0 * np.pi,
    },
]


# ---------------------------------------------------------------------------
# PDE residual:  u'' + (2pi)^2 sin(2pi x) = 0
# ---------------------------------------------------------------------------


def pde_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        return u_xx + (4.0 * np.pi**2) * torch.sin(2.0 * np.pi * xi[0])

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    x_pde = torch.linspace(-1, 1, N_PDE, device=device).view(-1, 1)
    x_bc = torch.tensor([[-1.0], [1.0]], device=device)
    y_bc = torch.full((2, 1), BC_ANCHOR, dtype=torch.float64, device=device)

    x_plot = torch.linspace(-1, 1, 500, device=device).view(-1, 1)
    x_np = x_plot.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Train warm-start base model
    # ------------------------------------------------------------------
    set_seed(40)
    model = MLP(n_in=1, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=nn.Tanh).to(
        device
    )

    print("Training regime-anchored base model (lambda_pde=1, lambda_bc=1e-3)...")
    train(
        model,
        pde_fn,
        x_pde,
        x_bc,
        y_bc,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lr=LR,
        lambdas=(1.0, 1e-3, 0.0),
        print_every=0,
    )

    x_pde_t = x_pde.clone().requires_grad_(True)
    base_pde = torch.mean(pde_fn(model, x_pde_t) ** 2).item()
    print(f"Post-train PDE residual (MSE): {base_pde:.2e}")
    with torch.no_grad():
        bc_vals = model(x_bc).cpu().numpy().flatten()
    print(f"BC values at base: u(-1)={bc_vals[0]:.4f}  u(1)={bc_vals[1]:.4f}")

    with torch.no_grad():
        u_base = model(x_plot).cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Physics null-space sampling
    # ------------------------------------------------------------------
    print("\nComputing physics null-space samples...")

    params = get_target_params(model, LAST_LAYER)
    theta_base = torch.cat([p.view(-1) for p in params]).detach()
    d = theta_base.shape[0]

    # Build J_pde: [N_PDE, d]  -- last-layer params only
    theta_r = theta_base.clone().requires_grad_(True)
    param_dict = build_param_dict(model, theta_r, LAST_LAYER)
    x_pde_t = x_pde.clone().requires_grad_(True)
    residual = pde_fn(model, x_pde_t, param_dict).view(-1)

    J_rows = []
    for i in range(len(residual)):
        g = torch.autograd.grad(
            residual[i], theta_r, retain_graph=True, allow_unused=True
        )[0]
        J_rows.append(g.detach() if g is not None else torch.zeros_like(theta_r))
    J_pde = torch.stack(J_rows)  # [N_PDE, d]

    K = J_pde @ J_pde.T
    K_inv = torch.linalg.pinv(K.cpu(), rcond=1e-10, hermitian=True).to(device)

    def project_null(v):
        return v - (v @ J_pde.T) @ K_inv @ J_pde

    torch.manual_seed(42)
    null_samples = []
    for _ in range(NUM_SAMPLES):
        c = torch.randn(d, dtype=torch.float64, device=device) * SAMPLE_SCALE
        delta = project_null(c)
        theta_new = theta_base + delta
        p_dict = build_param_dict(model, theta_new, LAST_LAYER)
        with torch.no_grad():
            u_samp = torch.func.functional_call(model, p_dict, x_plot)
        null_samples.append(u_samp.cpu().numpy().flatten())

    # Verify PDE residual is preserved for first sample
    theta_check = theta_base + project_null(
        torch.randn(d, dtype=torch.float64, device=device) * SAMPLE_SCALE
    )
    p_check = build_param_dict(model, theta_check, LAST_LAYER)
    x_pde_t2 = x_pde.clone().requires_grad_(True)
    res_check = pde_fn(model, x_pde_t2, p_check)
    print(
        f"Null-space sample PDE residual (MSE): {torch.mean(res_check**2).item():.2e}"
    )

    # ------------------------------------------------------------------
    # Figure: 1x2
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): regime-anchored base model
    ax = axes[0]
    ax.plot(x_np, u_base, color="tab:red", lw=2.2, zorder=5, label="Base model")
    ax.legend(fontsize=9)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_xlim(-1, 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.5, -0.13, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=13
    )

    # Panel (b): physics null-space samples
    ax = axes[1]
    for u_s in null_samples:
        ax.plot(x_np, u_s, color="steelblue", lw=1.0, alpha=0.6)
    ax.plot(x_np, u_base, color="tab:red", lw=2.2, zorder=5)
    ax.legend(
        handles=[
            mlines.Line2D(
                [], [], color="steelblue", lw=1.0, alpha=0.6, label="Null-space samples"
            ),
            mlines.Line2D([], [], color="tab:red", lw=2.2, label="Base model"),
        ],
        fontsize=9,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_xlim(-1, 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.5, -0.13, "(b)", transform=ax.transAxes, ha="center", va="top", fontsize=13
    )

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "pinn_1d_sine_comparison_kinda_close_bc.pdf"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Adaptation runs -- same three BC scenarios as exp_1d_adaptation.py
    # ------------------------------------------------------------------
    def exact_u(x_np, C1, C2):
        return np.sin(2.0 * np.pi * x_np) + C1 * x_np + C2

    def run_adaptation(use_corrector, layers, num_inc):
        tag = (
            "corr" if use_corrector else "no-corr"
        ) + f" {'last-layer' if layers == LAST_LAYER else 'full'}"
        print(f"\n{'#'*55}\n  {tag}\n{'#'*55}")
        preds, pde_res_list, rmse_list, times = [], [], [], []
        for sc in SCENARIOS:
            print(f"  Adapting to: {sc['label']}")
            t0 = time.perf_counter()
            specify_bcs(
                model,
                pde_fn,
                sc["bcs"],
                layers,
                x_pde,
                num_increments=num_inc,
                max_iter=15,
                tolerance=1e-16,
                use_corrector=use_corrector,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            with torch.no_grad():
                u_ad = model(x_plot).cpu().numpy().flatten()
            preds.append(u_ad)

            x_t = x_pde.clone().requires_grad_(True)
            pde_res = torch.mean(pde_fn(model, x_t) ** 2).item()
            pde_res_list.append(pde_res)

            rmse = np.sqrt(np.mean((u_ad - exact_u(x_np, sc["C1"], sc["C2"])) ** 2))
            rmse_list.append(rmse)
            print(f"    PDE MSE: {pde_res:.2e}    RMSE: {rmse:.2e}    Time: {elapsed:.2f}s")
        return preds, pde_res_list, rmse_list, times

    _, res_last, rmse_last, times_last = run_adaptation(False, LAST_LAYER, 1)
    _, res_no_corr, rmse_no_corr, times_no_corr = run_adaptation(False, ALL_LAYERS, NUM_INCREMENTS)
    _, res_corr, rmse_corr, times_corr = run_adaptation(True, ALL_LAYERS, NUM_INCREMENTS)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    col = 35
    print("\n" + "=" * 140)
    print(
        f"  {'Scenario':<{col}}  {'PDE MSE (last)':>14}  {'RMSE (last)':>11}  {'t (last)':>8}  {'PDE MSE (no-corr)':>17}  {'RMSE (no-corr)':>14}  {'t (no-corr)':>11}  {'PDE MSE (corr)':>14}  {'RMSE (corr)':>11}  {'t (corr)':>8}"
    )
    print("=" * 140)
    for i, sc in enumerate(SCENARIOS):
        print(
            f"  {sc['label']:<{col}}  {res_last[i]:>14.2e}  {rmse_last[i]:>11.2e}  {times_last[i]:>7.2f}s"
            f"  {res_no_corr[i]:>17.2e}  {rmse_no_corr[i]:>14.2e}  {times_no_corr[i]:>10.2f}s"
            f"  {res_corr[i]:>14.2e}  {rmse_corr[i]:>11.2e}  {times_corr[i]:>7.2f}s"
        )


if __name__ == "__main__":
    run()
