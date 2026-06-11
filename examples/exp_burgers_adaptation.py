"""
Experiment: Burgers' Equation -- Regime Anchoring and IC/BC Adaptation

Trains a 4-hidden-layer Tanh PINN on the viscous Burgers equation

    u_t + u*u_x - nu*u_xx = 0,   x in [-1,1], t in [0,1]
    u(x,0) = -sin(pi*x),  u(+/-1,t) = 0,  nu = 0.01/pi

with regime anchoring (soft BC weight during base training). The base model
is then adapted to two new scenarios via subspace projection and compared
against a timed adaptive fine-tuning baseline:

  (A) New IC: u(x,0) = -2 sin(pi*x)
  (B) New BC: u(-1,t) = 0.5t,  u(1,t) = -0.5t

All adaptations use the full network (all layers) with predictor-corrector
and are compared against an adaptive fine-tuning baseline.

Outputs:
    examples/burgers_ic_adaptation.pdf        (ref | ft-full | proj)
    examples/burgers_ic_snapshots.pdf
    examples/burgers_ic_ft_adapt.pdf          (ref | ft-adapt | proj)
    examples/burgers_ic_ft_adapt_snapshots.pdf
    examples/burgers_bc_adaptation.pdf        (ref | ft-full | proj)
    examples/burgers_bc_snapshots.pdf
    examples/burgers_bc_ft_adapt.pdf          (ref | ft-adapt | proj)
    examples/burgers_bc_ft_adapt_snapshots.pdf
    examples/burgers_pde_history.pdf          (2-col: IC | BC, all 3 methods)

Run:
    python examples/exp_burgers_adaptation.py
"""

import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from torch.func import functional_call

from adr.pinn import MLP
from adr.training import train
from adr.adaptation import specify_bcs
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

NU = 0.01 / np.pi

HIDDEN = 120
DEPTH = 4  # 5 nn.Linear layers (indices -5 ... -1)
N_PDE_X = 200
N_PDE_T = 160
LR = 1e-3
ADAM_EPOCHS = 10_000
LBFGS_MAX_ITER = 20_000
LAMBDA_PDE = 1.0
LAMBDA_BC = 1.0  # NTK scaling handles balancing dynamically

NUM_INCS = 100  # predictor steps per adaptation
N_ADAPT_IC = 40  # IC points in BC list
N_ADAPT_BC = 40  # boundary (left/right) t-points in BC list
N_ADAPT_GRID = 40  # grid resolution for PDE collocation (interior: (N-2)x(N-1) pts)

ALL_LAYERS = list(range(-(DEPTH + 1), 0))  # [-5,-4,-3,-2,-1]

# ---------------------------------------------------------------------------
# PDE residual factory (captures nu via closure)
# ---------------------------------------------------------------------------


def _make_pde_fn(nu):
    def fn(model, x, param_dict=None):
        pd = param_dict if param_dict is not None else dict(model.named_parameters())

        def u_fn(xi):  # xi: [2]
            return functional_call(model, pd, xi.unsqueeze(0)).squeeze()

        def residual_single(xi):
            u = u_fn(xi)
            grads = torch.func.grad(u_fn)(xi)  # [du/dx, du/dt]
            u_x, u_t = grads[0], grads[1]
            u_xx = torch.func.grad(
                lambda z: torch.func.grad(u_fn)(z)[0]  # d(du/dx)/dx
            )(xi)[0]
            return u_t + u * u_x - nu * u_xx

        return torch.func.vmap(residual_single)(x)

    return fn


pde_fn = _make_pde_fn(NU)


def compute_pde_loss(model, x_pde, fn=None):
    fn = fn or pde_fn
    return torch.mean(fn(model, x_pde) ** 2).item()


# ---------------------------------------------------------------------------
# Restore base state
# ---------------------------------------------------------------------------


def restore_base(model):
    """Write _adr_base_state back into model parameters."""
    if not hasattr(model, "_adr_base_state"):
        raise RuntimeError("No base state saved. Call train() first.")
    for name, p in model.named_parameters():
        if name in model._adr_base_state:
            p.data.copy_(model._adr_base_state[name])


# ---------------------------------------------------------------------------
# adaptive fine-tuning baseline
# ---------------------------------------------------------------------------


def _grad_norm(loss, model):
    model.zero_grad()
    loss.backward(retain_graph=True)
    norm = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
        ** 0.5
    )
    model.zero_grad()
    return max(norm, 1e-12)


def fine_tune(
    model,
    x_pde,
    x_bc,
    y_bc,
    fn=None,
    x_pde_log=None,
    snapshot_time=None,
    target_pde_loss=None,
    target_rmse=None,
    xt_eval=None,
    u_ref=None,
    N_eval=None,
    lr=1e-3,
    ntk_freq=200,
    log_freq=50,
    adam_max=10_000,
    lbfgs_max=30_000,
):
    """adaptive Adam + L-BFGS fine-tuning.

    x_pde_log: grid for logging PDE residual; defaults to x_pde when not given.
    Returns (history, total_time, snapshot_state) where history is a list of
    (wall_time, pde_mse, bc_mse) and snapshot_state is the model state dict
    the first time elapsed >= snapshot_time (or None).
    """
    fn = fn or pde_fn
    x_log = x_pde_log if x_pde_log is not None else x_pde
    w_pde, w_bc = 1.0, 1.0
    history = []
    t0 = time.time()
    snapshot_state = [None]

    def _losses():
        x_t = x_pde.clone().requires_grad_(True)
        l_pde = torch.mean(fn(model, x_t) ** 2)
        l_bc = torch.mean((model(x_bc) - y_bc) ** 2)
        return l_pde, l_bc

    def _log_pde():
        return torch.mean(fn(model, x_log.clone()) ** 2).item()

    def _log_bc():
        with torch.no_grad():
            return torch.mean((model(x_bc) - y_bc) ** 2).item()

    def _record():
        t = time.time() - t0
        pde_val = _log_pde()
        bc_val = _log_bc()
        history.append((t, pde_val, bc_val))
        if snapshot_time is not None and snapshot_state[0] is None and t >= snapshot_time:
            snapshot_state[0] = copy.deepcopy(model.state_dict())

    def _check_stop(pde_val):
        if target_pde_loss and pde_val <= target_pde_loss:
            return True
        if target_rmse is not None and xt_eval is not None and u_ref is not None:
            with torch.no_grad():
                u_pred = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
            rmse = np.sqrt(np.mean((u_pred - u_ref) ** 2))
            if rmse <= target_rmse:
                return True
        return False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)
    model.train()
    stopped = False
    _record()

    for epoch in range(adam_max):
        if epoch % ntk_freq == 0:
            l_p, l_b = _losses()
            n_pde = _grad_norm(l_p, model)
            n_bc = _grad_norm(l_b, model)
            w_bc = n_pde / n_bc
            model.train()

        optimizer.zero_grad()
        l_pde, l_bc = _losses()
        loss = w_pde * l_pde + w_bc * l_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_freq == 0:
            _record()
            if _check_stop(history[-1][1]):
                stopped = True
                break

    _record()

    if not stopped:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max,
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            l_pde, l_bc = _losses()
            loss = w_pde * l_pde + w_bc * l_bc
            loss.backward()
            _record()
            if _check_stop(history[-1][1]):
                raise StopIteration
            return loss

        try:
            optimizer_lbfgs.step(closure)
        except StopIteration:
            pass

    model.eval()
    elapsed = time.time() - t0
    _record()
    print(f"    Fine-tuning done in {elapsed:.2f}s | PDE MSE {history[-1][1]:.4e} | BC MSE {history[-1][2]:.4e}")
    return history, elapsed, snapshot_state[0]


# ---------------------------------------------------------------------------
# Finite-difference reference
# ---------------------------------------------------------------------------


def burgers_fd_reference(
    u0_fn, bc_left=None, bc_right=None, N_x=1000, N_t=500, nu=None
):
    if nu is None:
        nu = NU
    if bc_left is None:
        bc_left = lambda t: 0.0
    if bc_right is None:
        bc_right = lambda t: 0.0

    x = np.linspace(-1, 1, N_x)
    dx = x[1] - x[0]

    def rhs(t, u):
        dudt = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        u_x = np.zeros_like(u)
        for i in range(1, N_x - 1):
            u_x[i] = (u[i] - u[i - 1]) / dx if u[i] > 0 else (u[i + 1] - u[i]) / dx
        dudt[1:-1] = -u[1:-1] * u_x[1:-1] + nu * u_xx[1:-1]
        eps = 1e-7
        dudt[0] = (bc_left(t + eps) - bc_left(t)) / eps
        dudt[-1] = (bc_right(t + eps) - bc_right(t)) / eps
        return dudt

    u0 = u0_fn(x)
    u0[0] = bc_left(0.0)
    u0[-1] = bc_right(0.0)
    sol = solve_ivp(
        rhs,
        (0, 1),
        u0,
        t_eval=np.linspace(0, 1, N_t),
        method="BDF",
        max_step=0.001,
        rtol=1e-6,
        atol=1e-8,
    )
    if sol.status != 0:
        print(f"  FD warning: {sol.message}")
    return x, sol.t, sol.y


def interp_fd(x_ref, t_ref, u_ref, x_eval, t_eval):
    Xp, Tp = np.meshgrid(x_eval, t_eval, indexing="ij")
    interp = RegularGridInterpolator(
        (x_ref, t_ref), u_ref, method="linear", bounds_error=False, fill_value=0.0
    )
    u = interp(np.stack([Xp.flatten(), Tp.flatten()], axis=1)).reshape(
        len(x_eval), len(t_eval)
    )
    return u, Xp, Tp


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_spacetime(ref, ft, proj, Xp, Tp, filename):
    vabs = float(np.ceil(np.abs(ref).max() * 10) / 10)
    levels = np.linspace(-vabs, vabs, 60)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, u, lbl in zip(axes, [ref, ft, proj], ["(a)", "(b)", "(c)"]):
        c = ax.contourf(Xp, Tp, u, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_aspect("auto")
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes, ha="center", fontsize=13)
    fig.colorbar(
        c, ax=axes.tolist(), location="right", aspect=20, pad=0.02, label=r"$u(x,t)$"
    )
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Figure saved -> {filename}")
    plt.close(fig)


def plot_time_snapshots(ref, ft, proj, x_eval, t_eval, filename):
    fracs = [0.25, 0.50, 0.75]
    t_indices = [int(len(t_eval) * f) for f in fracs]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, t_idx, lbl in zip(axes, t_indices, ["(a)", "(b)", "(c)"]):
        t_v = t_eval[t_idx]
        ax.plot(x_eval, ref[:, t_idx], "k-", lw=2.5, label="Reference (FD)")
        ax.plot(x_eval, ft[:, t_idx], "r--", lw=2.0, label="Fine-tuned (NTK)")
        ax.plot(x_eval, proj[:, t_idx], "b-.", lw=2.0, label="Subspace proj.")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_title(rf"$t={t_v:.2f}$", fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.text(
            0.5,
            -0.18,
            lbl,
            transform=ax.transAxes,
            ha="center",
            fontsize=13,
        )
    axes[0].set_ylabel(r"$u(x,t)$")
    axes[0].legend(fontsize=10)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Figure saved -> {filename}")
    plt.close(fig)


def plot_combined_pde_history(
    ft_log_ic,
    ft_adapt_log_ic,
    t_proj_ic,
    pde_proj_ic,
    pde_proj_ic_adapt,
    ft_log_bc,
    ft_adapt_log_bc,
    t_proj_bc,
    pde_proj_bc,
    pde_proj_bc_adapt,
    base_pde,
    filename,
):
    def _xy(log):
        return [e[0] for e in log], [e[1] for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    scenarios = [
        (
            axes[0],
            ft_log_ic,
            ft_adapt_log_ic,
            t_proj_ic,
            pde_proj_ic,
            pde_proj_ic_adapt,
            "(a)",
        ),
        (
            axes[1],
            ft_log_bc,
            ft_adapt_log_bc,
            t_proj_bc,
            pde_proj_bc,
            pde_proj_bc_adapt,
            "(b)",
        ),
    ]

    for ax, ft_log, ft_adapt_log, t_proj, proj_full, proj_adapt, sublabel in scenarios:
        ft_t, ft_v = _xy(ft_log)
        fa_t, fa_v = _xy(ft_adapt_log)

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

        h_base = ax.axhline(base_pde, color="k", lw=1.0, linestyle="--", alpha=0.6)
        (h_ft,) = ax.plot(ft_t, ft_v, color="tab:orange", lw=2.0)
        (h_fa,) = ax.plot(fa_t, fa_v, color="tab:red", lw=2.0, linestyle="--")
        h_sub_full = ax.scatter(
            [t_proj], [proj_full], color="tab:green", s=80, marker="o", zorder=5
        )
        h_sub_adapt = ax.scatter(
            [t_proj], [proj_adapt], color="tab:green", s=80, marker="s", zorder=5
        )

        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("PDE residual (MSE)")
        ax.grid(True, which="both", alpha=0.25)
        ax.text(
            0.5,
            -0.13,
            sublabel,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    handles = [h_base, h_ft, h_fa, h_sub_full, h_sub_adapt]
    labels = [
        "Base model",
        "Fine-tune NTK (full grid)",
        "Fine-tune NTK (adapt grid)",
        "Subspace proj. (full grid)",
        "Subspace proj. (adapt grid)",
    ]
    axes[1].legend(handles, labels, fontsize=9, loc="upper right")

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Figure saved -> {filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def time_to_reach(log, col, target):
    """Return (entry, reached) where entry is the first log entry with entry[col] <= target.
    If never reached, entry is the entry with the minimum value in col (closest approach).
    """
    for entry in log:
        if entry[col] <= target:
            return entry, True
    return min(log, key=lambda e: e[col]), False


def snapshot_at_time(log, t_target):
    """Return the log entry closest to t_target."""
    times = np.array([e[0] for e in log])
    return log[np.argmin(np.abs(times - t_target))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    out = os.path.dirname(__file__)

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    x_grid = torch.linspace(-1, 1, N_PDE_X, device=device)
    t_grid = torch.linspace(0, 1, N_PDE_T, device=device)
    Xg, Tg = torch.meshgrid(x_grid, t_grid, indexing="ij")
    x_pde = torch.stack([Xg.flatten(), Tg.flatten()], dim=1)

    x_ic = torch.stack([x_grid, torch.zeros_like(x_grid)], dim=1)
    y_ic = (-torch.sin(np.pi * x_grid)).unsqueeze(1)

    x_sbc = torch.cat(
        [
            torch.stack([torch.full_like(t_grid, -1.0), t_grid], dim=1),
            torch.stack([torch.full_like(t_grid, 1.0), t_grid], dim=1),
        ]
    )
    y_sbc = torch.zeros(x_sbc.shape[0], 1, device=device)

    x_bc_base = torch.cat([x_ic, x_sbc])
    y_bc_base = torch.cat([y_ic, y_sbc])

    # Fine eval grid
    N_eval = 200
    x_eval = np.linspace(-1, 1, N_eval)
    t_eval = np.linspace(0, 1, N_eval)
    xt_eval = torch.tensor(
        np.stack(np.meshgrid(x_eval, t_eval, indexing="ij"), axis=-1).reshape(-1, 2),
        dtype=torch.float64,
        device=device,
    )

    # Collocation point grids for adaptation BCs (list-of-dicts format)
    x_bc_pts = np.linspace(-1, 1, N_ADAPT_IC)
    t_bc_pts = np.linspace(0, 1, N_ADAPT_BC)

    # uniform collocation grid for PDE constraint in subspace projection
    x_adapt = torch.linspace(-1, 1, N_ADAPT_GRID, device=device)
    t_adapt = torch.linspace(0, 1, N_ADAPT_GRID, device=device)
    Xa, Ta = torch.meshgrid(x_adapt, t_adapt, indexing="ij")
    x_pde_adapt = torch.stack([Xa.flatten(), Ta.flatten()], dim=1).to(
        dtype=torch.float64
    )

    # ------------------------------------------------------------------
    # Train base model  (nu = 0.01/pi)
    # ------------------------------------------------------------------
    set_seed(42)
    model = MLP(n_in=2, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=nn.Tanh).to(
        device
    )

    print("Training base model (nu=0.01/pi, regime anchoring lam_bc=1e-3, adaptive)...")
    train(
        model,
        pde_fn,
        x_pde,
        x_bc_base,
        y_bc_base,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lr=LR,
        lambdas=(LAMBDA_PDE, LAMBDA_BC, 0.0),
        print_every=2000,
    )

    base_pde = compute_pde_loss(model, x_pde, pde_fn)
    print(f"\nBase PDE residual (MSE): {base_pde:.4e}\n")

    with torch.no_grad():
        u_base = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)

    print("Computing FD reference (base IC)...")
    x_ref, t_ref, u_ref_raw = burgers_fd_reference(lambda x: -np.sin(np.pi * x))
    u_ref_base, Xp, Tp = interp_fd(x_ref, t_ref, u_ref_raw, x_eval, t_eval)
    print(f"Base RMSE vs FD: {np.sqrt(np.mean((u_base - u_ref_base) ** 2)):.4e}")

    # FD references for adaptation scenarios
    print("Computing FD reference (new IC: -2 sin(pi*x))...")
    _, _, u_ref_ic_raw = burgers_fd_reference(lambda x: -2.0 * np.sin(np.pi * x))
    u_ref_ic, _, _ = interp_fd(x_ref, t_ref, u_ref_ic_raw, x_eval, t_eval)

    print("Computing FD reference (new BCs: +/-0.5t)...")
    _, _, u_ref_bc_raw = burgers_fd_reference(
        lambda x: -np.sin(np.pi * x),
        bc_left=lambda t: 0.5 * t,
        bc_right=lambda t: -0.5 * t,
    )
    u_ref_bc, _, _ = interp_fd(x_ref, t_ref, u_ref_bc_raw, x_eval, t_eval)

    # BC tensors for fine-tuning (scenarios)
    y_ic_new = (-2.0 * torch.sin(np.pi * x_grid)).unsqueeze(1)
    x_bc_ic = torch.cat([x_ic, x_sbc])
    y_bc_ic = torch.cat([y_ic_new, y_sbc])

    y_sbc_new = torch.where(
        x_sbc[:, 0:1] < 0, 0.5 * x_sbc[:, 1:2], -0.5 * x_sbc[:, 1:2]
    )
    x_bc_bc = torch.cat([x_ic, x_sbc])
    y_bc_bc = torch.cat([y_ic, y_sbc_new])

    # BCs in list-of-dicts format for specify_bcs
    bcs_ic = []
    for xv in x_bc_pts:
        bcs_ic.append(
            {
                "type": "dirichlet",
                "coords": [float(xv), 0.0],
                "val": float(-2.0 * np.sin(np.pi * xv)),
            }
        )
    for tv in t_bc_pts:
        bcs_ic.append({"type": "dirichlet", "coords": [-1.0, float(tv)], "val": 0.0})
        bcs_ic.append({"type": "dirichlet", "coords": [1.0, float(tv)], "val": 0.0})

    bcs_bc = []
    for xv in x_bc_pts:
        bcs_bc.append(
            {
                "type": "dirichlet",
                "coords": [float(xv), 0.0],
                "val": float(-np.sin(np.pi * xv)),
            }
        )
    for tv in t_bc_pts:
        bcs_bc.append(
            {"type": "dirichlet", "coords": [-1.0, float(tv)], "val": 0.5 * float(tv)}
        )
        bcs_bc.append(
            {"type": "dirichlet", "coords": [1.0, float(tv)], "val": -0.5 * float(tv)}
        )

    # ==================================================================
    # (A) IC Adaptation: u(x,0) = -2 sin(pi*x)
    # ==================================================================
    print(f"\n{'#'*60}\n  (A) IC Adaptation: u(x,0) = -2 sin(pi*x)\n{'#'*60}")

    restore_base(model)
    t0 = time.time()
    specify_bcs(
        model,
        pde_fn,
        bcs_ic,
        ALL_LAYERS,
        x_pde_adapt,
        num_increments=NUM_INCS,
        use_corrector=True,
        tolerance=1e-8,
        max_iter=15,
        rcond=1e-8,
    )
    t_proj_ic = time.time() - t0

    pde_proj_ic = compute_pde_loss(model, x_pde, pde_fn)
    pde_proj_ic_adapt = compute_pde_loss(model, x_pde_adapt, pde_fn)
    with torch.no_grad():
        bc_proj_ic = torch.mean((model(x_bc_ic) - y_bc_ic) ** 2).item()
        u_proj_ic = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_proj_ic = np.sqrt(np.mean((u_proj_ic - u_ref_ic) ** 2))
    print(
        f"  Subspace proj (ALL_LAYERS): {t_proj_ic:.2f}s | "
        f"PDE MSE {pde_proj_ic:.4e} (adapt grid: {pde_proj_ic_adapt:.4e}) | "
        f"BC MSE {bc_proj_ic:.4e} | RMSE {rmse_proj_ic:.4e}"
    )

    restore_base(model)
    print(f"  Fine-tuning (full, adaptive, snapshot at t={t_proj_ic:.2f}s)...")
    ft_log_ic, ft_ic_total_time, ft_ic_snapshot = fine_tune(
        model, x_pde, x_bc_ic, y_bc_ic, fn=pde_fn, snapshot_time=t_proj_ic,
    )
    ft_ic_end = ft_log_ic[-1]
    ft_ic_tm = snapshot_at_time(ft_log_ic, t_proj_ic)
    if ft_ic_snapshot is not None:
        ft_ic_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ft_ic_snapshot)
        with torch.no_grad():
            u_ft_ic_timed = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_ic_timed = np.sqrt(np.mean((u_ft_ic_timed - u_ref_ic) ** 2))
        model.load_state_dict(ft_ic_final_state)
    else:
        u_ft_ic_timed = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_ic_timed = np.sqrt(np.mean((u_ft_ic_timed - u_ref_ic) ** 2))
    with torch.no_grad():
        u_ft_ic_end = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_ft_ic_end = np.sqrt(np.mean((u_ft_ic_end - u_ref_ic) ** 2))
    print(f"  Fine-tune @ t_sub: PDE MSE {ft_ic_tm[1]:.4e} | BC MSE {ft_ic_tm[2]:.4e} | RMSE {rmse_ft_ic_timed:.4e}")
    print(f"  Fine-tune @ end:   PDE MSE {ft_ic_end[1]:.4e} | BC MSE {ft_ic_end[2]:.4e} | RMSE {rmse_ft_ic_end:.4e}")

    plot_spacetime(
        u_ref_ic, u_ft_ic_timed, u_proj_ic, Xp, Tp,
        os.path.join(out, "burgers_ic_adaptation.pdf"),
    )
    plot_time_snapshots(
        u_ref_ic, u_ft_ic_timed, u_proj_ic, x_eval, t_eval,
        os.path.join(out, "burgers_ic_snapshots.pdf"),
    )

    restore_base(model)
    print(f"  Fine-tuning (adapt grid, adaptive, snapshot at t={t_proj_ic:.2f}s)...")
    ft_adapt_log_ic, ft_adapt_ic_total_time, ft_adapt_ic_snapshot = fine_tune(
        model, x_pde_adapt, x_bc_ic, y_bc_ic, fn=pde_fn, x_pde_log=x_pde, snapshot_time=t_proj_ic,
    )
    ft_adapt_ic_end = ft_adapt_log_ic[-1]
    ft_adapt_ic_tm = snapshot_at_time(ft_adapt_log_ic, t_proj_ic)
    if ft_adapt_ic_snapshot is not None:
        ft_adapt_ic_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ft_adapt_ic_snapshot)
        with torch.no_grad():
            u_ft_adapt_ic = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_adapt_ic = np.sqrt(np.mean((u_ft_adapt_ic - u_ref_ic) ** 2))
        model.load_state_dict(ft_adapt_ic_final_state)
    else:
        u_ft_adapt_ic = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_adapt_ic = np.sqrt(np.mean((u_ft_adapt_ic - u_ref_ic) ** 2))
    with torch.no_grad():
        u_ft_adapt_ic_end = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_ft_adapt_ic_end = np.sqrt(np.mean((u_ft_adapt_ic_end - u_ref_ic) ** 2))
    print(f"  FT adapt @ t_sub: PDE MSE {ft_adapt_ic_tm[1]:.4e} | BC MSE {ft_adapt_ic_tm[2]:.4e} | RMSE {rmse_ft_adapt_ic:.4e}")
    print(f"  FT adapt @ end:   PDE MSE {ft_adapt_ic_end[1]:.4e} | BC MSE {ft_adapt_ic_end[2]:.4e} | RMSE {rmse_ft_adapt_ic_end:.4e}")

    # BC-crossing analysis for IC scenario
    print(f"\n  Convergence to subspace BC target (bc_proj_ic={bc_proj_ic:.4e}):")
    for label, log in [("Full fine-tuning", ft_log_ic), ("FT adapt grid", ft_adapt_log_ic)]:
        cb, cb_r = time_to_reach(log, 2, bc_proj_ic)
        if cb_r:
            print(f"    {label}: BC reached at t={cb[0]:.1f}s ({cb[0]/t_proj_ic:.1f}x t_proj)  |  PDE MSE at crossing: {cb[1]:.4e}")
        else:
            print(f"    {label}: BC never reached -- best {cb[2]:.4e} at t={cb[0]:.1f}s  |  PDE MSE at that point: {cb[1]:.4e}")

    plot_spacetime(
        u_ref_ic,
        u_ft_adapt_ic,
        u_proj_ic,
        Xp,
        Tp,
        os.path.join(out, "burgers_ic_ft_adapt.pdf"),
    )
    plot_time_snapshots(
        u_ref_ic,
        u_ft_adapt_ic,
        u_proj_ic,
        x_eval,
        t_eval,
        os.path.join(out, "burgers_ic_ft_adapt_snapshots.pdf"),
    )

    # ==================================================================
    # (B) BC Adaptation: u(+/-1,t) = +/-0.5t
    # ==================================================================
    print(f"\n{'#'*60}\n  (B) BC Adaptation: u(+/-1,t) = +/-0.5t\n{'#'*60}")

    restore_base(model)
    t0 = time.time()
    specify_bcs(
        model,
        pde_fn,
        bcs_bc,
        ALL_LAYERS,
        x_pde_adapt,
        num_increments=NUM_INCS,
        use_corrector=True,
        tolerance=1e-8,
        max_iter=15,
        rcond=1e-8,
    )
    t_proj_bc = time.time() - t0

    pde_proj_bc = compute_pde_loss(model, x_pde, pde_fn)
    pde_proj_bc_adapt = compute_pde_loss(model, x_pde_adapt, pde_fn)
    with torch.no_grad():
        bc_proj_bc = torch.mean((model(x_bc_bc) - y_bc_bc) ** 2).item()
        u_proj_bc = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_proj_bc = np.sqrt(np.mean((u_proj_bc - u_ref_bc) ** 2))
    print(
        f"  Subspace proj (ALL_LAYERS): {t_proj_bc:.2f}s | "
        f"PDE MSE {pde_proj_bc:.4e} (adapt grid: {pde_proj_bc_adapt:.4e}) | "
        f"BC MSE {bc_proj_bc:.4e} | RMSE {rmse_proj_bc:.4e}"
    )

    restore_base(model)
    print(f"  Fine-tuning (full, adaptive, snapshot at t={t_proj_bc:.2f}s)...")
    ft_log_bc, ft_bc_total_time, ft_bc_snapshot = fine_tune(
        model, x_pde, x_bc_bc, y_bc_bc, fn=pde_fn, snapshot_time=t_proj_bc,
    )
    ft_bc_end = ft_log_bc[-1]
    ft_bc_tm = snapshot_at_time(ft_log_bc, t_proj_bc)
    if ft_bc_snapshot is not None:
        ft_bc_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ft_bc_snapshot)
        with torch.no_grad():
            u_ft_bc_timed = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_bc_timed = np.sqrt(np.mean((u_ft_bc_timed - u_ref_bc) ** 2))
        model.load_state_dict(ft_bc_final_state)
    else:
        u_ft_bc_timed = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_bc_timed = np.sqrt(np.mean((u_ft_bc_timed - u_ref_bc) ** 2))
    with torch.no_grad():
        u_ft_bc_end = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_ft_bc_end = np.sqrt(np.mean((u_ft_bc_end - u_ref_bc) ** 2))
    print(f"  Fine-tune @ t_sub: PDE MSE {ft_bc_tm[1]:.4e} | BC MSE {ft_bc_tm[2]:.4e} | RMSE {rmse_ft_bc_timed:.4e}")
    print(f"  Fine-tune @ end:   PDE MSE {ft_bc_end[1]:.4e} | BC MSE {ft_bc_end[2]:.4e} | RMSE {rmse_ft_bc_end:.4e}")

    plot_spacetime(
        u_ref_bc, u_ft_bc_timed, u_proj_bc, Xp, Tp,
        os.path.join(out, "burgers_bc_adaptation.pdf"),
    )
    plot_time_snapshots(
        u_ref_bc, u_ft_bc_timed, u_proj_bc, x_eval, t_eval,
        os.path.join(out, "burgers_bc_snapshots.pdf"),
    )

    restore_base(model)
    print(f"  Fine-tuning (adapt grid, adaptive, snapshot at t={t_proj_bc:.2f}s)...")
    ft_adapt_log_bc, ft_adapt_bc_total_time, ft_adapt_bc_snapshot = fine_tune(
        model, x_pde_adapt, x_bc_bc, y_bc_bc, fn=pde_fn, x_pde_log=x_pde, snapshot_time=t_proj_bc,
    )
    ft_adapt_bc_end = ft_adapt_log_bc[-1]
    ft_adapt_bc_tm = snapshot_at_time(ft_adapt_log_bc, t_proj_bc)
    if ft_adapt_bc_snapshot is not None:
        ft_adapt_bc_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ft_adapt_bc_snapshot)
        with torch.no_grad():
            u_ft_adapt_bc = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_adapt_bc = np.sqrt(np.mean((u_ft_adapt_bc - u_ref_bc) ** 2))
        model.load_state_dict(ft_adapt_bc_final_state)
    else:
        u_ft_adapt_bc = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        rmse_ft_adapt_bc = np.sqrt(np.mean((u_ft_adapt_bc - u_ref_bc) ** 2))
    with torch.no_grad():
        u_ft_adapt_bc_end = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
    rmse_ft_adapt_bc_end = np.sqrt(np.mean((u_ft_adapt_bc_end - u_ref_bc) ** 2))
    print(f"  FT adapt @ t_sub: PDE MSE {ft_adapt_bc_tm[1]:.4e} | BC MSE {ft_adapt_bc_tm[2]:.4e} | RMSE {rmse_ft_adapt_bc:.4e}")
    print(f"  FT adapt @ end:   PDE MSE {ft_adapt_bc_end[1]:.4e} | BC MSE {ft_adapt_bc_end[2]:.4e} | RMSE {rmse_ft_adapt_bc_end:.4e}")

    # BC-crossing analysis for BC scenario
    print(f"\n  Convergence to subspace BC target (bc_proj_bc={bc_proj_bc:.4e}):")
    for label, log in [("Full fine-tuning", ft_log_bc), ("FT adapt grid", ft_adapt_log_bc)]:
        cb, cb_r = time_to_reach(log, 2, bc_proj_bc)
        if cb_r:
            print(f"    {label}: BC reached at t={cb[0]:.1f}s ({cb[0]/t_proj_bc:.1f}x t_proj)  |  PDE MSE at crossing: {cb[1]:.4e}")
        else:
            print(f"    {label}: BC never reached -- best {cb[2]:.4e} at t={cb[0]:.1f}s  |  PDE MSE at that point: {cb[1]:.4e}")

    plot_spacetime(
        u_ref_bc, u_ft_adapt_bc, u_proj_bc, Xp, Tp,
        os.path.join(out, "burgers_bc_ft_adapt.pdf"),
    )
    plot_time_snapshots(
        u_ref_bc, u_ft_adapt_bc, u_proj_bc, x_eval, t_eval,
        os.path.join(out, "burgers_bc_ft_adapt_snapshots.pdf"),
    )

    plot_combined_pde_history(
        ft_log_ic, ft_adapt_log_ic, t_proj_ic, pde_proj_ic, pde_proj_ic_adapt,
        ft_log_bc, ft_adapt_log_bc, t_proj_bc, pde_proj_bc, pde_proj_bc_adapt,
        base_pde, os.path.join(out, "burgers_pde_history.pdf"),
    )

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n" + "=" * 90)
    print(f"  {'Scenario':<20} {'Method':<24} {'Time(s)':>8} {'PDE MSE':>12} {'BC MSE':>12} {'RMSE':>12}")
    print("=" * 90)
    rows_sum = [
        ("IC adapt", "Subspace proj.",      t_proj_ic,           pde_proj_ic,       bc_proj_ic,  rmse_proj_ic),
        ("IC adapt", "Full FT @ t_sub",     t_proj_ic,           ft_ic_tm[1],       ft_ic_tm[2], rmse_ft_ic_timed),
        ("IC adapt", "Full FT @ end",       ft_ic_total_time,    ft_ic_end[1],      ft_ic_end[2],rmse_ft_ic_end),
        ("IC adapt", "FT adapt @ t_sub",    t_proj_ic,           ft_adapt_ic_tm[1], ft_adapt_ic_tm[2], rmse_ft_adapt_ic),
        ("IC adapt", "FT adapt @ end",      ft_adapt_ic_total_time, ft_adapt_ic_end[1], ft_adapt_ic_end[2], rmse_ft_adapt_ic_end),
        ("BC adapt", "Subspace proj.",      t_proj_bc,           pde_proj_bc,       bc_proj_bc,  rmse_proj_bc),
        ("BC adapt", "Full FT @ t_sub",     t_proj_bc,           ft_bc_tm[1],       ft_bc_tm[2], rmse_ft_bc_timed),
        ("BC adapt", "Full FT @ end",       ft_bc_total_time,    ft_bc_end[1],      ft_bc_end[2],rmse_ft_bc_end),
        ("BC adapt", "FT adapt @ t_sub",    t_proj_bc,           ft_adapt_bc_tm[1], ft_adapt_bc_tm[2], rmse_ft_adapt_bc),
        ("BC adapt", "FT adapt @ end",      ft_adapt_bc_total_time, ft_adapt_bc_end[1], ft_adapt_bc_end[2], rmse_ft_adapt_bc_end),
    ]
    for sc, method, t, pde, bc, rmse in rows_sum:
        print(f"  {sc:<20} {method:<24} {t:>8.2f} {pde:>12.4e} {bc:>12.4e} {rmse:>12.4e}")
    print("=" * 90)


if __name__ == "__main__":
    run()
