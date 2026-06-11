"""
Experiment: Subspace Projection vs Fine-tuning -- 2D Poisson (SiLU ResNet)

Compares three BC adaptation strategies from the same physics-trained base:

  1. Subspace projection (all layers)    -- null-space constrained update
  2. Full fine-tuning  (Adam + L-BFGS)  -- adaptive, all parameters
  3. Last-layer fine-tuning             -- adaptive, output layer only

Summary table reports time-matched (at t_sub) and end-of-budget metrics with RMSE.
The trajectory plot x-axis is capped at PLOT_CAP * t_sub.

Run:
    python examples/exp_2d_poisson_comparison.py
"""

import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

from torch.func import functional_call

from adr.pinn import ResNet
from adr.training import train
from adr.adaptation import specify_bcs, save_base
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
# Hyperparameters -- match exp_2d_poisson_silu.py
# ---------------------------------------------------------------------------

N_PDE = 15
HIDDEN = 100
DEPTH = 3
LR = 5e-3
ADAM_EPOCHS = 10_000
LBFGS_MAX_ITER = 30_000
NUM_INCS = 100

FT_ADAM_EPOCHS = 10_000
FT_LBFGS_MAX_ITER = 30_000
NTK_UPDATE_EVERY = 200

PLOT_CAP = 10  # x-axis limit for trajectory plot


# ---------------------------------------------------------------------------
# PDE residual (matches exp_2d_poisson_silu.py)
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


def _pde_mse(model, x_pde):
    x_t = x_pde.clone().requires_grad_(True)
    return pde_fn(model, x_t).detach().pow(2).mean().item()


def _bc_mse(model, x_bc, y_bc):
    with torch.no_grad():
        return (model(x_bc) - y_bc).pow(2).mean().item()


# ---------------------------------------------------------------------------
# adaptive lambda (Wang et al. 2021)
# lambda_bc = ||grad L_pde|| / ||grad L_bc||
# ---------------------------------------------------------------------------


def compute_adaptive_lambda(model, x_pde, x_bc_new, y_bc_new):
    model.train()

    model.zero_grad()
    x_t = x_pde.clone().requires_grad_(True)
    pde_fn(model, x_t).pow(2).mean().backward()
    g_pde = (
        torch.cat(
            [
                p.grad.contiguous().view(-1)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        .norm()
        .item()
    )

    model.zero_grad()
    (model(x_bc_new) - y_bc_new).pow(2).mean().backward()
    g_bc = (
        torch.cat(
            [
                p.grad.contiguous().view(-1)
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        .norm()
        .item()
    )

    model.zero_grad()
    return g_pde / (g_bc + 1e-16)


# ---------------------------------------------------------------------------
# Subspace projection with per-increment trajectory logging.
# Equivalent to specify_bcs(num_increments=NUM_INCS) but logs after each step.
# ---------------------------------------------------------------------------


def subspace_with_trajectory(
    model,
    bcs_original,
    target_layers,
    x_pde,
    x_bc_new,
    y_bc_new,
    device,
    num_increments,
    **kwargs,
):
    """Run incremental subspace projection, logging (t, pde_mse, bc_mse) after each step."""
    # Initial BC values from current model (identical to _scalable_ntk's base_vals)
    with torch.no_grad():
        base_vals = [
            model(
                torch.tensor([bc["coords"]], dtype=torch.float64, device=device)
            ).item()
            for bc in bcs_original
        ]

    log = [(0.0, _pde_mse(model, x_pde), _bc_mse(model, x_bc_new, y_bc_new))]
    t0 = time.time()
    save_base(model)

    for k in range(1, num_increments + 1):
        frac = k / num_increments
        step_bcs = [
            {**bc, "val": base_vals[i] + frac * (bc["val"] - base_vals[i])}
            for i, bc in enumerate(bcs_original)
        ]
        specify_bcs(
            model,
            pde_fn,
            step_bcs,
            target_layers,
            x_pde,
            num_increments=1,
            **kwargs,
        )
        save_base(model)
        elapsed = time.time() - t0
        log.append(
            (elapsed, _pde_mse(model, x_pde), _bc_mse(model, x_bc_new, y_bc_new))
        )

    return log, time.time() - t0


# ---------------------------------------------------------------------------
# Fine-tuning with trajectory logging (Adam + L-BFGS, adaptive)
# ---------------------------------------------------------------------------


def finetune(
    model,
    x_pde,
    x_bc_new,
    y_bc_new,
    params,
    lr=LR,
    adam_epochs=FT_ADAM_EPOCHS,
    lbfgs_max_iter=FT_LBFGS_MAX_ITER,
    ntk_update_every=NTK_UPDATE_EVERY,
    snapshot_time=None,
):
    """Full Adam + L-BFGS fine-tuning. Logs (t, pde_mse, bc_mse) throughout.

    If snapshot_time is given, saves a copy of the model state the first time
    elapsed time crosses that threshold (used to compute RMSE at t_sub).
    Returns (log, total_time, snapshot_state_dict_or_None).
    """
    lam_bc = compute_adaptive_lambda(model, x_pde, x_bc_new, y_bc_new)
    log = []
    t0 = time.time()
    snapshot_state = [None]

    def _record():
        t = time.time() - t0
        p = _pde_mse(model, x_pde)
        b = _bc_mse(model, x_bc_new, y_bc_new)
        log.append((t, p, b))
        if (
            snapshot_time is not None
            and snapshot_state[0] is None
            and t >= snapshot_time
        ):
            snapshot_state[0] = copy.deepcopy(model.state_dict())

    def total_loss():
        x_t = x_pde.clone().requires_grad_(True)
        return (
            pde_fn(model, x_t).pow(2).mean()
            + lam_bc * (model(x_bc_new) - y_bc_new).pow(2).mean()
        )

    optimizer = optim.Adam(params, lr=lr)
    model.train()
    _record()

    for epoch in range(adam_epochs):
        if epoch > 0 and epoch % ntk_update_every == 0:
            lam_bc = compute_adaptive_lambda(model, x_pde, x_bc_new, y_bc_new)
        optimizer.zero_grad()
        total_loss().backward()
        optimizer.step()
        if epoch % 100 == 0:
            _record()

    _record()

    optimizer_lbfgs = optim.LBFGS(
        params,
        lr=1.0,
        max_iter=lbfgs_max_iter,
        max_eval=int(lbfgs_max_iter * 1.25),
        history_size=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-16,
        tolerance_change=1e-16,
    )
    lbfgs_iter = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = total_loss()
        loss.backward()
        if lbfgs_iter[0] % 50 == 0:
            _record()
        lbfgs_iter[0] += 1
        return loss

    model.train()
    optimizer_lbfgs.step(closure)
    _record()

    return log, time.time() - t0, snapshot_state[0]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def snapshot_at_time(log, t_target):
    """Return the log entry closest to t_target."""
    times = np.array([e[0] for e in log])
    return log[np.argmin(np.abs(times - t_target))]


def time_to_reach(log, col, target):
    """Return (entry, reached) where entry is the first log entry with entry[col] <= target.
    If never reached, entry is the entry with the minimum value in col (closest approach).
    """
    for entry in log:
        if entry[col] <= target:
            return entry, True
    return min(log, key=lambda e: e[col]), False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    edge = torch.linspace(-1, 1, N_PDE, device=device)
    x_grid, y_grid = torch.meshgrid(edge, edge, indexing="ij")
    x_pde = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

    bc_bottom = torch.stack([edge, torch.full_like(edge, -1.0)], dim=1)
    bc_top = torch.stack([edge, torch.full_like(edge, 1.0)], dim=1)
    bc_left = torch.stack([torch.full_like(edge, -1.0), edge], dim=1)
    bc_right = torch.stack([torch.full_like(edge, 1.0), edge], dim=1)
    x_bc_train = torch.cat([bc_bottom, bc_top, bc_left, bc_right], dim=0)
    y_bc_train = torch.zeros(x_bc_train.shape[0], 1, device=device)

    N_test = 50
    xt = torch.linspace(-1, 1, N_test)
    yt = torch.linspace(-1, 1, N_test)
    Xg, Yg = torch.meshgrid(xt, yt, indexing="ij")
    x_test = torch.stack([Xg.flatten(), Yg.flatten()], dim=1).to(device)
    U_exact = torch.sin(np.pi * Xg) * torch.sin(np.pi * Yg) + torch.sin(
        np.pi * Xg
    ) * torch.sinh(np.pi * (Yg + 1)) / np.sinh(2 * np.pi)
    U_exact_np = U_exact.numpy().flatten()

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

    x_bc_new = torch.tensor(
        [bc["coords"] for bc in bcs_new], dtype=torch.float64, device=device
    )
    y_bc_new = torch.tensor(
        [[bc["val"]] for bc in bcs_new], dtype=torch.float64, device=device
    )

    # ------------------------------------------------------------------
    # Train base model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Training base model (PDE only)...")
    print("=" * 60)
    set_seed(42)
    model = ResNet(n_in=2, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=nn.SiLU).to(
        device
    )
    train(
        model,
        pde_fn,
        x_pde,
        x_bc_train,
        y_bc_train,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lr=LR,
        lambdas=(1.0, 0.0, 0.0),
        print_every=2000,
    )
    base_pde = _pde_mse(model, x_pde)
    base_bc = _bc_mse(model, x_bc_new, y_bc_new)
    print(f"\nBase  PDE MSE: {base_pde:.2e}  |  BC MSE (new): {base_bc:.2e}\n")
    base_state = copy.deepcopy(model.state_dict())

    linear_names = [
        name for name, m in model.named_modules() if isinstance(m, nn.Linear)
    ]

    def params_for_layers(indices):
        targets = {linear_names[i] for i in indices}
        return [
            p
            for name, m in model.named_modules()
            if name in targets
            for p in m.parameters()
        ]

    # ------------------------------------------------------------------
    # Method 1: Subspace projection (all layers)
    # Step A -- clean timing run (no logging overhead)
    # Step B -- trajectory run for plotting, time axis rescaled to t_sub
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Method 1: Subspace projection (all layers) -- timing run")
    print("=" * 60)
    model.load_state_dict(base_state)
    save_base(model)
    t0 = time.time()
    specify_bcs(
        model,
        pde_fn,
        bcs_new,
        list(range(len(linear_names))),
        x_pde,
        num_increments=NUM_INCS,
        max_iter=15,
        tolerance=1e-8,
        use_corrector=True,
        rcond=1e-8,
    )
    t_sub = time.time() - t0
    sub_pde = _pde_mse(model, x_pde)
    sub_bc = _bc_mse(model, x_bc_new, y_bc_new)
    with torch.no_grad():
        u_sub = model(x_test).cpu().numpy().flatten()
    sub_rmse = np.sqrt(np.mean((u_sub - U_exact_np) ** 2))
    print(
        f"Time: {t_sub:.1f}s  |  PDE MSE: {sub_pde:.2e}  |  BC MSE: {sub_bc:.2e}  |  RMSE: {sub_rmse:.2e}"
    )

    print("\nMethod 1: Subspace projection -- trajectory run (for plotting)")
    model.load_state_dict(base_state)
    sub_log_raw, _ = subspace_with_trajectory(
        model,
        bcs_new,
        list(range(len(linear_names))),
        x_pde,
        x_bc_new,
        y_bc_new,
        device,
        num_increments=NUM_INCS,
        max_iter=15,
        tolerance=1e-8,
        use_corrector=True,
        rcond=1e-8,
    )
    # Rescale trajectory time axis so the final point aligns with t_sub
    n_pts = len(sub_log_raw)
    sub_log = [
        (t_sub * i / (n_pts - 1), sub_log_raw[i][1], sub_log_raw[i][2])
        for i in range(n_pts)
    ]

    # ------------------------------------------------------------------
    # Method 2: Full fine-tuning (adaptive, all parameters)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Method 2: Full fine-tuning (all parameters, adaptive)")
    print("=" * 60)
    model.load_state_dict(base_state)
    ft_log, ft_total_time, ft_snapshot = finetune(
        model, x_pde, x_bc_new, y_bc_new, list(model.parameters()), snapshot_time=t_sub
    )

    ft_tm = snapshot_at_time(ft_log, t_sub)
    ft_end = ft_log[-1]
    if ft_snapshot is not None:
        ft_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ft_snapshot)
        with torch.no_grad():
            u_ft_tm = model(x_test).cpu().numpy().flatten()
        ft_tm_rmse = np.sqrt(np.mean((u_ft_tm - U_exact_np) ** 2))
        model.load_state_dict(ft_final_state)
    else:
        ft_tm_rmse = None
    with torch.no_grad():
        u_ft_end = model(x_test).cpu().numpy().flatten()
    ft_end_rmse = np.sqrt(np.mean((u_ft_end - U_exact_np) ** 2))
    print(f"Time-matched ({t_sub:.1f}s): PDE MSE={ft_tm[1]:.2e}  BC MSE={ft_tm[2]:.2e}")
    print(
        f"End of budget ({ft_total_time:.1f}s): PDE MSE={ft_end[1]:.2e}  BC MSE={ft_end[2]:.2e}  RMSE={ft_end_rmse:.2e}"
    )

    # ------------------------------------------------------------------
    # Method 3: Last-layer fine-tuning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Method 3: Last-layer fine-tuning (adaptive)")
    print("=" * 60)
    model.load_state_dict(base_state)
    ll_log, ll_total_time, ll_snapshot = finetune(
        model, x_pde, x_bc_new, y_bc_new, params_for_layers([-1]), snapshot_time=t_sub
    )

    ll_tm = snapshot_at_time(ll_log, t_sub)
    ll_end = ll_log[-1]
    if ll_snapshot is not None:
        ll_final_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ll_snapshot)
        with torch.no_grad():
            u_ll_tm = model(x_test).cpu().numpy().flatten()
        ll_tm_rmse = np.sqrt(np.mean((u_ll_tm - U_exact_np) ** 2))
        model.load_state_dict(ll_final_state)
    else:
        ll_tm_rmse = None
    with torch.no_grad():
        u_ll_end = model(x_test).cpu().numpy().flatten()
    ll_end_rmse = np.sqrt(np.mean((u_ll_end - U_exact_np) ** 2))
    print(f"Time-matched ({t_sub:.1f}s): PDE MSE={ll_tm[1]:.2e}  BC MSE={ll_tm[2]:.2e}")
    print(
        f"End of budget ({ll_total_time:.1f}s): PDE MSE={ll_end[1]:.2e}  BC MSE={ll_end[2]:.2e}  RMSE={ll_end_rmse:.2e}"
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    ft_tm_rmse_str = f"{ft_tm_rmse:.2e}" if ft_tm_rmse is not None else "--"
    ll_tm_rmse_str = f"{ll_tm_rmse:.2e}" if ll_tm_rmse is not None else "--"
    print("\n" + "=" * 80)
    print(
        f"  {'Method':<38} {'Time (s)':>9} {'PDE MSE':>10} {'BC MSE':>10} {'RMSE':>10}"
    )
    print("-" * 80)
    print(f"  {'Base model':<38} {'--':>9} {base_pde:>10.2e} {base_bc:>10.2e} {'--':>10}")
    print(
        f"  {'Subspace projection':<38} {t_sub:>9.1f} {sub_pde:>10.2e} {sub_bc:>10.2e} {sub_rmse:>10.2e}"
    )
    print(
        f"  {'Full fine-tuning @ t_sub':<38} {t_sub:>9.1f} {ft_tm[1]:>10.2e} {ft_tm[2]:>10.2e} {ft_tm_rmse_str:>10}"
    )
    print(
        f"  {'Full fine-tuning @ end':<38} {ft_total_time:>9.1f} {ft_end[1]:>10.2e} {ft_end[2]:>10.2e} {ft_end_rmse:>10.2e}"
    )
    print(
        f"  {'Last-layer fine-tuning @ t_sub':<38} {t_sub:>9.1f} {ll_tm[1]:>10.2e} {ll_tm[2]:>10.2e} {ll_tm_rmse_str:>10}"
    )
    print(
        f"  {'Last-layer fine-tuning @ end':<38} {ll_total_time:>9.1f} {ll_end[1]:>10.2e} {ll_end[2]:>10.2e} {ll_end_rmse:>10.2e}"
    )
    print("=" * 80)

    # ------------------------------------------------------------------
    # Convergence to subspace BC target (BC MSE is the meaningful signal;
    # PDE MSE at that same log entry shows whether physics held up)
    # ------------------------------------------------------------------
    ft_cross_bc, ft_cross_bc_reached = time_to_reach(ft_log, 2, sub_bc)
    ll_cross_bc, ll_cross_bc_reached = time_to_reach(ll_log, 2, sub_bc)

    print(f"\nConvergence to subspace BC target  (sub_bc={sub_bc:.2e}):")
    for method_label, cb, cb_r in [
        ("Full fine-tuning", ft_cross_bc, ft_cross_bc_reached),
        ("Last-layer fine-tuning", ll_cross_bc, ll_cross_bc_reached),
    ]:
        print(f"  {method_label}:")
        if cb_r:
            print(
                f"    BC  MSE: reached at t={cb[0]:.1f}s  ({cb[0]/t_sub:.1f}x t_sub)"
                f"  |  PDE MSE at crossing: {cb[1]:.2e}"
            )
        else:
            print(
                f"    BC  MSE: never reached -- best {cb[2]:.2e} at t={cb[0]:.1f}s"
                f"  |  PDE MSE at that point: {cb[1]:.2e}"
            )

    # ------------------------------------------------------------------
    # Figure: 2-panel trajectory, x-axis capped at PLOT_CAP * t_sub
    # ------------------------------------------------------------------
    x_lim = PLOT_CAP * t_sub

    def _xy(log, col):
        t = [e[0] for e in log]
        v = [e[col] for e in log]
        return t, v

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    handles, labels = [], []

    for ax_idx, (ax, col, ylabel) in enumerate(
        [
            (axes[0], 1, "PDE residual (MSE)"),
            (axes[1], 2, "BC residual (MSE)"),
        ]
    ):
        sub_t, sub_v = _xy(sub_log, col)
        ft_t, ft_v = _xy(ft_log, col)
        ll_t, ll_v = _xy(ll_log, col)

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

        h_base = ax.axhline(
            base_pde if col == 1 else base_bc,
            color="k",
            lw=1.0,
            linestyle="--",
            alpha=0.5,
        )
        (h_ft,) = ax.plot(ft_t, ft_v, color="tab:orange", lw=2.0)
        (h_ll,) = ax.plot(ll_t, ll_v, color="tab:blue", lw=2.0)

        if col == 1:
            # PDE panel: full subspace trajectory
            (h_sub,) = ax.plot(sub_t, sub_v, color="tab:green", lw=2.0)
        else:
            # BC panel: draw a horizontal line at the final subspace BC value
            # so the log axis range and ticks register properly, then mark the
            # end-point with a scatter marker
            final_bc = sub_log[-1][2]
            (h_sub,) = ax.plot(
                [0, sub_log[-1][0]],
                [final_bc, final_bc],
                color="tab:green",
                lw=1.5,
                linestyle=":",
            )
            ax.scatter(
                [sub_log[-1][0]],
                [final_bc],
                color="tab:green",
                s=80,
                zorder=5,
            )

        # Reference line at the subspace projection target level
        target_val = sub_pde if col == 1 else sub_bc
        h_target = ax.axhline(
            target_val, color="tab:green", lw=1.0, linestyle="--", alpha=0.6
        )

        # Markers at the BC-crossing entry on both panels (same time, different y).
        # On the PDE panel this shows the PDE health at the moment BC is satisfied.
        cross_ft = (ft_cross_bc, ft_cross_bc_reached)
        cross_ll = (ll_cross_bc, ll_cross_bc_reached)
        for (entry, reached), color in [
            (cross_ft, "tab:orange"),
            (cross_ll, "tab:blue"),
        ]:
            if not reached and entry[0] <= x_lim:
                ax.scatter(
                    [entry[0]],
                    [entry[col]],
                    color=color,
                    marker="x",
                    s=80,
                    zorder=6,
                    linewidths=2,
                )

        ax.set_xlim(0, x_lim)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

        label = "(a)" if ax_idx == 0 else "(b)"
        ax.text(
            0.5,
            -0.13,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

        if ax_idx == 1:
            handles = [h_base, h_sub, h_target, h_ft, h_ll]
            labels = [
                "Base model (PDE only)",
                "Subspace projection",
                "Subspace target level",
                "Full fine-tuning",
                "Last-layer fine-tuning",
            ]

    # Single legend on the right panel
    axes[1].legend(handles, labels, fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "poisson_comparison.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
