"""
Demo: The Conceptual Arc of the Paper

Problem:  u_xx + (2*pi)^2 * sin(2*pi*x) = 0  on [-1, 1]
Kernel of u_xx:  span{1, x}  ->  analytical kernel dimension = 2.

The demo shows, in order:
  1. d_eff^pde after PDE-only training   ->  ~2   (matches the analytical
     kernel dimension, independent of architecture)
  2. d_eff^total as a well-posedness certificate:
       two BCs        -> ~0   (constraints absorb the kernel: certified)
       one BC         -> ~1   (counts exactly one missing constraint)
       one BC, x3     -> ~1   (redundant rows absorb nothing)
  3. Under-resolved training inflates d_eff to its ceiling N_p while the
     discrete loss reads ~1e-29 (spectral hallucination, detected label-free)
  4. Subspace-projection adaptation to new BCs u(-1)=10, u(1)=2 in <1 s,
     preserving the learned physics (exact solution: sin(2*pi*x) - 4x + 6)

Run:
    python demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
import torch.nn as nn

from torch.func import functional_call

from adr.pinn import MLP
from adr.training import train
from adr.deff import get_d_eff
from adr.adaptation import specify_bcs
from adr.utils import set_seed, use_float64

use_float64()


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_PDE = 100
HIDDEN = 50
ADAM_EPOCHS = 2_000
LBFGS_ITER = 3_000

N_SPARSE = 6
HIDDEN_SP = 50
DEPTH_SP = 4
ADAM_SP = 2_000
LBFGS_SP = 4_000

NUM_INCS_ADAPT = 20


# ---------------------------------------------------------------------------
# PDE residual:  u_xx + (2*pi)^2 * sin(2*pi*x) = 0
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


def _pde_mse(model, x):
    return pde_fn(model, x.clone().requires_grad_(True)).detach().pow(2).mean().item()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Demo -- device: {device}\n")
    print("Problem: u_xx + (2*pi)^2 * sin(2*pi*x) = 0,  ker(u_xx) = {1, x},  dim = 2")

    x_pde = torch.linspace(-1, 1, N_PDE, device=device).view(-1, 1)
    x_bc = torch.tensor([[-1.0], [1.0]], device=device)
    y_bc = torch.tensor([[0.0], [0.0]], device=device)
    layers = [0, 1]  # both Linear layers of the depth-1 MLP

    # ------------------------------------------------------------------
    # 1. d_eff^pde recovers the analytical kernel dimension
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print(f"  1. d_eff^pde after PDE-only training")
    print(f"{'#'*60}")
    set_seed(42)
    model = MLP(n_in=1, n_out=1, hidden=HIDDEN, depth=1, activation=nn.Tanh).to(device)

    print(f"\n  training on {N_PDE} collocation points ...")

    train(
        model,
        pde_fn,
        x_pde,
        x_bc,
        y_bc,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_max_iter=LBFGS_ITER,
        lambdas=(1.0, 0.0, 0.0),
        print_every=0,
    )

    d1 = get_d_eff(model, pde_fn, layers, x_pde, x_bc, mode="pde")
    print(f"\n  d_eff^pde after training  = {d1:.3f}   (analytical kernel dim = 2)")
    print(f"  PDE residual MSE          = {_pde_mse(model, x_pde):.2e}")

    # ------------------------------------------------------------------
    # 2. Well-posedness certificate: d_eff^total counts constraint deficits
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print(f"  2. d_eff^total as a well-posedness certificate")
    print(f"{'#'*60}")

    cases = [
        ("two BCs  u(-1), u(1)  ", torch.tensor([[-1.0], [1.0]]), "~0  -- certified"),
        (
            "one BC   u(-1) only   ",
            torch.tensor([[-1.0]]),
            "~1  -- one constraint missing",
        ),
        (
            "one BC duplicated x3  ",
            torch.tensor([[-1.0], [-1.0], [-1.0]]),
            "~1  -- redundancy absorbs nothing",
        ),
    ]
    print()
    for label, xb, expect in cases:
        d = get_d_eff(
            model, pde_fn, layers, x_pde, xb.to(device), mode="total", engine="ntk"
        )
        print(f"  {label}  d_eff^total = {d:.3f}   ({expect})")

    # ------------------------------------------------------------------
    # 3. The spatial sieve: under-resolved training inflates d_eff^pde
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print(f"  3. Spectral hallucination on a {N_SPARSE}-point grid")
    print(f"{'#'*60}")
    print(f"\n  training on {N_SPARSE} collocation points ...")

    set_seed(42)
    x_sparse = torch.linspace(-1, 1, N_SPARSE, device=device).view(-1, 1)
    m_sparse = MLP(
        n_in=1, n_out=1, hidden=HIDDEN_SP, depth=DEPTH_SP, activation=nn.SiLU
    ).to(device)
    train(
        m_sparse,
        pde_fn,
        x_sparse,
        x_bc,
        torch.ones_like(y_bc),
        adam_epochs=ADAM_SP,
        lbfgs_max_iter=LBFGS_SP,
        lambdas=(1.0, 1.0, 0.0),
        print_every=0,
    )

    d_sp = get_d_eff(
        m_sparse,
        pde_fn,
        list(range(DEPTH_SP + 1)),
        x_sparse,
        x_bc,
        mode="pde",
        engine="ntk",
    )
    disc = _pde_mse(m_sparse, x_sparse)
    cont = _pde_mse(m_sparse, torch.linspace(-1, 1, 1000, device=device).view(-1, 1))

    print(f"\n  discrete PDE loss ({N_SPARSE} pts)      = {disc:.2e}   (looks perfect)")
    print(f"  continuous PDE loss (1000 pts)  = {cont:.2e}   (it is not)")
    print(
        f"  d_eff^pde                       = {d_sp:.3f}   (saturated at ceiling {N_SPARSE}; kernel floor is 2)"
    )

    # ------------------------------------------------------------------
    # 4. Physics-preserving adaptation to new BCs
    # ------------------------------------------------------------------
    print(f"\n{'#'*60}")
    print(f"  4. Subspace-projection adaptation to new BCs  u(-1)=10, u(1)=2")
    print(f"{'#'*60}")

    bcs_new = [
        {"coords": [-1.0], "type": "dirichlet", "val": 10.0},
        {"coords": [1.0], "type": "dirichlet", "val": 2.0},
    ]
    base_mse = _pde_mse(model, x_pde)
    t0 = time.perf_counter()
    specify_bcs(
        model,
        pde_fn,
        bcs_new,
        layers,
        x_pde,
        num_increments=NUM_INCS_ADAPT,
        max_iter=10,
        tolerance=1e-14,
    )
    t_adapt = time.perf_counter() - t0

    x_plot = torch.linspace(-1, 1, 500, device=device).view(-1, 1)
    u_exact = (torch.sin(2 * np.pi * x_plot) - 4 * x_plot + 6).flatten()
    with torch.no_grad():
        u_pred = model(x_plot).flatten()
        u_b = model(x_bc).flatten()
    rmse = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
    bc_err = max(abs(u_b[0].item() - 10.0), abs(u_b[1].item() - 2.0))

    print(f"\n  adaptation time             = {t_adapt:.2f} s ")
    print(
        f"  PDE residual MSE            = {_pde_mse(model, x_pde):.2e}   (base was {base_mse:.2e})"
    )
    print(f"  BC error                    = {bc_err:.2e}")
    print(f"  RMSE vs exact sin(2pi x)-4x+6 = {rmse:.2e}")


if __name__ == "__main__":
    run()
