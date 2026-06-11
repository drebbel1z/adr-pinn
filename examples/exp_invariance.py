"""
Experiment: d_eff Invariance Across ODE Orders

Trains a PINN purely on the PDE residual (no BC loss) for 1st-, 2nd-, and
3rd-order ODEs that all share sin(2pi x) as a particular solution.  Shows
that d_eff is invariant to the architecture

Run:
    python examples/exp_invariance.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.func import functional_call

from adr.pinn import MLP, ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64

use_float64()


# ---------------------------------------------------------------------------
# ODE residuals (all share sin(2pi x) as a particular solution)
# ---------------------------------------------------------------------------


def ode_residual(order):
    """Return pde_fn for a 1D ODE of the given order."""

    def pde_fn(model, x, param_dict=None):
        pd = param_dict if param_dict is not None else dict(model.named_parameters())

        def u_fn(xi):
            return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

        def residual_single(xi):
            g1 = torch.func.grad(lambda y: u_fn(y)[0])(xi)[0]
            if order == 1:
                return g1 - (2.0 * np.pi) * torch.cos(2.0 * np.pi * xi[0])
            g2 = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
            if order == 2:
                return g2 + (4.0 * np.pi**2) * torch.sin(2.0 * np.pi * xi[0])
            g3 = torch.func.grad(lambda w: torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(w)[0])(xi)[0]
            return g3 + (8.0 * np.pi**3) * torch.cos(2.0 * np.pi * xi[0])

        return torch.func.vmap(residual_single)(x).unsqueeze(1)

    return pde_fn


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"d_eff Invariance Study -- device: {device}\n")

    x_pde = torch.linspace(-1, 1, 100, device=device).view(-1, 1)

    # Dummy BC tensors -- training is PDE-only (lam_bc = 0)
    x_bc = torch.tensor([[-1.0]], device=device)
    y_bc = torch.tensor([[0.0]], device=device)

    orders = [1, 2, 3]
    architectures = [
        {"depth": 1, "hidden": 50},
        {"depth": 3, "hidden": 100},
    ]
    activations = [
        {"name": "SiLU", "cls": nn.SiLU},
        {"name": "Tanh", "cls": nn.Tanh},
    ]
    model_classes = [
        {"name": "MLP", "cls": MLP},
        {"name": "ResNet", "cls": ResNet},
    ]

    results = []

    for mc in model_classes:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {mc['name']}")
        print(f"{'#'*60}")

        for order in orders:
            pde_fn = ode_residual(order)
            print(f"\n{'='*55}")
            print(f"  ODE ORDER {order}")
            print(f"{'='*55}")

            for arch in architectures:
                for act in activations:
                    tag = f"{act['name']:4s} | depth={arch['depth']} | width={arch['hidden']}"
                    print(f"\n  [{tag}]")

                    set_seed(42)
                    model = mc["cls"](
                        n_in=1,
                        n_out=1,
                        hidden=arch["hidden"],
                        depth=arch["depth"],
                        activation=act["cls"],
                    ).to(device)

                    train(
                        model,
                        pde_fn,
                        x_pde,
                        x_bc,
                        y_bc,
                        adam_epochs=3000,
                        lbfgs_max_iter=5000,
                        lambdas=(1.0, 0.0, 0.0),
                        print_every=0,
                    )

                    n_linear = arch["depth"] + 1  # depth hidden layers + 1 output layer

                    row = {
                        "model": mc["name"],
                        "order": order,
                        "activation": act["name"],
                        "depth": arch["depth"],
                        "hidden": arch["hidden"],
                    }

                    # All layers combined
                    d_all = get_d_eff(
                        model, pde_fn, list(range(n_linear)), x_pde, x_bc, mode="pde"
                    )
                    row["all"] = round(d_all, 3)
                    print(f"    all layers  d_eff = {d_all:.4f}")

                    # Per layer
                    for i in range(n_linear):
                        d = get_d_eff(model, pde_fn, [i], x_pde, x_bc, mode="pde")
                        row[f"L{i}"] = round(d, 3)
                        print(f"    layer {i}      d_eff = {d:.4f}")

                    results.append(row)

    df = pd.DataFrame(results)
    print("\n\n" + "=" * 80)
    print("RESULTS: d_eff INVARIANCE ACROSS ODE ORDERS")
    print("=" * 80)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
