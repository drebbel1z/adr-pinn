"""
Experiment: d_eff Invariance for a Variable-Coefficient Operator (Bessel)

Tests that d_eff^pde converges to the analytical kernel dimension for an
operator with non-constant coefficients.  The order-zero Bessel equation

    x^2 u'' + x u' + x^2 u = 0,   x in [1, 8]

has a two-dimensional kernel spanned by J_0(x) and Y_0(x), so the expected
result is d_eff^pde -> 2 regardless of architecture.

The equation is homogeneous, so without BCs the network collapses to u=0.
Two Dirichlet BCs pin the solution to J_0(x):
    u(1) = J_0(1) ~ 0.7652,  u(8) = J_0(8) ~ 0.1717

d_eff^pde is still computed from the PDE Jacobian only, so it measures the
null-space dimension of the operator irrespective of which particular solution
the network is anchored to.

Run:
    python examples/exp_bessel_invariance.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pandas as pd
from torch.func import functional_call

from adr.pinn import MLP, ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64

use_float64()


def bessel_pde(model, x, param_dict=None):
    """Residual of x^2 u'' + x u' + x^2 u = 0 (Bessel, n=0)."""
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u = u_fn(xi)[0]
        du = torch.func.grad(lambda y: u_fn(y)[0])(xi)[0]
        d2u = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        return xi[0]**2 * d2u + xi[0] * du + xi[0]**2 * u

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Bessel d_eff Invariance -- device: {device}\n")

    x_pde = torch.linspace(1.0, 8.0, 100, device=device).view(-1, 1)

    # Two Dirichlet BCs from J_0 to anchor the network to a non-trivial solution.
    # scipy.special.j0(1) ~ 0.7652, j0(8) ~ 0.1717
    x_bc = torch.tensor([[1.0], [8.0]], device=device)
    y_bc = torch.tensor([[0.7652], [0.1717]], device=device)

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

        for arch in architectures:
            for act in activations:
                tag = (
                    f"{act['name']:4s} | depth={arch['depth']} | width={arch['hidden']}"
                )
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
                    bessel_pde,
                    x_pde,
                    x_bc,
                    y_bc,
                    adam_epochs=5000,
                    lbfgs_max_iter=5000,
                    lambdas=(1.0, 1.0, 0.0),
                    print_every=0,
                )

                # x_t = x_pde.clone().requires_grad_(True)
                # pde_mse = bessel_pde(model, x_t).detach().pow(2).mean().item()
                # print(f"    PDE residual MSE = {pde_mse:.2e}")

                n_linear = arch["depth"] + 1  # hidden layers + output layer

                row = {
                    "model": mc["name"],
                    "activation": act["name"],
                    "depth": arch["depth"],
                    "hidden": arch["hidden"],
                }

                d_all = get_d_eff(
                    model,
                    bessel_pde,
                    list(range(n_linear)),
                    x_pde,
                    x_bc,
                    mode="pde",
                )
                row["all"] = round(d_all, 3)
                print(f"    all layers  d_eff = {d_all:.4f}")

                for i in range(n_linear):
                    d = get_d_eff(model, bessel_pde, [i], x_pde, x_bc, mode="pde")
                    row[f"L{i}"] = round(d, 3)
                    print(f"    layer {i}      d_eff = {d:.4f}")

                results.append(row)

    df = pd.DataFrame(results)
    print("\n\n" + "=" * 60)
    print("RESULTS: d_eff INVARIANCE -- BESSEL OPERATOR (expected: 2.0)")
    print("=" * 60)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
