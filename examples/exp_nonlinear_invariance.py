"""
Experiment: d_eff Invariance for Nonlinear Operators

Tests that d_eff^pde converges to 2 (the analytical solution-manifold dimension)
for two genuinely nonlinear second-order ODEs that share the same exact solution
on the same domain, isolating the effect of operator structure.

Both equations use u* = sin(pi x) on [-1, 1],  BCs: u(-1) = u(1) = 0.

  EQ1 -- Forced Burgers (convective nonlinearity):
      u'' + u u' = f(x)
    f(x) = -pi^2 sin(pi x) + (pi/2) sin(2 pi x)
    Linearized Jacobian: v'' + u* v' + u*' v  (has a v' term)

  EQ2 -- Forced quadratic (algebraic nonlinearity):
      u'' + u^2 = f(x)
    f(x) = -pi^2 sin(pi x) + sin^2(pi x)
    Linearized Jacobian: v'' + 2 u* v  (no v' term)

Both have the same order and the same exact solution; the only structural
difference is the presence or absence of the convective v' term in the
linearized operator.  Expected d_eff = 2 for both.

Run:
    python examples/exp_nonlinear_invariance.py
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


def forced_burgers_pde(model, x, param_dict=None):
    """Residual of u'' + u u' - f(x) = 0, f chosen so u* = sin(pi x)."""
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u = u_fn(xi)[0]
        du = torch.func.grad(lambda y: u_fn(y)[0])(xi)[0]
        d2u = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        f = -(torch.pi**2) * torch.sin(torch.pi * xi[0]) + (torch.pi / 2) * torch.sin(2 * torch.pi * xi[0])
        return d2u + u * du - f

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def quadratic_pde(model, x, param_dict=None):
    """Residual of u'' + u^2 - f(x) = 0, f chosen so u* = sin(pi x)."""
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u = u_fn(xi)[0]
        d2u = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        f = -(torch.pi**2) * torch.sin(torch.pi * xi[0]) + torch.sin(torch.pi * xi[0]) ** 2
        return d2u + u**2 - f

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


EQUATIONS = [
    {
        "name": "EQ1: u''+uu'=f(x) (forced Burgers, convective)",
        "pde_fn": forced_burgers_pde,
    },
    {
        "name": "EQ2: u''+u^2=f(x) (quadratic, algebraic)",
        "pde_fn": quadratic_pde,
    },
]

X_PDE = None  # built inside run() once device is known


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Nonlinear d_eff Invariance -- device: {device}\n")
    print("Both equations: u* = sin(pi x) on [-1, 1],  BCs: u(-1)=u(1)=0\n")

    x_pde = torch.linspace(-1.0, 1.0, 100, device=device).view(-1, 1)
    x_bc = torch.tensor([[-1.0], [1.0]], dtype=torch.float64, device=device)
    y_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float64, device=device)

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

    all_results = {}

    for eq in EQUATIONS:
        print(f"\n{'='*65}")
        print(f"  {eq['name']}")
        print(f"{'='*65}")

        pde_fn = eq["pde_fn"]
        results = []

        for mc in model_classes:
            print(f"\n{'#'*60}")
            print(f"  MODEL: {mc['name']}")
            print(f"{'#'*60}")

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
                        adam_epochs=10000,
                        lbfgs_max_iter=10000,
                        lambdas=(1.0, 1.0, 0.0),
                        print_every=0,
                    )

                    x_t = x_pde.clone().requires_grad_(True)
                    pde_mse = pde_fn(model, x_t).detach().pow(2).mean().item()
                    print(f"    PDE MSE     = {pde_mse:.2e}")

                    n_linear = arch["depth"] + 1

                    row = {
                        "model": mc["name"],
                        "activation": act["name"],
                        "depth": arch["depth"],
                        "hidden": arch["hidden"],
                        "pde_mse": f"{pde_mse:.2e}",
                    }

                    d_all = get_d_eff(
                        model, pde_fn, list(range(n_linear)), x_pde, x_bc, mode="pde"
                    )
                    row["all"] = round(d_all, 3)
                    print(f"    all layers  d_eff = {d_all:.4f}")

                    for i in range(n_linear):
                        d = get_d_eff(model, pde_fn, [i], x_pde, x_bc, mode="pde")
                        row[f"L{i}"] = round(d, 3)
                        print(f"    layer {i}      d_eff = {d:.4f}")

                    results.append(row)

        all_results[eq["name"]] = pd.DataFrame(results)

    print("\n\n" + "=" * 65)
    print("SUMMARY (expected d_eff = 2.0 for both)")
    print("=" * 65)
    for name, df in all_results.items():
        print(f"\n--- {name} ---")
        print(df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    run()
