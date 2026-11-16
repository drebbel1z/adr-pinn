"""
Experiment: sensitivity of per-layer d_eff^pde to the pseudo-inverse cutoff (rcond)

Checks the Layer 0 entry of Table 1 (3rd-order ODE, Tanh, 3 hidden layers x 100).
Trains the feedforward and skip-connection networks exactly as in
exp_invariance.py (seed 42, PDE loss only, Adam 3000 + L-BFGS 5000, N = 100) and
evaluates d_eff^pde for every layer and for all layers together over a range of
rcond values.

Every layer of both networks contains all three kernel functions of d^3/dx^3
(d_eff^pde -> 3.000 as rcond -> 1e-14). In the feedforward network two of them
are represented in the input layer only through very small singular values, so
its value depends on the cutoff (2.85 at the paper's rcond = 1e-10); the
skip-connection network keeps them better conditioned.

Run:
    python examples/exp_rcond_sweep.py
    python examples/exp_rcond_sweep.py --order 2 --activation SiLU --lam 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from adr.pinn import MLP, ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64
from exp_invariance import ode_residual

use_float64()

RCONDS = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
ACTIVATIONS = {"Tanh": nn.Tanh, "SiLU": nn.SiLU}
MODELS = {"Feedforward": MLP, "Skip Connections": ResNet}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--activation", default="Tanh", choices=list(ACTIVATIONS))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde_fn = ode_residual(args.order)
    x_pde = torch.linspace(-1, 1, 100, device=device).view(-1, 1)
    x_bc = torch.tensor([[-1.0]], device=device)
    y_bc = torch.tensor([[0.0]], device=device)
    n_linear = args.depth + 1

    print(
        f"order {args.order}, {args.activation}, {args.depth} x {args.width}, "
        f"seed {args.seed}, lam = {args.lam:g}, device {device}"
    )
    header = f"{'':<28}" + "".join(f"{rc:>9.0e}" for rc in RCONDS)

    for name, cls in MODELS.items():
        set_seed(args.seed)
        model = cls(
            n_in=1, n_out=1, hidden=args.width, depth=args.depth,
            activation=ACTIVATIONS[args.activation],
        ).to(device)
        train(
            model, pde_fn, x_pde, x_bc, y_bc,
            adam_epochs=3000, lbfgs_max_iter=5000,
            lambdas=(1.0, 0.0, 0.0), print_every=0,
        )
        print(f"\n{name}  (rows: layers, columns: rcond)")
        print(header)
        for layers in [[i] for i in range(n_linear)] + [list(range(n_linear))]:
            label = f"Layer {layers[0]}" if len(layers) == 1 else "All layers"
            vals = [
                get_d_eff(model, pde_fn, layers, x_pde, x_bc, mode="pde",
                          engine="ntk", rcond=rc, lam=args.lam)
                for rc in RCONDS
            ]
            print(f"  {label:<26}" + "".join(f"{v:>9.4f}" for v in vals), flush=True)


if __name__ == "__main__":
    main()
