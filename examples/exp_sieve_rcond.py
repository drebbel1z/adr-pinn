"""
Experiment: sensitivity of the spatial-sieve table (Table 4) to the pseudo-inverse cutoff

Same networks, training and grids as exp_spatial_sieve.py (2nd-order ODE with Dirichlet BCs,
4 hidden layers x 100, SiLU, feedforward and skip-connection, seed 42, 6..26 points). For
every trained network, d_eff_total (before and after training) and d_eff_pde (after) are
evaluated at the training points for each rcond in --rconds (default 1e-6 ... 1e-14; the
paper uses 1e-10).

Output: results/sieve_rcond/sieve_rcond.csv and a printed table.
Run:
    python examples/exp_sieve_rcond.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import argparse

import pandas as pd
import torch
import torch.nn as nn

from exp_spatial_sieve import pde_fn
from adr.pinn import MLP, ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64

use_float64()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rconds", type=float, nargs="+", default=[1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sieve_rcond"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_bc = torch.tensor([[-1.0], [1.0]], device=device)
    y_bc = torch.tensor([[1.0], [1.0]], device=device)
    rows = []
    for name, cls in (("Feedforward", MLP), ("Skip Connection", ResNet)):
        for n_pts in (6, 10, 14, 18, 22, 26):
            set_seed(42)
            x_pde = torch.linspace(-1, 1, n_pts, device=device).view(-1, 1)
            model = cls(n_in=1, n_out=1, hidden=100, depth=4, activation=nn.SiLU).to(device)
            layers = list(range(5))

            def deff(mode, rc):
                x_t = x_pde.clone().requires_grad_(True)
                return get_d_eff(model, pde_fn, layers, x_t, x_bc, mode=mode, engine="ntk", rcond=rc)

            pre = {rc: deff("total", rc) for rc in args.rconds}
            train(model, pde_fn, x_pde, x_bc, y_bc, adam_epochs=2000, lbfgs_max_iter=5000,
                  lambdas=(1.0, 1.0, 0.0), print_every=0)
            for rc in args.rconds:
                r = {"architecture": name, "grid_pts": n_pts, "rcond": rc, "d_eff_pre": pre[rc],
                     "d_eff_post": deff("total", rc), "d_eff_pde": deff("pde", rc)}
                rows.append(r)
                print(f"{name:15s} N={n_pts:2d} rcond={rc:.0e}: pre {r['d_eff_pre']:.3f}  "
                      f"post {r['d_eff_post']:.3f}  pde {r['d_eff_pde']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "sieve_rcond.csv"), index=False)
    df = pd.DataFrame(rows)
    for col in ("d_eff_pre", "d_eff_post", "d_eff_pde"):
        print(f"\n{col}")
        print(df.pivot_table(index=["architecture", "grid_pts"], columns="rcond", values=col)
              .round(3).to_string())


if __name__ == "__main__":
    main()
