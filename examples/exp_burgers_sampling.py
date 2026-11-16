"""
Experiment: collocation sampling during Burgers IC/BC adaptation

Same base model, targets and finite-difference references as
exp_burgers_adaptation.py (Section 6.3). Only the subspace projection is run,
with three choices of the PDE collocation points used by the continuation
(100 increments, predictor-corrector, all layers, rcond 1e-8, 1600 points per
increment in every case):

  fixed    the 40 x 40 adaptation grid of the paper, the same points at every
           increment, no final corrector (reproduces the paper's subspace rows)
  uniform  1600 new i.i.d. uniform points at every increment, plus a final
           corrector on a fresh draw
  hybrid   800 scrambled-Sobol points plus 800 points drawn from the 200 x 160
           training grid with probability ~ |r|^2 + 0.2 mean|r|^2 (r = current
           PDE residual), new at every increment, plus a final corrector

New points are drawn only at the start of an increment, so each corrector step
sees points the previous increment did not constrain (adr/sampling.py).

Reported: wall time, PDE MSE on the full 200 x 160 grid, IC/BC MSE of the
adapted constraint, RMSE against the finite-difference reference (200 x 200).

Run:
    python examples/exp_burgers_sampling.py
    python examples/exp_burgers_sampling.py --scenario ic --strategies fixed hybrid
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn

import exp_burgers_adaptation as B
from adr.pinn import MLP
from adr.training import train
from adr.adaptation import specify_bcs
from adr.sampling import UniformSampler, HybridSampler
from adr.utils import set_seed

N_POINTS = B.N_ADAPT_GRID**2  # 1600, as in the paper
LOWER, UPPER = [-1.0, 0.0], [1.0, 1.0]


def setup(device):
    x_grid = torch.linspace(-1, 1, B.N_PDE_X, device=device)
    t_grid = torch.linspace(0, 1, B.N_PDE_T, device=device)
    Xg, Tg = torch.meshgrid(x_grid, t_grid, indexing="ij")
    x_pde = torch.stack([Xg.flatten(), Tg.flatten()], dim=1)
    x_ic = torch.stack([x_grid, torch.zeros_like(x_grid)], dim=1)
    y_ic = (-torch.sin(np.pi * x_grid)).unsqueeze(1)
    x_sbc = torch.cat([torch.stack([torch.full_like(t_grid, -1.0), t_grid], 1),
                       torch.stack([torch.full_like(t_grid, 1.0), t_grid], 1)])
    y_sbc = torch.zeros(len(x_sbc), 1, device=device)
    xa = torch.linspace(-1, 1, B.N_ADAPT_GRID, device=device)
    ta = torch.linspace(0, 1, B.N_ADAPT_GRID, device=device)
    Xa, Ta = torch.meshgrid(xa, ta, indexing="ij")
    x_pde_adapt = torch.stack([Xa.flatten(), Ta.flatten()], dim=1)
    return x_grid, x_pde, x_ic, y_ic, x_sbc, y_sbc, x_pde_adapt


def scenarios(x_grid, x_ic, y_ic, x_sbc, y_sbc):
    """Adaptation BC lists and the dense constraint tensors used for IC/BC MSE (as in the paper)."""
    x_bc_pts = np.linspace(-1, 1, B.N_ADAPT_IC)
    t_bc_pts = np.linspace(0, 1, B.N_ADAPT_BC)
    side = lambda fl, fr: [d for tv in t_bc_pts for d in (
        {"type": "dirichlet", "coords": [-1.0, float(tv)], "val": fl(tv)},
        {"type": "dirichlet", "coords": [1.0, float(tv)], "val": fr(tv)})]
    ic = lambda f: [{"type": "dirichlet", "coords": [float(xv), 0.0], "val": float(f(xv))} for xv in x_bc_pts]
    bcs_ic = ic(lambda x: -2.0 * np.sin(np.pi * x)) + side(lambda t: 0.0, lambda t: 0.0)
    bcs_bc = ic(lambda x: -np.sin(np.pi * x)) + side(lambda t: 0.5 * float(t), lambda t: -0.5 * float(t))
    y_ic_new = (-2.0 * torch.sin(np.pi * x_grid)).unsqueeze(1)
    y_sbc_new = torch.where(x_sbc[:, 0:1] < 0, 0.5 * x_sbc[:, 1:2], -0.5 * x_sbc[:, 1:2])
    x_c = torch.cat([x_ic, x_sbc])
    return {
        "ic": dict(bcs=bcs_ic, x_c=x_c, y_c=torch.cat([y_ic_new, y_sbc]),
                   ref=dict(u0_fn=lambda x: -2.0 * np.sin(np.pi * x))),
        "bc": dict(bcs=bcs_bc, x_c=x_c, y_c=torch.cat([y_ic, y_sbc_new]),
                   ref=dict(u0_fn=lambda x: -np.sin(np.pi * x), bc_left=lambda t: 0.5 * t,
                            bc_right=lambda t: -0.5 * t)),
    }


def base_model(device, x_pde, x_bc_base, y_bc_base, path):
    set_seed(42)
    model = MLP(n_in=2, n_out=1, hidden=B.HIDDEN, depth=B.DEPTH, activation=nn.Tanh).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model._adr_base_state = {n: p.detach().clone() for n, p in model.named_parameters()}
        print(f"loaded base model {path}")
    else:
        train(model, B.pde_fn, x_pde, x_bc_base, y_bc_base, adam_epochs=B.ADAM_EPOCHS,
              lbfgs_max_iter=B.LBFGS_MAX_ITER, lr=B.LR, lambdas=(B.LAMBDA_PDE, B.LAMBDA_BC, 0.0),
              print_every=2000)
        torch.save(model.state_dict(), path)
        print(f"trained and saved base model {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["ic", "bc", "both"], default="both")
    ap.add_argument("--strategies", nargs="+", default=["fixed", "uniform", "hybrid"],
                    choices=["fixed", "uniform", "hybrid"])
    ap.add_argument("--seed", type=int, default=0, help="sampler seed")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "burgers_sampling"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_grid, x_pde, x_ic, y_ic, x_sbc, y_sbc, x_pde_adapt = setup(device)
    model = base_model(device, x_pde, torch.cat([x_ic, x_sbc]), torch.cat([y_ic, y_sbc]),
                       os.path.join(args.out, "burgers_base.pt"))
    print(f"base PDE MSE (full grid): {B.compute_pde_loss(model, x_pde, B.pde_fn):.4e}")

    N_eval = 200
    x_eval, t_eval = np.linspace(-1, 1, N_eval), np.linspace(0, 1, N_eval)
    xt_eval = torch.tensor(np.stack(np.meshgrid(x_eval, t_eval, indexing="ij"), -1).reshape(-1, 2),
                           dtype=torch.float64, device=device)
    scen = scenarios(x_grid, x_ic, y_ic, x_sbc, y_sbc)

    def sampler(name):
        if name == "fixed":
            return None
        if name == "uniform":
            return UniformSampler(N_POINTS, LOWER, UPPER, seed=args.seed, device=device)
        return HybridSampler(N_POINTS, LOWER, UPPER, pool=x_pde, frac_residual=0.5,
                             power=2.0, floor=0.2, seed=args.seed)

    results = []
    for sc in (["ic", "bc"] if args.scenario == "both" else [args.scenario]):
        S = scen[sc]
        x_ref, t_ref, u_raw = B.burgers_fd_reference(**S["ref"])
        u_ref, _, _ = B.interp_fd(x_ref, t_ref, u_raw, x_eval, t_eval)
        for name in args.strategies:
            B.restore_base(model)
            t0 = time.time()
            specify_bcs(model, B.pde_fn, S["bcs"], B.ALL_LAYERS, x_pde_adapt,
                        num_increments=B.NUM_INCS, use_corrector=True, tolerance=1e-8,
                        max_iter=15, rcond=1e-8, x_pde_sampler=sampler(name),
                        final_corrector=(name != "fixed"))
            elapsed = time.time() - t0
            with torch.no_grad():
                c_mse = torch.mean((model(S["x_c"]) - S["y_c"]) ** 2).item()
                u = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
            row = {
                "scenario": sc, "strategy": name, "time_s": elapsed,
                "pde_mse_full": B.compute_pde_loss(model, x_pde, B.pde_fn),
                "pde_mse_adapt_grid": B.compute_pde_loss(model, x_pde_adapt, B.pde_fn),
                "constraint_mse": c_mse, "rmse": float(np.sqrt(np.mean((u - u_ref) ** 2))),
            }
            results.append(row)
            print(f"[{sc}] {name:<8} time {elapsed:7.1f}s  full PDE MSE {row['pde_mse_full']:.3e}  "
                  f"adapt-grid PDE MSE {row['pde_mse_adapt_grid']:.3e}  IC/BC MSE {c_mse:.3e}  "
                  f"RMSE {row['rmse']:.3e}", flush=True)
            np.save(os.path.join(args.out, f"u_{sc}_{name}.npy"), u)
        np.save(os.path.join(args.out, f"u_ref_{sc}.npy"), u_ref)

    with open(os.path.join(args.out, f"burgers_sampling_seed{args.seed}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n" + "\n".join(f"{r['scenario']:<3} {r['strategy']:<8} {r['time_s']:7.1f}s  "
                           f"{r['pde_mse_full']:.3e}  {r['constraint_mse']:.3e}  {r['rmse']:.3e}"
                           for r in results))


if __name__ == "__main__":
    main()
