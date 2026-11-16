"""
Burgers sparse-grid fine-tuning baseline with resampled collocation points.

Re-runs only the sparse-grid ("adapt grid") fine-tuning baseline of
exp_burgers_adaptation.py, with the PDE collocation points drawn by the same hybrid
sampler as the subspace projection (1,600 points: half scrambled Sobol, half from the
200 x 160 training grid with probability ~ |r|^2 + 0.2 mean(|r|^2)) instead of a fixed
40 x 40 grid. Points are redrawn every 100 Adam steps, and L-BFGS is restarted on a
fresh draw every 300 iterations.

The equal-time budgets t_sub and the subspace BC targets are those of the paper run
(IC: examples/results/burgers_hybrid, BC: examples/results/paper_runs), so the subspace projection
and full-grid fine-tuning do not need to be re-run.

Run:
    python examples/exp_burgers_ft_resampled.py
Output:
    examples/results/burgers_ft_resampled/summary.json
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn

import exp_burgers_adaptation as B
from adr.pinn import MLP

# From the runs reported in the paper (IC: results/burgers_hybrid, BC: results/paper_runs)
T_SUB = {"ic": 80.27, "bc": 79.75}
BC_TARGET = {"ic": 4.8993e-08, "bc": 4.6278e-07}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    out_dir = os.path.join(HERE, "results", "burgers_ft_resampled")
    os.makedirs(out_dir, exist_ok=True)

    x_grid = torch.linspace(-1, 1, B.N_PDE_X, device=device)
    t_grid = torch.linspace(0, 1, B.N_PDE_T, device=device)
    Xg, Tg = torch.meshgrid(x_grid, t_grid, indexing="ij")
    x_pde = torch.stack([Xg.flatten(), Tg.flatten()], dim=1)

    x_ic = torch.stack([x_grid, torch.zeros_like(x_grid)], dim=1)
    y_ic = (-torch.sin(np.pi * x_grid)).unsqueeze(1)
    x_sbc = torch.cat([
        torch.stack([torch.full_like(t_grid, -1.0), t_grid], dim=1),
        torch.stack([torch.full_like(t_grid, 1.0), t_grid], dim=1),
    ])
    y_sbc = torch.zeros(x_sbc.shape[0], 1, device=device)

    scen = {
        "ic": (torch.cat([x_ic, x_sbc]),
               torch.cat([(-2.0 * torch.sin(np.pi * x_grid)).unsqueeze(1), y_sbc])),
        "bc": (torch.cat([x_ic, x_sbc]),
               torch.cat([y_ic, torch.where(x_sbc[:, 0:1] < 0, 0.5 * x_sbc[:, 1:2],
                                            -0.5 * x_sbc[:, 1:2])])),
    }

    N_eval = 200
    x_eval = np.linspace(-1, 1, N_eval)
    t_eval = np.linspace(0, 1, N_eval)
    xt_eval = torch.tensor(np.stack(np.meshgrid(x_eval, t_eval, indexing="ij"), axis=-1)
                           .reshape(-1, 2), dtype=torch.float64, device=device)

    print("Computing FD references...")
    x_ref, t_ref, _ = B.burgers_fd_reference(lambda x: -np.sin(np.pi * x))
    _, _, u_ic_raw = B.burgers_fd_reference(lambda x: -2.0 * np.sin(np.pi * x))
    _, _, u_bc_raw = B.burgers_fd_reference(lambda x: -np.sin(np.pi * x),
                                            bc_left=lambda t: 0.5 * t,
                                            bc_right=lambda t: -0.5 * t)
    u_ref = {"ic": B.interp_fd(x_ref, t_ref, u_ic_raw, x_eval, t_eval)[0],
             "bc": B.interp_fd(x_ref, t_ref, u_bc_raw, x_eval, t_eval)[0]}

    model = MLP(n_in=2, n_out=1, hidden=B.HIDDEN, depth=B.DEPTH, activation=nn.Tanh).to(device)
    base_path = os.path.join(HERE, "results", "burgers_sampling", "burgers_base.pt")
    base_state = torch.load(base_path, map_location=device)

    def rmse_now():
        with torch.no_grad():
            u = model(xt_eval).cpu().numpy().reshape(N_eval, N_eval)
        return u

    summary = {}
    for key in ["ic", "bc"]:
        x_bc, y_bc = scen[key]
        model.load_state_dict(base_state)
        print(f"\n### {key.upper()} adaptation: sparse-grid FT, resampled (t_sub={T_SUB[key]}s)")
        log, total, snap = B.fine_tune(
            model, None, x_bc, y_bc, fn=B.pde_fn, x_pde_log=x_pde,
            snapshot_time=T_SUB[key], x_pde_sampler=B.hybrid_sampler(x_pde),
        )
        u_end = rmse_now()
        end = log[-1]
        tm = B.snapshot_at_time(log, T_SUB[key])
        if snap is not None:
            model.load_state_dict(snap)
        u_tm = rmse_now()
        r_tm = float(np.sqrt(np.mean((u_tm - u_ref[key]) ** 2)))
        r_end = float(np.sqrt(np.mean((u_end - u_ref[key]) ** 2)))
        cb, reached = B.time_to_reach(log, 2, BC_TARGET[key])
        res = {
            "t_sub": T_SUB[key],
            "at_t_sub": {"time": tm[0], "pde_mse": tm[1], "bc_mse": tm[2], "rmse": r_tm},
            "end": {"time": total, "pde_mse": end[1], "bc_mse": end[2], "rmse": r_end},
            "bc_target": BC_TARGET[key],
            "bc_target_reached": reached,
            "bc_target_entry": {"time": cb[0], "pde_mse": cb[1], "bc_mse": cb[2]},
        }
        summary[key] = res
        print(f"  @ t_sub: PDE {tm[1]:.4e} | BC {tm[2]:.4e} | RMSE {r_tm:.4e}")
        print(f"  @ end ({total:.1f}s): PDE {end[1]:.4e} | BC {end[2]:.4e} | RMSE {r_end:.4e}")
        if reached:
            print(f"  BC target reached at t={cb[0]:.1f}s ({cb[0]/T_SUB[key]:.1f}x t_sub), PDE {cb[1]:.4e}")
        else:
            print(f"  BC target never reached; best BC {cb[2]:.4e} at t={cb[0]:.1f}s, PDE {cb[1]:.4e}")
        np.save(os.path.join(out_dir, f"log_{key}.npy"), np.array(log))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\nSaved", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
