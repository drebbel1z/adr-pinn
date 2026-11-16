"""
Experiment: layer-selection diagnostics for 2D Poisson boundary adaptation

Retrains the base models of exp_2d_poisson_{tanh,silu}.py (ResNet 3 x 100,
seed 42, 20 x 20 collocation grid, PDE-only) and, for every adaptation
configuration used there, reports diagnostics computed at the base state,
before any adaptation:

  d_eff^pde, d_eff^bc, d_eff^total   (adr.deff, NTK engine, rcond = 1e-8)
  ratio      = d_eff^bc / d_eff^pde  (heuristic used in the paper)
  gap        = d_eff^pde - d_eff^total
  rank, s_min, cond of the projected boundary Jacobian
             J_proj = J_bc (I - J_pde^+ J_pde), built exactly as in
             adr.adaptation._scalable_ntk; its Gram J_proj J_proj^T is the
             matrix pseudo-inverted in each Gauss--Newton boundary step
  reach      = RMS of the part of the required boundary change
             Delta b = u_b - u_base(x_b) outside range(J_proj), relative to
             RMS(Delta b): the first-order best boundary error the layer can
             attain without changing the PDE residual

Run:
    python examples/exp_layer_selection.py            # both activations
    python examples/exp_layer_selection.py --act SiLU
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from adr.pinn import ResNet
from adr.training import train
from adr.deff import get_d_eff
from adr.utils import set_seed, use_float64, get_target_params, compute_jacobian

use_float64()

N_PDE, HIDDEN, DEPTH, LR = 20, 100, 3, 5e-3
ADAM_EPOCHS, LBFGS_MAX_ITER = 10_000, 30_000
RCOND = 1e-8
CONFIGS = [
    ("Layer 3 (output)", [-1]),
    ("Layer 2 (penultimate)", [-2]),
    ("Layer 1", [-3]),
    ("Layer 0 (input)", [-4]),
    ("Layers 3, 2", [-2, -1]),
    ("Layers 3, 2, 1", [-3, -2, -1]),
    ("All layers", [-4, -3, -2, -1]),
]
ACTS = {"Tanh": nn.Tanh, "SiLU": nn.SiLU}


def pde_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        u_yy = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[1])(xi)[1]
        f = -2.0 * np.pi**2 * torch.sin(np.pi * xi[0]) * torch.sin(np.pi * xi[1])
        return u_xx + u_yy - f

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def domain(device):
    edge = torch.linspace(-1, 1, N_PDE, device=device)
    xg, yg = torch.meshgrid(edge, edge, indexing="ij")
    x_pde = torch.stack([xg.flatten(), yg.flatten()], dim=1)
    x_bc = torch.cat([
        torch.stack([edge, torch.full_like(edge, -1)], 1),
        torch.stack([edge, torch.full_like(edge, 1)], 1),
        torch.stack([torch.full_like(edge, -1), edge], 1),
        torch.stack([torch.full_like(edge, 1), edge], 1),
    ])
    # adaptation targets, in the order of exp_2d_poisson_*.py
    coords, vals = [], []
    for v in edge.tolist():
        coords += [[v, -1.0], [v, 1.0]]
        vals += [0.0, float(np.sin(np.pi * v))]
        if not np.isclose(abs(v), 1.0):
            coords += [[-1.0, v], [1.0, v]]
            vals += [0.0, 0.0]
    x_adapt = torch.tensor(coords, device=device)
    u_target = torch.tensor(vals, device=device)
    return x_pde, x_bc, x_adapt, u_target


def projection_diagnostics(model, layers, x_pde, x_adapt, u_target):
    params = get_target_params(model, layers)
    all_named = dict(model.named_parameters())
    ids = {id(p) for p in params}
    names = [n for n, p in all_named.items() if id(p) in ids]
    static = {n: p for n, p in all_named.items() if id(p) not in ids}

    def pd(*p):
        return {**static, **dict(zip(names, p))}

    xp = x_pde.clone().requires_grad_(True)
    J_pde = compute_jacobian(lambda *p: pde_fn(model, xp, param_dict=pd(*p)).view(-1), params)
    J_bc = compute_jacobian(lambda *p: functional_call(model, pd(*p), x_adapt).view(-1), params)

    # projector exactly as in adr.adaptation._scalable_ntk
    G = J_pde @ J_pde.T
    scale = G.diagonal().mean().clamp(min=1e-30).sqrt()
    K_inv = torch.linalg.pinv(G / scale, rcond=RCOND, hermitian=True) / scale
    J_proj = J_bc - (J_bc @ J_pde.T) @ (K_inv @ J_pde)

    A = J_proj @ J_proj.T  # pseudo-inverted in adr.adaptation._lstsq_step
    ev = torch.linalg.eigvalsh(A).flip(0).clamp(min=0)
    keep = ev > RCOND * ev[0]
    rank = int(keep.sum())
    s = ev[keep].sqrt()

    with torch.no_grad():
        u_base = model(x_adapt).view(-1)
    db = u_target - u_base
    A_pinv = torch.linalg.pinv(A, rcond=RCOND, hermitian=True)
    db_out = db - A @ (A_pinv @ db)
    reach = (db_out.norm() / db.norm()).item()
    return {
        "n_params": sum(p.numel() for p in params),
        "rank": rank, "n_bc": len(db),
        "s_max": s[0].item(), "s_min": s[-1].item(), "cond": (s[0] / s[-1]).item(),
        "reach": reach,
    }


def run(act_name, device, out_dir):
    x_pde, x_bc, x_adapt, u_target = domain(device)
    set_seed(42)
    model = ResNet(n_in=2, n_out=1, hidden=HIDDEN, depth=DEPTH, activation=ACTS[act_name]).to(device)
    train(model, pde_fn, x_pde, x_bc, torch.zeros(len(x_bc), 1, device=device),
          adam_epochs=ADAM_EPOCHS, lbfgs_max_iter=LBFGS_MAX_ITER, lr=LR,
          lambdas=(1.0, 0.0, 0.0), print_every=0)
    xp = x_pde.clone().requires_grad_(True)
    base_pde = torch.mean(pde_fn(model, xp) ** 2).item()
    print(f"\n{act_name} ResNet {DEPTH} x {HIDDEN}: base PDE MSE {base_pde:.4e}", flush=True)

    rows = []
    for label, layers in CONFIGS:
        d = get_d_eff(model, pde_fn, layers, x_pde, x_bc, mode="all", engine="ntk", rcond=RCOND)
        row = {"label": label, "layers": layers, "d_pde": d["pde"], "d_bc": d["bc"],
               "d_total": d["total"], "ratio": d["bc"] / d["pde"], "gap": d["pde"] - d["total"]}
        row.update(projection_diagnostics(model, layers, x_pde, x_adapt, u_target))
        rows.append(row)
        print(f"  {label:<22} d_pde {row['d_pde']:7.3f}  d_bc {row['d_bc']:8.3f}  "
              f"d_tot {row['d_total']:7.3f}  ratio {row['ratio']:6.3f}  gap {row['gap']:7.3f}  "
              f"rank {row['rank']:2d}/{row['n_bc']}  s_min {row['s_min']:.2e}  "
              f"cond {row['cond']:.2e}  reach {row['reach']:.2e}", flush=True)
    with open(os.path.join(out_dir, f"layer_selection_{act_name.lower()}.json"), "w") as f:
        json.dump({"act": act_name, "base_pde": base_pde, "rows": rows}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", choices=list(ACTS), default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for act in [args.act] if args.act else list(ACTS):
        run(act, device, args.out)


if __name__ == "__main__":
    main()
