"""
Per-layer numerical rank of the output Jacobian next to d_eff^pde and d_eff^bc (Table 7)

Networks: the 2D Poisson base models (ResNet 3 x 100, Tanh and SiLU, seed 42, PDE-only,
20 x 20 grid; results/gamma_directions/poisson_ResNet_<act>_d3_w100_s42_col20.pt). Points
and cutoff as in exp_2d_poisson_tanh.py / _silu.py: 20 x 20 grid, 80 boundary points,
NTK engine, rcond 1e-8, lambda = 1. The numerical rank of Q_layer is d_eff with no
constraint (lambda = 0), which equals the rank of the output Jacobian at the same cutoff.
BC consumption = rank - d_eff^bc (at most N_b = 80).

Run:
    python examples/layer_rank_check.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import torch, torch.nn as nn
from adr.pinn import ResNet
from adr.deff import get_d_eff
from adr.utils import use_float64
use_float64()
import exp_2d_poisson_tanh as P

edge = torch.linspace(-1, 1, 20)
X, Y = torch.meshgrid(edge, edge, indexing="ij")
x_pde = torch.stack([X.flatten(), Y.flatten()], 1)
x_bc = torch.cat([torch.stack([edge, torch.full_like(edge, -1)], 1), torch.stack([edge, torch.full_like(edge, 1)], 1),
                  torch.stack([torch.full_like(edge, -1), edge], 1), torch.stack([torch.full_like(edge, 1), edge], 1)])
def base_model_path(act):
    return os.path.join(HERE, "results", "gamma_directions", f"poisson_ResNet_{act}_d3_w100_s42_col20.pt")


missing = [a for a in ("Tanh", "SiLU") if not os.path.exists(base_model_path(a))]
if missing:
    sys.exit("Base models not found. Train them first:\n" + "\n".join(
        f"    python examples/exp_gamma_directions_2d.py --n-col 20" + ("" if a == "Tanh" else f" --act {a}")
        for a in missing))

print(f"{'net':5s} {'layer':18s} {'rank':>6s} {'d_pde':>7s} {'d_bc':>7s} {'ratio':>6s} {'consumed':>8s}")
for act, cls in (("Tanh", nn.Tanh), ("SiLU", nn.SiLU)):
    m = ResNet(n_in=2, n_out=1, hidden=100, depth=3, activation=cls)
    # lam = 0 below: d_eff = tr(Q Q^+) = rank(Q), the numerical rank at the same cutoff
    m.load_state_dict(torch.load(base_model_path(act), map_location="cpu"))
    for name, L in (("Layer 3 (output)", [-1]), ("Layer 2", [-2]), ("Layer 1", [-3]), ("Layer 0 (input)", [-4])):
        d = get_d_eff(m, P.pde_fn, L, x_pde, x_bc, mode="all", engine="ntk", rcond=1e-8)
        r = get_d_eff(m, P.pde_fn, L, x_pde, x_bc, mode="bc", engine="ntk", rcond=1e-8, lam=0.0)
        r = r["bc"] if isinstance(r, dict) else r
        print(f"{act:5s} {name:18s} {r:6.1f} {d['pde']:7.3f} {d['bc']:7.3f} {d['bc']/d['pde']:6.3f} {r-d['bc']:8.2f}", flush=True)
