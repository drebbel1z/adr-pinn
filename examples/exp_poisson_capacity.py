"""
Experiment: d_eff^pde grows with network size for the 2D Poisson operator
(infinite-dimensional kernel), but not for a finite-kernel operator.

Each network is trained on the PDE residual alone with the protocol of
exp_2d_poisson_{tanh,silu}.py (Adam 10k + L-BFGS 30k, lr 5e-3; the ODE control
uses the exp_invariance.py budget of Adam 3k + L-BFGS 5k). The all-layer
d_eff^pde is then evaluated at the trained state.

  Poisson (infinite kernel: harmonic functions)
      u_xx + u_yy = -2 pi^2 sin(pi x) sin(pi y),   (x, y) in [-1, 1]^2
      trained on an n_train x n_train uniform grid (default 30 x 30)

  Control (finite kernel, dim = 2)
      u'' = -4 pi^2 sin(2 pi x),   x in [-1, 1]
      trained on 100 uniform points (as in Section 3.1)

d_eff^pde is evaluated with Psi and Phi_p on a dense grid that is separate from
the training points (Poisson 41 x 41, ODE 401 points). On the training points
alone, d_eff^pde is capped by the number of points and also counts directions
that keep the residual zero only at those points (collocation aliasing); the
dense grid removes both effects. The training-grid value is stored as well.

By Eq. (deff_spectral), d_eff^pde = sum_i (1 + lam*gamma_i)^-2 over the Rayleigh
quotients gamma_i of the operator on the network's tangent space V. For the
ODE, V contains only the two kernel functions exactly, so d_eff^pde stays at
~2 for every size. For Poisson, a larger V contains more (approximately)
harmonic functions with small gamma_i, so d_eff^pde should grow with size.

Sweeps (Tanh ResNet and SiLU ResNet, as in Section 6.1), seeds 42, 43, 44:
  width sweep : depth 3, width in {25, 50, 100, 200, 400}
  depth sweep : width 100, depth in {1, 2, 3, 4, 5}

Each trained network is evaluated at lam in {1, 100} and rcond in
{1e-8, 1e-10}; the paper's setting is lam = 1, rcond = 1e-8.

Usage
-----
  python examples/exp_poisson_capacity.py --worker 0 --nworkers 2 --out out
  python examples/exp_poisson_capacity.py --aggregate --out out
  python examples/exp_poisson_capacity.py --quick --out out_quick   # smoke test
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from adr.pinn import ResNet
from adr.training import train
from adr.utils import set_seed, use_float64, compute_jacobian

use_float64()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
WIDTHS = [25, 50, 100, 200, 400]  # at depth 3
DEPTHS = [1, 2, 3, 4, 5]  # at width 100
FAMILIES = [
    {"name": "Tanh ResNet", "cls": ResNet, "act": nn.Tanh},
    {"name": "SiLU ResNet", "cls": ResNet, "act": nn.SiLU},
]
LAMS = [1.0, 100.0]
RCONDS = [1e-8, 1e-10]
N_TRAIN = 30  # Poisson training grid per side (overridden by --n-train)
N_EVAL = {"poisson": 41, "ode2": 401}  # dense d_eff grid (per side for Poisson)
CHUNK = 64  # points per Jacobian chunk

PROBLEMS = {
    "poisson": {"adam": 10_000, "lbfgs": 30_000, "lr": 5e-3, "seeds": SEEDS},
    "ode2": {"adam": 3_000, "lbfgs": 5_000, "lr": 5e-3, "seeds": SEEDS},
}


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def poisson_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        u_xx = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        u_yy = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[1])(xi)[1]
        f = -2.0 * np.pi**2 * torch.sin(np.pi * xi[0]) * torch.sin(np.pi * xi[1])
        return u_xx + u_yy - f

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def ode2_fn(model, x, param_dict=None):
    pd = param_dict if param_dict is not None else dict(model.named_parameters())

    def u_fn(xi):
        return functional_call(model, pd, xi.unsqueeze(0)).squeeze(0)

    def residual_single(xi):
        g2 = torch.func.grad(lambda z: torch.func.grad(lambda y: u_fn(y)[0])(z)[0])(xi)[0]
        return g2 + (4.0 * np.pi**2) * torch.sin(2.0 * np.pi * xi[0])

    return torch.func.vmap(residual_single)(x).unsqueeze(1)


def grid(problem, n, device):
    """Uniform grid: n x n on [-1, 1]^2 for Poisson, n points on [-1, 1] for the ODE."""
    edge = torch.linspace(-1, 1, n, device=device)
    if problem == "poisson":
        xg, yg = torch.meshgrid(edge, edge, indexing="ij")
        return torch.stack([xg.flatten(), yg.flatten()], dim=1)
    return edge.view(-1, 1)


def setup(problem, device, n_train):
    """Return (pde_fn, x_train, x_eval, x_bc, y_bc, n_in). Training is PDE-only."""
    if problem == "poisson":
        x_bc = torch.tensor([[-1.0, -1.0]], device=device)
        return (poisson_fn, grid(problem, n_train, device), grid(problem, N_EVAL[problem], device),
                x_bc, torch.zeros(1, 1, device=device), 2)
    x_bc = torch.tensor([[-1.0]], device=device)
    return (ode2_fn, grid(problem, 100, device), grid(problem, N_EVAL[problem], device),
            x_bc, torch.zeros(1, 1, device=device), 1)


def deff_pde_sweep(model, pde_fn, x, lams, rconds, chunk=CHUNK):
    """All-layer d_eff^pde on points x for every (lam, rcond).

    Same formula as adr.deff (NTK engine, mode='pde'): K = [Psi_n; sqrt(lam) J_n],
    G = K K^T, d_eff = ||(G G^+)_{11}||_F^2. Psi and J are computed once, in chunks
    of points, and the Gram blocks are reused for every (lam, rcond).
    """
    model.eval()
    names = [n for n, _ in model.named_parameters()]
    params = [p.detach() for _, p in model.named_parameters()]

    def pd(*p):
        return dict(zip(names, p))

    Psi, J = [], []
    for i in range(0, len(x), chunk):
        xc = x[i:i + chunk].clone().requires_grad_(True)
        Psi.append(compute_jacobian(lambda *p: functional_call(model, pd(*p), xc).view(-1), params))
        J.append(compute_jacobian(lambda *p: pde_fn(model, xc, param_dict=pd(*p)).view(-1), params))
    N = len(x)
    Psi = torch.cat(Psi) / N**0.5
    J = torch.cat(J) / N**0.5
    G11, G12, G22 = Psi @ Psi.T, Psi @ J.T, J @ J.T
    del Psi, J
    out = {}
    for lam in lams:
        G = torch.cat([torch.cat([G11, lam**0.5 * G12], 1),
                       torch.cat([lam**0.5 * G12.T, lam * G22], 1)], 0)
        scale = G.diagonal().mean().clamp(min=1e-30).sqrt()
        for rc in rconds:
            G_pinv = torch.linalg.pinv(G / scale, rcond=rc, hermitian=True) / scale
            P_11 = (G @ G_pinv)[:N, :N]
            out[f"lam{lam:g}_rcond{rc:g}"] = torch.sum(P_11**2).item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def all_jobs():
    sizes = sorted({(3, w) for w in WIDTHS} | {(d, 100) for d in DEPTHS})
    jobs = []
    for problem, cfg in PROBLEMS.items():
        for fam in FAMILIES:
            for depth, width in sizes:
                for seed in cfg["seeds"]:
                    jobs.append((problem, fam["name"], depth, width, seed))
    # longest first, so parallel workers finish at similar times
    jobs.sort(key=lambda j: (j[0] != "poisson", -j[2] * j[3] ** 2))
    return jobs


def job_name(problem, family, depth, width, seed):
    return f"{problem}_{family.replace(' ', '')}_d{depth}_w{width}_s{seed}"


def run_job(job, out_dir, device, n_train, quick=False, do_train=True, claim=False):
    problem, family, depth, width, seed = job
    name = job_name(*job)
    path = os.path.join(out_dir, name + ".json")
    if os.path.exists(path):
        print(f"[skip] {name}")
        return
    if claim:  # several workers sharing one job list: first to create the lock file runs the job
        try:
            os.close(os.open(path + ".claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            print(f"[claimed] {name}")
            return

    fam = next(f for f in FAMILIES if f["name"] == family)
    cfg = PROBLEMS[problem]
    pde_fn, x_train, x_eval, x_bc, y_bc, n_in = setup(problem, device, n_train)

    set_seed(seed)
    model = fam["cls"](n_in=n_in, n_out=1, hidden=width, depth=depth, activation=fam["act"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    if do_train:
        train(
            model, pde_fn, x_train, x_bc, y_bc,
            adam_epochs=50 if quick else cfg["adam"],
            lbfgs_max_iter=50 if quick else cfg["lbfgs"],
            lr=cfg["lr"], lambdas=(1.0, 0.0, 0.0), print_every=0,
        )
    t_train = time.perf_counter() - t0

    def mse(x):
        xt = x.clone().requires_grad_(True)
        return torch.mean(pde_fn(model, xt) ** 2).item()

    def sweep(x):
        for attempt in range(20):  # wait if another process holds GPU memory
            try:
                return deff_pde_sweep(model, pde_fn, x, LAMS, RCONDS)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(60)
        raise RuntimeError(f"{name}: out of GPU memory in d_eff evaluation")

    t1 = time.perf_counter()
    deff_dense = sweep(x_eval)
    deff_train = sweep(x_train)
    t_deff = time.perf_counter() - t1

    rec = {
        "problem": problem, "family": family, "depth": depth, "width": width,
        "seed": seed, "trained": do_train, "n_params": n_params,
        "n_train": len(x_train), "n_eval": len(x_eval),
        "pde_mse": mse(x_train), "pde_mse_dense": mse(x_eval),
        "train_s": t_train, "deff_s": t_deff,
        "deff_pde": deff_dense, "deff_pde_train": deff_train,
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(
        f"[done] {name}: params={n_params} mse train/dense={rec['pde_mse']:.1e}/"
        f"{rec['pde_mse_dense']:.1e} deff dense/train(lam1,rc1e-8)="
        f"{deff_dense['lam1_rcond1e-08']:.3f}/{deff_train['lam1_rcond1e-08']:.3f} "
        f"train={t_train:.0f}s deff={t_deff:.0f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(out_dir):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        r = json.load(open(p))
        dense, trn = r.pop("deff_pde"), r.pop("deff_pde_train")
        for key, val in dense.items():
            lam, rc = key.replace("lam", "").split("_rcond")
            rows.append({**r, "lam": float(lam), "rcond": float(rc), "deff": val,
                         "deff_train": trn[key]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "poisson_capacity_all.csv"), index=False)

    g = (
        df.groupby(["problem", "family", "depth", "width", "lam", "rcond"])
        .agg(n_params=("n_params", "first"), deff_mean=("deff", "mean"),
             deff_std=("deff", "std"), deff_train=("deff_train", "mean"),
             pde_mse=("pde_mse", "median"), pde_mse_dense=("pde_mse_dense", "median"),
             n_seeds=("seed", "nunique"))
        .reset_index()
    )
    g.to_csv(os.path.join(out_dir, "poisson_capacity_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    paper = g[(g.lam == 1.0) & (g.rcond == 1e-8)]
    print("\n=== lam = 1, rcond = 1e-8 (paper setting) ===")
    print(paper.to_string(index=False))
    print("\n=== Poisson: d_eff vs rcond and lam (mean over seeds) ===")
    piv = g[g.problem == "poisson"].pivot_table(
        index=["family", "depth", "width"], columns=["lam", "rcond"], values="deff_mean")
    print(piv.round(2).to_string())

    # LaTeX rows for the paper (lam = 1, rcond = 1e-8)
    with open(os.path.join(out_dir, "poisson_capacity_table.tex"), "w") as f:
        for _, r in paper[paper.problem == "poisson"].sort_values(
                ["family", "depth", "width"]).iterrows():
            ode = paper[(paper.problem == "ode2") & (paper.family == r.family)
                        & (paper.depth == r.depth) & (paper.width == r.width)]
            ode_val = f"{ode.deff_mean.iloc[0]:.3f}" if len(ode) else "--"
            f.write(f"{r.family} & {r.depth} & {r.width} & {int(r.n_params)} & "
                    f"${r.deff_mean:.2f} \\pm {r.deff_std:.2f}$ & {ode_val} & "
                    f"{r.deff_train:.2f} & ${r.pde_mse_dense:.1e}$ \\\\\n")

    # Figure: d_eff^pde vs width (depth 3) and vs depth (width 100), lam = 1.
    # Solid: Poisson, rcond 1e-8 (mean +- std over seeds); dotted: Poisson, rcond 1e-10;
    # dashed, open markers: u'' control, rcond 1e-8.
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.labelsize": 14,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.linewidth": 1.2, "mathtext.fontset": "cm"})
    lam1 = g[g.lam == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"Tanh ResNet": ("C0", "o", "Tanh"), "SiLU ResNet": ("C3", "s", "SiLU")}
    for ax, (xkey, fixed, xlabel) in zip(
        axes, [("width", ("depth", 3), "Width (depth 3)"),
               ("depth", ("width", 100), "Depth (width 100)")]):
        for fam, (c, m, short) in styles.items():
            sel = (lam1.family == fam) & (lam1[fixed[0]] == fixed[1])
            for prob, rc, ls, mfc, lbl, err in [
                    ("poisson", 1e-8, "-", c, f"{short}, Poisson", True),
                    ("poisson", 1e-10, ":", c, f"{short}, Poisson, rcond $10^{{-10}}$", False),
                    ("ode2", 1e-8, "--", "white", f"{short}, $u''$ (control)", False)]:
                sub = lam1[sel & (lam1.problem == prob) & (lam1.rcond == rc)].sort_values(xkey)
                if sub.empty:
                    continue
                ax.errorbar(sub[xkey], sub.deff_mean,
                            yerr=sub.deff_std.fillna(0) if err else None,
                            color=c, ls=ls, marker=m, mfc=mfc, ms=6, capsize=3, lw=1.5,
                            alpha=1.0 if err else 0.75, label=lbl)
        ax.set_xlabel(xlabel)
        if xkey == "width":
            ax.set_xscale("log", base=2)
            ax.set_xticks(WIDTHS)
            ax.set_xticklabels([str(w) for w in WIDTHS])
            ax.minorticks_off()
        else:
            ax.set_xticks(DEPTHS)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$d_{\mathrm{eff}}^{\mathrm{pde}}$")
    axes[1].legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.09), ncol=2)
    for ax, letter in zip(axes, "ab"):
        ax.text(0.5, -0.17, f"({letter})", transform=ax.transAxes, ha="center", va="top",
                fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(os.path.join(out_dir, "poisson_capacity.pdf"), bbox_inches="tight")
    print(f"\nWrote CSV, table and figure to {out_dir}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_poisson_capacity")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--no-train", action="store_true", help="evaluate d_eff at initialization")
    ap.add_argument("--n-train", type=int, default=N_TRAIN, help="Poisson training grid per side")
    ap.add_argument("--quick", action="store_true", help="two small jobs")
    ap.add_argument("--family", default=None, help="run only this family, e.g. 'SiLU MLP'")
    ap.add_argument("--smallest-first", action="store_true")
    ap.add_argument("--claim", action="store_true",
                    help="claim jobs with lock files so several workers can share one job list")
    ap.add_argument("--tag", default="", help="suffix for the worker's DONE file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        aggregate(args.out)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = all_jobs()
    if args.quick:
        jobs = [("poisson", "Tanh ResNet", 1, 25, 0), ("ode2", "SiLU ResNet", 1, 25, 0)]
    if args.family:
        jobs = [j for j in jobs if j[1] == args.family]
    if args.smallest_first:
        jobs = jobs[::-1]
    mine = jobs[args.worker::args.nworkers]
    print(f"device={device}  worker {args.worker}/{args.nworkers}: {len(mine)} of {len(jobs)} jobs",
          flush=True)
    for job in mine:
        run_job(job, args.out, device, args.n_train, quick=args.quick, do_train=not args.no_train,
                claim=args.claim)
    tag = f"{args.family.replace(' ', '')}_" if args.family else ""
    open(os.path.join(args.out, f"WORKER_{tag}{args.worker}{args.tag}_DONE"), "w").close()


if __name__ == "__main__":
    main()
