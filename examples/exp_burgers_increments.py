"""
Experiment: Burgers IC/BC adaptation with the incremental predictor-corrector, varying the
number of increments and the corrector damping

Same base model, targets, hybrid sampler (1600 points per increment) and settings as
exp_burgers_sampling.py (subspace projection, predictor-corrector, all layers, rcond 1e-8,
BC tolerance 1e-8, 15 predictor iterations, final corrector); num_increments and the
damping of the corrector (fraction of the Gauss--Newton step, 1.0 = the full step) vary.

Reported per run: wall time, PDE MSE on the full grid, IC/BC MSE, RMSE against the FD
reference.

Run:
    python examples/exp_burgers_increments.py --incs 100 300
    python examples/exp_burgers_increments.py --incs 100 --damping 0.2 --seed 2
    python examples/exp_burgers_increments.py --scenario ic --incs 100 --damping 1 0.2
"""

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import numpy as np
import torch

from adr.adaptation import specify_bcs


def main():
    import exp_burgers_adaptation as B
    import exp_burgers_sampling as BS
    from adr.sampling import HybridSampler

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="bc", choices=["bc", "ic"])
    ap.add_argument("--incs", type=int, nargs="+", default=[100])
    ap.add_argument("--damping", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "burgers_increments"))
    ap.add_argument("--burgers-base", default=os.path.join(HERE, "results", "burgers_sampling",
                                                            "burgers_base.pt"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    x_grid, x_pde, x_ic, y_ic, x_sbc, y_sbc, x_pde_adapt = BS.setup(device)
    model = BS.base_model(device, x_pde, torch.cat([x_ic, x_sbc]), torch.cat([y_ic, y_sbc]),
                          args.burgers_base)
    base = copy.deepcopy(model.state_dict())
    S = BS.scenarios(x_grid, x_ic, y_ic, x_sbc, y_sbc)[args.scenario]
    x_ref, t_ref, u_raw = B.burgers_fd_reference(**S["ref"])
    xe, te = np.linspace(-1, 1, 200), np.linspace(0, 1, 200)
    u_ref, _, _ = B.interp_fd(x_ref, t_ref, u_raw, xe, te)
    xt_eval = torch.tensor(np.stack(np.meshgrid(xe, te, indexing="ij"), -1).reshape(-1, 2),
                           dtype=torch.float64, device=device)

    def metrics():
        with torch.no_grad():
            c = torch.mean((model(S["x_c"]) - S["y_c"]) ** 2).item()
            u = model(xt_eval).cpu().numpy().reshape(200, 200)
        return {"pde_mse": B.compute_pde_loss(model, x_pde, B.pde_fn), "c_mse": c,
                "rmse": float(np.sqrt(np.mean((u - u_ref) ** 2)))}

    results = {}
    for n in args.incs:
        for a in args.damping:
            model.load_state_dict(base)
            B.restore_base(model)
            sampler = HybridSampler(BS.N_POINTS, BS.LOWER, BS.UPPER, pool=x_pde, frac_residual=0.5,
                                    power=2.0, floor=0.2, seed=args.seed)
            t0 = time.time()
            specify_bcs(model, B.pde_fn, S["bcs"], B.ALL_LAYERS, x_pde_adapt, num_increments=n,
                        use_corrector=True, tolerance=1e-8, max_iter=15, rcond=1e-8,
                        damping=a, x_pde_sampler=sampler, final_corrector=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            r = {"scenario": args.scenario, "increments": n, "damping": a, "seed": args.seed,
                 "t": time.time() - t0, **metrics()}
            results[f"n{n}_d{a:g}"] = r
            print(f"[{args.scenario}] increments {n:4d} damping {a:g} seed {args.seed}  time {r['t']:6.1f}s  "
                  f"PDE {r['pde_mse']:.3e}  constraint {r['c_mse']:.3e}  RMSE {r['rmse']:.3e}", flush=True)
            with open(os.path.join(args.out, f"burgers_{args.scenario}_seed{args.seed}.json"), "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
