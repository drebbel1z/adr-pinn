# Reproducing the paper's figures and tables

Every figure and table in `paper/main.tex` is produced by a script in this folder.
Run the commands from the repository root. Figures are written under the file name
the paper uses, so they can be copied to `paper/` unchanged.

The paper's timings were measured on a single A100 GPU with nothing else running on it.
Results on other hardware match up to floating-point and timing differences.

## Section 3: invariance and resolution of d_eff

| Output | Command | Written to |
|---|---|---|
| Table 1 | `python examples/exp_invariance.py` | console |
| Figure 1 (`gamma_directions_ode2.pdf`) | `python examples/exp_gamma_directions_1d.py` | `examples/results/gamma_directions/` |
| Table 2 | `python examples/exp_bessel_invariance.py` | console |
| Table 3 | `python examples/exp_nonlinear_invariance.py` | console |
| Figure 2 (`spatial_sieve_feedforward.pdf`), Table 4 | `python examples/exp_spatial_sieve.py` | `examples/` |
| Table B.10 | `python examples/exp_sieve_rcond.py` | `examples/results/sieve_rcond/` |

Supporting runs for statements in the text:

- `python examples/exp_rcond_sweep.py`: d_eff^pde as a function of `rcond`
  ("the invariance results hold for any `rcond` down to 1e-15").

## Section 4–5: 1D boundary adaptation and refinements

| Output | Command | Written to |
|---|---|---|
| Figure 3 (`pinn_1d_sine_comparison.pdf`), Table 5 | `python examples/exp_1d_adaptation.py` | `examples/` |
| Figure 4 (`pinn_1d_sine_comparison_kinda_close_bc.pdf`) | `python examples/exp_regime_anchoring.py` | `examples/` |

## Section 6.1: 2D Poisson

| Output | Command | Written to |
|---|---|---|
| Figure 5 (`pinn_poisson_tanh.pdf`), Table 6 (Tanh rows) | `python examples/exp_2d_poisson_tanh.py` | `examples/` |
| Figure 6 (`pinn_poisson_silu.pdf`), Table 6 (SiLU rows) | `python examples/exp_2d_poisson_silu.py` | `examples/` |
| Figure 7 (`poisson_capacity.pdf`) | see below | `<out>/` |
| Figure 8 (`gamma_directions_poisson.pdf`) | `python examples/exp_gamma_directions_2d.py --n-col 20` | `examples/results/gamma_directions/` |
| Table 7 | see below | console |

**Figure 7.** The sweep trains many networks and can be split across workers (or run with one):

```bash
python examples/exp_poisson_capacity.py --worker 0 --nworkers 2 --out out_poisson_capacity
python examples/exp_poisson_capacity.py --worker 1 --nworkers 2 --out out_poisson_capacity
python examples/exp_poisson_capacity.py --aggregate --out out_poisson_capacity
```

**Table 7.** The d_eff^pde, d_eff^bc, RMSE and PDE residual columns are printed by
`exp_2d_poisson_tanh.py` and `exp_2d_poisson_silu.py`. The Rank column comes from
`layer_rank_check.py`, which loads the base models saved by the gamma-direction runs:

```bash
python examples/exp_gamma_directions_2d.py --n-col 20
python examples/exp_gamma_directions_2d.py --n-col 20 --act SiLU
python examples/layer_rank_check.py
```

## Section 6.2: subspace projection vs gradient-based adaptation

| Output | Command | Written to |
|---|---|---|
| Figure 9 (`poisson_comparison.pdf`), Table 8 | `python examples/exp_2d_poisson_comparison.py` | `examples/` |

## Section 6.3: Burgers' equation

| Output | Command | Written to |
|---|---|---|
| Figures 10–13 (`burgers_{ic,bc}_{adaptation,snapshots}.pdf`), Table 9 | `python examples/exp_burgers_adaptation.py` | `examples/` |
| Table 9, sparse-grid rows only | `python examples/exp_burgers_ft_resampled.py` | `examples/results/burgers_ft_resampled/` |

`exp_burgers_adaptation.py` loads the base model from
`examples/results/burgers_sampling/burgers_base.pt`, or trains and saves it if the
file is missing. `exp_burgers_ft_resampled.py` re-runs only the sparse-grid
fine-tuning baseline at the paper's equal-time budgets.

Supporting runs for statements in the text:

```bash
# Fixed 40 x 40 grid instead of resampled points (PDE MSE on that grid vs the full grid)
python examples/exp_burgers_sampling.py --strategies fixed hybrid

# BC adaptation: full corrector step vs damped (alpha = 0.2), five sampler seeds
for k in 0 1 2 3 4; do
  python examples/exp_burgers_increments.py --incs 100 --damping 1 0.2 --seed $k
done

# Other damping values
python examples/exp_burgers_increments.py --incs 100 --damping 0.1 0.4 0.6
```

## Shared modules

`gamma_utils.py` (Jacobians and Rayleigh-quotient spectra) and `exp_layer_selection.py`
(2D Poisson setup) are imported by the scripts above and are not run on their own.
