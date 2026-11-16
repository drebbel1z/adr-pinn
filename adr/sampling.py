"""Collocation-point samplers for specify_bcs(..., x_pde_sampler=...).

A sampler is called once at the start of every continuation increment as
sampler(step, residual_fn) and returns the collocation points [N, d_in] used for
that increment's corrector, null-space projector and predictor iterations.
residual_fn(x) returns the PDE residual at the current parameters.

Drawing new points every increment means that each corrector step evaluates the
PDE on points the previous increment did not constrain, so residual that the
previous increment created between its collocation points is detected and
removed instead of accumulating (collocation aliasing).
"""

import torch
from torch.quasirandom import SobolEngine


def _scale(u, lower, upper):
    return lower + u * (upper - lower)


class UniformSampler:
    """n i.i.d. uniform points in the box [lower, upper], new points every call."""

    def __init__(self, n, lower, upper, seed=0, device="cpu", dtype=torch.float64):
        self.n = n
        self.lower = torch.as_tensor(lower, dtype=dtype, device=device)
        self.upper = torch.as_tensor(upper, dtype=dtype, device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.dtype, self.device = dtype, device

    def __call__(self, step, residual_fn):
        u = torch.rand(self.n, len(self.lower), generator=self.gen, dtype=self.dtype)
        return _scale(u.to(self.device), self.lower, self.upper)


class HybridSampler:
    """Quasi-random coverage plus residual-weighted refinement.

    Every call returns n points:
      n_qmc = round((1 - frac_residual) * n) points from a scrambled Sobol sequence in
          [lower, upper], with a new scramble every call (even coverage of the domain);
      n - n_qmc points drawn without replacement from the candidate pool with
          probability proportional to |r|^power + floor * mean(|r|^power), where r is
          the PDE residual at the current parameters (points where the residual is
          largest, e.g. near a shock, are sampled more often; the floor keeps a
          fraction of the draw spread over the whole pool).
    """

    def __init__(self, n, lower, upper, pool, frac_residual=0.5, power=2.0, floor=0.2,
                 seed=0, chunk=4096):
        self.n = n
        self.lower = torch.as_tensor(lower, dtype=pool.dtype, device=pool.device)
        self.upper = torch.as_tensor(upper, dtype=pool.dtype, device=pool.device)
        self.pool = pool
        self.n_res = int(round(frac_residual * n))
        self.n_qmc = n - self.n_res
        self.power, self.floor, self.seed, self.chunk = power, floor, seed, chunk
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, step, residual_fn):
        parts = []
        if self.n_qmc > 0:
            sobol = SobolEngine(len(self.lower), scramble=True, seed=self.seed * 100_003 + step)
            u = sobol.draw(self.n_qmc, dtype=self.pool.dtype).to(self.pool.device)
            parts.append(_scale(u, self.lower, self.upper))
        if self.n_res > 0:
            r = torch.cat([residual_fn(self.pool[i:i + self.chunk])
                           for i in range(0, len(self.pool), self.chunk)])
            w = r.abs() ** self.power
            w = w + self.floor * w.mean().clamp(min=1e-300)
            idx = torch.multinomial(w.cpu(), self.n_res, replacement=False, generator=self.gen)
            parts.append(self.pool[idx.to(self.pool.device)])
        return torch.cat(parts)
