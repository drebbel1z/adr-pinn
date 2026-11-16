import torch
from torch.func import functional_call, jacrev
from .utils import get_target_params, build_param_dict

# ---------------------------------------------------------------------------
# Base state management
# ---------------------------------------------------------------------------


def save_base(model):
    """Snapshot all model parameters as the adaptation base.

    Called automatically by train(). All subsequent specify_bcs calls
    restart from this snapshot regardless of how many adaptations have run.
    """
    model._adr_base_state = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }


def _get_base_theta(model, target_layers):
    """Extract target layer params from the saved base state."""
    if not hasattr(model, "_adr_base_state"):
        return _get_theta(model, target_layers)
    target_ids = {id(p) for p in get_target_params(model, target_layers)}
    chunks = [
        model._adr_base_state[name].view(-1)
        for name, p in model.named_parameters()
        if id(p) in target_ids
    ]
    return torch.cat(chunks).detach()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_theta(model, target_layers):
    return torch.cat(
        [p.view(-1) for p in get_target_params(model, target_layers)]
    ).detach()


def _write_theta(model, theta, target_layers):
    """Copy theta back into model parameters in-place."""
    idx = 0
    for p in get_target_params(model, target_layers):
        numel = p.numel()
        p.data.copy_(theta[idx : idx + numel].view_as(p))
        idx += numel


def _eval_bc(model, param_dict, bc, device):
    """Evaluate model output (or derivative) at a BC location.

    Returns (value, x_t) where x_t retains grad for Neumann BCs.
    """
    dtype = next(iter(param_dict.values())).dtype
    x_t = torch.tensor([bc["coords"]], dtype=dtype, device=device, requires_grad=True)
    pred = functional_call(model, param_dict, x_t)[0, 0]

    if bc["type"] == "dirichlet":
        return pred, x_t
    elif bc["type"] == "neumann":
        # Index both axes so this is 0-d like the Dirichlet branch, else
        # torch.stack in _bc_jac fails on mixed BCs.
        val = torch.autograd.grad(pred, x_t, create_graph=True)[0][0, 0]
        return val, x_t
    else:
        raise ValueError(
            f"Unknown BC type '{bc['type']}'. Expected 'dirichlet' or 'neumann'."
        )


def _jac_wrt_flat(output, theta_r):
    """Compute Jacobian of output vector w.r.t. flat theta_r, row by row."""
    rows = []
    for i in range(len(output)):
        g = torch.autograd.grad(
            output[i], theta_r, retain_graph=True, allow_unused=True
        )[0]
        rows.append(g.detach() if g is not None else torch.zeros_like(theta_r))
    return torch.stack(rows)


def _pde_jac(model, pde_fn, theta, target_layers, x_pde):
    """PDE residual and its Jacobian w.r.t. theta."""

    def residual_fn(th, x=x_pde):
        pd = build_param_dict(model, th, target_layers)
        return pde_fn(model, x.detach(), pd).view(-1)

    if len(x_pde) <= 2048:
        J = jacrev(residual_fn, chunk_size=900)(theta)
    else:  # large point sets: the same rows, computed per block of points to bound memory
        J = torch.cat(
            [
                jacrev(lambda th: residual_fn(th, x_pde[i : i + 1024]), chunk_size=256)(
                    theta
                )
                for i in range(0, len(x_pde), 1024)
            ]
        )
    with torch.no_grad():
        r = residual_fn(theta)
    return J.detach(), r.detach()


def _bc_jac(model, theta, target_layers, bcs, target_vals, device):
    """BC Jacobian rows and residuals w.r.t. theta."""
    # Fast path: batch all Dirichlet BCs into one forward pass + vectorized jacrev.
    if all(bc["type"] == "dirichlet" for bc in bcs):
        dtype = theta.dtype
        coords = torch.tensor([bc["coords"] for bc in bcs], dtype=dtype, device=device)
        tv = torch.tensor(target_vals, dtype=dtype, device=device)

        def fwd(th):
            pd = build_param_dict(model, th, target_layers)
            return functional_call(model, pd, coords).view(-1)

        J_bc = jacrev(fwd, chunk_size=900)(theta)
        with torch.no_grad():
            r_bc = tv - fwd(theta)
        return J_bc.detach(), r_bc.detach()

    # Fallback for Neumann / mixed BCs: row-by-row implementation.
    theta_r = theta.clone().requires_grad_(True)
    param_dict = build_param_dict(model, theta_r, target_layers)
    outputs, residuals = [], []
    for i, bc in enumerate(bcs):
        val, _ = _eval_bc(model, param_dict, bc, device)
        outputs.append(val)
        residuals.append(target_vals[i] - val.item())
    J_bc = _jac_wrt_flat(torch.stack(outputs), theta_r)
    r_bc = torch.tensor(residuals, dtype=theta.dtype, device=device)
    return J_bc, r_bc


def _incremental_targets(bcs, base_vals, step, num_increments):
    return [
        base_vals[i] + (step / num_increments) * (bc["val"] - base_vals[i])
        for i, bc in enumerate(bcs)
    ]


def _lstsq_step(J_proj, r, rcond=1e-8):
    A = J_proj @ J_proj.T
    sol = torch.linalg.pinv(A, hermitian=True, rcond=rcond) @ r
    return J_proj.T @ sol


# ---------------------------------------------------------------------------
# Method: scalable_ntk (predictor-corrector in function space)
# ---------------------------------------------------------------------------


def _scalable_ntk(
    model,
    pde_fn,
    bcs,
    target_layers,
    x_pde,
    num_increments,
    max_iter,
    tolerance,
    step_size,
    use_corrector,
    rcond,
    damping=1.0,
    x_pde_sampler=None,
    final_corrector=False,
):
    device = next(model.parameters()).device
    theta_base = _get_base_theta(model, target_layers)

    param_dict_base = build_param_dict(model, theta_base, target_layers)
    base_vals = [_eval_bc(model, param_dict_base, bc, device)[0].item() for bc in bcs]
    delta_theta = torch.zeros_like(theta_base)

    def _points(step, theta):
        """Collocation points for this increment: fixed, or drawn by the sampler at theta."""
        if x_pde_sampler is None:
            return x_pde
        pd_cur = build_param_dict(model, theta, target_layers)

        def residual(x):
            return pde_fn(model, x, param_dict=pd_cur).reshape(-1).detach()

        return x_pde_sampler(step, residual)

    def _projector(theta, x):
        """PDE Jacobian, residual and C = (J J^T)^+ J at theta on points x."""
        J_pde, r_pde = _pde_jac(model, pde_fn, theta, target_layers, x)
        Gram_pde = J_pde @ J_pde.T
        # Normalise before inverting so rcond is relative to O(1) singular values
        # rather than the raw scale of J_pde, which varies across devices.
        scale = Gram_pde.diagonal().mean().clamp(min=1e-30).sqrt()
        K_inv = torch.linalg.pinv(Gram_pde / scale, rcond=rcond, hermitian=True) / scale
        return J_pde, r_pde, K_inv @ J_pde  # C: [N, d], reused in corrector and projection

    def _increment(theta_cur, x, target_vals):
        """Damped corrector, then predictor iterations in the PDE null space."""
        J_pde, r_pde, C = _projector(theta_cur, x)

        # Corrector: damped Gauss--Newton step back towards the PDE manifold
        # (damping = 1: the full step)
        if use_corrector:
            theta_inner = (theta_cur - damping * (C.T @ r_pde)).detach()
        else:
            theta_inner = theta_cur.detach()

        # Predictor: BC null-space steps
        converged = False
        for _ in range(max_iter):
            J_bc, r_bc = _bc_jac(model, theta_inner, target_layers, bcs, target_vals, device)
            if r_bc.pow(2).mean().item() < tolerance:
                converged = True
                break
            J_bc_proj = J_bc - (J_bc @ J_pde.T) @ C
            direction = _lstsq_step(J_bc_proj, r_bc, rcond=rcond)
            theta_inner = (theta_inner + step_size * direction).detach()
        return theta_inner, converged, r_bc.pow(2).mean().item()

    for step in range(1, num_increments + 1):
        theta_cur = theta_base + delta_theta
        x_step = _points(step, theta_cur)
        target_vals = _incremental_targets(bcs, base_vals, step, num_increments)
        theta_inner, converged, bc_mse = _increment(theta_cur, x_step, target_vals)
        if not converged:
            print(f"  Warning: step {step}/{num_increments} did not converge. BC MSE: {bc_mse:.2e}")
        delta_theta = theta_inner - theta_base

    if final_corrector:
        # One more increment at the final target, on a fresh point set when a sampler is
        # given: corrector, then predictor iterations, so the PDE correction does not leave
        # the BCs off target
        theta_cur = theta_base + delta_theta
        x_step = _points(num_increments + 1, theta_cur)
        target_vals = [bc["val"] for bc in bcs]
        theta_inner, _, _ = _increment(theta_cur, x_step, target_vals)
        delta_theta = theta_inner - theta_base

    return theta_base + delta_theta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def specify_bcs(
    model,
    pde_fn,
    bcs,
    target_layers,
    x_pde,
    num_increments=10,
    max_iter=10,
    tolerance=1e-10,
    step_size=1.0,
    use_corrector=True,
    rcond=1e-10,
    damping=1.0,
    x_pde_sampler=None,
    final_corrector=False,
):
    """Adapt model to new boundary conditions.

    Projects parameter updates into the null space of the PDE Hessian so the
    learned physics is preserved while the new BCs are enforced.

    Parameters
    ----------
    model         : nn.Module — trained model
    pde_fn        : callable(model, x, param_dict=None) -> residual [N]
    bcs           : list of dicts with keys:
                      'coords' : coordinate list, e.g. [-1.0] or [-1.0, 0.0]
                      'type'   : 'dirichlet' or 'neumann'
                      'val'    : target float value
    target_layers : list of int or str — layers to adapt
    x_pde         : tensor [M, d_in] — PDE collocation points
    num_increments: incremental BC steps
    max_iter      : inner iterations per increment
    tolerance     : BC MSE convergence threshold (early-exit when r_bc².mean() < tolerance)
    step_size     : inner BC update step size
    use_corrector : restore PDE residual after predictor step
    rcond         : pseudoinverse rank truncation threshold
    damping       : fraction of the Gauss--Newton corrector step taken in every increment
                    (1.0: the full step)
    x_pde_sampler : optional callable(step, residual_fn) -> tensor [N, d_in]. If given, a new
                    set of collocation points is drawn at the start of every increment
                    (step = 1..num_increments) and used for that increment's corrector,
                    projector and predictor iterations; residual_fn(x) returns the PDE
                    residual at the current parameters. x_pde is then unused.
    final_corrector: run one more increment at the final target after the last one
                    (corrector, then predictor iterations; on a fresh point set when a
                    sampler is given, step = num_increments + 1)

    Updates the model in-place.
    """
    was_training = model.training
    model.eval()
    try:
        # Restore all params to the base state so non-target layers are consistent
        # when build_param_dict reads them for Jacobian computation.
        if hasattr(model, "_adr_base_state"):
            for name, p in model.named_parameters():
                if name in model._adr_base_state:
                    p.data.copy_(model._adr_base_state[name])

        # Output layer map is linear in params — one projected lstsq step is exact
        if target_layers == [-1]:
            num_increments = 1

        theta_final = _scalable_ntk(
            model,
            pde_fn,
            bcs,
            target_layers,
            x_pde,
            num_increments,
            max_iter,
            tolerance,
            step_size,
            use_corrector,
            rcond,
            damping=damping,
            x_pde_sampler=x_pde_sampler,
            final_corrector=final_corrector,
        )
        _write_theta(model, theta_final, target_layers)
    finally:
        model.train(was_training)
