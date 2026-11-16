import random
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev

EPS = 1e-16  # numerical stability


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def use_float64():
    torch.set_default_dtype(torch.float64)


def resolve_target_layers(model, target_layers):
    """Convert integer indices to Linear layer name strings.

    Accepts a mixed list of ints (index into all nn.Linear layers in the model)
    or strings (module names from model.named_modules()). Returns a list of strings.
    """
    if not target_layers:
        return []
    if isinstance(target_layers[0], str):
        return list(target_layers)
    linear_names = [
        name for name, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    n = len(linear_names)
    return [linear_names[i] for i in target_layers if -n <= i < n]


def get_target_params(model, target_layers):
    """Return the list of parameters belonging to the specified layers.

    target_layers: list of str (module names) or int (index into Linear layers).
    """
    layer_names = set(resolve_target_layers(model, target_layers))

    params = []
    for name, m in model.named_modules():
        if name in layer_names:
            params.extend(m.parameters())
    return list(dict.fromkeys(params))  # remove duplicates while preserving order


def build_param_dict(model, theta, target_layers):
    """Build a full parameter dict for functional_call with target params replaced by theta.

    Non-target parameters are kept as-is (shared with the live model).
    """
    target_params = get_target_params(model, target_layers)
    target_ids = {id(p) for p in target_params}

    param_dict = {}
    idx = 0
    for name, p in model.named_parameters():
        if id(p) in target_ids and theta is not None:
            numel = p.numel()
            param_dict[name] = theta[idx : idx + numel].view_as(p)
            idx += numel
        else:
            param_dict[name] = p
    return param_dict


def compute_jacobian(output_fn, params):
    """Compute Jacobian of output_fn(*params) -> [N] w.r.t. params via jacrev.

    output_fn: callable(*params) -> [N] tensor
    params: list of parameter tensors to differentiate w.r.t.

    Returns J of shape [N, d] — data points as rows, parameters as columns.
    """
    argnums = tuple(range(len(params)))
    J_tuple = jacrev(output_fn, argnums=argnums)(*params)
    N = J_tuple[0].shape[0]
    return torch.cat([j.reshape(N, -1) for j in J_tuple], dim=1).detach()


def spectral_scale(J1, J2):
    """Return max_eig(J1 J1^T) / max_eig(J2 J2^T) for balancing two Jacobians."""

    def _max_eig(J):
        return torch.linalg.svdvals(J)[0].item() ** 2 + EPS

    return _max_eig(J1) / _max_eig(J2)
