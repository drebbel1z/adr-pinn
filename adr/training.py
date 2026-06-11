import torch
import torch.optim as optim
from .adaptation import save_base


def train(model, pde_fn, x_pde, x_bc, y_bc,
          adam_epochs=5000, lbfgs_max_iter=1000,
          lr=5e-3, lambdas=(1.0, 1.0, 0.0),
          print_every=1000):
    """Train a PINN with Adam followed by L-BFGS.

    Parameters
    ----------
    model          : nn.Module
    pde_fn         : callable(model, x) -> residual [N]
    x_pde          : tensor [M, d_in] — PDE collocation points
    x_bc           : tensor [K, d_in] — BC collocation points
    y_bc           : tensor [K, d_out] — BC target values
    adam_epochs    : number of Adam steps
    lbfgs_max_iter : max L-BFGS iterations
    lr             : Adam learning rate
    lambdas        : (lam_pde, lam_bc, lam_data) loss weights
    print_every    : print interval for Adam (0 to suppress)
    """
    lam_pde, lam_bc, _ = lambdas

    def pde_loss():
        x_t = x_pde.clone().requires_grad_(True)
        return torch.mean(pde_fn(model, x_t).view(-1) ** 2)

    def bc_loss():
        return torch.mean((model(x_bc) - y_bc) ** 2)

    def total_loss():
        loss = 0.0
        if lam_pde > 0:
            loss = loss + lam_pde * pde_loss()
        if lam_bc > 0:
            loss = loss + lam_bc * bc_loss()
        return loss

    best_loss = float('inf')
    best_state = None

    def _maybe_save(loss_val):
        nonlocal best_loss, best_state
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)
    model.train()

    for epoch in range(adam_epochs):
        optimizer.zero_grad()
        loss = total_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        _maybe_save(loss.item())

        if print_every and epoch % print_every == 0:
            print(f"  Adam {epoch:05d}/{adam_epochs} | LR: {scheduler.get_last_lr()[0]:.2e} | Loss: {loss.item():.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # L-BFGS
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_max_iter,
        max_eval=int(lbfgs_max_iter * 1.25),
        history_size=100,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-16,
        tolerance_change=1e-16,
    )

    lbfgs_iter = 0

    def closure():
        nonlocal lbfgs_iter
        optimizer_lbfgs.zero_grad()
        loss = total_loss()
        loss.backward()
        _maybe_save(loss.item())
        if print_every and lbfgs_iter % 100 == 0:
            print(f"  L-BFGS {lbfgs_iter:05d} | Loss: {loss.item():.2e}")
        lbfgs_iter += 1
        return loss

    model.train()
    optimizer_lbfgs.step(closure)

    if best_state is not None:
        model.load_state_dict(best_state)

    if print_every:
        lp = pde_loss().item()
        lb = bc_loss().item()
        print(f"  Training complete. Best loss: {best_loss:.2e} | pde {lp:.2e} | bc {lb:.2e}")

    model.eval()
    save_base(model)
