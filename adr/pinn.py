import torch.nn as nn


class MLP(nn.Module):
    """Standard feedforward PINN: input → [hidden × depth] → output."""

    def __init__(self, n_in, n_out=1, hidden=100, depth=3, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """PINN with residual skip connections: h = act(W h + b) + h at each hidden layer.

    All hidden layers share the same width.
    """

    def __init__(self, n_in, n_out=1, hidden=100, depth=3, activation=nn.Tanh):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(n_in, hidden))
        for _ in range(depth - 1):
            self.linears.append(nn.Linear(hidden, hidden))
        self.linears.append(nn.Linear(hidden, n_out))
        self.activation = activation()

    def forward(self, x):
        h = self.activation(self.linears[0](x))
        for linear in self.linears[1:-1]:
            h = self.activation(linear(h)) + h
        return self.linears[-1](h)
