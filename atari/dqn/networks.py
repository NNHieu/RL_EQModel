import torch
import torch.nn.functional as F
from torch import nn

from .deq.mon import mon
from .deq.mon import splitting as sp


class NetworkName(object):
    BASELINE = "baseline"
    MONDEQ = "mondeq"
    RECUR = "recur"


# ALGO LOGIC: initialize agent here:
class BaselineQNetwork(nn.Module):
    def __init__(self, out_features):
        super(BaselineQNetwork, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
        )
        self.core = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.post = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        out = self.pre(x / 255.0)
        out = self.core(out)
        out = self.post(out)
        return out


# ------------------------------------- Mon DEQ -----------------------------------------
MON_DEFAULTS = {"alpha": 1.0, "tol": 1e-5, "max_iter": 50}


def expand_args(defaults, kwargs):
    d = defaults.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d


class MonDEQConv2d(nn.Module):
    def __init__(self, splittingMethod: sp.MONPeacemanRachford, in_dim, in_channels, out_channels, m=0.1, **kwargs):
        super().__init__()
        n = in_dim
        shp = (n, n)
        # self.pool = 4
        # self.out_dim = out_channels * (n // self.pool) ** 2
        linear_module = mon.MONSingleConv(in_channels, out_channels, shp, m=m)
        nonlin_module = mon.MONBorderReLU(linear_module.pad[0])
        self.mon = splittingMethod(linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        # self.Wout = nn.Linear(self.out_dim, 10)

    def forward(self, x):
        # x = F.pad(x, (1, 1, 1, 1))
        z = self.mon(x)
        # z = F.avg_pool2d(z[-1], self.pool)
        return z[-1]


# ALGO LOGIC: initialize agent here:
class QMon(nn.Module):
    def __init__(self, out_features, m, alpha, tol, max_iter):
        super(QMon, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
        )
        self.core = MonDEQConv2d(
            sp.MONPeacemanRachford, in_dim=9, in_channels=64, out_channels=64, m=m, alpha=alpha, tol=tol, max_iter=max_iter
        )
        self.post = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            # nn.Linear(512, env.single_action_space.n),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        out = self.pre(x / 255.0)
        out = self.core(out)
        out = out[:, :, 1:-1, 1:-1]
        out = self.post(out)
        return out


# -------------------------- Recur ----------------------------
class InjectLayer(nn.Module):
    def __init__(self, n_chanels) -> None:
        super(InjectLayer, self).__init__()
        self.conv_z = nn.Conv2d(n_chanels, n_chanels, 3, stride=1, padding="same", bias=False)

    def forward(self, z, bias):
        out = self.conv_z(z)
        return F.relu(out + bias)


# ALGO LOGIC: initialize agent here:
class QRecurNetwork(nn.Module):
    def __init__(self, out_feature, num_iters=10):
        super(QRecurNetwork, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding="same", bias=False),
        )
        self.core = InjectLayer(64)
        self.post = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_feature),
        )
        self.num_iters = num_iters

    def forward(self, x, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iters

        x = self.pre(x / 255.0)
        out = torch.zeros_like(x)
        for i in range(num_iters):
            out = self.core(out, x)
        out = out[:, :, 1:-1, 1:-1]
        out = self.post(out)
        return out
