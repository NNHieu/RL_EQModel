import numpy as np
import torch
from deq.mon import mon
from deq.mon import splitting as sp
from torch import nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaselineAgent(nn.Module):
    def __init__(self, envs):
        super(BaselineAgent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ------------------------ Mon DEQ ------------------------------------
MON_DEFAULTS = {"alpha": 1.0, "tol": 1e-5, "max_iter": 50}


def expand_args(defaults, kwargs):
    d = defaults.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d


class SingleFcMonLayer(nn.Module):
    def __init__(self, splittingMethod, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        linear_module = mon.MONSingleFc(in_dim, out_dim, m=m)
        nonlin_module = mon.MONReLU()
        self.mon = splittingMethod(linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        return z[-1]


class MonNet(nn.Module):
    def __init__(self, in_features, out_features, out_std, m=0.1, **kwargs) -> None:
        super(MonNet, self).__init__()
        self.pre = nn.Sequential(
            layer_init(nn.Linear(in_features, 64)),
            nn.ReLU(),
        )
        self.core = SingleFcMonLayer(sp.MONPeacemanRachford, in_dim=64, out_dim=64, m=0.1, **kwargs)
        self.post = layer_init(nn.Linear(64, out_features), std=out_std)

    def forward(self, x):
        out = self.pre(x)
        out = self.core(out)
        out = self.post(out)
        return out


class MonAgent(nn.Module):
    def __init__(self, envs):
        super(MonAgent, self).__init__()
        self.critic = MonNet(np.array(envs.single_observation_space.shape).prod(), 1, 1.0)
        self.actor_mean = MonNet(
            np.array(envs.single_observation_space.shape).prod(), np.prod(envs.single_action_space.shape), 0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
