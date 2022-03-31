from torch import nn


# ALGO LOGIC: initialize agent here:
class BaselineQNetwork(nn.Module):
    def __init__(self, env):
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
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        out = self.pre(x / 255.0)
        out = self.core(out)
        out = self.post(out)
        return out
