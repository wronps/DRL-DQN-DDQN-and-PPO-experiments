"""Q-networks used in the experiments."""

import torch.nn as nn


class CartPoleQNetwork(nn.Module):
    """Two hidden layers of 64 ReLU units."""

    def __init__(self, input_size, num_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, num_actions))

    def forward(self, input):
        return self.network(input)


# Adapted from https://github.com/hungtuchen/pytorch-dqn/blob/master/dqn_model.py
class JumpingTaskQNetwork(nn.Module):
    """MLP on the flattened 60x60 binary frame (3600 inputs)."""

    def __init__(self, num_actions=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3600, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.fc(x)
