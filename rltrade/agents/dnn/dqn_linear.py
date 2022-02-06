#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import torch.nn as nn

class SimpleFFDQN(nn.Module):

    def __init__(self, input_shape, action_space):
        """Init DQNlinear class
        Args:
            input_shape (int): shape on the convolution
            action_space (int): degree of liberty
        """
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        """Get the linear dqn to the next step
        Args:
            shape (obj): the convolution
        """
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))
