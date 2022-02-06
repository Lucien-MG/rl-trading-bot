#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import torch.nn as nn

class SimpleFFDQN(nn.Module):

    def __init__(self, input_shape, action_space):
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
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))
