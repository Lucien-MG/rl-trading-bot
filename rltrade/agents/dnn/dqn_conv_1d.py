#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import torch
import torch.nn as nn

class DQNConv1D(nn.Module):

    def __init__(self, input_shape, action_space):
        super(DQNConv1D, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(input_shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(input_shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            #nn.Sigmoid()
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
            #nn.Sigmoid()
        )

    def _get_conv_out(self, shape):
        out = self.conv1d(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(out.size())))

    def forward(self, x):
        conv1d = self.conv1d(x).view(x.size()[0], -1)
        val = self.fc_val(conv1d)
        adv = self.fc_adv(conv1d)
        return val + (adv - adv.mean(dim=1, keepdim=True))
