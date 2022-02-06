#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import torch
import torch.nn as nn

class DQNConv2D(nn.Module):

    def __init__(self, input_shape, action_space):
        """Init DQNConv2D class
        Args:
            input_shape (int): shape on the convolution
            action_space (int): degree of liberty
        """
        super(DQNConv2D, self).__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(input_shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def _get_conv_out(self, shape):
        """Get return value of conv2D
        Args:
            shape (int): shape of conv2D
        """
        out = self.conv2d(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(out.size())))

    def forward(self, x):
        """Get the conv2 to the next step
        Args:
            shape (obj): the convolution
        """
        conv1d = self.conv1d(x).view(x.size()[0], -1)
        val = self.fc_val(conv1d)
        adv = self.fc_adv(conv1d)
        return val + (adv - adv.mean(dim=1, keepdim=True))
