import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels=6625, fc_decay=0.0004, mid_channels=None, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels

    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            predicts = self.fc1(x)
            predicts = self.fc2(predicts)

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts