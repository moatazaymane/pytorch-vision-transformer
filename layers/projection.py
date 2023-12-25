import torch
import torch.nn as nn


class Projection(nn.Module):

    def __init__(self, width, num_classes, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.width = width
        self.projection = nn.Linear(width * n_channels, num_classes)

    def forward(self, inp):
        assert inp[:, :, 0, :].flatten(1).shape[1] == self.width * self.n_channels
        out = self.projection(inp[:, :, 0, :].flatten(1))
        out = torch.softmax(out, dim=1)

        return out
    