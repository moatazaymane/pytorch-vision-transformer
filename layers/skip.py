import torch.nn as nn
from layers.norm import Norm


class Skip(nn.Module):

    def __init__(self, dropout, layer):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer = layer
        self.norm = Norm()

    def forward(self, inp):
        out = self.norm(inp)
        out = self.layer(out)
        out = self.dropout_layer(out)

        return inp + out
