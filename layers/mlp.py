import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, width, Dmlp, dropout):
        super().__init__()
        self.width = width
        self.Dmlp = Dmlp
        self.layer_in = nn.Linear(width, Dmlp)
        self.layer_out = nn.Linear(Dmlp, width)
        self.dropout_layer = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, inp):
        out = self.layer_in(inp)
        out = self.tanh(out)
        out = self.layer_out(out)

        return out
