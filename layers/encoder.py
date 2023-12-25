import torch.nn as nn
from layers.msa import MultiHeadAttention
from layers.mlp import MLP
from layers.skip import Skip
from layers.norm import Norm


class EncoderMd(nn.Module):

    def __init__(self, multi_head_attention_layer: MultiHeadAttention, mlp: MLP, dropout):
        super().__init__()
        self.skip_connexion_first = Skip(dropout, multi_head_attention_layer)
        self.skip_connexion_second = Skip(dropout, mlp)

    def forward(self, inp):
        out = self.skip_connexion_first.forward(inp)
        out = self.skip_connexion_second.forward(out)

        return out


class Encoder(nn.Module):

    def __init__(self, L: int, encoders: nn.ModuleList):
        super().__init__()
        self.L = L
        self.encoders = encoders
        self.norm = Norm()

    def forward(self, inp):
        out = inp
        for encoder in self.encoders:
            out = encoder(out)

        return self.norm(out)
