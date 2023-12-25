import torch.nn as nn
import math


def attention(q, k, v, dropout_layer):
    dk = q.shape[-1]
    att = q @ k.transpose(-2, -1) / math.sqrt(dk)
    scores = att.softmax(dim=2)
    return dropout_layer(scores) @ v, scores


class MultiHeadAttention(nn.Module):

    def __init__(self, h, width, dropout):

        super().__init__()
        self.h = h
        self.width = width
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(self.width, self.width)

        self.Wk = nn.Linear(self.width, self.width)

        self.Wo = nn.Linear(self.width, self.width)

    def forward(self, inp):

        q, k, v = inp, inp, inp  # [bs, num_channels, num patches + 1 , D]
        q, k, v = self.Wq(q), self.Wk(k), self.Wo(v)
        # shape (bs, num_channels, h, num_patches + cls, D // h)
        qs = q.view(q.shape[0], q.shape[1], q.shape[2], 4, 512 // 4).transpose(-3, -2)
        ks = k.view(k.shape[0], k.shape[1], k.shape[2], 4, 512 // 4).transpose(-3, -2)
        vs = v.view(v.shape[0], v.shape[1], v.shape[2], 4, 512 // 4).transpose(-3, -2)

        x, maps = attention(qs, ks, vs, self.dropout)
        x = x.transpose(2, 3)
        x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], self.width)

        return self.Wo(x)
