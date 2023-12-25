import torch
import torch.nn as nn
from layers.image_embedding import ImageEmbedding
from layers.encoder import Encoder, EncoderMd
from layers.projection import Projection
from layers.msa import MultiHeadAttention
from layers.mlp import MLP


class VIT(nn.Module):

    def __init__(self, embedding: ImageEmbedding, encoder: Encoder, projection: Projection):
        super().__init__()
        self.embedding = embedding
        self.Encoder = encoder
        self.projection = projection

    def forward(self, inp):
        out = self.embedding.forward(inp)
        out = self.Encoder.forward(out)
        out = self.projection.forward(out)

        return out


def vit_instance(imgsize: int, patch_size: int, n_channels: int, width: int, L: int, k: int, Dmlp: int,
                 num_classes: int, dropout: float):
    embedding_layer = ImageEmbedding(imgsize, n_channels, patch_size, width)
    multi_head_attention_layer = MultiHeadAttention(k, width, dropout)
    encoder_s = nn.ModuleList([])

    for _ in range(L):
        mlp = MLP(width, Dmlp, dropout=dropout)
        encoder = EncoderMd(multi_head_attention_layer, mlp, dropout)
        encoder_s.append(encoder)

    l_encoder = Encoder(L, encoder_s)
    projection_layer = Projection(width, num_classes, n_channels)
    vit = VIT(embedding_layer, l_encoder, projection_layer)

    # mlp was zero initialized in the paper (initializing all params to zero)

    for param in vit.parameters():
      torch.nn.init.normal_(param, 0, 0.02)  # following the gpt1 paper

    return vit
