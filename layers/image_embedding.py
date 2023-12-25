import torch
import torch.nn as nn
from torch.nn import Linear


class ImageEmbedding(nn.Module):

    def __init__(self, imsize, n_channels, patch_size, D):

        super().__init__()
        self.imsize = imsize
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.D = D
        self.num_patches = self.imsize**2//self.patch_size**2
        self.linear_layer = Linear(patch_size**2, D)
        self.cls_token = torch.nn.Parameter(
                torch.randn(1, 1, n_channels, D))
        self.pos_embedding = torch.nn.Parameter(
                torch.randn(1, self.num_patches + 1, n_channels, D))

    def forward(self, image):

        bs = image.size(0)
        x = self.linear_layer(image)
        x = torch.cat([self.cls_token.expand(bs, -1, -1, -1).transpose(1, 2), x], dim=-2)
        x = x + self.pos_embedding.expand(bs, -1, -1, -1).transpose(1, 2)

        return x
