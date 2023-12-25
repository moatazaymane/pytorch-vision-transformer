import torch.nn


class Norm(torch.nn.Module):

    def __init__(self, epsilon: float = 10 ** -6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inp):
        mean = inp.mean(dim=(-2, -1), keepdim=True)
        std = inp.std(dim=(-2, -1), keepdim=True)
        return self.gamma * (inp - mean) / (std + self.epsilon) + self.beta
