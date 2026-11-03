import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet1d import ProjectionHead, ResNet1D18


def nt_xent(z1, z2, tau=0.2):
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, target)


class SimCLR(nn.Module):
    def __init__(self, width=64, proj_dim=128):
        super().__init__()
        self.encoder = ResNet1D18(width)
        self.head = ProjectionHead(self.encoder.out_dim, proj_dim)

    def forward(self, x):
        return self.head(self.encoder(x))