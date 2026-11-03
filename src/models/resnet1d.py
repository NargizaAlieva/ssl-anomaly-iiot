import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm1d(c_out))
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(c_out)))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))


class ResNet1D18(nn.Module):

    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1))
        layers, c_in = [], width
        for i, c_out in enumerate([width, 2 * width, 4 * width, 8 * width]):
            layers += [BasicBlock(c_in, c_out, stride=1 if i == 0 else 2), BasicBlock(c_out, c_out)]
            c_in = c_out
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_in

    def forward(self, x):
        return self.pool(self.layers(self.stem(x))).flatten(1)


class ProjectionHead(nn.Module):

    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, h):
        return self.net(h)


if __name__ == "__main__":
    for width in (64, 32):
        enc = ResNet1D18(width)
        h = enc(torch.randn(4, 1, 2048))
        z = ProjectionHead(enc.out_dim)(h)
        n = sum(p.numel() for p in enc.parameters())
        print(f"width {width}: embedding {tuple(h.shape)}, projection {tuple(z.shape)}, "
              f"encoder params {n / 1e6:.2f} M")