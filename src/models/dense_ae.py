import torch.nn as nn

class DenseAE(nn.Module):
    def __init__(self, n_in=1024, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))