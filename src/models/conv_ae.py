import torch.nn as nn

def down(c_in, c_out):
    return nn.Sequential(nn.Conv1d(c_in, c_out, 7, stride=2, padding=3),
                         nn.BatchNorm1d(c_out), nn.ReLU())

def up(c_in, c_out):
    return nn.Sequential(nn.ConvTranspose1d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm1d(c_out), nn.ReLU())

class ConvAE(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(                       # (N, 1, 2048) -> (N, latent)
            down(1, 16), down(16, 32), down(32, 64), down(64, 64), down(64, 64),
            nn.Flatten(), nn.Linear(64 * 64, latent))
        self.decoder_fc = nn.Linear(latent, 64 * 64)
        self.decoder = nn.Sequential(                       # (N, 64, 64) -> (N, 1, 2048)
            up(64, 64), up(64, 64), up(64, 32), up(32, 16),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.decoder_fc(z).view(-1, 64, 64))