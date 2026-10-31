import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, n_mels=32, hidden=64, latent=64):
        super().__init__()
        self.encoder = nn.LSTM(n_mels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_mels)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        seq = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.decoder(seq)
        return self.out(y)