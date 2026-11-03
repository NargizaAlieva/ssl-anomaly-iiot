import torch
import torch.nn.functional as F


def time_shift(x, max_shift=512):
    N, _, L = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (N,), device=x.device)
    idx = (torch.arange(L, device=x.device)[None, :] - shifts[:, None]) % L
    return torch.gather(x, 2, idx[:, None, :])


def amplitude_scale(x, low=0.8, high=1.2):
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(low, high)
    return x * scale


def add_noise(x, sigma=0.05):
    return x + sigma * x.std(dim=-1, keepdim=True) * torch.randn_like(x)


def time_mask(x, max_frac=0.1):
    N, _, L = x.shape
    width = (torch.rand(N, device=x.device) * max_frac * L).long()
    start = (torch.rand(N, device=x.device) * (L - width)).long()
    pos = torch.arange(L, device=x.device)[None, :]
    keep = (pos < start[:, None]) | (pos >= (start + width)[:, None])
    return x * keep[:, None, :]


def time_stretch(x, max_rate=0.05):
    N, _, L = x.shape
    out = torch.empty_like(x)
    for i, rate in enumerate(torch.empty(N).uniform_(1 - max_rate, 1 + max_rate).tolist()):
        n = int(round(L * rate))
        y = F.interpolate(x[i:i + 1], size=n, mode="linear", align_corners=False)
        out[i] = y[..., :L] if n >= L else F.pad(y, (0, L - n), mode="circular")
    return out


AUGMENTATIONS = {"shift": time_shift, "scale": amplitude_scale, "noise": add_noise,
                 "mask": time_mask, "stretch": time_stretch}


def augment(x, names=tuple(AUGMENTATIONS), p=0.8):
    for name in names:
        apply = torch.rand(x.size(0), 1, 1, device=x.device) < p
        x = torch.where(apply, AUGMENTATIONS[name](x), x)
    return x