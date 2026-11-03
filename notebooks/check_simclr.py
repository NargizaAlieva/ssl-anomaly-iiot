import math
import sys

import torch

sys.path.insert(0, "src")
from models.simclr import SimCLR, nt_xent

torch.manual_seed(0)
N = 8
z = torch.randn(N, 128)

print(f"identical views: loss {nt_xent(z, z).item():.3f}  (expected close to 0)")

print(f"unrelated views: loss {nt_xent(z, torch.randn(N, 128)).item():.3f}  "
      f"(expected about log({2 * N - 1}) = {math.log(2 * N - 1):.2f})")

model = SimCLR()
print("model output:", tuple(model(torch.randn(4, 1, 2048)).shape), " (expected (4, 128))")