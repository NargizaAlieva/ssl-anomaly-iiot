import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
from augment import AUGMENTATIONS, augment

torch.manual_seed(0)
d = np.load(os.path.join("data", "processed", "cwru_S1.npz"))
x = torch.from_numpy((d["X_train"][:1] - d["mean"]) / d["std"]).float().unsqueeze(1)

panels = ([("original", x)]
          + [(name, f(x)) for name, f in AUGMENTATIONS.items()]
          + [("view 1 (all, random)", augment(x)), ("view 2 (all, random)", augment(x))])

fig, axes = plt.subplots(len(panels), 1, figsize=(10, 13), sharex=True, sharey=True)
for ax, (title, y) in zip(axes, panels):
    ax.plot(y[0, 0].numpy(), linewidth=0.5)
    ax.set_title(title, fontsize=9, loc="left")
axes[-1].set_xlabel("sample")
plt.tight_layout()
plt.savefig(os.path.join("results", "augmentations.png"), dpi=150)
print("saved results/augmentations.png")