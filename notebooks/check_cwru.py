import csv
import os

import matplotlib.pyplot as plt
import scipy.io as sio

DATA_DIR = os.path.join("data", "raw", "cwru")


def load_de_signal(file_name):
    m = sio.loadmat(os.path.join(DATA_DIR, file_name))
    num = int(file_name.split(".")[0])
    key = f"X{num:03d}_DE_time"
    if key not in m:
        de_keys = [k for k in m if k.endswith("DE_time")]
        if not de_keys:
            return None, None
        key = de_keys[0]
    return key, m[key].ravel()


with open(os.path.join(DATA_DIR, "files.csv")) as f:
    rows = list(csv.DictReader(f))

print(f"{'file':>8}  {'type':>6} {'size':>5} load  {'key':<14} {'samples':>8}")
for r in rows:
    key, x = load_de_signal(r["file"])
    if x is None:
        print(f"{r['file']:>8}  NO Drive End channel found!")
        continue
    print(f"{r['file']:>8}  {r['type']:>6} {r['size']:>5} {r['load_hp']}HP   "
          f"{key:<14} {len(x):>8}")

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, name, title in [(axes[0], "97.mat", "Normal, 0 HP"),
                        (axes[1], "105.mat", "Inner Race fault 0.007 in, 0 HP")]:
    _, x = load_de_signal(name)
    ax.plot(x[:2048], linewidth=0.7)
    ax.set_title(title)
    ax.set_ylabel("acceleration, g")
axes[1].set_xlabel("sample")
plt.tight_layout()
plt.savefig(os.path.join("results", "cwru_normal_vs_fault.png"), dpi=150)
plt.show()