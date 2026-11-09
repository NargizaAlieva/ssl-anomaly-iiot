import csv
import os

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