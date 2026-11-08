import csv
import os

import numpy as np
import scipy.io as sio

DATA_DIR = os.path.join("data", "raw", "cwru")
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def peak_near(x, fs, f_target):
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs >= f_target * 0.95) & (freqs <= f_target * 1.05)
    return freqs[band][np.argmax(spec[band])]


with open(os.path.join(DATA_DIR, "files.csv")) as f:
    rows = list(csv.DictReader(f))

errors = {"normal": [], "fault": []}
print("file   type    load   shaft Hz | peak if 12 kHz | peak if 48 kHz")
for r in rows:
    num = int(r["file"].split(".")[0])
    m = sio.loadmat(os.path.join(DATA_DIR, r["file"]))
    x = m[f"X{num:03d}_DE_time"].ravel()
    rpm_key = f"X{num:03d}RPM"
    rpm = float(np.ravel(m[rpm_key])[0]) if rpm_key in m else NOMINAL_RPM[int(r["load_hp"])]
    shaft = rpm / 60
    p12, p48 = peak_near(x, 12000, shaft), peak_near(x, 48000, shaft)
    print(f"{num:4d}   {r['type']:6}  {r['load_hp']}HP   {shaft:7.2f}  | {p12:14.2f} | {p48:14.2f}")
    group = "normal" if r["type"] == "normal" else "fault"
    errors[group].append((abs(p12 - shaft), abs(p48 - shaft)))

for group, values in errors.items():
    e12, e48 = np.mean(values, axis=0)
    print(f"\n{group}: mean error {e12:.3f} Hz if 12 kHz, {e48:.3f} Hz if 48 kHz")