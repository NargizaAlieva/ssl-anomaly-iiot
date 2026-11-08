import os
import urllib.request

import numpy as np
import scipy.io as sio

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"
OUT_DIR = os.path.join("data", "raw", "cwru_48k_control")
NOMINAL_RPM = {109: 1797, 110: 1772, 111: 1750, 112: 1730}


def peak_near(x, fs, f_target):
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs >= f_target * 0.95) & (freqs <= f_target * 1.05)
    return freqs[band][np.argmax(spec[band])]

os.makedirs(OUT_DIR, exist_ok=True)
errors = []
print("file   shaft Hz | peak if 12 kHz | peak if 48 kHz")
for num, nominal in NOMINAL_RPM.items():
    path = os.path.join(OUT_DIR, f"{num}.mat")
    if not os.path.exists(path):
        print(f"Downloading {num}.mat...")
        urllib.request.urlretrieve(BASE_URL.format(num), path)
    m = sio.loadmat(path)
    x = m[f"X{num:03d}_DE_time"].ravel()
    rpm_key = f"X{num:03d}RPM"
    rpm = float(np.ravel(m[rpm_key])[0]) if rpm_key in m else nominal
    shaft = rpm / 60
    p12, p48 = peak_near(x, 12000, shaft), peak_near(x, 48000, shaft)
    print(f"{num:4d}   {shaft:7.2f}  | {p12:14.2f} | {p48:14.2f}")
    errors.append((abs(p12 - shaft), abs(p48 - shaft)))

e12, e48 = np.mean(errors, axis=0)
print(f"\ncontrol: mean error {e12:.3f} Hz if 12 kHz, {e48:.3f} Hz if 48 kHz")