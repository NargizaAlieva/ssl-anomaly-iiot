import os

import numpy as np
import scipy.io as sio

DATA_DIR = os.path.join("data", "raw", "cwru")


def load(num):
    """Return (signal, rpm) for a CWRU file number."""
    m = sio.loadmat(os.path.join(DATA_DIR, f"{num}.mat"))
    x = m[f"X{num:03d}_DE_time"].ravel()
    rpm_keys = [k for k in m if k.endswith("RPM")]
    rpm = float(np.ravel(m[rpm_keys[0]])[0]) if rpm_keys else 1797.0
    return x, rpm


def peak_near(x, fs, f_target):
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs > f_target * 0.85) & (freqs < f_target * 1.15)
    return freqs[band][np.argmax(spec[band])], spec[band].max() / np.median(spec[band])


for num in (97, 99, 105):
    x, rpm = load(num)
    f_shaft = rpm / 60
    print(f"\n{num}.mat  RPM={rpm:.0f}  expected shaft frequency {f_shaft:.1f} Hz")
    print(f"   duration: {len(x) / 12000:.1f} s at 12 kHz, {len(x) / 48000:.1f} s at 48 kHz")
    for fs in (12000, 48000):
        f, ratio = peak_near(x, fs, f_shaft)
        print(f"   assuming fs={fs:>5}: peak at {f:6.1f} Hz, peak strength x{ratio:.0f}")