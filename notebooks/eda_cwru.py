import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.join("src", "data"))
from cwru import NORMAL_FILES, RAW_DIR, WIN, load_signal

OUT_DIR = "results"
FS = 12000
HIGH_HZ = 2000
NAMES = {"normal": "Normal", "IR": "Inner race", "B": "Ball", "OR": "Outer race"}

os.makedirs(OUT_DIR, exist_ok=True)
files = pd.read_csv(os.path.join(RAW_DIR, "files.csv"))
signals = {f: load_signal(f) for f in files["file"]}


def high_share(x):
    """Share of signal power above HIGH_HZ."""
    freqs, psd = welch(x, fs=FS, nperseg=4096)
    return psd[freqs > HIGH_HZ].sum() / psd.sum()

inventory = []
for _, r in files.iterrows():
    num = int(r["file"].split(".")[0])
    m = sio.loadmat(os.path.join(RAW_DIR, r["file"]))
    x = signals[r["file"]]
    rpm_key = f"X{num:03d}RPM"
    inventory.append({
        "file": r["file"], "type": r["type"], "size_in": r["size"], "load_hp": r["load_hp"],
        "has_FE": f"X{num:03d}_FE_time" in m, "has_BA": f"X{num:03d}_BA_time" in m,
        "recorded_khz": 48 if r["file"] in NORMAL_FILES else 12,
        "samples_12k": len(x), "duration_s": round(len(x) / FS, 1),
        "rpm": float(np.ravel(m[rpm_key])[0]) if rpm_key in m else np.nan,
        "high_share": round(high_share(x), 3),
    })
inventory = pd.DataFrame(inventory)
inventory.to_csv(os.path.join(OUT_DIR, "cwru_inventory.csv"), index=False)

print("=== Inventory ===")
print(inventory.to_string(index=False))
print(f"\nTotal: {len(inventory)} files, {inventory.duration_s.sum():.0f} s of signal; "
      f"normal only: {inventory[inventory.type == 'normal'].duration_s.sum():.1f} s")

def window_features(x):
    n = len(x) // WIN
    w = x[:n * WIN].reshape(n, WIN)
    rms = np.sqrt((w ** 2).mean(axis=1))
    peak = np.abs(w).max(axis=1)
    return pd.DataFrame({"rms": rms, "peak": peak, "crest": peak / rms,
                         "kurtosis": kurtosis(w, axis=1, fisher=False),  # Gaussian = 3
                         "skew": skew(w, axis=1)})

parts = []
for _, r in files.iterrows():
    df = window_features(signals[r["file"]])
    df["file"], df["type"], df["size_in"], df["load_hp"] = r["file"], r["type"], r["size"], r["load_hp"]
    parts.append(df)
features = pd.concat(parts, ignore_index=True)
features.to_csv(os.path.join(OUT_DIR, "cwru_window_features.csv"), index=False)

print("\n=== Mean features by fault type and size (windows of 2048 samples) ===")
summary = (features.groupby(["type", "size_in"])[["rms", "peak", "crest", "kurtosis"]]
           .mean().round(3))
summary["high_share"] = inventory.groupby(["type", "size_in"])["high_share"].mean().round(3)
summary["windows"] = features.groupby(["type", "size_in"]).size()
print(summary.to_string())
summary.to_csv(os.path.join(OUT_DIR, "cwru_summary_by_type.csv"))

print("\n=== Normal data: does the load change the signal? ===")
by_load = features[features.type == "normal"].groupby("load_hp")[["rms", "kurtosis"]].mean()
by_load["high_share"] = inventory[inventory.type == "normal"].set_index("load_hp")["high_share"]
print(by_load.round(3).to_string())

examples = [("97.mat", "Normal"), ("105.mat", "Inner race 0.007 in"),
            ("118.mat", "Ball 0.007 in"), ("130.mat", "Outer race 0.007 in")]
t_ms = np.arange(WIN) / FS * 1000

fig, axes = plt.subplots(len(examples), 1, figsize=(10, 8), sharex=True)
for ax, (f, title) in zip(axes, examples):
    ax.plot(t_ms, signals[f][:WIN], linewidth=0.6)
    ax.set_title(f"{title}, 0 HP", fontsize=10)
    ax.set_ylabel("g")
axes[-1].set_xlabel("time, ms (one model window = 2048 samples)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_time_signals.png"), dpi=150)

fig, ax = plt.subplots(figsize=(10, 5))
for f, title in examples:
    freqs, psd = welch(signals[f], fs=FS, nperseg=4096)
    ax.semilogy(freqs, psd, linewidth=0.8, label=title)
ax.set_xlabel("frequency, Hz")
ax.set_ylabel("power spectral density, g²/Hz")
ax.set_title("Power spectra at 0 HP (all signals at 12 kHz)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_spectra.png"), dpi=150)

order = ["normal", "IR", "B", "OR"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, label in [(axes[0], "rms", "RMS, g"), (axes[1], "kurtosis", "kurtosis")]:
    ax.boxplot([features.loc[features.type == t, col] for t in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), [NAMES[t] for t in order])
    ax.set_ylabel(label)
axes[1].axhline(3, color="grey", linestyle="--", linewidth=0.8)  # Gaussian noise level
plt.suptitle("Window features by fault type (all sizes and loads)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_feature_boxplots.png"), dpi=150)
plt.close("all")

print(f"\nSaved tables and figures to '{OUT_DIR}/'")