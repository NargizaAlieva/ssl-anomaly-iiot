import csv
import os

import numpy as np
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfiltfilt

RAW_DIR = os.path.join("data", "raw", "cwru")
OUT_DIR = os.path.join("data", "processed")

WIN = 2048
TRAIN_STEP = 64
NORMAL_EVAL_STEP = 128
FAULT_EVAL_STEP = 2048
SPLIT = (0.6, 0.2)
ALL_PARTS = ("train", "val", "test")

NORMAL_FILES = {"97.mat", "98.mat", "99.mat", "100.mat"}
LOWPASS = butter(8, 5000, fs=12000, output="sos")

def load_signal(file_name):
    m = sio.loadmat(os.path.join(RAW_DIR, file_name))
    key = f"X{int(file_name.split('.')[0]):03d}_DE_time"
    if key not in m:
        key = [k for k in m if k.endswith("DE_time")][0]
    x = m[key].ravel()
    if file_name in NORMAL_FILES:
        x = resample_poly(x, 1, 4)
    return sosfiltfilt(LOWPASS, x).astype(np.float32)

def split_by_time(x):
    a, b = int(len(x) * SPLIT[0]), int(len(x) * (SPLIT[0] + SPLIT[1]))
    return {"train": x[:a], "val": x[a:b], "test": x[b:]}

def make_windows(x, step):
    n = (len(x) - WIN) // step + 1
    idx = np.arange(WIN)[None, :] + step * np.arange(n)[:, None]
    return x[idx]

def load_records():
    with open(os.path.join(RAW_DIR, "files.csv")) as f:
        rows = list(csv.DictReader(f))
    return [{"file": r["file"], "type": r["type"], "size": float(r["size"]),
             "load": int(r["load_hp"]), "parts": split_by_time(load_signal(r["file"]))}
            for r in rows]

def collect(records, parts, condition, step=None):
    X, meta = [], []
    for rec in records:
        if not condition(rec):
            continue
        s = step or (NORMAL_EVAL_STEP if rec["type"] == "normal" else FAULT_EVAL_STEP)
        for part in parts:
            w = make_windows(rec["parts"][part], s)
            X.append(w)
            meta += [(rec["file"], rec["type"], rec["size"], rec["load"])] * len(w)
    return np.concatenate(X), np.array(meta, dtype=object)

def build_scenario(records, name, train_cond, test_cond, test_parts=("test",)):
    normal_train = lambda r: r["type"] == "normal" and train_cond(r)

    X_train, _ = collect(records, ("train",), normal_train, step=TRAIN_STEP)
    X_val, _ = collect(records, ("val",), normal_train)

    X_val_mon, m_val_mon = collect(records, ("val",), test_cond)
    X_test, m_test = collect(records, test_parts, test_cond)

    y_val_mon = (m_val_mon[:, 1] != "normal").astype(np.int64)
    y_test = (m_test[:, 1] != "normal").astype(np.int64)

    mean, std = float(X_train.mean()), float(X_train.std())

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"cwru_{name}.npz")
    np.savez_compressed(path, X_train=X_train, X_val=X_val,
                        X_val_mon=X_val_mon, y_val_mon=y_val_mon,
                        X_test=X_test, y_test=y_test,
                        meta_test=m_test.astype(str), mean=mean, std=std)

    print(f"{name:7}: train {len(X_train):>5} | val {len(X_val):>4} | "
          f"test {len(X_test):>5} (normal {int((y_test == 0).sum()):>4}, "
          f"fault {int(y_test.sum()):>4})  -> {path}")

def main():
    records = load_records()

    build_scenario(records, "S1",
                   train_cond=lambda r: r["load"] == 0,
                   test_cond=lambda r: r["load"] == 0)

    build_scenario(records, "S2",
                   train_cond=lambda r: r["load"] == 0,
                   test_cond=lambda r: r["load"] in (1, 2, 3))

    build_scenario(records, "S3",
                   train_cond=lambda r: True,
                   test_cond=lambda r: r["type"] == "normal" or r["size"] == 0.007)

    for k in range(4):
        build_scenario(records, f"LOLO{k}",
                       train_cond=lambda r, k=k: r["load"] != k,
                       test_cond=lambda r, k=k: r["load"] == k,
                       test_parts=ALL_PARTS)

if __name__ == "__main__":
    main()