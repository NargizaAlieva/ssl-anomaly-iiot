import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join("data", "processed")
OUT_DIR = "results"
SCENARIOS = ["S1", "S2", "S3", "LOLO0", "LOLO1", "LOLO2", "LOLO3"]
LOLO = [f"LOLO{k}" for k in range(4)]

def features(X):
    rms = np.sqrt((X ** 2).mean(axis=1))
    peak = np.abs(X).max(axis=1)
    return np.column_stack([rms, peak, peak / rms,
                            kurtosis(X, axis=1, fisher=False), skew(X, axis=1)])

def evaluate(s_val, s_test, y_test):
    thr = np.percentile(s_val, 99)
    alarm = s_test > thr
    return {"auc": roc_auc_score(y_test, s_test),
            "far": alarm[y_test == 0].mean(),
            "tpr": alarm[y_test == 1].mean()}

rows = []
for name in SCENARIOS:
    d = np.load(os.path.join(DATA_DIR, f"cwru_{name}.npz"))
    F_train, F_val, F_test = (features(d[k]) for k in ("X_train", "X_val", "X_test"))
    y = d["y_test"]

    for col, method in [(0, "RMS"), (3, "Kurtosis")]:
        rows.append({"scenario": name, "method": method,
                     **evaluate(F_val[:, col], F_test[:, col], y)})

    iso = IsolationForest(n_estimators=200, random_state=0).fit(F_train)
    rows.append({"scenario": name, "method": "IsolationForest",
                 **evaluate(-iso.score_samples(F_val), -iso.score_samples(F_test), y)})

res = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
res.to_csv(os.path.join(OUT_DIR, "baseline_features.csv"), index=False)

for metric in ("auc", "far", "tpr"):
    table = res.pivot(index="method", columns="scenario", values=metric)[SCENARIOS]
    table["LOLO_mean"] = table[LOLO].mean(axis=1)
    print(f"\n=== {metric.upper()} ===")
    print(table.round(3).to_string())