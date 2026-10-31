import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(s_val, s_test, y_test, pct=99):
    thr = np.percentile(s_val, pct)
    alarm = s_test > thr
    return {"auc": float(roc_auc_score(y_test, s_test)),
            "far": float(alarm[y_test == 0].mean()),
            "tpr": float(alarm[y_test == 1].mean())}

def convergence_epoch(auc_history, ref, frac=0.95):
    target = 0.5 + frac * (ref - 0.5)
    for epoch, auc in enumerate(auc_history, start=1):
        if auc >= target:
            return epoch
    return len(auc_history)