import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics import convergence_epoch, evaluate
from models.conv_ae import ConvAE

DATA_DIR = os.path.join("data", "processed")
SCENARIOS = ["S1", "S2", "S3", "LOLO0", "LOLO1", "LOLO2", "LOLO3"]

def to_tensor(X, mean, std):
    return torch.from_numpy((X - mean) / std).float().unsqueeze(1)

@torch.no_grad()
def scores(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = X[i:i + batch].to(device)
        out.append(((model(x) - x) ** 2).mean(dim=(1, 2)).cpu())
    return torch.cat(out).numpy()

def run(name, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(os.path.join(DATA_DIR, f"cwru_{name}.npz"))
    mean, std = float(d["mean"]), float(d["std"])
    X_train, X_val, X_mon, X_test = (to_tensor(d[k], mean, std)
                                     for k in ("X_train", "X_val", "X_val_mon", "X_test"))
    loader = DataLoader(TensorDataset(X_train), batch_size=args.batch, shuffle=True)

    model = ConvAE(args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best, best_state, wait = [], float("inf"), None, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)

        s_val = scores(model, X_val, device)

        mon_auc = evaluate(s_val, scores(model, X_mon, device), d["y_val_mon"])["auc"]
        history.append({"epoch": epoch, "train_loss": train_loss / len(X_train),
                        "val_loss": float(s_val.mean()), "mon_auc": mon_auc})

        if s_val.mean() < best:
            best, wait = s_val.mean(), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    res = evaluate(scores(model, X_val, device), scores(model, X_test, device), d["y_test"])
    res.update(scenario=name, seed=args.seed, best_epoch=best_epoch, epochs_run=len(history),
               convergence_epoch=convergence_epoch([h["mon_auc"] for h in history],
                                                   history[best_epoch - 1]["mon_auc"]),
               train_time_s=round(time.time() - start, 1))

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(args.out, f"{name}_seed{args.seed}_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.out, f"{name}_seed{args.seed}.pt"))
    print(f"{name:6} AUC {res['auc']:.3f}  FAR {res['far']:.3f}  TPR {res['tpr']:.3f}  "
          f"best epoch {best_epoch:3d}  convergence epoch {res['convergence_epoch']:3d}  "
          f"{res['train_time_s']:.0f} s")
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results", "conv_ae"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    rows = [run(name, args, device) for name in args.scenarios]
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"summary_seed{args.seed}.csv"), index=False)

if __name__ == "__main__":
    main()