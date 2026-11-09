import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from augment import AUGMENTATIONS, augment
from metrics import convergence_epoch, evaluate
from models.simclr import SimCLR, nt_xent
from train_ae import DATA_DIR, SCENARIOS, prepare

@torch.no_grad()
def embed(model, X, device, batch=512):
    model.eval()
    return torch.cat([F.normalize(model.encoder(X[i:i + batch].to(device)), dim=1)
                      for i in range(0, len(X), batch)])

@torch.no_grad()
def knn_scores(bank, E, k):
    return (1 - (E @ bank.t()).topk(k, dim=1).values.mean(dim=1)).cpu().numpy()

@torch.no_grad()
def val_loss(model, X, args, device):
    model.eval()
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
        torch.manual_seed(0)
        x = X.to(device)
        return nt_xent(model(augment(x, args.augs)), model(augment(x, args.augs)), args.tau).item()

def run(name, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(os.path.join(DATA_DIR, f"cwru_{name}.npz"))
    X_train, X_val, X_mon, X_test = prepare(d, "conv", args.norm)
    loader = DataLoader(TensorDataset(X_train), batch_size=args.batch, shuffle=True, drop_last=True)

    model = SimCLR(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history, best, best_state, wait = [], float("inf"), None, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss = nt_xent(model(augment(x, args.augs)), model(augment(x, args.augs)), args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        sched.step()

        v_loss = val_loss(model, X_val, args, device)
        bank = embed(model, X_train, device)

        mon_auc = evaluate(knn_scores(bank, embed(model, X_val, device), args.k),
                           knn_scores(bank, embed(model, X_mon, device), args.k),
                           d["y_val_mon"])["auc"]
        history.append({"epoch": epoch, "train_loss": train_loss / len(loader),
                        "val_loss": v_loss, "mon_auc": mon_auc})
        if epoch % 10 == 0:
            print(f"   epoch {epoch:3d}  train loss {train_loss / len(loader):.3f}  "
                  f"val loss {v_loss:.3f}  monitoring AUC {mon_auc:.3f}")

        if v_loss < best:
            best, wait = v_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    best_epoch = int(np.argmin([h["val_loss"] for h in history])) + 1
    bank = embed(model, X_train, device)
    res = evaluate(knn_scores(bank, embed(model, X_val, device), args.k),
                   knn_scores(bank, embed(model, X_test, device), args.k), d["y_test"])
    res.update(model="simclr", norm=args.norm, scenario=name, seed=args.seed, best_epoch=best_epoch,
               epochs_run=len(history),
               convergence_epoch=convergence_epoch([h["mon_auc"] for h in history[:best_epoch]],
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
    p.add_argument("--norm", choices=["global", "window"], default="global")
    p.add_argument("--augs", nargs="+", choices=list(AUGMENTATIONS), default=list(AUGMENTATIONS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.out = args.out or os.path.join("results", f"simclr_{args.norm}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| model: simclr | norm:", args.norm, "| augs:", " ".join(args.augs))
    rows = []
    for name in args.scenarios:
        rows.append(run(name, args, device))
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"summary_seed{args.seed}.csv"), index=False)

if __name__ == "__main__":
    main()