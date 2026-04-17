"""Train the FraudGCN model on the Elliptic Bitcoin dataset."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.bitcoin_loader import load_bitcoin_dataset
from src.gnn_model import FraudGCN


def train_on_bitcoin():
    """Train FraudGCN on Elliptic Bitcoin dataset and report metrics."""
    print("\n" + "=" * 55)
    print("  Bitcoin Dataset — FraudGCN Training")
    print("=" * 55)

    data, _ = load_bitcoin_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    data = data.to(device)

    model = FraudGCN(num_features=data.x.shape[1], hidden_dim=config.GNN_HIDDEN_DIM).to(device)

    y_train = data.y[data.train_mask].cpu().numpy()
    y_train = y_train[y_train >= 0]
    counts = np.bincount(y_train, minlength=config.GNN_NUM_CLASSES)
    counts[counts == 0] = 1
    weights = torch.tensor([len(y_train) / c for c in counts], dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_f1 = -1.0
    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    history = {"loss": [], "val_acc": [], "val_f1": []}

    print(f"\n  Training for {config.NUM_EPOCHS} epochs...\n")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        train_mask = data.train_mask & (data.y != -1)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            preds = out_eval.argmax(dim=1)
            val_mask = data.val_mask & (data.y != -1)
            y_val = data.y[val_mask].cpu().numpy()
            p_val = preds[val_mask].cpu().numpy()
            val_acc = accuracy_score(y_val, p_val)
            val_f1 = f1_score(y_val, p_val, zero_division=0)

        history["loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}"
            )

    model.load_state_dict(best_weights)
    config.ensure_dirs()
    ckpt_path = os.path.join(config.MODELS_DIR, "bitcoin_gcn.pt")
    torch.save(best_weights, ckpt_path)
    print(f"\n  Best model saved: {ckpt_path} (val F1={best_val_f1:.4f})")

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        test_mask = data.test_mask & (data.y != -1)
        y_test = data.y[test_mask].cpu().numpy()
        p_test = out[test_mask].argmax(dim=1).cpu().numpy()
        prob_test = probs[test_mask][:, 1].cpu().numpy()

    acc = accuracy_score(y_test, p_test)
    prec = precision_score(y_test, p_test, zero_division=0)
    rec = recall_score(y_test, p_test, zero_division=0)
    f1 = f1_score(y_test, p_test, zero_division=0)
    auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.5

    print("\n" + "=" * 55)
    print("  Bitcoin Test Set Results")
    print("=" * 55)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print("=" * 55)

    results = pd.DataFrame(
        [
            {
                "model": "FraudGCN (Bitcoin)",
                "dataset": "Elliptic Bitcoin",
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }
        ]
    )
    results_path = os.path.join(config.RESULTS_DIR, "bitcoin_metrics.csv")
    results.to_csv(results_path, index=False)
    print(f"\n  Results saved: {results_path}")

    history_path = os.path.join(config.RESULTS_DIR, "bitcoin_training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    return model, history, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }


if __name__ == "__main__":
    train_on_bitcoin()
