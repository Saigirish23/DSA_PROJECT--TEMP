"""Tuned Elliptic training with staged ablations (Fix1 -> Fix4)."""

import os
import sys

# Allow running as `python src/bitcoin_train_tuned.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.bitcoin_model import EllipticGNN
from src.elliptic_loader import load_elliptic_full
from src.gnn_model import FraudGCN
from src.train import get_gnn_predictions, train_model


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary fraud classification."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        return fl.mean()


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float(y_true.mean()),
    }
    return metrics


def _build_model(model_type, num_features, hidden, dropout):
    if model_type == "fraudgcn":
        return FraudGCN(num_features=num_features, hidden_dim=hidden, dropout=dropout)
    if model_type == "ellipticgnn":
        return EllipticGNN(num_features=num_features, hidden=hidden, dropout=dropout)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _compute_class_alpha(y_train, device):
    counts = np.bincount(y_train, minlength=2)
    counts[counts == 0] = 1
    alpha_np = np.array([len(y_train) / (2.0 * c) for c in counts], dtype=np.float32)
    # Normalize and clip weights so weighted loss remains numerically stable.
    alpha_np = alpha_np / max(alpha_np.mean(), 1e-8)
    alpha_np = np.clip(alpha_np, 0.25, 4.0)
    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)
    return alpha


def train_stage(
    stage_name,
    model_type,
    loss_type,
    hidden=128,
    dropout=0.4,
    lr=0.001,
    weight_decay=1e-4,
    epochs=300,
    gamma=2.0,
    threshold=0.35,
    patience=40,
    ce_label_smoothing=0.02,
):
    """Train one stage configuration and return trained model + metrics."""
    data, scaler, feat_cols = load_elliptic_full()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)
    edge_attr = getattr(data, "edge_attr", None)

    model = _build_model(model_type=model_type, num_features=data.x.shape[1], hidden=hidden, dropout=dropout).to(device)

    train_mask = data.train_mask & (data.y >= 0)
    val_mask = data.val_mask & (data.y >= 0)
    test_mask = data.test_mask & (data.y >= 0)

    y_train = data.y[train_mask].detach().cpu().numpy()
    alpha = _compute_class_alpha(y_train, device)

    if loss_type == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=alpha, label_smoothing=ce_label_smoothing)
    elif loss_type == "focal":
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch_idx):
        warmup = 20
        if epoch_idx < warmup:
            return max(epoch_idx, 1) / float(warmup)
        progress = (epoch_idx - warmup) / float(max(epochs - warmup, 1))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = -1.0
    best_weights = None
    patience_count = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, edge_attr=edge_attr)
        loss = loss_fn(logits[train_mask], data.y[train_mask])
        raw_ce_loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x, data.edge_index, edge_attr=edge_attr)
            probs = torch.softmax(logits_eval, dim=1)[:, 1]

            y_val = data.y[val_mask].detach().cpu().numpy()
            prob_val = probs[val_mask].detach().cpu().numpy()
            pred_val = (prob_val >= threshold).astype(int)

            val_f1 = f1_score(y_val, pred_val, zero_division=0)
            val_auc = roc_auc_score(y_val, prob_val) if len(np.unique(y_val)) > 1 else 0.5

        history.append(
            {
                "stage": stage_name,
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_raw_ce": float(raw_ce_loss.item()),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    if best_weights is None:
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits_test = model(data.x, data.edge_index, edge_attr=edge_attr)
        probs_test = torch.softmax(logits_test, dim=1)[:, 1]

        y_test = data.y[test_mask].detach().cpu().numpy()
        prob_test = probs_test[test_mask].detach().cpu().numpy()
        pred_test = (prob_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, pred_test, prob_test)
    metrics["stage"] = stage_name
    metrics["threshold"] = threshold
    metrics["best_val_f1"] = float(best_val_f1)
    metrics["epochs_ran"] = int(len(history))
    metrics["model_type"] = model_type
    metrics["loss_type"] = loss_type
    metrics["num_features"] = int(data.x.shape[1])

    return model, metrics, history, data, feat_cols


def sweep_threshold(model, data, device, thresholds=None):
    """Sweep classification threshold on validation split and return best by F1."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.70, 0.05)

    model.eval()
    data = data.to(device)
    edge_attr = getattr(data, "edge_attr", None)

    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_attr=edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1]

    val_mask = (data.val_mask & (data.y >= 0)).detach().cpu().numpy()
    y_val = data.y.detach().cpu().numpy()[val_mask]
    prob_val = probs.detach().cpu().numpy()[val_mask]

    rows = []
    for t in thresholds:
        pred = (prob_val >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "f1": f1_score(y_val, pred, zero_division=0),
                "precision": precision_score(y_val, pred, zero_division=0),
                "recall": recall_score(y_val, pred, zero_division=0),
            }
        )

    best = max(rows, key=lambda x: x["f1"])
    return best["threshold"], rows


def run_fix_sequence(
    hidden=128,
    dropout=0.4,
    lr=0.001,
    weight_decay=1e-4,
    epochs=300,
    gamma=2.0,
    threshold=0.35,
    patience=40,
):
    """Run Fix 1->4 in order and persist stagewise metrics."""
    config.ensure_dirs()

    stage_results = []
    combined_history = []

    # Fix 1: Full features, baseline architecture/loss.
    m1, r1, h1, data, _ = train_stage(
        stage_name="fix1_full_features",
        model_type="fraudgcn",
        loss_type="ce",
        hidden=64,
        dropout=0.3,
        lr=0.001,
        weight_decay=weight_decay,
        epochs=epochs,
        gamma=gamma,
        threshold=0.50,
        patience=patience,
    )
    stage_results.append(r1)
    combined_history.extend(h1)

    # Fix 2: Deeper architecture, CE loss.
    m2, r2, h2, data, _ = train_stage(
        stage_name="fix2_deeper_model",
        model_type="ellipticgnn",
        loss_type="ce",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        gamma=gamma,
        threshold=0.50,
        patience=patience,
    )
    stage_results.append(r2)
    combined_history.extend(h2)

    # Fix 3: Deeper architecture + focal loss.
    m3, r3, h3, data, _ = train_stage(
        stage_name="fix3_focal_loss",
        model_type="ellipticgnn",
        loss_type="focal",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        gamma=gamma,
        threshold=threshold,
        patience=patience,
    )
    stage_results.append(r3)
    combined_history.extend(h3)

    # Fix 4: Threshold sweep on Fix 3 model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_t, sweep_rows = sweep_threshold(m3, data, device)

    m4, r4, h4, _, _ = train_stage(
        stage_name="fix4_threshold_tuned",
        model_type="ellipticgnn",
        loss_type="focal",
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        gamma=gamma,
        threshold=float(best_t),
        patience=patience,
    )
    stage_results.append(r4)
    combined_history.extend(h4)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    stage_df = pd.DataFrame(stage_results)
    hist_df = pd.DataFrame(combined_history)
    sweep_df = pd.DataFrame(sweep_rows)

    stage_path = os.path.join(config.RESULTS_DIR, "elliptic_tuned_metrics.csv")
    hist_path = os.path.join(config.RESULTS_DIR, "elliptic_training_history.csv")
    sweep_path = os.path.join(config.RESULTS_DIR, "elliptic_threshold_sweep.csv")
    model_path = os.path.join(config.MODELS_DIR, "elliptic_tuned.pt")

    stage_df.to_csv(stage_path, index=False)
    hist_df.to_csv(hist_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    torch.save(m4.state_dict(), model_path)

    return {
        "stage_metrics": stage_df,
        "history": hist_df,
        "threshold_sweep": sweep_df,
        "best_threshold": float(best_t),
        "checkpoint": model_path,
    }


def tune_and_train(
    hidden=128,
    dropout=0.4,
    lr=0.001,
    weight_decay=1e-4,
    epochs=300,
    gamma=2.0,
    threshold=0.35,
    patience=40,
):
    """Train AMLDetector on Elliptic data and return (model, metrics, history)."""
    del hidden, dropout, lr, weight_decay, epochs, gamma, threshold, patience

    data, _, _ = load_elliptic_full()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_model(data, device=device)

    preds, probs = get_gnn_predictions(model, data, device=device)
    y = data.y.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy() & (y >= 0)

    best_threshold = float(history.get("best_threshold", 0.35))
    y_true = y[test_mask]
    y_prob = probs.numpy()[test_mask, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = best_threshold

    # Persist outputs compatible with dashboard/result consumers.
    config.ensure_dirs()
    pd.DataFrame([metrics]).to_csv(os.path.join(config.RESULTS_DIR, "elliptic_tuned_metrics.csv"), index=False)

    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.get("loss", [])) + 1),
            "loss": history.get("loss", []),
            "train_f1": history.get("train_f1", []),
            "val_f1": history.get("val_f1", []),
            "val_pr_auc": history.get("val_pr_auc", []),
            "val_roc_auc": history.get("val_roc_auc", []),
            "lr": history.get("lr", []),
        }
    )
    hist_df.to_csv(os.path.join(config.RESULTS_DIR, "elliptic_training_history.csv"), index=False)

    return model, metrics, history


if __name__ == "__main__":
    out = run_fix_sequence()
    print("\nStage metrics:")
    print(out["stage_metrics"][["stage", "f1", "roc_auc", "pr_auc", "threshold"]].to_string(index=False))
    print("\nBest threshold:", out["best_threshold"])
    print("Checkpoint:", out["checkpoint"])
