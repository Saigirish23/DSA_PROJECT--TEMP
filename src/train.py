"""
train.py — GNN Training Loop

Handles:
  - Model initialization with configurable hyperparameters
  - Adam optimizer with weight decay for L2 regularization
  - Class-weighted CrossEntropyLoss to handle fraud/normal imbalance
  - Per-epoch training with periodic logging
  - Best model checkpoint saving based on test F1 score
  - Training history tracking for loss curves

Usage:
    from src.train import train_model
    model, history = train_model(data, device)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import config
from src.gnn_model import FraudGCN

logger = config.setup_logging(__name__)


def compute_class_weights(y, train_mask):
    """
    Compute inverse-frequency class weights to handle class imbalance.

    Fraud is rare (~15%), so the model would achieve high accuracy by
    predicting everything as 'normal'. Class weights penalize
    misclassification of the minority class more heavily.

    Args:
        y (torch.Tensor): Label tensor [N].
        train_mask (torch.Tensor): Boolean mask for training nodes.

    Returns:
        torch.Tensor: Class weights [num_classes].
    """
    train_labels = y[train_mask].numpy()
    class_counts = np.bincount(train_labels, minlength=config.GNN_NUM_CLASSES)
    total = class_counts.sum()

    # Inverse frequency: rarer class gets higher weight
    weights = total / (len(class_counts) * class_counts + 1e-8)
    weights = torch.tensor(weights, dtype=torch.float32)

    logger.info("  Class weights: %s (normal=%.2f, fraud=%.2f)",
                weights.tolist(), weights[0].item(), weights[1].item())
    return weights


def train_model(data, device="cpu"):
    """
    Train the GCN model on the prepared PyG data.

    Training procedure:
      1. Initialize FraudGCN with num_features from data
      2. Use Adam optimizer (lr=0.01, weight_decay=5e-4)
      3. Use NLLLoss with class weights for imbalanced fraud detection
      4. Train for NUM_EPOCHS, logging every LOG_INTERVAL epochs
      5. Save best model based on test F1 score

    Args:
        data (torch_geometric.data.Data): PyG data with x, edge_index, y, masks.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (trained_model, training_history_dict)
            history keys: 'train_loss', 'train_acc', 'test_acc', 'test_f1'
    """
    logger.info("=" * 60)
    logger.info("Starting GNN Training")
    logger.info("=" * 60)

    # Move data to device
    data = data.to(device)

    # Initialize model
    num_features = data.x.shape[1]
    model = FraudGCN(num_features).to(device)

    # Optimizer: Adam with L2 regularization via weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    logger.info("  Optimizer: Adam (lr=%.4f, weight_decay=%.4f)",
                config.LEARNING_RATE, config.WEIGHT_DECAY)

    # Loss function: NLLLoss with class weights
    class_weights = compute_class_weights(data.y.cpu(), data.train_mask.cpu())
    class_weights = class_weights.to(device)
    criterion = torch.nn.NLLLoss(weight=class_weights)
    logger.info("  Loss: NLLLoss with class weights")
    logger.info("  Epochs: %d, Log interval: %d", config.NUM_EPOCHS, config.LOG_INTERVAL)
    logger.info("-" * 60)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": [],
    }

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # --- Training ---
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            # Training accuracy
            train_correct = pred[data.train_mask] == data.y[data.train_mask]
            train_acc = train_correct.sum().item() / data.train_mask.sum().item()

            # Test accuracy
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = test_correct.sum().item() / data.test_mask.sum().item()

            # Test F1
            test_f1 = f1_score(
                data.y[data.test_mask].cpu().numpy(),
                pred[data.test_mask].cpu().numpy(),
                zero_division=0
            )

        # Record history
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch
            config.ensure_dirs()
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

        # Periodic logging
        if epoch % config.LOG_INTERVAL == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d/%d | Loss: %.4f | Train Acc: %.4f | Test Acc: %.4f | Test F1: %.4f",
                epoch, config.NUM_EPOCHS, loss.item(), train_acc, test_acc, test_f1
            )

    logger.info("-" * 60)
    logger.info("  Training complete!")
    logger.info("  Best Test F1: %.4f (epoch %d)", best_f1, best_epoch)
    logger.info("  Model saved to %s", config.BEST_MODEL_PATH)

    # Load best model
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    return model, history


def get_gnn_predictions(model, data, device="cpu"):
    """
    Get GNN predictions and probabilities for all nodes.

    Args:
        model (FraudGCN): Trained GCN model.
        data (torch_geometric.data.Data): PyG data object.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (predictions [N], probabilities [N, 2])
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)  # Convert log-probs to probabilities
        preds = out.argmax(dim=1)

    return preds.cpu(), probs.cpu()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, set_seeds, build_pyg_data
    from src.features import compute_all_features
    from src.heuristics import compute_fraud_scores, generate_heuristic_labels

    set_seeds()
    config.ensure_dirs()

    # Full pipeline to get data
    df, account_labels = load_dataset()
    G = build_graph(df)
    features_df = compute_all_features(G)
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)
    labels_series = labels_df.set_index("node_id")["heuristic_label"]
    data, scaler, node_to_idx = build_pyg_data(G, features_df, labels_series)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Train
    model, history = train_model(data, device)

    # Get predictions
    preds, probs = get_gnn_predictions(model, data, device)
    logger.info("\n✅ GNN training and prediction complete!")
    logger.info("  Predictions shape: %s", preds.shape)
