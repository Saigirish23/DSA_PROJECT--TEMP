#!/usr/bin/env python3
"""
Quick training script for Elliptic Bitcoin dataset with progress tracking.
Exports all required artifacts for dashboard integration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import config
from src.data_loader import load_data, build_graph, build_pyg_data, set_seeds
from src.features import compute_all_features
from src.train import train_model, get_gnn_predictions
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

def export_artifacts(model, history, data, device, node_to_idx):
    """Export training artifacts for dashboard integration."""
    config.ensure_dirs()
    
    # Export training history
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'val_f1': history['val_f1'],
        'test_acc': history['test_acc'],
        'test_f1': history['test_f1'],
    })
    history_path = os.path.join(config.RESULTS_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"      ✓ Training history exported to {history_path}")
    
    # Get final metrics
    preds, probs = get_gnn_predictions(model, data, device)
    labeled_mask = (data.y >= 0).cpu().numpy()
    test_mask = data.test_mask.cpu().numpy() & labeled_mask
    test_y = data.y.cpu().numpy()[test_mask]
    test_pred = preds.numpy()[test_mask]
    test_probs = probs.numpy()[test_mask, 1]
    
    acc = accuracy_score(test_y, test_pred)
    prec = precision_score(test_y, test_pred, zero_division=0)
    rec = recall_score(test_y, test_pred, zero_division=0)
    f1 = f1_score(test_y, test_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(test_y, test_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(test_y, test_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Export final metrics
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives'],
        'value': [acc, prec, rec, f1, auc, tn, fp, fn, tp],
    })
    metrics_path = os.path.join(config.RESULTS_DIR, 'final_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"      ✓ Final metrics exported to {metrics_path}")
    
    # Export node-level predictions
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    all_y = data.y.cpu().numpy()
    all_pred = preds.numpy()
    all_probs = probs.numpy()
    
    nodes = [idx_to_node.get(i, str(i)) for i in range(len(all_y))]
    
    predictions_df = pd.DataFrame({
        'node_id': nodes,
        'true_label': all_y,
        'predicted_label': all_pred,
        'fraud_probability': all_probs[:, 1],
    })
    predictions_path = os.path.join(config.RESULTS_DIR, 'node_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"      ✓ Node predictions exported to {predictions_path}")
    
    return acc, prec, rec, f1, auc, tn, fp, fn, tp

def main():
    set_seeds()
    config.ensure_dirs()

    print("\n" + "="*70)
    print("  ELLIPTIC BITCOIN DATASET - GCN TRAINING")
    print("="*70)

    # Load data
    print("\n[1/5] Loading Elliptic Bitcoin dataset...")
    df, account_labels = load_data()
    print(f"      ✓ Loaded {len(df)} transactions")

    print("\n[2/5] Building transaction graph...")
    G = build_graph(df)
    print(f"      ✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[3/5] Computing graph features (this may take a few minutes)...")
    features_df = compute_all_features(G)
    print(f"      ✓ Features computed: {features_df.shape}")

    print("\n[4/5] Building PyTorch Geometric data structure...")
    data, scaler, node_to_idx = build_pyg_data(G, features_df)
    print(f"      ✓ PyG Data ready")
    print(f"        - Nodes: {data.num_nodes()}")
    print(f"        - Edges: {data.num_edges()}")
    print(f"        - Features: {data.x.shape[1]}")
    print(f"        - Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    labeled_mask = data.y >= 0
    fraud_ratio = ((data.y[labeled_mask] == 1).float().mean().item() if labeled_mask.sum() > 0 else 0.0)
    print(f"        - Ground-truth fraud ratio (labeled only): {fraud_ratio:.2%}")

    print("\n[5/5] Training GCN model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Using device: {device}")

    model, history = train_model(data, device)

    print("\n" + "="*70)
    print("  TRAINING COMPLETE - EXPORTING ARTIFACTS")
    print("="*70)

    # Export artifacts
    print("\n[6/5] Exporting training artifacts...")
    acc, prec, rec, f1, auc, tn, fp, fn, tp = export_artifacts(model, history, data, device, node_to_idx)

    print("\n" + "="*70)
    print("  📊 TEST SET METRICS:")
    print("="*70)
    print(f"     Accuracy:   {acc:.4f}")
    print(f"     Precision:  {prec:.4f}")
    print(f"     Recall:     {rec:.4f}")
    print(f"     F1-Score:   {f1:.4f}")
    print(f"     ROC-AUC:    {auc:.4f}")

    print(f"\n  📋 CONFUSION MATRIX:")
    print(f"     True Negatives:  {tn}")
    print(f"     False Positives: {fp}")
    print(f"     False Negatives: {fn}")
    print(f"     True Positives:  {tp}")

    print(f"\n  📈 TRAINING HISTORY (last 5 epochs):")
    n_epochs = len(history['test_f1'])
    for i in range(max(0, n_epochs-5), n_epochs):
        print(f"     Epoch {i+1}: Loss={history['train_loss'][i]:.4f}, "
              f"Train Acc={history['train_acc'][i]:.4f}, "
              f"Test F1={history['test_f1'][i]:.4f}")

    print("\n" + "="*70)
    print(f"  ✅ Model saved to {config.BEST_MODEL_PATH}")
    print(f"  ✅ All artifacts exported to {config.RESULTS_DIR}/")
    print("="*70 + "\n")

    return model, history, {
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'test_auc': auc,
    }

if __name__ == "__main__":
    main()
