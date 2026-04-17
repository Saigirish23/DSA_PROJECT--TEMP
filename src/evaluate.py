"""
evaluate.py — Metrics & Evaluation

Computes comprehensive evaluation metrics and generates visualizations:
    - Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Confusion Matrix
  - Training loss curves
  - ROC curves for all model variants
  - Fraud graph visualization (red=fraud, blue=normal)
  - Feature distribution histograms (fraud vs. normal)

All plots are saved to outputs/plots/.
All results are saved to outputs/results/.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, average_precision_score
)

import config

logger = config.setup_logging(__name__)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for ROC-AUC.

    Returns:
        dict: Metrics dictionary.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labeled_mask = y_true != -1
    y_true = y_true[labeled_mask]
    y_pred = y_pred[labeled_mask]

    if y_prob is not None:
        y_prob = np.asarray(y_prob)[labeled_mask]

    if y_true.size == 0:
        empty_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        if y_prob is not None:
            empty_metrics["roc_auc"] = 0.5
            empty_metrics["pr_auc"] = 0.0
        return empty_metrics

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = float(y_true.mean()) if len(y_true) else 0.0

    return metrics


def plot_loss_curve(history, save_path=None):
    """
    Plot training loss over epochs.

    Args:
        history (dict): Training history with 'train_loss' key.
        save_path (str, optional): Path to save the plot.
    """
    logger.info("Generating loss curve plot...")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    loss_key = "train_loss" if "train_loss" in history else "loss"
    if loss_key not in history:
        logger.warning("History missing loss key; skipping loss curve plot")
        return

    epochs = range(1, len(history[loss_key]) + 1)

    # Loss curve
    color_loss = "#e74c3c"
    ax1.plot(epochs, history[loss_key], color=color_loss, linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Loss", fontsize=13, color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    # F1 curve on secondary y-axis
    if "test_f1" in history or "val_f1" in history:
        f1_key = "test_f1" if "test_f1" in history else "val_f1"
        f1_label = "Test F1" if f1_key == "test_f1" else "Validation F1"
        ax2 = ax1.twinx()
        color_f1 = "#2ecc71"
        ax2.plot(epochs, history[f1_key], color=color_f1, linewidth=2,
                 linestyle="--", label=f1_label)
        ax2.set_ylabel("F1 Score", fontsize=13, color=color_f1)
        ax2.tick_params(axis="y", labelcolor=color_f1)

    fig.suptitle("GCN Training: Loss & F1 Over Epochs", fontsize=15, fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=11)
    fig.tight_layout()

    save_path = save_path or os.path.join(config.PLOTS_DIR, "loss_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", save_path)


def plot_roc_curves(results_dict, save_path=None):
    """
    Plot ROC curves for multiple model variants.

    Args:
        results_dict (dict): {model_name: (y_true, y_prob)} for each model.
        save_path (str, optional): Path to save the plot.
    """
    logger.info("Generating ROC curve plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, (y_true, y_prob)) in enumerate(results_dict.items()):
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label="{} (AUC = {:.3f})".format(name, auc))
        except Exception as e:
            logger.warning("  Could not compute ROC for %s: %s", name, e)

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = save_path or os.path.join(config.PLOTS_DIR, "roc_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", save_path)


def plot_fraud_graph(G, labels, node_to_idx=None, save_path=None):
    """
    Visualize the transaction graph with fraud nodes highlighted.

    Red nodes = fraud, Blue nodes = normal.
    Edge opacity proportional to transaction amount.

    Args:
        G (nx.DiGraph): Transaction graph.
        labels (dict or pd.Series): {node_id: 0 or 1} labels.
        node_to_idx (dict, optional): Node to index mapping.
        save_path (str, optional): Path to save the plot.
    """
    logger.info("Generating fraud graph visualization...")

    # Use a subgraph for readability (top 120 nodes by degree)
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:120]
    subG = G.subgraph(top_nodes).copy()

    # Node colors: red for fraud, blue for normal
    node_colors = []
    for node in subG.nodes():
        label = labels.get(node, 0) if isinstance(labels, dict) else labels.get(node, 0)
        node_colors.append("#e74c3c" if label == 1 else "#3498db")

    # Edge transparency proportional to amount
    edge_amounts = []
    for u, v, d in subG.edges(data=True):
        edge_amounts.append(d.get("amount", 100))

    if edge_amounts:
        max_amount = max(edge_amounts)
        edge_alphas = [min(0.8, 0.1 + 0.7 * (a / max_amount)) for a in edge_amounts]
    else:
        edge_alphas = [0.3]

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(subG, seed=config.RANDOM_SEED, k=0.3, iterations=50)

    # Draw edges with varying opacity
    for (u, v), alpha in zip(subG.edges(), edge_alphas):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->", color="#888888",
                alpha=alpha, connectionstyle="arc3,rad=0.1",
                lw=0.5
            )
        )

    # Draw nodes
    node_sizes = [60 + degrees.get(n, 0) * 10 for n in subG.nodes()]
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        edgecolors="#333333",
        linewidths=0.5,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="#333", label="Fraud"),
        Patch(facecolor="#3498db", edgecolor="#333", label="Normal"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=13,
              framealpha=0.8, edgecolor="#ccc")

    ax.set_title(
        "Transaction Graph — Fraud Detection Results\n(🔴 Fraud  🔵 Normal  |  Edge opacity ∝ amount)",
        fontsize=15, fontweight="bold", pad=20
    )
    ax.axis("off")

    save_path = save_path or os.path.join(config.PLOTS_DIR, "fraud_graph.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", save_path)


def plot_feature_distributions(features_df, labels, save_path=None):
    """
    Plot histograms of key features, split by fraud vs. normal.

    Args:
        features_df (pd.DataFrame): Node features with 'node_id' column.
        labels (dict or pd.Series): {node_id: 0 or 1}.
        save_path (str, optional): Path to save the plot.
    """
    logger.info("Generating feature distribution histograms...")

    df = features_df.copy()
    if isinstance(labels, dict):
        df["is_fraud"] = df["node_id"].map(labels).fillna(0).astype(int)
    else:
        df["is_fraud"] = df["node_id"].map(labels).fillna(0).astype(int)

    feature_specs = [
        ("degree", "Degree", "total_degree"),
        ("in_degree", "In Degree", None),
        ("out_degree", "Out Degree", None),
        ("clustering", "Clustering", "clustering_coefficient"),
        ("pagerank", "PageRank", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (canonical_col, label, legacy_col) in enumerate(feature_specs):
        ax = axes[i]

        source_col = canonical_col
        if source_col not in df.columns and legacy_col is not None and legacy_col in df.columns:
            source_col = legacy_col
        if source_col not in df.columns:
            ax.set_title(label + " (missing)", fontsize=13, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        fraud_data = df[df["is_fraud"] == 1][source_col]
        normal_data = df[df["is_fraud"] == 0][source_col]

        ax.hist(normal_data, bins=25, alpha=0.6, color="#3498db", label="Normal", density=True)
        ax.hist(fraud_data, bins=25, alpha=0.6, color="#e74c3c", label="Fraud", density=True)

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot (2x3 grid for 5 features).
    axes[-1].axis("off")

    fig.suptitle("Feature Distributions: Fraud vs. Normal Accounts",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()

    save_path = save_path or os.path.join(config.PLOTS_DIR, "feature_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", save_path)


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Model name for the title.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_title("Confusion Matrix — {}".format(model_name),
                 fontsize=14, fontweight="bold", pad=15)

    tick_labels = ["Normal", "Fraud"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_yticklabels(tick_labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("  Saved: %s", save_path)
    else:
        plt.close(fig)

    return cm


def generate_final_report(comparison_df, save_path=None):
    """
    Save the final metrics comparison table as a CSV.

    Args:
        comparison_df (pd.DataFrame): Model comparison table.
        save_path (str, optional): Path to save the CSV.
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "final_metrics.csv")
    comparison_df.to_csv(save_path, index=False)
    logger.info("  Final metrics saved to %s", save_path)
    return save_path


def export_node_predictions(data, gnn_preds, gnn_probs, node_to_idx=None, save_path=None):
    """
    Export node-level predictions for dashboard consumption.

    Required columns:
      - node_id (original dataset IDs)
      - true_label
      - predicted_label
      - predicted_probability (fraud class probability)
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "node_predictions.csv")

    y_true = data.y.detach().cpu().numpy().astype(int)
    y_pred = gnn_preds.detach().cpu().numpy().astype(int)
    y_prob = gnn_probs.detach().cpu().numpy()[:, 1].astype(float)
    num_nodes = len(y_true)

    if node_to_idx is not None:
        idx_to_node = {int(idx): str(node_id) for node_id, idx in node_to_idx.items()}
        node_ids = [idx_to_node.get(i, str(i)) for i in range(num_nodes)]
    else:
        logger.warning("No node_to_idx mapping provided; exporting fallback index-based node IDs")
        node_ids = [str(i) for i in range(num_nodes)]

    pred_df = pd.DataFrame(
        {
            "node_id": node_ids,
            "true_label": y_true,
            "predicted_label": y_pred,
            "predicted_probability": y_prob,
        }
    )

    # Backward-compatible alias for existing dashboard adapters.
    pred_df["fraud_probability"] = pred_df["predicted_probability"]

    pred_df.to_csv(save_path, index=False)
    logger.info("  Node predictions exported to %s", save_path)
    return save_path


def run_full_evaluation(G, features_df, labels_df, data, model, history,
                        account_labels, device="cpu", node_to_idx=None):
    """
    Run the complete evaluation pipeline with all visualizations.

    Args:
        G (nx.DiGraph): Transaction graph.
        features_df (pd.DataFrame): Node features.
        labels_df (pd.DataFrame): Heuristic labels.
        data: PyG Data object.
        model: Trained GCN model.
        history (dict): Training history.
        account_labels (dict): Ground truth labels.
        device (str): 'cuda' or 'cpu'.
        node_to_idx (dict | None): Optional node-to-index mapping from build_pyg_data.

    Returns:
        pd.DataFrame: Final comparison table.
    """
    import torch
    from src.train import get_gnn_predictions
    from src.hybrid import run_hybrid_comparison

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8 — FULL EVALUATION & VISUALIZATION")
    logger.info("=" * 70)

    config.ensure_dirs()

    # 1. Loss curve
    plot_loss_curve(history)

    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested evaluation device '%s' but CUDA is unavailable; using CPU", requested_device)
        eval_device = torch.device("cpu")
    else:
        eval_device = torch.device(requested_device)

    # Keep model/data on the same device for evaluation to avoid silent CPU execution.
    model = model.to(eval_device)
    data_eval = data.to(eval_device)
    logger.info("Evaluation device verification:")
    logger.info("  - Using torch device: %s", eval_device)
    logger.info("  - torch.cuda.is_available(): %s", torch.cuda.is_available())
    logger.info("  - model parameter device: %s", next(model.parameters()).device)
    logger.info("  - data.x device: %s", data_eval.x.device)
    logger.info("  - data.edge_index device: %s", data_eval.edge_index.device)

    # 2. Get GNN predictions for ROC curves
    gnn_preds, gnn_probs = get_gnn_predictions(model, data_eval, str(eval_device))

    # Export dashboard predictions early so file is always available.
    export_node_predictions(data_eval, gnn_preds, gnn_probs, node_to_idx=node_to_idx)

    # 3. Run hybrid comparison
    comparison, fusion_results, best_alpha = run_hybrid_comparison(
        G, features_df, labels_df, data_eval, model, str(eval_device)
    )

    # 4. ROC curves (strictly labeled test nodes only).
    test_mask_cpu = data_eval.test_mask.detach().cpu()
    labeled_mask_cpu = data_eval.y.detach().cpu() != -1
    eval_mask_cpu = test_mask_cpu & labeled_mask_cpu
    y_true_test = data_eval.y.detach().cpu()[eval_mask_cpu].numpy()
    if y_true_test.size == 0:
        logger.warning("No labeled nodes available in evaluation test split; skipping ROC and confusion plots")
        generate_final_report(comparison)
        return comparison

    # Heuristic scores for test nodes
    labels_work = labels_df.copy()
    labels_work["node_id"] = labels_work["node_id"].astype(str)
    fraud_scores = labels_work.set_index("node_id")["fraud_score"]
    nodes = sorted(G.nodes())
    graph_node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    h_scores = np.zeros(len(nodes), dtype=np.float32)
    for node in nodes:
        h_scores[graph_node_to_idx[node]] = fraud_scores.get(str(node), 0.0)
    h_scores_test = h_scores[eval_mask_cpu.numpy()]

    # Normalize heuristic scores
    h_min, h_max = h_scores_test.min(), h_scores_test.max()
    if h_max > h_min:
        h_scores_norm = (h_scores_test - h_min) / (h_max - h_min)
    else:
        h_scores_norm = h_scores_test

    gnn_probs_test = gnn_probs[eval_mask_cpu][:, 1].numpy()

    # Late fusion scores for best alpha
    fusion_scores_test = best_alpha * gnn_probs_test + (1 - best_alpha) * h_scores_norm

    roc_data = {
        "Heuristic": (y_true_test, h_scores_norm),
        "GNN": (y_true_test, gnn_probs_test),
        "Hybrid (Late Fusion, α={:.1f})".format(best_alpha): (y_true_test, fusion_scores_test),
    }
    plot_roc_curves(roc_data)

    # 5. Fraud graph visualization (using ground truth labels)
    plot_fraud_graph(G, account_labels)

    # 6. Feature distributions (using ground truth labels)
    plot_feature_distributions(features_df, account_labels)

    # 7. Confusion matrices
    gnn_pred_test = gnn_preds[eval_mask_cpu].numpy()
    plot_confusion_matrix(y_true_test, gnn_pred_test, "GNN",
                          os.path.join(config.PLOTS_DIR, "confusion_matrix_gnn.png"))

    # 8. Save final report
    generate_final_report(comparison)

    logger.info("\n" + "=" * 70)
    logger.info("✅ EVALUATION COMPLETE — All plots and results saved!")
    logger.info("  Plots: %s", config.PLOTS_DIR)
    logger.info("  Results: %s", config.RESULTS_DIR)
    logger.info("=" * 70)

    return comparison


if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, set_seeds, build_pyg_data
    from src.features import compute_all_features
    from src.heuristics import compute_fraud_scores, generate_heuristic_labels
    from src.train import train_model

    set_seeds()
    config.ensure_dirs()

    # Full pipeline
    df, account_labels = load_dataset()
    G = build_graph(df)
    features_df = compute_all_features(G)
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)
    labels_series = labels_df.set_index("node_id")["heuristic_label"]
    data, scaler, node_to_idx = build_pyg_data(G, features_df, labels_series)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_model(data, device)

    # Run full evaluation
    comparison = run_full_evaluation(
        G, features_df, labels_df, data, model, history, account_labels, device
    )

    logger.info("\n✅ Phase 8 standalone test complete!")
