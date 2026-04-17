"""
hybrid.py — Heuristic + GNN Combination Logic

Implements two hybrid strategies for combining heuristic fraud scores
with GNN predictions:

Strategy A — Feature Augmentation:
    Adds the heuristic fraud score as an additional input feature to the GNN.
    The GNN then learns to use both structural features AND the heuristic score.

Strategy B — Late Fusion (Ensemble):
    Combines GNN probability with heuristic score using a weighted average:
        final_score = α × GNN_probability + (1 - α) × heuristic_score
    Sweeps α from 0 to 1 to find the optimal blending weight.

Both strategies are compared against the standalone heuristic and GNN baselines.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import config
from src.gnn_model import FraudGCN
from src.data_loader import set_seeds

logger = config.setup_logging(__name__)


def _safe_roc_auc(y_true, y_prob):
    """Compute ROC-AUC safely for potentially single-class slices."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true, y_prob):
    """Compute PR-AUC safely for potentially single-class slices."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float(y_true.mean()) if len(y_true) else 0.0
    return float(average_precision_score(y_true, y_prob))


def _verify_labels_are_ground_truth(data):
    """Guardrail against evaluating against heuristic-derived labels."""
    y = data.y.detach().cpu().numpy()
    y = y[y != -1]
    fraud_ratio = float(y.mean()) if len(y) else 0.0
    assert 0.03 <= fraud_ratio <= 0.30, (
        "LABEL LEAKAGE RISK: Fraud ratio {:.2%} is suspicious. "
        "Verify data.y comes from account_ground_truth.csv".format(fraud_ratio)
    )
    return True


def strategy_a_feature_augmentation(G, features_df, labels_series, fraud_scores, device="cpu"):
    """
    Strategy A: Add heuristic fraud score as an extra input feature to the GNN.

    Rationale:
        The GNN receives the heuristic score as prior knowledge. If the heuristic
        is informative, the GNN can learn to amplify it. If not, it can learn to
        downweight it. This is strictly more powerful than either alone.

    Args:
        G (nx.DiGraph): Transaction graph.
        features_df (pd.DataFrame): Node features with 'node_id' column.
        labels_series (pd.Series): Node labels indexed by node_id.
        fraud_scores (pd.Series): Heuristic fraud scores indexed by node_id.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (predictions, probabilities, metrics_dict, model, history)
    """
    from torch_geometric.data import Data
    from sklearn.model_selection import train_test_split
    from src.train import train_model, get_gnn_predictions

    logger.info("=" * 60)
    logger.info("Strategy A: Feature Augmentation (heuristic score as input feature)")
    logger.info("=" * 60)

    set_seeds()

    # Get sorted node list
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # Build augmented feature matrix (original features + heuristic score)
    feature_cols = [col for col in features_df.columns if col != "node_id"]
    num_original_features = len(feature_cols)

    feature_matrix = np.zeros((num_nodes, num_original_features + 1), dtype=np.float32)

    features_work = features_df.copy()
    features_work["node_id"] = features_work["node_id"].astype(str)

    for _, row in features_work.iterrows():
        node_id = str(row["node_id"])
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            feature_matrix[idx, :num_original_features] = [row[col] for col in feature_cols]
            # Append heuristic fraud score as the last feature
            feature_matrix[idx, -1] = fraud_scores.get(node_id, 0.0)

    # Build edge index
    edge_list = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Labels
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for node_id, label in labels_series.items():
        node_key = str(node_id)
        if node_key in node_to_idx:
            y[node_to_idx[node_key]] = int(label)

    labeled_mask = y != -1
    labeled_idx = np.where(labeled_mask.numpy())[0]
    if len(labeled_idx) < 3:
        raise ValueError("Not enough labeled nodes to run Strategy A split")

    # Match reference-style split: 70/30 then 80/20 on train.
    indices = labeled_idx
    stratify_targets = y.numpy()[indices]
    stratify_targets = stratify_targets if len(np.unique(stratify_targets)) > 1 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=stratify_targets,
        random_state=config.RANDOM_SEED,
    )

    train_labels = y.numpy()[train_idx]
    stratify_train = train_labels if len(np.unique(train_labels)) > 1 else None
    tr_rel_idx, val_rel_idx = train_test_split(
        np.arange(len(train_idx)),
        test_size=0.2,
        stratify=stratify_train,
        random_state=config.RANDOM_SEED,
    )
    val_idx = train_idx[val_rel_idx]
    train_idx = train_idx[tr_rel_idx]

    # Normalize all features (including heuristic score) with train-only fit.
    scaler = StandardScaler()
    train_mask_np = np.zeros(num_nodes, dtype=bool)
    train_mask_np[train_idx] = True
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    feature_matrix[train_mask_np] = scaler.fit_transform(feature_matrix[train_mask_np])
    feature_matrix[~train_mask_np] = scaler.transform(feature_matrix[~train_mask_np])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    x = torch.tensor(feature_matrix, dtype=torch.float32)
    logger.info("  Augmented feature matrix: %s (original + heuristic_score)", x.shape)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # Train augmented GNN with memory-safe minibatching.
    model, history = train_model(data, device=device)
    preds, probs = get_gnn_predictions(model, data, device=device)

    # Compute metrics on test set
    test_eval_mask = (data.test_mask & (data.y != -1)).cpu().numpy()
    y_true = y.cpu().numpy()[test_eval_mask]
    y_pred = preds.cpu().numpy()[test_eval_mask]
    y_prob = probs.cpu().numpy()[test_eval_mask, 1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
    }

    logger.info("  Strategy A Results:")
    for k, v in metrics.items():
        logger.info("    %-12s: %.4f", k.capitalize(), v)

    return preds, probs, metrics, model, history


def strategy_b_late_fusion(gnn_probs, heuristic_scores, labels, val_mask, test_mask):
    """
    Strategy B: Late Fusion — weighted average of GNN and heuristic scores.

    Sweeps alpha from 0 to 1 and, for each alpha, sweeps decision threshold
    on validation data to select the best operating point:
        final_score = α × GNN_fraud_prob + (1 - α) × heuristic_score
        If final_score > threshold → predict fraud

    Args:
        gnn_probs (torch.Tensor): GNN probabilities [N, 2].
        heuristic_scores (np.ndarray or torch.Tensor): Heuristic fraud scores [N].
        labels (torch.Tensor): Ground truth labels [N].
        val_mask (torch.Tensor): Boolean mask for validation nodes.
        test_mask (torch.Tensor): Boolean mask for test nodes.

    Returns:
        tuple: (best_alpha, best_threshold, best_test_metrics, all_results_df)
    """
    logger.info("=" * 60)
    logger.info("Strategy B: Late Fusion (sweep α from 0.0 to 1.0)")
    logger.info("=" * 60)

    labels_np = labels.detach().cpu().numpy()
    labeled_mask_np = labels_np != -1
    fraud_ratio = float(labels_np[labeled_mask_np].mean()) if labeled_mask_np.any() else 0.0
    assert 0.03 <= fraud_ratio <= 0.30, (
        "LABEL LEAKAGE RISK: Fraud ratio {:.2%} is suspicious. "
        "Verify labels are true ground truth".format(fraud_ratio)
    )

    if isinstance(heuristic_scores, torch.Tensor):
        h_scores = heuristic_scores.detach().cpu().float().numpy()
    else:
        h_scores = np.array(heuristic_scores, dtype=np.float32)

    # GNN fraud probability (class 1)
    gnn_fraud_prob = gnn_probs[:, 1].detach().cpu().numpy()

    # Normalize heuristic scores to [0, 1]
    h_min, h_max = h_scores.min(), h_scores.max()
    if h_max > h_min:
        h_scores_norm = (h_scores - h_min) / (h_max - h_min)
    else:
        h_scores_norm = h_scores

    val_mask_np = val_mask.detach().cpu().numpy()
    test_mask_np = test_mask.detach().cpu().numpy()
    eval_val_mask_np = val_mask_np & labeled_mask_np
    eval_test_mask_np = test_mask_np & labeled_mask_np

    y_val_true = labels_np[eval_val_mask_np]
    y_test_true = labels_np[eval_test_mask_np]

    if y_val_true.size == 0:
        raise ValueError("No labeled validation nodes available for late fusion selection")
    if y_test_true.size == 0:
        raise ValueError("No labeled test nodes available for late fusion evaluation")

    results = []

    alphas = np.arange(
        config.ALPHA_SWEEP_START,
        config.ALPHA_SWEEP_END + config.ALPHA_SWEEP_STEP / 2,
        config.ALPHA_SWEEP_STEP
    )
    threshold_grid = np.arange(0.10, 0.95 + 0.001, 0.05)

    for alpha in alphas:
        # Blend scores
        final_score = alpha * gnn_fraud_prob + (1 - alpha) * h_scores_norm
        val_score = final_score[eval_val_mask_np]
        test_score = final_score[eval_test_mask_np]

        best_thr = 0.50
        best_val_f1 = -1.0
        best_val_precision = 0.0
        best_val_recall = 0.0

        for thr in threshold_grid:
            y_val_pred = (val_score > thr).astype(int)
            val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)
            val_precision = precision_score(y_val_true, y_val_pred, zero_division=0)
            val_recall = recall_score(y_val_true, y_val_pred, zero_division=0)
            if (val_f1 > best_val_f1) or (
                np.isclose(val_f1, best_val_f1) and val_precision > best_val_precision
            ):
                best_val_f1 = val_f1
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_thr = float(thr)

        y_test_pred = (test_score > best_thr).astype(int)

        metrics = {
            "alpha": round(float(alpha), 2),
            "threshold": round(float(best_thr), 2),
            "val_precision": best_val_precision,
            "val_recall": best_val_recall,
            "val_f1": best_val_f1,
            "val_roc_auc": _safe_roc_auc(y_val_true, val_score),
            "val_pr_auc": _safe_pr_auc(y_val_true, val_score),
            "accuracy": accuracy_score(y_test_true, y_test_pred),
            "precision": precision_score(y_test_true, y_test_pred, zero_division=0),
            "recall": recall_score(y_test_true, y_test_pred, zero_division=0),
            "f1": f1_score(y_test_true, y_test_pred, zero_division=0),
            "roc_auc": _safe_roc_auc(y_test_true, test_score),
            "pr_auc": _safe_pr_auc(y_test_true, test_score),
        }
        results.append(metrics)
        logger.info(
            "  α=%.2f | thr*=%.2f | Val F1: %.4f | Test Acc: %.4f | Test P: %.4f | Test R: %.4f | Test F1: %.4f",
            alpha,
            best_thr,
            best_val_f1,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

    results_df = pd.DataFrame(results)

    # Select alpha + threshold by validation objective, then report test metrics.
    best_idx = results_df.sort_values(["val_f1", "val_pr_auc"], ascending=False).index[0]
    best_alpha = results_df.loc[best_idx, "alpha"]
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_metrics = results_df.loc[best_idx].to_dict()

    logger.info("-" * 60)
    logger.info(
        "  Selected α = %.2f | threshold = %.2f | Val F1 = %.4f | Test F1 = %.4f",
        best_alpha,
        best_threshold,
        best_metrics["val_f1"],
        best_metrics["f1"],
    )

    return best_alpha, best_threshold, best_metrics, results_df


def run_hybrid_comparison(G, features_df, labels_df, data, model, device="cpu"):
    """
    Run both hybrid strategies and produce a comparison table.

    Args:
        G (nx.DiGraph): Transaction graph.
        features_df (pd.DataFrame): Node features.
        labels_df (pd.DataFrame): Labels with 'node_id', 'fraud_score', 'heuristic_label'.
        data (torch_geometric.data.Data): PyG data object.
        model (FraudGCN): Trained GNN model.
        device (str): 'cuda' or 'cpu'.

    Returns:
        pd.DataFrame: Comparison table with metrics for all models.
    """
    from src.train import get_gnn_predictions

    logger.info("\n" + "=" * 70)
    logger.info("HYBRID MODEL COMPARISON")
    logger.info("=" * 70)

    _verify_labels_are_ground_truth(data)

    true_label_series = pd.Series(
        data.y.cpu().numpy(),
        index=sorted(G.nodes()),
    )
    heuristic_series = labels_df.set_index("node_id")["heuristic_label"]
    fraud_scores = labels_df.set_index("node_id")["fraud_score"]

    # --- Heuristic Only ---
    labels_np = data.y.cpu().numpy()
    test_mask_np = data.test_mask.cpu().numpy()
    eval_mask_np = test_mask_np & (labels_np != -1)
    if eval_mask_np.sum() == 0:
        raise ValueError("No labeled nodes in test split for hybrid comparison")
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    y_true_test = labels_np[eval_mask_np]

    # Build heuristic scores aligned with node ordering
    h_scores_aligned = np.zeros(len(nodes), dtype=np.float32)
    for node in nodes:
        idx = node_to_idx[node]
        h_scores_aligned[idx] = fraud_scores.get(node, 0.0)

    # Heuristic predictions on test set
    h_preds_test = []
    for node in nodes:
        if node in node_to_idx:
            idx = node_to_idx[node]
            if eval_mask_np[idx]:
                h_preds_test.append(int(heuristic_series.get(node, 0)))
    h_preds_test = np.array(h_preds_test)

    heuristic_metrics = {
        "accuracy": accuracy_score(y_true_test, h_preds_test),
        "precision": precision_score(y_true_test, h_preds_test, zero_division=0),
        "recall": recall_score(y_true_test, h_preds_test, zero_division=0),
        "f1": f1_score(y_true_test, h_preds_test, zero_division=0),
        "roc_auc": _safe_roc_auc(y_true_test, h_scores_aligned[eval_mask_np]),
        "pr_auc": _safe_pr_auc(y_true_test, h_scores_aligned[eval_mask_np]),
    }

    # --- GNN Only ---
    gnn_preds, gnn_probs = get_gnn_predictions(model, data, device)
    y_pred_gnn = gnn_preds.numpy()[eval_mask_np]
    gnn_probs_test = gnn_probs.numpy()[eval_mask_np, 1]

    gnn_metrics = {
        "accuracy": accuracy_score(y_true_test, y_pred_gnn),
        "precision": precision_score(y_true_test, y_pred_gnn, zero_division=0),
        "recall": recall_score(y_true_test, y_pred_gnn, zero_division=0),
        "f1": f1_score(y_true_test, y_pred_gnn, zero_division=0),
        "roc_auc": _safe_roc_auc(y_true_test, gnn_probs_test),
        "pr_auc": _safe_pr_auc(y_true_test, gnn_probs_test),
    }

    # --- Strategy A: Feature Augmentation ---
    _, _, strategy_a_metrics, _, _ = strategy_a_feature_augmentation(
        G, features_df, true_label_series, fraud_scores.to_dict(), device
    )

    # --- Strategy B: Late Fusion ---
    best_alpha, best_threshold, strategy_b_metrics, fusion_results = strategy_b_late_fusion(
        gnn_probs,
        h_scores_aligned,
        data.y.cpu(),
        data.val_mask.cpu(),
        data.test_mask.cpu(),
    )

    # --- Comparison Table ---
    strategy_b_test_metrics = {
        k: strategy_b_metrics[k]
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    }

    comparison = pd.DataFrame([
        {"Model": "Heuristic Only", **heuristic_metrics},
        {"Model": "GNN Only", **gnn_metrics},
        {"Model": "Hybrid (Strategy A)", **strategy_a_metrics},
        {
            "Model": "Hybrid (Strategy B, α={:.2f}, thr={:.2f})".format(best_alpha, best_threshold),
            **strategy_b_test_metrics,
        },
    ])

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 70)
    for _, row in comparison.iterrows():
        logger.info(
            "  %-35s  Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f",
            row["Model"], row["accuracy"], row["precision"],
            row["recall"], row["f1"], row["roc_auc"], row["pr_auc"]
        )
    logger.info("=" * 70)

    # Save results
    config.ensure_dirs()
    comparison.to_csv(
        config.RESULTS_DIR + "/model_comparison.csv", index=False
    )
    fusion_results.to_csv(
        config.RESULTS_DIR + "/alpha_sweep_results.csv", index=False
    )
    logger.info("  Results saved to outputs/results/")

    return comparison, fusion_results, best_alpha


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, build_pyg_data
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
    data, scaler, node_to_idx = build_pyg_data(G, features_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_model(data, device)

    # Run hybrid comparison
    comparison, fusion_results, best_alpha = run_hybrid_comparison(
        G, features_df, labels_df, data, model, device
    )

    logger.info("\n✅ Hybrid model comparison complete!")
