"""
heuristics.py — Rule-Based Fraud Scoring

Implements a transparent, explainable fraud scoring function that combines
normalized graph-structural features with configurable weights.

Scoring intuition:
  - High degree → suspicious (hub account, many transactions)
  - Low clustering → suspicious (broker/mule connecting disparate groups)
  - High PageRank → suspicious (central node receiving many flows)

The scoring function is:
    fraud_score(node) = w1 * normalize(degree)
                                        + w2 * (1 - normalize(clustering))
                    + w3 * normalize(pagerank)

where weights w1, w2, w3 are loaded from config.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config

logger = config.setup_logging(__name__)


def normalize_series(series):
    """
    Min-max normalize a pandas Series to [0, 1] range.

    Args:
        series (pd.Series): Raw feature values.

    Returns:
        pd.Series: Normalized values in [0, 1].
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_fraud_scores(features_df):
    """
    Compute a heuristic fraud score for each node based on graph features.

    The score combines three normalized features with configurable weights:
            - w1 * norm(degree): high degree → suspicious hub
      - w2 * (1 - norm(clustering)): low clustering → suspicious broker/mule
      - w3 * norm(pagerank): high PageRank → suspicious central node

    Args:
        features_df (pd.DataFrame): Node features with columns:
            node_id, degree, clustering, pagerank

    Returns:
        pd.DataFrame: Features augmented with 'fraud_score' column,
            sorted by fraud_score descending.
    """
    weights = config.HEURISTIC_WEIGHTS
    w1, w2, w3 = weights["w1"], weights["w2"], weights["w3"]

    logger.info("Computing heuristic fraud scores...")
    logger.info("  Weights: w1=%.2f (degree), w2=%.2f (1-clustering), w3=%.2f (pagerank)",
                w1, w2, w3)

    df = features_df.copy()

    # Handle legacy feature exports while preferring canonical names.
    degree_col = "degree" if "degree" in df.columns else "total_degree"
    clustering_col = "clustering" if "clustering" in df.columns else "clustering_coefficient"

    # Normalize features
    norm_degree = normalize_series(df[degree_col])
    norm_clustering = normalize_series(df[clustering_col])
    norm_pagerank = normalize_series(df["pagerank"])

    # Compute fraud score
    # High degree = suspicious → multiply by w1
    # Low clustering = suspicious → use (1 - normalized) → multiply by w2
    # High PageRank = suspicious → multiply by w3
    df["fraud_score"] = (
        w1 * norm_degree
        + w2 * (1.0 - norm_clustering)
        + w3 * norm_pagerank
    )

    # Sort by fraud score (descending)
    df = df.sort_values("fraud_score", ascending=False).reset_index(drop=True)

    logger.info("  Fraud score range: [%.4f, %.4f]", df["fraud_score"].min(), df["fraud_score"].max())
    logger.info("  Mean fraud score: %.4f", df["fraud_score"].mean())

    return df


def generate_heuristic_labels(scored_df, threshold=None):
    """
    Generate binary fraud labels based on heuristic fraud scores.

    Nodes with fraud scores in the top `threshold` fraction are labeled
    as fraudulent (y=1), the rest as normal (y=0).

    Args:
        scored_df (pd.DataFrame): DataFrame with 'node_id' and 'fraud_score' columns.
        threshold (float, optional): Fraction of top-scoring nodes to label as fraud.
            Defaults to config.HEURISTIC_FRAUD_THRESHOLD (0.15).

    Returns:
        pd.DataFrame: DataFrame with columns: node_id, fraud_score, heuristic_label
    """
    if threshold is None:
        threshold = config.HEURISTIC_FRAUD_THRESHOLD

    logger.info("Generating heuristic labels (top %.0f%% → fraud)...", threshold * 100)

    df = scored_df.copy()
    n_fraud = int(len(df) * threshold)

    # Top n_fraud scores → label 1, rest → label 0
    df = df.sort_values("fraud_score", ascending=False).reset_index(drop=True)
    df["heuristic_label"] = 0
    df.loc[:n_fraud - 1, "heuristic_label"] = 1

    n_labeled_fraud = df["heuristic_label"].sum()
    logger.info("  Labeled %d nodes as fraud, %d as normal",
                n_labeled_fraud, len(df) - n_labeled_fraud)

    # Save labels
    labels_df = df[["node_id", "fraud_score", "heuristic_label"]].copy()
    config.ensure_dirs()
    labels_df.to_csv(config.LABELS_PATH, index=False)
    logger.info("  Saved labels to %s", config.LABELS_PATH)

    return labels_df


def evaluate_heuristic(labels_df, ground_truth_labels):
    """
    Evaluate heuristic fraud detection against ground truth.

    Computes baseline classification metrics to establish a performance
    floor that the GNN must beat.

    Args:
        labels_df (pd.DataFrame): Heuristic labels with 'node_id', 'heuristic_label'.
        ground_truth_labels (dict): {node_id: 0 or 1} ground truth fraud labels.

    Returns:
        dict: Metrics dictionary with accuracy, precision, recall, f1.
    """
    logger.info("Evaluating heuristic labels against ground truth...")

    # Align labels, ignoring unlabeled ground-truth nodes (label=-1).
    y_true = []
    y_pred = []
    for _, row in labels_df.iterrows():
        node_id = str(row["node_id"])
        if node_id in ground_truth_labels and int(ground_truth_labels[node_id]) != -1:
            y_true.append(int(ground_truth_labels[node_id]))
            y_pred.append(int(row["heuristic_label"]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.size == 0:
        logger.warning("  No labeled nodes available for heuristic evaluation")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    logger.info("  Heuristic Baseline Metrics:")
    logger.info("  %s", "-" * 40)
    for metric_name, value in metrics.items():
        logger.info("    %-12s: %.4f", metric_name.capitalize(), value)
    logger.info("  %s", "-" * 40)

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_dataset, build_graph, set_seeds
    from src.features import compute_all_features

    set_seeds()
    config.ensure_dirs()

    # Load data and build graph
    df, account_labels = load_dataset()
    G = build_graph(df)

    # Compute features
    features_df = compute_all_features(G)

    # Compute fraud scores and labels
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)

    # Evaluate against ground truth
    metrics = evaluate_heuristic(labels_df, account_labels)

    logger.info("\n✅ Heuristic fraud detection complete!")
