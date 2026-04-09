"""
main.py — Pipeline Orchestrator

Runs the complete fraud detection pipeline end-to-end:
  Phase 2: Data loading & graph construction
  Phase 3: Feature engineering
  Phase 4: Heuristic fraud detection
  Phase 5: PyG data preparation
  Phase 6: GNN training
  Phase 7: Hybrid model evaluation
  Phase 8: Final evaluation & visualization

Usage:
    python main.py
"""

import sys
import torch

import config
from src.data_loader import load_dataset, build_graph, set_seeds, build_pyg_data
from src.features import compute_all_features
from src.heuristics import compute_fraud_scores, generate_heuristic_labels
from src.train import train_model
from src.evaluate import run_full_evaluation

logger = config.setup_logging(__name__)


def main():
    """Run the entire fraud detection pipeline."""
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION IN TRANSACTION GRAPHS (END-TO-END)")
    logger.info("=" * 60)

    # Phase 1: Configuration & Setup
    config.ensure_dirs()
    set_seeds()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running on device: %s", device)

    # Phase 2: Data Loading & Graph Construction
    logger.info("\n>>> PHASE 2: Data Loading & Graph Construction")
    df, account_labels = load_dataset()
    G = build_graph(df)

    # Phase 3: Feature Engineering (DSA Core)
    logger.info("\n>>> PHASE 3: Feature Engineering")
    features_df = compute_all_features(G)

    # Phase 4: Heuristic Fraud Detection
    logger.info("\n>>> PHASE 4: Rule-based Heuristics")
    scored_df = compute_fraud_scores(features_df)
    labels_df = generate_heuristic_labels(scored_df)
    
    # Extract labels series for downstream
    labels_series = labels_df.set_index("node_id")["heuristic_label"]

    # Phase 5: PyTorch Geometric Data Prep
    logger.info("\n>>> PHASE 5: PyG Data Preparation")
    data, scaler, node_to_idx = build_pyg_data(G, features_df, labels_series)

    # Phase 6: GNN Model Training
    logger.info("\n>>> PHASE 6: GNN Model Training")
    model, history = train_model(data, device)

    # Phase 7 & 8: Hybrid Model and Full Evaluation
    logger.info("\n>>> PHASE 7 & 8: Evaluation & Hybrid Model")
    comparison_df = run_full_evaluation(
        G, features_df, labels_df, data, model, history, account_labels, device
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info("View outputs in:")
    logger.info("  - %s (Data & Labels)", config.PROCESSED_DATA_DIR)
    logger.info("  - %s (Plots & Visualizations)", config.PLOTS_DIR)
    logger.info("  - %s (Metrics & Results)", config.RESULTS_DIR)
    logger.info("  - %s (Trained Models)", config.MODELS_DIR)


if __name__ == "__main__":
    main()
