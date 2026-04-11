"""config.py - central settings for the fraud detection project."""

import os
import logging

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Model directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Specific file paths
RAW_TRANSACTIONS_PATH = os.path.join(RAW_DATA_DIR, "transactions.csv")
ACCOUNT_GROUND_TRUTH_PATH = os.path.join(RAW_DATA_DIR, "account_ground_truth.csv")
NODE_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "node_features.csv")
LABELS_PATH = os.path.join(PROCESSED_DATA_DIR, "labels.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_gcn.pt")

# Compatibility aliases used by audit scripts
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

# =============================================================================
# Random Seeds (for reproducibility)
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Dataset Generation Parameters
# =============================================================================
NUM_ACCOUNTS = 500          # Number of nodes (accounts)
NUM_TRANSACTIONS = 2000     # Number of edges (transactions)
FRAUD_RATIO = 0.10          # 10% of accounts are fraudulent (ground truth)

# Transaction amount range
MIN_TRANSACTION_AMOUNT = 10.0
MAX_TRANSACTION_AMOUNT = 10000.0

# =============================================================================
# Heuristic Fraud Scoring Weights
# =============================================================================
# fraud_score = w1 * norm(degree) + w2 * (1 - norm(clustering)) + w3 * norm(pagerank)
HEURISTIC_WEIGHTS = {
    "w1": 0.35,   # Weight for normalized degree (high degree = suspicious)
    "w2": 0.35,   # Weight for (1 - normalized clustering) (low clustering = suspicious)
    "w3": 0.30,   # Weight for normalized PageRank (high centrality = suspicious)
}

# Fraction of top-scoring nodes to label as fraudulent
HEURISTIC_FRAUD_THRESHOLD = 0.15  # Top 15% → fraud = 1

# =============================================================================
# GNN Model Hyperparameters
# =============================================================================
GNN_HIDDEN_DIM = 64         # Hidden layer dimension
GNN_NUM_CLASSES = 2         # Binary classification (normal, fraud)
GNN_DROPOUT = 0.5           # Dropout rate
GNN_NUM_LAYERS = 2          # Number of GCN layers

# Compatibility alias used by some harnesses
HIDDEN_DIM = GNN_HIDDEN_DIM
SEED = RANDOM_SEED

# =============================================================================
# Training Hyperparameters
# =============================================================================
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 200
LOG_INTERVAL = 10           # Log metrics every N epochs
# Leakage-safe split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# Hybrid Model Parameters
# =============================================================================
ALPHA_SWEEP_START = 0.0     # Late fusion alpha sweep range
ALPHA_SWEEP_END = 1.0
ALPHA_SWEEP_STEP = 0.1

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name=None):
    """
    Configure logging with consistent format across all modules.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name or "fraud_detection")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


def ensure_dirs():
    """Create all required directories if they don't exist."""
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_dirs():
    """Short alias used by the main pipeline setup step."""
    ensure_dirs()
