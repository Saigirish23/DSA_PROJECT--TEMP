"""
data_loader.py — Data Loading & Graph Construction

Responsibilities:
  1. Normalize available raw datasets into a common schema.
  2. Build a directed NetworkX graph from transactions.
  3. Convert graph features + true ground-truth labels into PyG Data.
"""

import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

logger = config.setup_logging(__name__)


REQUIRED_TX_COLUMNS = ["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]


def set_seeds():
    """Set all random seeds for reproducibility."""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)


def generate_enhanced_synthetic_dataset():
    """Generate enhanced synthetic fallback with account-level ground truth labels."""
    set_seeds()
    logger.info("Generating enhanced synthetic fallback dataset...")

    n_accounts = 1000
    n_transactions = 5000
    n_fraud_accounts = int(n_accounts * 0.10)

    account_ids = ["ACC_{:04d}".format(i) for i in range(n_accounts)]
    fraud_accounts = set(np.random.choice(account_ids, n_fraud_accounts, replace=False))

    rows = []
    normal_accounts = list(set(account_ids) - fraud_accounts)
    fraud_accounts_list = list(fraud_accounts)
    
    # 80% of transactions involve fraud accounts (makes their degree very high)
    for i in range(n_transactions):
        if np.random.rand() < 0.8:
            sender = np.random.choice(fraud_accounts_list)
            receiver = np.random.choice(account_ids)
        else:
            sender = np.random.choice(normal_accounts)
            receiver = np.random.choice(normal_accounts)
            
        while receiver == sender:
            receiver = np.random.choice(account_ids)

        is_fraud_tx = (sender in fraud_accounts) or (receiver in fraud_accounts)
        if is_fraud_tx:
            amount = np.random.exponential(scale=5000)
            hour = np.random.choice([1, 2, 3, 23, 0], p=[0.2] * 5)
        else:
            amount = np.random.exponential(scale=500)
            hour = np.random.randint(6, 22)

        rows.append(
            {
                "transaction_id": "TX_{:06d}".format(i),
                "sender_id": sender,
                "receiver_id": receiver,
                "amount": round(float(amount), 2),
                "timestamp": pd.Timestamp("2023-01-01")
                + pd.Timedelta(
                    days=int(np.random.randint(0, 365)),
                    hours=int(hour),
                    minutes=int(np.random.randint(0, 60)),
                ),
                "transaction_type": np.random.choice(
                    ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT"], p=[0.4, 0.3, 0.2, 0.1]
                ),
                "is_fraud": int(is_fraud_tx),
            }
        )

    tx_df = pd.DataFrame(rows)
    tx_df.to_csv(config.RAW_TRANSACTIONS_PATH, index=False)

    gt_df = pd.DataFrame(
        {
            "account_id": account_ids,
            "is_fraud": [1 if a in fraud_accounts else 0 for a in account_ids],
        }
    )
    gt_df.to_csv(config.ACCOUNT_GROUND_TRUTH_PATH, index=False)

    logger.info("  Generated %d transactions and %d accounts", len(tx_df), len(gt_df))
    logger.info("  Fraud account ratio: %.2f%%", gt_df["is_fraud"].mean() * 100)
    return tx_df, gt_df, "Enhanced Synthetic"


def _normalize_common_schema(tx_df, gt_df, source_name):
    """Normalize schema and persist canonical raw files."""
    df = tx_df.copy()

    # Required transaction schema coercion
    rename_map = {
        "nameOrig": "sender_id",
        "nameDest": "receiver_id",
        "step": "timestamp",
    }
    df = df.rename(columns=rename_map)

    if "transaction_id" not in df.columns:
        df["transaction_id"] = ["TX_{:08d}".format(i) for i in range(len(df))]

    if "amount" not in df.columns:
        df["amount"] = 0.0

    # Parse/standardize timestamp
    if np.issubdtype(df["timestamp"].dtype, np.number):
        base_ts = pd.Timestamp("2023-01-01")
        df["timestamp"] = base_ts + pd.to_timedelta(df["timestamp"].astype(int), unit="h")
    else:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        parsed = parsed.fillna(pd.Timestamp("2023-01-01", tz="UTC"))
        df["timestamp"] = parsed.dt.tz_convert(None)

    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = df[REQUIRED_TX_COLUMNS].copy()
    df = df[df["sender_id"] != df["receiver_id"]]
    df = df.drop_duplicates(subset=["sender_id", "receiver_id", "amount", "timestamp"])
    df = df.reset_index(drop=True)

    # Ground truth schema coercion
    gdf = gt_df.copy()
    gdf = gdf.rename(columns={"node_id": "account_id", "class": "is_fraud"})
    gdf["account_id"] = gdf["account_id"].astype(str)
    gdf["is_fraud"] = pd.to_numeric(gdf["is_fraud"], errors="coerce").fillna(0).astype(int)
    gdf["is_fraud"] = (gdf["is_fraud"] > 0).astype(int)
    gdf = gdf[["account_id", "is_fraud"]].drop_duplicates(subset=["account_id"])

    # Ensure every node in transactions has a label
    tx_accounts = set(df["sender_id"]).union(set(df["receiver_id"]))
    known_accounts = set(gdf["account_id"])
    missing = sorted(tx_accounts - known_accounts)
    if missing:
        gdf = pd.concat(
            [gdf, pd.DataFrame({"account_id": missing, "is_fraud": 0})],
            ignore_index=True,
        )

    # Persist canonical files
    config.ensure_dirs()
    df.to_csv(config.RAW_TRANSACTIONS_PATH, index=False)
    gdf.to_csv(config.ACCOUNT_GROUND_TRUTH_PATH, index=False)

    fraud_ratio = float(gdf["is_fraud"].mean()) if len(gdf) else 0.0
    logger.info("Dataset source: %s", source_name)
    logger.info("  transactions rows: %d", len(df))
    logger.info("  account labels rows: %d", len(gdf))
    logger.info("  ground-truth fraud ratio: %.2f%%", fraud_ratio * 100)
    logger.info("  columns: %s", df.columns.tolist())

    return df, dict(zip(gdf["account_id"], gdf["is_fraud"]))


def _adapt_paysim_dataset(paysim_path):
    """Adapt PaySim transactions into canonical transaction + account label schema."""
    df = pd.read_csv(paysim_path)
    required = {"step", "nameOrig", "nameDest", "amount", "isFraud"}
    if not required.issubset(df.columns):
        raise ValueError("PaySim file missing required columns: {}".format(sorted(required - set(df.columns))))

    tx_df = df[["step", "nameOrig", "nameDest", "amount"]].copy()
    tx_df["transaction_id"] = ["PS_{:08d}".format(i) for i in range(len(tx_df))]

    fraud_accounts = set(df.loc[df["isFraud"] == 1, "nameOrig"].astype(str))
    fraud_accounts.update(set(df.loc[df["isFraud"] == 1, "nameDest"].astype(str)))
    all_accounts = set(df["nameOrig"].astype(str)).union(set(df["nameDest"].astype(str)))
    gt_df = pd.DataFrame(
        {
            "account_id": sorted(all_accounts),
            "is_fraud": [1 if a in fraud_accounts else 0 for a in sorted(all_accounts)],
        }
    )
    return tx_df, gt_df


def _adapt_elliptic_dataset(features_path, edges_path, labels_path):
    """Adapt Elliptic files into canonical schema."""
    edges = pd.read_csv(edges_path)
    labels = pd.read_csv(labels_path)

    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Elliptic edges file missing txId1/txId2 columns")

    # Elliptic has no amount/timestamp in edge list, so create deterministic placeholders.
    tx_df = edges[["txId1", "txId2"]].copy()
    tx_df = tx_df.rename(columns={"txId1": "sender_id", "txId2": "receiver_id"})
    tx_df["amount"] = 1.0
    tx_df["timestamp"] = pd.Timestamp("2023-01-01")
    tx_df["transaction_id"] = ["EL_{:08d}".format(i) for i in range(len(tx_df))]

    # classes: unknown=2, licit=1, illicit=0 in some versions; map illicit->1 (fraud)
    labels = labels.rename(columns={"txId": "account_id", "class": "is_fraud"})
    labels["account_id"] = labels["account_id"].astype(str)
    labels["is_fraud"] = pd.to_numeric(labels["is_fraud"], errors="coerce").fillna(0).astype(int)
    labels["is_fraud"] = labels["is_fraud"].map({0: 1, 1: 0, 2: 0}).fillna(0).astype(int)
    gt_df = labels[["account_id", "is_fraud"]].copy()

    return tx_df, gt_df


def normalize_dataset_sources():
    """Select available source and normalize to canonical transactions + ground truth files."""
    config.ensure_dirs()

    # Prefer externally sourced files if present.
    paysim_path = os.path.join(config.RAW_DATA_DIR, "paysim_transactions.csv")
    elliptic_features_path = os.path.join(config.RAW_DATA_DIR, "elliptic_features.csv")
    elliptic_edges_path = os.path.join(config.RAW_DATA_DIR, "elliptic_edges.csv")
    elliptic_labels_path = os.path.join(config.RAW_DATA_DIR, "elliptic_labels.csv")

    if os.path.exists(paysim_path):
        tx_df, gt_df = _adapt_paysim_dataset(paysim_path)
        return _normalize_common_schema(tx_df, gt_df, "PaySim")

    if os.path.exists(elliptic_features_path) and os.path.exists(elliptic_edges_path) and os.path.exists(elliptic_labels_path):
        tx_df, gt_df = _adapt_elliptic_dataset(elliptic_features_path, elliptic_edges_path, elliptic_labels_path)
        return _normalize_common_schema(tx_df, gt_df, "Elliptic")

    # If canonical files already exist, normalize them in place.
    if os.path.exists(config.RAW_TRANSACTIONS_PATH) and os.path.exists(config.ACCOUNT_GROUND_TRUTH_PATH):
        tx_df = pd.read_csv(config.RAW_TRANSACTIONS_PATH)
        gt_df = pd.read_csv(config.ACCOUNT_GROUND_TRUTH_PATH)
        return _normalize_common_schema(tx_df, gt_df, "Existing Canonical")

    tx_df, gt_df, source = generate_enhanced_synthetic_dataset()
    return _normalize_common_schema(tx_df, gt_df, source)


def load_data():
    """
    Load normalized transactions and true account labels.

    Returns:
        tuple: (transactions_df, account_labels_dict)
    """
    tx_df, account_labels = normalize_dataset_sources()
    return tx_df, account_labels


def load_true_labels(G):
    """
    Load TRUE labels from account_ground_truth.csv.
    These must NEVER be generated from heuristics.
    """
    gt_path = config.ACCOUNT_GROUND_TRUTH_PATH
    gt_df = pd.read_csv(gt_path)
    gt_df["account_id"] = gt_df["account_id"].astype(str)
    gt_df["is_fraud"] = pd.to_numeric(gt_df["is_fraud"], errors="coerce").fillna(0).astype(int)
    gt_map = dict(zip(gt_df["account_id"], gt_df["is_fraud"]))
    nodes = sorted(G.nodes())
    return {str(n): int(gt_map.get(str(n), 0)) for n in nodes}


def build_graph(df):
    """Build directed transaction graph from normalized transactions DataFrame."""
    logger.info("Building directed transaction graph...")

    # directed because each transaction has a sender and a receiver
    G = nx.DiGraph()

    work = df.copy()
    work["sender_id"] = work["sender_id"].astype(str)
    work["receiver_id"] = work["receiver_id"].astype(str)
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce").fillna(pd.Timestamp("2023-01-01"))

    for _, row in work.iterrows():
        sender = row["sender_id"]
        receiver = row["receiver_id"]
        amount = float(row["amount"])
        timestamp = int(pd.Timestamp(row["timestamp"]).timestamp())

        if sender == receiver:
            continue

        if G.has_edge(sender, receiver):
            G[sender][receiver]["amount"] += amount
            G[sender][receiver]["timestamp"] = max(G[sender][receiver]["timestamp"], timestamp)
            G[sender][receiver]["count"] += 1
        else:
            G.add_edge(sender, receiver, amount=amount, timestamp=timestamp, count=1)

    logger.info("  Graph Statistics:")
    logger.info("    Nodes: %d", G.number_of_nodes())
    logger.info("    Edges: %d", G.number_of_edges())
    logger.info("    Density: %.6f", nx.density(G) if G.number_of_nodes() > 1 else 0.0)

    undirected = G.to_undirected()
    num_components = nx.number_connected_components(undirected) if G.number_of_nodes() else 0
    logger.info("    Connected components (undirected): %d", num_components)

    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        logger.warning("    Found %d self-loops, removing", len(self_loops))
        G.remove_edges_from(self_loops)
    else:
        logger.info("    No self-loops detected")

    isolated = list(nx.isolates(G))
    if isolated:
        logger.warning("    Found %d isolated nodes", len(isolated))
    else:
        logger.info("    No isolated nodes")

    return G


def build_pyg_data(G, features_df, labels_series=None):
    """
    Convert graph + features into a leakage-safe PyG Data object.

    Uses account_ground_truth.csv as the ONLY source for y labels.
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, remove_self_loops

    logger.info("Building PyTorch Geometric Data object...")
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    feature_cols = [col for col in features_df.columns if col != "node_id"]
    feature_matrix = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)

    work = features_df.copy()
    work["node_id"] = work["node_id"].astype(str)

    for _, row in work.iterrows():
        node_id = row["node_id"]
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            feature_matrix[idx] = [float(row[col]) for col in feature_cols]

    # Build y from TRUE labels only.
    ground_truth = load_true_labels(G)
    y_np = np.array([int(ground_truth.get(str(n), 0)) for n in nodes], dtype=np.int64)
    y = torch.tensor(y_np, dtype=torch.long)

    # 70/15/15 split to avoid selecting model by test set.
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - config.TRAIN_RATIO),
        stratify=y_np,
        random_state=config.RANDOM_SEED,
    )

    temp_labels = y_np[temp_idx]
    val_relative_idx, test_relative_idx = train_test_split(
        np.arange(len(temp_idx)),
        test_size=0.5,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED,
    )
    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # fit scaler only on train nodes to avoid leakage
    scaler = StandardScaler()
    train_mask_np = train_mask.numpy()
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    feature_matrix[train_mask_np] = scaler.fit_transform(feature_matrix[train_mask_np])
    feature_matrix[~train_mask_np] = scaler.transform(feature_matrix[~train_mask_np])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    x = torch.tensor(feature_matrix, dtype=torch.float32)

    edge_list = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    logger.info("  x (node features): shape=%s, dtype=%s", data.x.shape, data.x.dtype)
    logger.info("  edge_index: shape=%s", data.edge_index.shape)
    logger.info("  y (labels): shape=%s", data.y.shape)

    fraud_ratio = float(data.y.float().mean().item()) if data.num_nodes else 0.0
    logger.info("  Ground-truth fraud ratio: %.2f%%", fraud_ratio * 100)
    logger.info(
        "  Split sizes - train: %d, val: %d, test: %d",
        int(train_mask.sum().item()),
        int(val_mask.sum().item()),
        int(test_mask.sum().item()),
    )

    if labels_series is not None:
        logger.info("  Note: labels_series argument ignored to prevent label leakage")

    return data, scaler, node_to_idx


def get_pyg_data():
    """
    Convenience zero-argument wrapper for test harnesses.

    Returns:
        tuple: (data, scaler, node_to_idx)
    """
    from src.features import compute_all_features

    df, _ = load_data()
    G = build_graph(df)
    features_df = compute_all_features(G)
    data, scaler, node_to_idx = build_pyg_data(G, features_df)

    return data, scaler, node_to_idx


# Backward-compatible aliases for existing imports.
load_dataset = load_data
load_ground_truth_labels = load_true_labels
load_pyg_data = get_pyg_data


if __name__ == "__main__":
    config.ensure_dirs()
    set_seeds()

    df, labels = load_data()
    G = build_graph(df)

    logger.info("Data load test complete")
    logger.info("  Transactions: %d", len(df))
    logger.info("  Graph nodes: %d, edges: %d", G.number_of_nodes(), G.number_of_edges())
