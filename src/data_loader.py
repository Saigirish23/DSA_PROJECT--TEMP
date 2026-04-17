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


def _normalize_common_schema(tx_df, gt_df, source_name, allow_unlabeled=False):
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
    gdf["is_fraud"] = pd.to_numeric(gdf["is_fraud"], errors="coerce").fillna(-1 if allow_unlabeled else 0).astype(int)
    if allow_unlabeled:
        gdf.loc[~gdf["is_fraud"].isin([-1, 0, 1]), "is_fraud"] = 0
    else:
        gdf["is_fraud"] = (gdf["is_fraud"] > 0).astype(int)
    gdf = gdf[["account_id", "is_fraud"]].drop_duplicates(subset=["account_id"])

    # Ensure every node in transactions has a label
    tx_accounts = set(df["sender_id"]).union(set(df["receiver_id"]))
    known_accounts = set(gdf["account_id"])
    missing = sorted(tx_accounts - known_accounts)
    if missing:
        missing_label_value = -1 if allow_unlabeled else 0
        gdf = pd.concat(
            [gdf, pd.DataFrame({"account_id": missing, "is_fraud": missing_label_value})],
            ignore_index=True,
        )

    # Persist canonical files
    config.ensure_dirs()
    df.to_csv(config.RAW_TRANSACTIONS_PATH, index=False)
    gdf.to_csv(config.ACCOUNT_GROUND_TRUTH_PATH, index=False)

    labeled_ratio = float((gdf["is_fraud"] >= 0).mean()) if len(gdf) else 0.0
    fraud_ratio = float((gdf["is_fraud"] == 1).mean()) if len(gdf) else 0.0
    logger.info("Dataset source: %s", source_name)
    logger.info("  transactions rows: %d", len(df))
    logger.info("  account labels rows: %d", len(gdf))
    logger.info("  labeled ratio: %.2f%%", labeled_ratio * 100)
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


def _resolve_elliptic_sources():
    """Return available Elliptic/Bitcoin raw file paths if present."""
    candidates = [
        {
            "name": "Elliptic",
            "features": os.path.join(config.RAW_DATA_DIR, "elliptic_features.csv"),
            "edges": os.path.join(config.RAW_DATA_DIR, "elliptic_edges.csv"),
            "labels": os.path.join(config.RAW_DATA_DIR, "elliptic_labels.csv"),
        },
        {
            "name": "Bitcoin/Elliptic",
            "features": os.path.join(config.RAW_DATA_DIR, "bitcoin", "elliptic_txs_features.csv"),
            "edges": os.path.join(config.RAW_DATA_DIR, "bitcoin", "elliptic_txs_edgelist.csv"),
            "labels": os.path.join(config.RAW_DATA_DIR, "bitcoin", "elliptic_txs_classes.csv"),
        },
    ]

    for source in candidates:
        if all(os.path.exists(source[k]) for k in ("features", "edges", "labels")):
            return source
    return None


def _read_elliptic_metadata(features_path, labels_path):
    """Load Elliptic node metadata (txId, timestep, label) with robust mapping."""
    features = pd.read_csv(features_path, header=None)
    if features.shape[1] < 2:
        raise ValueError(f"Elliptic features file must have at least 2 columns. Found shape={features.shape}")

    features = features.rename(columns={0: "account_id", 1: "time_step"})
    features["account_id"] = features["account_id"].astype(str)
    features["time_step"] = pd.to_numeric(features["time_step"], errors="coerce").astype("Int64")

    labels = pd.read_csv(labels_path)
    if not {"txId", "class"}.issubset(labels.columns):
        raise ValueError(f"Elliptic labels file missing txId/class columns. Found: {list(labels.columns)}")

    labels = labels.rename(columns={"txId": "account_id", "class": "raw_class"})
    labels["account_id"] = labels["account_id"].astype(str)
    labels["raw_class"] = labels["raw_class"].astype(str).str.strip().str.lower()

    # Common Elliptic mappings:
    # v1: 1=illicit, 2=licit, unknown=unknown
    # v2: 0=illicit, 1=licit, 2=unknown
    has_unknown = (labels["raw_class"] == "unknown").any()
    if has_unknown:
        mapped = labels["raw_class"].map({"1": 1, "2": 0, "unknown": -1})
        mapping_name = "v1"
    else:
        raw_num = pd.to_numeric(labels["raw_class"], errors="coerce").fillna(2).astype(int)
        if set(raw_num.unique()).issubset({0, 1, 2}):
            mapped = raw_num.map({0: 1, 1: 0, 2: -1})
            mapping_name = "v2"
        else:
            # Conservative fallback: treat non-positive as unknown, 1 as fraud, others as licit.
            mapped = raw_num.map(lambda v: 1 if v == 1 else (0 if v > 1 else -1))
            mapping_name = "fallback"

    labels["is_fraud"] = mapped.fillna(-1).astype(int)
    labels["is_labeled"] = (labels["is_fraud"] >= 0).astype(int)

    metadata = features[["account_id", "time_step"]].merge(
        labels[["account_id", "is_fraud", "is_labeled"]], on="account_id", how="outer"
    )
    metadata["time_step"] = pd.to_numeric(metadata["time_step"], errors="coerce").astype("Int64")
    metadata["is_fraud"] = pd.to_numeric(metadata["is_fraud"], errors="coerce").fillna(-1).astype(int)
    metadata["is_labeled"] = (metadata["is_fraud"] >= 0).astype(int)

    class_counts = metadata["is_fraud"].value_counts().sort_index().to_dict()
    logger.info("  Elliptic mapping format: %s", mapping_name)
    logger.info("  Mapped class distribution (-1=unknown,0=licit,1=illicit): %s", class_counts)
    return metadata


def _adapt_elliptic_dataset(features_path, edges_path, labels_path):
    """Adapt Elliptic files into canonical schema with robust validation."""
    logger.info("Adapting Elliptic dataset...")
    
    # Validate and load edges
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Elliptic edges file not found: {edges_path}")
    edges = pd.read_csv(edges_path)
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError(f"Elliptic edges file missing txId1/txId2 columns. Found: {list(edges.columns)}")
    logger.info("  ✓ Edges loaded: %d rows", len(edges))

    # Validate and load labels + node-level metadata
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Elliptic labels file not found: {labels_path}")
    metadata = _read_elliptic_metadata(features_path, labels_path)
    logger.info("  ✓ Labels loaded: %d rows", len(metadata))

    # Elliptic has no amount in edge list. Timestamp is derived from sender tx timestep.
    timestep_map = (
        metadata[["account_id", "time_step"]]
        .dropna(subset=["time_step"])
        .drop_duplicates(subset=["account_id"])
        .set_index("account_id")["time_step"]
        .astype(int)
        .to_dict()
    )
    tx_df = edges[["txId1", "txId2"]].copy()
    tx_df = tx_df.rename(columns={"txId1": "sender_id", "txId2": "receiver_id"})
    tx_df["amount"] = 1.0
    base_ts = pd.Timestamp("2023-01-01")
    sender_steps = tx_df["sender_id"].astype(str).map(timestep_map).fillna(1).astype(int)
    tx_df["timestamp"] = base_ts + pd.to_timedelta((sender_steps - 1) * 7, unit="D")
    tx_df["transaction_id"] = ["EL_{:08d}".format(i) for i in range(len(tx_df))]

    gt_df = metadata[["account_id", "is_fraud"]].copy()
    labeled_df = gt_df[gt_df["is_fraud"] >= 0]
    fraud_ratio = (labeled_df["is_fraud"] == 1).mean() if len(labeled_df) > 0 else 0.0
    logger.info("  Labeled node ratio: %.2f%%", (len(labeled_df) / len(gt_df) * 100) if len(gt_df) else 0.0)
    logger.info("  Fraud ratio among labeled nodes: %.2f%%", fraud_ratio * 100)

    return tx_df, gt_df


def _time_based_split_indices(time_step_np, labeled_mask_np, y_np):
    """Create train/val/test splits by time_step; return None when invalid."""
    time_step_np = np.asarray(time_step_np)
    labeled_mask_np = np.asarray(labeled_mask_np, dtype=bool)

    usable_mask = (time_step_np > 0) & labeled_mask_np
    if usable_mask.sum() < 10:
        return None

    usable_steps = np.unique(time_step_np[usable_mask])
    if usable_steps.size < 3:
        return None

    min_ts = int(usable_steps.min())
    max_ts = int(usable_steps.max())

    # Canonical Elliptic split: train [1..34], val [35..39], test [40..49]
    if min_ts == 1 and max_ts == 49:
        train_steps = usable_steps[usable_steps <= 34]
        val_steps = usable_steps[(usable_steps >= 35) & (usable_steps <= 39)]
        test_steps = usable_steps[usable_steps >= 40]
    else:
        n_steps = usable_steps.size
        train_cut = max(1, int(n_steps * 0.70))
        val_cut = max(train_cut + 1, int(n_steps * 0.85))
        val_cut = min(val_cut, n_steps - 1)

        train_steps = usable_steps[:train_cut]
        val_steps = usable_steps[train_cut:val_cut]
        test_steps = usable_steps[val_cut:]

    if len(train_steps) == 0 or len(val_steps) == 0 or len(test_steps) == 0:
        return None

    train_idx = np.where(labeled_mask_np & np.isin(time_step_np, train_steps))[0]
    val_idx = np.where(labeled_mask_np & np.isin(time_step_np, val_steps))[0]
    test_idx = np.where(labeled_mask_np & np.isin(time_step_np, test_steps))[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        return None

    # Ensure train set has both classes for weighted CE and stable optimization.
    if np.unique(y_np[train_idx]).size < 2:
        return None

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def normalize_dataset_sources():
    """Select available source and normalize to canonical transactions + ground truth files."""
    config.ensure_dirs()

    # Prefer externally sourced files if present.
    paysim_path = os.path.join(config.RAW_DATA_DIR, "paysim_transactions.csv")
    elliptic_source = _resolve_elliptic_sources()

    if os.path.exists(paysim_path):
        tx_df, gt_df = _adapt_paysim_dataset(paysim_path)
        return _normalize_common_schema(tx_df, gt_df, "PaySim")

    if elliptic_source is not None:
        tx_df, gt_df = _adapt_elliptic_dataset(
            elliptic_source["features"],
            elliptic_source["edges"],
            elliptic_source["labels"],
        )
        return _normalize_common_schema(tx_df, gt_df, elliptic_source["name"], allow_unlabeled=True)

    # If canonical files already exist, normalize them in place.
    if os.path.exists(config.RAW_TRANSACTIONS_PATH) and os.path.exists(config.ACCOUNT_GROUND_TRUTH_PATH):
        tx_df = pd.read_csv(config.RAW_TRANSACTIONS_PATH)
        gt_df = pd.read_csv(config.ACCOUNT_GROUND_TRUTH_PATH)
        allow_unlabeled = (pd.to_numeric(gt_df.get("is_fraud"), errors="coerce").fillna(0) < 0).any()
        return _normalize_common_schema(tx_df, gt_df, "Existing Canonical", allow_unlabeled=bool(allow_unlabeled))

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
    gt_df["is_fraud"] = pd.to_numeric(gt_df["is_fraud"], errors="coerce").fillna(-1).astype(int)
    gt_df.loc[~gt_df["is_fraud"].isin([-1, 0, 1]), "is_fraud"] = 0
    gt_map = dict(zip(gt_df["account_id"], gt_df["is_fraud"]))
    default_label = -1 if (gt_df["is_fraud"] < 0).any() else 0
    nodes = sorted(G.nodes())
    return {str(n): int(gt_map.get(str(n), default_label)) for n in nodes}


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


def build_edge_features(df, node_idx):
    """
    Build 3-dim edge feature tensor for each transaction:
    [normalized_amount, temporal_delta_normalized, direction_flag]

    direction_flag:
      - 1.0 for forward edge (sender -> receiver)
      - 0.0 for reverse edge (receiver -> sender)
    """
    required = {"sender_id", "receiver_id", "amount", "timestamp"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Cannot build edge features; missing columns: {missing}")

    work = df.copy()
    work["sender_id"] = work["sender_id"].astype(str)
    work["receiver_id"] = work["receiver_id"].astype(str)
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)

    ts = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    ts = ts.fillna(pd.Timestamp("2023-01-01", tz="UTC"))
    ts_int = (ts.astype("int64") // 10**9).to_numpy(dtype=np.int64)

    amounts = work["amount"].to_numpy(dtype=np.float64)
    amt_norm = (amounts - amounts.mean()) / (amounts.std() + 1e-8)
    time_norm = (ts_int - ts_int.min()) / (ts_int.max() - ts_int.min() + 1e-8)

    edge_pairs = []
    edge_attrs = []

    sender_vals = work["sender_id"].to_numpy(dtype=object)
    receiver_vals = work["receiver_id"].to_numpy(dtype=object)

    for idx in range(len(work)):
        s = node_idx.get(str(sender_vals[idx]))
        r = node_idx.get(str(receiver_vals[idx]))
        if s is None or r is None or s == r:
            continue

        # forward edge
        edge_pairs.append([s, r])
        edge_attrs.append([float(amt_norm[idx]), float(time_norm[idx]), 1.0])

        # reverse edge
        edge_pairs.append([r, s])
        edge_attrs.append([float(amt_norm[idx]), float(time_norm[idx]), 0.0])

    if not edge_pairs:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return edge_index, edge_attr


def build_pyg_data(G, features_df, labels_series=None):
    """
    Convert graph + features into a leakage-safe PyG Data object.

    Uses account_ground_truth.csv as the ONLY source for y labels.
    Exports labels to data/processed/labels.csv for dashboard compatibility.
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, remove_self_loops

    logger.info("Building PyTorch Geometric Data object...")
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    canonical_feature_cols = ["degree", "in_degree", "out_degree", "clustering", "pagerank"]
    full_feature_priority = [
        "degree",
        "in_degree",
        "out_degree",
        "clustering",
        "pagerank",
        "tx_count_window",
        "amount_std_window",
        "unique_peers_window",
        "night_tx_ratio",
        "neighbor_fraud_ratio",
        "neighbor_degree_mean",
        "in_out_ratio",
        "is_bridge",
        "local_efficiency",
        "heuristic_score",
    ]
    fallback_feature_map = {
        "degree": ["degree", "total_degree"],
        "in_degree": ["in_degree"],
        "out_degree": ["out_degree"],
        "clustering": ["clustering", "clustering_coefficient"],
        "pagerank": ["pagerank", "page_rank"],
    }

    resolved_columns = {}
    for canonical_name, candidates in fallback_feature_map.items():
        for candidate in candidates:
            if candidate in features_df.columns:
                resolved_columns[canonical_name] = candidate
                break

    missing = [c for c in canonical_feature_cols if c not in resolved_columns]
    if missing:
        raise ValueError(
            "Missing required feature columns for GNN input: {}. Available columns: {}".format(
                missing, list(features_df.columns)
            )
        )

    rename_map = {source: canonical for canonical, source in resolved_columns.items()}
    work = features_df.rename(columns=rename_map).copy()
    work["node_id"] = work["node_id"].astype(str)

    feature_cols = [c for c in full_feature_priority if c in work.columns]
    # Include any additional numeric columns that are not already selected.
    extra_numeric = [
        c for c in work.columns
        if c != "node_id" and c not in feature_cols and np.issubdtype(work[c].dtype, np.number)
    ]
    feature_cols.extend(extra_numeric)

    feature_matrix = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)

    logger.info("  Using feature columns for GNN (%d): %s", len(feature_cols), feature_cols)

    for _, row in work.iterrows():
        node_id = row["node_id"]
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            feature_matrix[idx] = [float(row[col]) for col in feature_cols]

    # Build y from TRUE labels only. Unlabeled nodes use y=-1.
    ground_truth = load_true_labels(G)
    y_np = np.array([int(ground_truth.get(str(n), 0)) for n in nodes], dtype=np.int64)
    y = torch.tensor(y_np, dtype=torch.long)
    labeled_mask_np = y_np >= 0

    # Export labels.csv for dashboard compatibility
    labels_export_df = pd.DataFrame({"node_id": nodes, "is_fraud": y_np})
    labels_export_path = config.LABELS_PATH
    config.ensure_dirs()
    labels_export_df.to_csv(labels_export_path, index=False)
    logger.info("  ✓ Exported labels to %s", labels_export_path)

    # Elliptic: time-based split using node timesteps from raw features.
    train_idx = np.array([], dtype=np.int64)
    val_idx = np.array([], dtype=np.int64)
    test_idx = np.array([], dtype=np.int64)

    elliptic_source = _resolve_elliptic_sources()
    time_step_np = np.full(num_nodes, -1, dtype=np.int64)

    if elliptic_source is not None:
        try:
            meta = _read_elliptic_metadata(elliptic_source["features"], elliptic_source["labels"])
            ts_map = (
                meta[["account_id", "time_step"]]
                .dropna(subset=["time_step"])
                .drop_duplicates(subset=["account_id"])
                .set_index("account_id")["time_step"]
                .astype(int)
                .to_dict()
            )
            for idx, node in enumerate(nodes):
                time_step_np[idx] = int(ts_map.get(str(node), -1))

            split_indices = _time_based_split_indices(time_step_np, labeled_mask_np, y_np)
            if split_indices is not None:
                train_idx, val_idx, test_idx = split_indices
                logger.info(
                    "  Using time-based split for %s dataset (%d train / %d val / %d test)",
                    elliptic_source["name"],
                    len(train_idx),
                    len(val_idx),
                    len(test_idx),
                )
            else:
                logger.warning("  Time-based split not viable (insufficient temporal coverage or class support)")
        except Exception as exc:
            logger.warning("  Could not apply time-based Elliptic split (%s); falling back to stratified random split", exc)

    # Fallback: stratified random split on labeled nodes.
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        labeled_idx = np.where(labeled_mask_np)[0]
        labeled_y = y_np[labeled_idx]
        stratify_targets = labeled_y if len(np.unique(labeled_y)) > 1 else None

        train_idx, test_idx = train_test_split(
            labeled_idx,
            test_size=0.3,
            stratify=stratify_targets,
            random_state=config.RANDOM_SEED,
        )

        train_labels = y_np[train_idx]
        stratify_train = train_labels if len(np.unique(train_labels)) > 1 else None
        tr_relative_idx, val_relative_idx = train_test_split(
            np.arange(len(train_idx)),
            test_size=0.2,
            stratify=stratify_train,
            random_state=config.RANDOM_SEED,
        )
        val_idx = train_idx[val_relative_idx]
        train_idx = train_idx[tr_relative_idx]
        logger.info("  Using stratified random split on labeled nodes")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Fit scaler only on train nodes to avoid leakage.
    scaler = StandardScaler()
    train_mask_np = train_mask.numpy()
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    if train_mask_np.sum() > 0:
        feature_matrix[train_mask_np] = scaler.fit_transform(feature_matrix[train_mask_np])
        feature_matrix[~train_mask_np] = scaler.transform(feature_matrix[~train_mask_np])
    else:
        feature_matrix = scaler.fit_transform(feature_matrix)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    x = torch.tensor(feature_matrix, dtype=torch.float32)

    # Prefer transaction-level edges so edge attributes align with real events.
    if os.path.exists(config.RAW_TRANSACTIONS_PATH):
        tx_df = pd.read_csv(config.RAW_TRANSACTIONS_PATH)
        edge_index, edge_attr = build_edge_features(tx_df, node_to_idx)
    else:
        edge_pairs = []
        edge_attrs = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx and u != v:
                su = node_to_idx[u]
                sv = node_to_idx[v]
                edge_pairs.append([su, sv])
                edge_attrs.append([0.0, 0.0, 1.0])
                edge_pairs.append([sv, su])
                edge_attrs.append([0.0, 0.0, 0.0])

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.0, num_nodes=num_nodes)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labeled_mask=torch.tensor(labeled_mask_np, dtype=torch.bool),
        time_step=torch.tensor(time_step_np, dtype=torch.long),
    )

    logger.info("  x (node features): shape=%s, dtype=%s", data.x.shape, data.x.dtype)
    logger.info("  edge_index: shape=%s", data.edge_index.shape)
    logger.info("  edge_attr: shape=%s", data.edge_attr.shape)
    logger.info("  y (labels): shape=%s", data.y.shape)

    labeled_count = int(data.labeled_mask.sum().item())
    if labeled_count:
        fraud_ratio = float((data.y[data.labeled_mask] == 1).float().mean().item())
    else:
        fraud_ratio = 0.0
    logger.info("  Labeled nodes: %d / %d", labeled_count, data.num_nodes)
    logger.info("  Fraud ratio (labeled only): %.2f%%", fraud_ratio * 100)
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
