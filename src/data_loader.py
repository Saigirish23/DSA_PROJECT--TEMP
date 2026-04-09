"""
data_loader.py — Data Loading & Graph Construction

Responsibilities:
  1. Generate or load a synthetic transaction dataset.
  2. Build a directed NetworkX DiGraph from the transaction data.
  3. Convert the graph + features into a PyTorch Geometric Data object.

The synthetic dataset models realistic financial transaction patterns:
  - Fraudulent accounts tend to have burst transaction patterns
  - Normal accounts have more regular, lower-value transactions
  - Fraudulent accounts sometimes form clusters (fraud rings)
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


def set_seeds():
    """Set all random seeds for reproducibility."""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)


def generate_synthetic_dataset():
    """
    Generate a realistic synthetic transaction dataset.

    The dataset models financial transactions between accounts, where a
    fraction of accounts are fraudulent. Fraudulent accounts exhibit:
      - Higher transaction volumes (more edges)
      - Connections to other fraud accounts (fraud rings)
      - Higher-value transactions on average

    Returns:
        pd.DataFrame: Transaction data with columns:
            - transaction_id: unique transaction identifier
            - sender_id: source account (node)
            - receiver_id: destination account (node)
            - amount: transaction amount in USD
            - timestamp: Unix timestamp of the transaction
            - sender_is_fraud: ground truth label for sender (0 or 1)
            - receiver_is_fraud: ground truth label for receiver (0 or 1)

    Complexity:
        O(E) where E = NUM_TRANSACTIONS, since we generate each edge once.
    """
    set_seeds()
    logger.info("Generating synthetic transaction dataset...")

    num_accounts = config.NUM_ACCOUNTS
    num_transactions = config.NUM_TRANSACTIONS
    fraud_ratio = config.FRAUD_RATIO

    # Assign ground-truth fraud labels to accounts
    num_fraud = int(num_accounts * fraud_ratio)
    account_ids = list(range(num_accounts))
    fraud_accounts = set(random.sample(account_ids, num_fraud))

    logger.info(
        "  Accounts: %d total, %d fraudulent (%.1f%%)",
        num_accounts, num_fraud, fraud_ratio * 100
    )

    # Build ground-truth label map
    account_labels = {acc: (1 if acc in fraud_accounts else 0) for acc in account_ids}

    transactions = []
    base_timestamp = 1700000000  # Nov 2023 epoch

    for txn_id in range(num_transactions):
        # Fraudulent accounts are more likely to be involved in transactions
        # This creates a realistic class imbalance in activity
        if random.random() < 0.3 and len(fraud_accounts) > 1:
            # 30% of transactions involve at least one fraud account
            sender = random.choice(list(fraud_accounts))
            if random.random() < 0.5:
                # Fraud-to-fraud transaction (fraud ring)
                receiver = random.choice(list(fraud_accounts - {sender}))
            else:
                # Fraud-to-normal transaction
                normal_accounts = [a for a in account_ids if a not in fraud_accounts]
                receiver = random.choice(normal_accounts)
        else:
            # Normal transaction between any two accounts
            sender, receiver = random.sample(account_ids, 2)

        # Avoid self-loops
        while receiver == sender:
            receiver = random.choice(account_ids)

        # Transaction amount — fraud transactions tend to be higher value
        is_fraud_txn = (sender in fraud_accounts) or (receiver in fraud_accounts)
        if is_fraud_txn:
            # Fraud-related transactions: skewed toward higher amounts
            amount = round(
                np.random.lognormal(mean=7.5, sigma=1.2), 2
            )
        else:
            # Normal transactions: more moderate amounts
            amount = round(
                np.random.lognormal(mean=5.0, sigma=1.0), 2
            )

        # Clip to configured range
        amount = max(config.MIN_TRANSACTION_AMOUNT, min(config.MAX_TRANSACTION_AMOUNT, amount))

        # Timestamp: transactions spread over ~30 days
        timestamp = base_timestamp + random.randint(0, 30 * 24 * 3600)

        transactions.append({
            "transaction_id": txn_id,
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "timestamp": timestamp,
            "sender_is_fraud": account_labels[sender],
            "receiver_is_fraud": account_labels[receiver],
        })

    df = pd.DataFrame(transactions)

    # Save to raw data directory
    config.ensure_dirs()
    df.to_csv(config.RAW_TRANSACTIONS_PATH, index=False)
    logger.info("  Saved %d transactions to %s", len(df), config.RAW_TRANSACTIONS_PATH)

    # Save ground-truth account labels
    labels_df = pd.DataFrame([
        {"node_id": acc, "is_fraud": label}
        for acc, label in account_labels.items()
    ])
    ground_truth_path = os.path.join(config.PROCESSED_DATA_DIR, "ground_truth_labels.csv")
    labels_df.to_csv(ground_truth_path, index=False)
    logger.info("  Saved ground-truth labels to %s", ground_truth_path)

    return df, account_labels


def load_dataset():
    """
    Load transaction dataset from disk, or generate synthetic data if not found.

    Returns:
        tuple: (pd.DataFrame of transactions, dict of account_id -> fraud_label)
    """
    if os.path.exists(config.RAW_TRANSACTIONS_PATH):
        logger.info("Loading existing dataset from %s", config.RAW_TRANSACTIONS_PATH)
        df = pd.read_csv(config.RAW_TRANSACTIONS_PATH)

        # Reconstruct account labels from the data
        all_accounts = set(df["sender_id"].unique()) | set(df["receiver_id"].unique())
        fraud_senders = set(df[df["sender_is_fraud"] == 1]["sender_id"].unique())
        fraud_receivers = set(df[df["receiver_is_fraud"] == 1]["receiver_id"].unique())
        fraud_accounts = fraud_senders | fraud_receivers
        account_labels = {acc: (1 if acc in fraud_accounts else 0) for acc in all_accounts}

        return df, account_labels
    else:
        logger.info("No dataset found in data/raw/. Generating synthetic data...")
        return generate_synthetic_dataset()


def build_graph(df):
    """
    Build a directed NetworkX DiGraph from transaction data.

    Each node represents an account. Each edge represents a transaction
    with attributes for amount and timestamp.

    Args:
        df (pd.DataFrame): Transaction data with sender_id, receiver_id, amount, timestamp.

    Returns:
        nx.DiGraph: Directed graph with edge attributes.

    Complexity:
        O(V + E) where V = unique accounts, E = transactions.
        Building the graph requires iterating over all edges once.
    """
    logger.info("Building directed transaction graph...")

    G = nx.DiGraph()

    # Add all edges with attributes
    for _, row in df.iterrows():
        sender = int(row["sender_id"])
        receiver = int(row["receiver_id"])

        # If edge already exists, we keep the latest transaction (multi-graph simplified)
        if G.has_edge(sender, receiver):
            # Aggregate: sum amounts, keep latest timestamp
            G[sender][receiver]["amount"] += row["amount"]
            G[sender][receiver]["timestamp"] = max(
                G[sender][receiver]["timestamp"], row["timestamp"]
            )
            G[sender][receiver]["count"] += 1
        else:
            G.add_edge(
                sender, receiver,
                amount=row["amount"],
                timestamp=int(row["timestamp"]),
                count=1
            )

    # Sanity checks
    logger.info("  Graph Statistics:")
    logger.info("    Nodes: %d", G.number_of_nodes())
    logger.info("    Edges: %d", G.number_of_edges())
    logger.info("    Density: %.6f", nx.density(G))

    # Connected components (on undirected version)
    undirected = G.to_undirected()
    num_components = nx.number_connected_components(undirected)
    logger.info("    Connected components (undirected): %d", num_components)

    # Self-loop check
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        logger.warning("    ⚠️ Found %d self-loops — removing them", len(self_loops))
        G.remove_edges_from(self_loops)
    else:
        logger.info("    ✅ No self-loops detected")

    # Isolated nodes check (nodes with no edges at all)
    isolated = list(nx.isolates(G))
    if isolated:
        logger.warning(
            "    ⚠️ Found %d isolated nodes (no transactions)", len(isolated)
        )
    else:
        logger.info("    ✅ No isolated nodes")

    return G


def build_pyg_data(G, features_df, labels_series):
    """
    Convert a NetworkX graph with features and labels into a PyTorch Geometric Data object.

    This function:
      1. Normalizes node features using StandardScaler
      2. Builds the COO edge_index tensor
      3. Creates train/test masks with stratified splitting
      4. Packages everything into a PyG Data object

    Args:
        G (nx.DiGraph): The transaction graph.
        features_df (pd.DataFrame): Node features indexed by node_id.
            Columns: degree, clustering_coefficient, pagerank, betweenness_centrality
        labels_series (pd.Series): Node labels indexed by node_id (0 = normal, 1 = fraud).

    Returns:
        torch_geometric.data.Data: PyG data object with:
            - x: normalized node feature matrix (float32, shape [N, F])
            - edge_index: COO edge list (long, shape [2, E])
            - y: label tensor (long, shape [N])
            - train_mask: boolean mask for training nodes
            - test_mask: boolean mask for test nodes

    Complexity:
        O(V·F + E) — feature normalization is O(V·F), edge list construction is O(E).
    """
    from torch_geometric.data import Data

    logger.info("Building PyTorch Geometric Data object...")

    # Get sorted node list to ensure consistent ordering
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # --- Node features (x) ---
    # Reindex features to match node ordering
    feature_cols = [col for col in features_df.columns if col != "node_id"]
    feature_matrix = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)

    for _, row in features_df.iterrows():
        node_id = int(row["node_id"])
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            feature_matrix[idx] = [row[col] for col in feature_cols]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix).astype(np.float32)
    x = torch.tensor(feature_matrix, dtype=torch.float32)
    logger.info("  x (node features): shape=%s, dtype=%s", x.shape, x.dtype)

    # --- Edge index ---
    edge_list = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    logger.info("  edge_index: shape=%s", edge_index.shape)

    # --- Labels (y) ---
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node_id, label in labels_series.items():
        if node_id in node_to_idx:
            y[node_to_idx[node_id]] = int(label)
    logger.info("  y (labels): shape=%s", y.shape)

    # Class distribution
    num_fraud = int(y.sum().item())
    num_normal = num_nodes - num_fraud
    logger.info(
        "  Class distribution: %d normal (%.1f%%), %d fraud (%.1f%%)",
        num_normal, num_normal / num_nodes * 100,
        num_fraud, num_fraud / num_nodes * 100
    )

    # --- Train/Test masks (80/20 stratified split) ---
    indices = np.arange(num_nodes)
    labels_np = y.numpy()

    train_idx, test_idx = train_test_split(
        indices,
        test_size=(1.0 - config.TRAIN_RATIO),
        stratify=labels_np,
        random_state=config.RANDOM_SEED
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    logger.info(
        "  Train: %d nodes (%d fraud), Test: %d nodes (%d fraud)",
        train_mask.sum().item(), y[train_mask].sum().item(),
        test_mask.sum().item(), y[test_mask].sum().item()
    )

    # --- Build Data object ---
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    logger.info("  ✅ PyG Data object built successfully")
    return data, scaler, node_to_idx


if __name__ == "__main__":
    # Quick test: generate data and build graph
    config.ensure_dirs()
    set_seeds()

    df, account_labels = load_dataset()
    G = build_graph(df)

    logger.info("\n✅ Data loading and graph construction complete!")
    logger.info("  Transactions: %d", len(df))
    logger.info("  Graph nodes: %d, edges: %d", G.number_of_nodes(), G.number_of_edges())
