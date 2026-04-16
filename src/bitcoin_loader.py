"""Load and adapt the Elliptic Bitcoin dataset into PyG Data format."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import config


def _stratified_split_indices(labeled_idx, y_labeled):
    """Create train/val/test splits on labeled nodes with stable stratification."""
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        labeled_idx,
        y_labeled,
        test_size=0.30,
        stratify=y_labeled,
        random_state=config.RANDOM_SEED,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )
    return train_idx, val_idx, test_idx


def load_bitcoin_dataset():
    """
    Load Elliptic Bitcoin dataset and return a PyG Data object.

    Dataset facts:
    - 203,769 transactions (nodes)
    - 234,355 payment flows (edges)
    - 166 features per node (first is time step, rest are aggregated)
    - Labels: 1=illicit (fraud), 2=licit (normal), 0=unknown
    - Only a subset of nodes are labeled
    """
    raw = os.path.join(config.RAW_DATA_DIR, "bitcoin")

    print("Loading Elliptic Bitcoin dataset...")

    feat_df = pd.read_csv(os.path.join(raw, "elliptic_txs_features.csv"), header=None)
    edges_df = pd.read_csv(os.path.join(raw, "elliptic_txs_edgelist.csv"))
    class_df = pd.read_csv(os.path.join(raw, "elliptic_txs_classes.csv"))

    print(f"  Transactions: {len(feat_df)}")
    print(f"  Edges: {len(edges_df)}")
    print(f"  Labels: {len(class_df)}")

    feat_df.columns = ["tx_id", "time_step"] + [f"f_{i}" for i in range(feat_df.shape[1] - 2)]

    class_df.columns = ["tx_id", "class"]
    merged = feat_df.merge(class_df, on="tx_id", how="left")

    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    merged["label"] = merged["class"].map(label_map).fillna(-1).astype(int)

    print(f"  Illicit (fraud): {(merged['label'] == 1).sum()}")
    print(f"  Licit (normal):  {(merged['label'] == 0).sum()}")
    print(f"  Unknown:         {(merged['label'] == -1).sum()}")

    feature_cols = [f"f_{i}" for i in range(93)]
    X = merged[feature_cols].values.astype(np.float32)

    tx_ids = merged["tx_id"].values
    tx_idx = {tid: i for i, tid in enumerate(tx_ids)}
    n_nodes = len(tx_ids)

    src_ids = edges_df.iloc[:, 0].values
    dst_ids = edges_df.iloc[:, 1].values

    src_list, dst_list = [], []
    for s, d in zip(src_ids, dst_ids):
        if s in tx_idx and d in tx_idx:
            src_list.append(tx_idx[s])
            dst_list.append(tx_idx[d])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    print(f"  Valid edges in graph: {edge_index.shape[1]}")

    y = torch.tensor(merged["label"].values, dtype=torch.long)

    labeled_idx = np.where(merged["label"].values >= 0)[0]
    y_labeled = merged["label"].values[labeled_idx]
    train_idx, val_idx, test_idx = _stratified_split_indices(labeled_idx, y_labeled)

    scaler = StandardScaler()
    X[train_idx] = scaler.fit_transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.tensor(X, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    fraud_ratio = y_labeled.mean() if len(y_labeled) else 0.0
    print(f"  Fraud ratio in labeled set: {fraud_ratio:.2%}")
    print(
        f"  Train: {int(data.train_mask.sum())} | "
        f"Val: {int(data.val_mask.sum())} | "
        f"Test: {int(data.test_mask.sum())}"
    )
    print(f"  Feature shape: {x.shape}")

    return data, scaler
