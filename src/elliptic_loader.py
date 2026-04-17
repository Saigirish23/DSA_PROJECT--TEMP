"""Loader for full-feature Elliptic Bitcoin dataset (tuned workflow only)."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data

import config


def load_elliptic_full():
    """
    Load full Elliptic dataset with all available numeric node features.

    Layout:
      - col 0: txId
      - col 1: time_step
      - remaining cols: node features

    Label mapping:
      - illicit -> 1
      - licit -> 0
      - unknown/unlabeled -> -1

    Split protocol (time-based):
      - train: t <= 27
      - val:   28 <= t <= 34
      - test:  t >= 35
    """
    raw = os.path.join(config.RAW_DATA_DIR, "bitcoin")

    feat_path = os.path.join(raw, "elliptic_txs_features.csv")
    edge_path = os.path.join(raw, "elliptic_txs_edgelist.csv")
    class_path = os.path.join(raw, "elliptic_txs_classes.csv")

    feat_df = pd.read_csv(feat_path, header=None)
    edges_df = pd.read_csv(edge_path)
    class_df = pd.read_csv(class_path)

    if feat_df.shape[1] < 3:
        raise ValueError("Elliptic feature matrix is malformed: expected at least 3 columns")

    # Name columns: txId, time_step, f_0..f_k
    feature_dim = feat_df.shape[1] - 2
    feat_df.columns = ["txId", "time_step"] + [f"f_{i}" for i in range(feature_dim)]

    class_df.columns = ["txId", "class"]
    df = feat_df.merge(class_df, on="txId", how="left")

    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    df["label"] = df["class"].map(label_map).fillna(-1).astype(int)

    tx_ids = df["txId"].astype(str).values
    tx_idx = {tid: i for i, tid in enumerate(tx_ids)}
    n_nodes = len(tx_ids)

    # Use all 166 numeric features available in Elliptic file (time_step + f_*).
    feat_cols = ["time_step"] + [f"f_{i}" for i in range(feature_dim)]
    x_np = df[feat_cols].values.astype(np.float32)

    time_np = pd.to_numeric(df["time_step"], errors="coerce").fillna(-1).astype(int).values
    label_np = df["label"].values.astype(np.int64)
    labeled = label_np >= 0

    train_mask_np = labeled & (time_np <= 27)
    val_mask_np = labeled & (time_np >= 28) & (time_np <= 34)
    test_mask_np = labeled & (time_np >= 35)

    train_idx = np.where(train_mask_np)[0]
    val_idx = np.where(val_mask_np)[0]
    test_idx = np.where(test_mask_np)[0]

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "Time-based split is empty in one or more partitions: "
            f"train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
        )

    scaler = RobustScaler()
    x_np[train_idx] = scaler.fit_transform(x_np[train_idx])
    x_np[val_idx] = scaler.transform(x_np[val_idx])
    x_np[test_idx] = scaler.transform(x_np[test_idx])
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=3.0, neginf=-3.0)
    x_np = np.clip(x_np, -10.0, 10.0)

    src_ids = edges_df.iloc[:, 0].astype(str).values
    dst_ids = edges_df.iloc[:, 1].astype(str).values

    # Build edge attributes: [normalized_amount, temporal_delta_normalized, direction_flag].
    # Elliptic edge list has no explicit amount, so amount channel is 0 after normalization.
    edge_pairs = []
    edge_attr_rows = []

    node_time = pd.to_numeric(df["time_step"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    max_delta = 1.0
    raw_deltas = []
    for s, d in zip(src_ids, dst_ids):
      if s in tx_idx and d in tx_idx:
        si = tx_idx[s]
        di = tx_idx[d]
        raw_deltas.append(abs(float(node_time[si] - node_time[di])))
    if raw_deltas:
      max_delta = max(max(raw_deltas), 1.0)

    for s, d in zip(src_ids, dst_ids):
      if s not in tx_idx or d not in tx_idx:
        continue

      si = tx_idx[s]
      di = tx_idx[d]
      dt_norm = abs(float(node_time[si] - node_time[di])) / max_delta

      # forward edge
      edge_pairs.append([si, di])
      edge_attr_rows.append([0.0, dt_norm, 1.0])

      # reverse edge
      edge_pairs.append([di, si])
      edge_attr_rows.append([0.0, dt_norm, 0.0])

    if edge_pairs:
      edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
      edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
    else:
      edge_index = torch.empty((2, 0), dtype=torch.long)
      edge_attr = torch.empty((0, 3), dtype=torch.float32)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(label_np, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    data.tx_ids = tx_ids

    return data, scaler, feat_cols
