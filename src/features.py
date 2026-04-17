"""Feature engineering for AML graph learning."""

import importlib
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

import config

logger = config.setup_logging(__name__)


CANONICAL_FEATURE_COLUMNS = ["degree", "in_degree", "out_degree", "clustering", "pagerank"]
FULL_FEATURE_COLUMNS = [
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
]


def _coerce_transactions(transactions_df=None):
    """Load or normalize transactions with timestamp and amount fields."""
    if transactions_df is None:
        if not os.path.exists(config.RAW_TRANSACTIONS_PATH):
            return None
        transactions_df = pd.read_csv(config.RAW_TRANSACTIONS_PATH)

    required = {"sender_id", "receiver_id", "amount", "timestamp"}
    if not required.issubset(set(transactions_df.columns)):
        logger.warning(
            "Transactions missing required columns for temporal features: %s",
            sorted(required - set(transactions_df.columns)),
        )
        return None

    tx = transactions_df.copy()
    tx["sender_id"] = tx["sender_id"].astype(str)
    tx["receiver_id"] = tx["receiver_id"].astype(str)
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)

    ts = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True)
    ts = ts.fillna(pd.Timestamp("2023-01-01", tz="UTC"))
    tx["timestamp"] = ts.dt.tz_convert(None)
    tx["timestamp_int"] = tx["timestamp"].astype("int64") // 10**9
    tx["hour"] = tx["timestamp"].dt.hour.astype(int)
    return tx


def compute_degree(G):
    """Compute in/out/total degree."""
    degree_features = {}
    for node in G.nodes():
        in_deg = int(G.in_degree(node))
        out_deg = int(G.out_degree(node))
        degree_features[node] = {
            "degree": in_deg + out_deg,
            "in_degree": in_deg,
            "out_degree": out_deg,
        }
    return degree_features


def compute_clustering(G):
    """Compute clustering coefficient on undirected projection."""
    return nx.clustering(G.to_undirected())


def compute_pagerank(G):
    """Compute PageRank for directed graph."""
    return nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)


def _compute_base_features_cpp_or_nx(G):
    """Compute canonical structural features with C++ fast path and NetworkX fallback."""
    records = None

    try:
        cpp_dir = os.path.join(os.path.dirname(__file__), "..", "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)

        runner_module = importlib.import_module("graph_runner")
        run_cpp_algorithms = getattr(runner_module, "run_cpp_algorithms")

        cpp_result = run_cpp_algorithms(config.RAW_TRANSACTIONS_PATH)
        if cpp_result is not None:
            graph_nodes = set(str(n) for n in G.nodes())
            cpp_nodes = set(str(n) for n in cpp_result.index)
            if graph_nodes == cpp_nodes:
                logger.info("Using C++ structural features")
                records = []
                for node in sorted(G.nodes()):
                    n = str(node)
                    in_deg = int(G.in_degree(node))
                    out_deg = int(G.out_degree(node))
                    records.append(
                        {
                            "node_id": n,
                            "degree": int(cpp_result.at[n, "degree"]),
                            "in_degree": in_deg,
                            "out_degree": out_deg,
                            "clustering": float(cpp_result.at[n, "clustering"]),
                            "pagerank": float(cpp_result.at[n, "pagerank"]),
                        }
                    )
            else:
                logger.warning("C++ node mismatch, falling back to NetworkX")
    except Exception as exc:
        logger.warning("C++ backend unavailable; fallback to NetworkX (%s)", exc)

    if records is None:
        logger.info("Using NetworkX structural features")
        degree_features = compute_degree(G)
        clustering = compute_clustering(G)
        pagerank = compute_pagerank(G)

        records = []
        for node in sorted(G.nodes()):
            records.append(
                {
                    "node_id": str(node),
                    "degree": float(degree_features[node]["degree"]),
                    "in_degree": float(degree_features[node]["in_degree"]),
                    "out_degree": float(degree_features[node]["out_degree"]),
                    "clustering": float(clustering.get(node, 0.0)),
                    "pagerank": float(pagerank.get(node, 0.0)),
                }
            )

    return pd.DataFrame(records)


def _compute_temporal_features(nodes, tx):
    """Compute temporal burst and activity features over 7-day sliding windows."""
    default_row = {
        "tx_count_window": 0.0,
        "amount_std_window": 0.0,
        "unique_peers_window": 0.0,
        "night_tx_ratio": 0.0,
    }
    if tx is None or len(tx) == 0:
        return pd.DataFrame([{"node_id": str(n), **default_row} for n in nodes])

    # Build endpoint-wise transaction view so each transaction contributes to both nodes.
    left = tx[["sender_id", "receiver_id", "amount", "timestamp_int", "hour"]].rename(
        columns={"sender_id": "node_id", "receiver_id": "peer_id"}
    )
    right = tx[["sender_id", "receiver_id", "amount", "timestamp_int", "hour"]].rename(
        columns={"receiver_id": "node_id", "sender_id": "peer_id"}
    )
    node_tx = pd.concat([left, right], axis=0, ignore_index=True)
    node_tx["node_id"] = node_tx["node_id"].astype(str)
    node_tx["peer_id"] = node_tx["peer_id"].astype(str)

    window_seconds = 7 * 24 * 60 * 60
    out_rows = []

    grouped = node_tx.groupby("node_id", sort=False)
    for node in nodes:
        node_key = str(node)
        if node_key not in grouped.groups:
            out_rows.append({"node_id": node_key, **default_row})
            continue

        g = grouped.get_group(node_key).sort_values("timestamp_int", kind="mergesort")
        ts = g["timestamp_int"].to_numpy(dtype=np.int64)
        amt = g["amount"].to_numpy(dtype=np.float64)
        peers = g["peer_id"].to_numpy(dtype=object)
        hours = g["hour"].to_numpy(dtype=np.int64)

        start = 0
        best_count = 0
        best_start = 0
        best_end = -1
        for end in range(len(ts)):
            while ts[end] - ts[start] > window_seconds:
                start += 1
            count = end - start + 1
            if count > best_count:
                best_count = count
                best_start = start
                best_end = end

        if best_end >= best_start:
            seg_amt = amt[best_start : best_end + 1]
            seg_peers = peers[best_start : best_end + 1]
            amount_std = float(np.std(seg_amt)) if seg_amt.size > 1 else 0.0
            unique_peers = float(len(set(seg_peers.tolist())))
        else:
            amount_std = 0.0
            unique_peers = 0.0

        night_ratio = float(np.mean((hours >= 0) & (hours < 6))) if len(hours) else 0.0

        out_rows.append(
            {
                "node_id": node_key,
                "tx_count_window": float(best_count),
                "amount_std_window": amount_std,
                "unique_peers_window": unique_peers,
                "night_tx_ratio": night_ratio,
            }
        )

    return pd.DataFrame(out_rows)


def _derive_heuristic_flags(base_df):
    """Generate heuristic flags for neighbor risk aggregation."""
    w = config.HEURISTIC_WEIGHTS

    tmp = base_df.copy()
    for c in ["degree", "clustering", "pagerank"]:
        cmin = float(tmp[c].min())
        cmax = float(tmp[c].max())
        if cmax > cmin:
            tmp[f"norm_{c}"] = (tmp[c] - cmin) / (cmax - cmin)
        else:
            tmp[f"norm_{c}"] = 0.0

    tmp["heuristic_score"] = (
        float(w["w1"]) * tmp["norm_degree"]
        + float(w["w2"]) * (1.0 - tmp["norm_clustering"])
        + float(w["w3"]) * tmp["norm_pagerank"]
    )

    q = max(0.0, min(1.0, 1.0 - float(config.HEURISTIC_FRAUD_THRESHOLD)))
    threshold = float(tmp["heuristic_score"].quantile(q)) if len(tmp) else 1.0
    tmp["heuristic_flag"] = (tmp["heuristic_score"] >= threshold).astype(int)

    return tmp[["node_id", "heuristic_flag"]]


def _local_efficiency_from_neighbors(undirected_graph, node):
    """Compute harmonic mean of inverse shortest paths inside node's neighbor ego-subgraph."""
    neighbors = list(undirected_graph.neighbors(node))
    if len(neighbors) < 2:
        return 0.0

    sub = undirected_graph.subgraph(neighbors)
    n = sub.number_of_nodes()
    if n < 2:
        return 0.0

    lengths = dict(nx.all_pairs_shortest_path_length(sub))
    nodes = list(sub.nodes())
    pair_count = n * (n - 1) / 2.0
    if pair_count <= 0:
        return 0.0

    inv_sum = 0.0
    for i, u in enumerate(nodes):
        du = lengths.get(u, {})
        for v in nodes[i + 1 :]:
            d = du.get(v, None)
            if d is not None and d > 0:
                inv_sum += 1.0 / float(d)

    return float(inv_sum / pair_count)


def _compute_ego_topological_features(G, base_df):
    """Compute ego-neighborhood and topological descriptors."""
    undirected = G.to_undirected()
    bridges = set(nx.articulation_points(undirected)) if undirected.number_of_nodes() else set()

    deg_map = dict(zip(base_df["node_id"].astype(str), base_df["degree"].astype(float)))
    flag_df = _derive_heuristic_flags(base_df)
    flag_map = dict(zip(flag_df["node_id"].astype(str), flag_df["heuristic_flag"].astype(float)))

    rows = []
    for node in sorted(G.nodes()):
        node_key = str(node)
        in_deg = float(G.in_degree(node))
        out_deg = float(G.out_degree(node))
        neigh = set(str(n) for n in G.predecessors(node)).union(str(n) for n in G.successors(node))

        if neigh:
            neighbor_flags = [float(flag_map.get(n, 0.0)) for n in neigh]
            neighbor_deg = [float(deg_map.get(n, 0.0)) for n in neigh]
            neighbor_fraud_ratio = float(np.mean(neighbor_flags))
            neighbor_degree_mean = float(np.mean(neighbor_deg))
        else:
            neighbor_fraud_ratio = 0.0
            neighbor_degree_mean = 0.0

        rows.append(
            {
                "node_id": node_key,
                "neighbor_fraud_ratio": neighbor_fraud_ratio,
                "neighbor_degree_mean": neighbor_degree_mean,
                "in_out_ratio": float(in_deg / (out_deg + 1e-6)),
                "is_bridge": float(1.0 if node in bridges else 0.0),
                "local_efficiency": _local_efficiency_from_neighbors(undirected, node),
            }
        )

    return pd.DataFrame(rows)


def _correlation_report(features_df):
    """Log feature-label correlations against ground truth labels when available."""
    gt_path = config.ACCOUNT_GROUND_TRUTH_PATH
    if not os.path.exists(gt_path):
        logger.warning("Ground truth file not found for correlation report: %s", gt_path)
        return

    gt = pd.read_csv(gt_path)
    if "account_id" not in gt.columns or "is_fraud" not in gt.columns:
        logger.warning("Ground truth file missing account_id/is_fraud columns")
        return

    gt = gt[["account_id", "is_fraud"]].copy()
    gt["account_id"] = gt["account_id"].astype(str)
    gt["is_fraud"] = pd.to_numeric(gt["is_fraud"], errors="coerce").fillna(-1).astype(int)

    merged = features_df.merge(gt, left_on="node_id", right_on="account_id", how="left")
    merged = merged[merged["is_fraud"] >= 0].copy()
    if merged.empty:
        logger.warning("No labeled rows available for correlation report")
        return

    corr_rows = []
    numeric_cols = [c for c in features_df.columns if c != "node_id"]
    for col in numeric_cols:
        x = pd.to_numeric(merged[col], errors="coerce")
        if x.notna().sum() < 3:
            continue
        corr = float(x.corr(merged["is_fraud"]))
        if np.isfinite(corr):
            corr_rows.append((col, corr))

    corr_rows.sort(key=lambda t: abs(t[1]), reverse=True)
    logger.info("Feature correlation with ground truth labels (top 15 by |corr|):")
    for col, corr in corr_rows[:15]:
        logger.info("  %-24s corr=%.6f", col, corr)


def compute_features(G, use_dynamic=False, transactions_df=None):
    """Compute canonical and upgraded AML features for each node."""
    logger.info("Computing AML feature matrix...")

    nodes = [str(n) for n in sorted(G.nodes())]

    if use_dynamic and transactions_df is not None:
        from src.dynamic_graph import DynamicFraudGraph

        logger.info("Using dynamic feature path for canonical structural base")
        dg = DynamicFraudGraph(window_size=7)
        tx_in = _coerce_transactions(transactions_df)
        if tx_in is not None:
            for row in tx_in.itertuples(index=False):
                dg.add_transaction(row.sender_id, row.receiver_id, float(row.amount), int(row.timestamp_int))

        base_df = dg.get_all_features()
        if base_df.empty:
            base_df = _compute_base_features_cpp_or_nx(G)
    else:
        base_df = _compute_base_features_cpp_or_nx(G)

    base_df["node_id"] = base_df["node_id"].astype(str)
    base_df = pd.DataFrame({"node_id": nodes}).merge(base_df, on="node_id", how="left")
    for c in CANONICAL_FEATURE_COLUMNS:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce").fillna(0.0)

    tx = _coerce_transactions(transactions_df)
    temporal_df = _compute_temporal_features(nodes, tx)
    ego_topo_df = _compute_ego_topological_features(G, base_df)

    full_df = base_df.merge(temporal_df, on="node_id", how="left")
    full_df = full_df.merge(ego_topo_df, on="node_id", how="left")

    for c in FULL_FEATURE_COLUMNS:
        full_df[c] = pd.to_numeric(full_df[c], errors="coerce").fillna(0.0)

    nan_count = int(full_df.isna().sum().sum())

    config.ensure_dirs()
    full_path = os.path.join(config.PROCESSED_DATA_DIR, "node_features_full.csv")
    full_df.to_csv(full_path, index=False)

    # Keep existing downstream compatibility path.
    full_df.to_csv(config.NODE_FEATURES_PATH, index=False)

    logger.info("Saved full feature matrix: %s", full_path)
    logger.info("Saved compatibility feature matrix: %s", config.NODE_FEATURES_PATH)
    logger.info("Feature matrix shape: %s", tuple(full_df.shape))
    logger.info("Total NaN count: %d", nan_count)

    _correlation_report(full_df)

    return full_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from src.data_loader import build_graph, load_data, set_seeds

    set_seeds()
    config.ensure_dirs()

    df, _ = load_data()
    G = build_graph(df)
    features_df = compute_features(G, transactions_df=df)

    logger.info("Feature engineering complete")
    logger.info("Shape: %s", features_df.shape)


# Backward-compatible aliases.
compute_clustering_coefficient = compute_clustering
compute_all_features = compute_features
