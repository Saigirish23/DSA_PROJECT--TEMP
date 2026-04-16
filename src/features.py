"""
features.py — Graph Feature Engineering (DSA Core)

Computes structural graph features for every node in the transaction graph.
Each feature is implemented with a clear explanation of:
  - What the algorithm does
  - Why it matters for fraud detection
  - Its time complexity

Features computed:
  1. Degree (in + out)     — O(V + E)  — Hub detection
  2. Clustering Coefficient — O(V · d²) — Broker/mule detection
  3. PageRank              — O(k·(V+E)) — Centrality in flow network
  4. Betweenness Centrality — O(V · E)  — Bridges between clusters
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
import sys

import config

logger = config.setup_logging(__name__)


def compute_degree(G):
    """
    Compute in-degree, out-degree, and total degree for each node.

    Algorithm:
        For a directed graph, in-degree = number of incoming edges,
        out-degree = number of outgoing edges. Total degree = in + out.
        This is computed by a single traversal of the adjacency list.

    Why it matters:
        High-degree nodes are hubs in the transaction network.
        Fraudulent accounts often have unusually high degree because
        they transact with many victims or launder through many accounts.

    Complexity: O(V + E)
        Each edge contributes to one in-degree and one out-degree count.

    Args:
        G (nx.DiGraph): Directed transaction graph.

    Returns:
        dict: {node_id: {"in_degree": int, "out_degree": int, "total_degree": int}}
    """
    logger.info("  Computing degree features... [O(V+E)]")
    # high-degree nodes interact with many accounts, which can be a fraud signal

    degree_features = {}
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        degree_features[node] = {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": in_deg + out_deg,
        }

    return degree_features


def compute_clustering(G):
    """
    Compute the clustering coefficient for each node.

    Algorithm:
        The clustering coefficient measures the fraction of a node's neighbors
        that are also connected to each other (triangle density). For a directed
        graph, we use the undirected version to compute triangles.

        For each node v with degree d(v):
            C(v) = 2 * |triangles(v)| / (d(v) * (d(v) - 1))

        Triangle counting is done by checking, for each pair of neighbors (u, w),
        whether an edge (u, w) exists.

    Why it matters:
        Legitimate accounts tend to form tight-knit communities (high clustering).
        Fraudulent accounts acting as brokers or mules connect disparate groups,
        resulting in LOW clustering coefficients. This makes clustering coefficient
        inversely correlated with fraud likelihood.

    Complexity: O(V · d²)
        For each node, we examine all pairs of neighbors. In the worst case
        (star graph), d can be O(V), making it O(V³). In sparse graphs with
        bounded degree, it's much faster.

    Args:
        G (nx.DiGraph): Directed transaction graph.

    Returns:
        dict: {node_id: float} — clustering coefficient in [0, 1].
    """
    logger.info("  Computing clustering coefficients... [O(V·d²)]")
    # low clustering often means a node connects otherwise unrelated accounts

    # Use undirected version for triangle counting
    undirected = G.to_undirected()
    clustering = nx.clustering(undirected)

    return clustering


def compute_pagerank(G):
    """
    Compute PageRank for each node using the power iteration method.

    Algorithm:
        PageRank models a random walk on the graph. A surfer at node u follows
        a random outgoing edge with probability (1 - alpha), or jumps to any
        random node with probability alpha (damping factor, typically 0.85).

        The score represents the steady-state probability of the surfer being
        at each node. Computed iteratively:
            PR(v) = alpha/N + (1-alpha) * Σ PR(u) / out_deg(u)
                    for all u with edge u→v

        Iteration continues until convergence (change < epsilon) or max
        iterations reached.

    Why it matters:
        PageRank measures a node's importance in the flow network. Fraudulent
        accounts that receive money from many sources (e.g., in money laundering)
        will have high PageRank because they accumulate "importance" from
        incoming transactions.

    Complexity: O(k · (V + E))
        Each iteration traverses all edges once. k = number of iterations
        until convergence (typically 50–100 for sparse graphs).

    Args:
        G (nx.DiGraph): Directed transaction graph.

    Returns:
        dict: {node_id: float} — PageRank score (sums to 1.0).
    """
    logger.info("  Computing PageRank... [O(k·(V+E)), k=100 max iterations]")

    # standard pagerank with damping factor 0.85
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

    return pagerank


def compute_betweenness(G):
    """
    Compute betweenness centrality for each node.

    Algorithm:
        Betweenness centrality measures how often a node lies on the shortest
        path between all pairs of other nodes. For each pair (s, t), we run
        BFS (unweighted) from s, then accumulate the fraction of shortest
        paths through each intermediate node.

        BC(v) = Σ σ(s,t|v) / σ(s,t)
                for all s ≠ v ≠ t

        where σ(s,t) = number of shortest paths from s to t
              σ(s,t|v) = number of those paths passing through v

        Brandes' algorithm computes this efficiently by combining forward BFS
        with backward accumulation.

    Why it matters:
        Nodes with high betweenness are bridges between clusters. In fraud
        networks, these are often money mules or intermediaries connecting
        fraud rings to legitimate accounts. Disrupting these nodes can
        break the fraud network.

    Complexity: O(V · E)
        Brandes' algorithm runs BFS from each node (O(V+E) per source),
        for V sources total. For sparse graphs, this is approximately O(V·E).

    Args:
        G (nx.DiGraph): Directed transaction graph.

    Returns:
        dict: {node_id: float} — betweenness centrality in [0, 1].
    """
    logger.info("  Computing betweenness centrality... [O(V·E)] (may take a moment)")

    # normalized=True keeps values comparable across graph sizes
    betweenness = nx.betweenness_centrality(G, normalized=True)

    return betweenness


def compute_features(G, use_dynamic=False, transactions_df=None):
    """
    Compute all graph-structural features for every node.

    Combines degree, clustering coefficient, PageRank, and betweenness
    centrality into a single DataFrame for downstream use.

    Args:
        G (nx.DiGraph): Directed transaction graph.
        use_dynamic (bool): Use incremental dynamic feature path when True.
        transactions_df (pd.DataFrame | None): Transaction rows required for
            dynamic mode with sender_id, receiver_id, amount, timestamp/timestamp_int.

    Returns:
        pd.DataFrame: Feature matrix with columns:
            node_id, in_degree, out_degree, total_degree,
            clustering_coefficient, pagerank, betweenness_centrality
    """
    logger.info("Computing all graph-structural features...")

    if use_dynamic and transactions_df is not None:
        logger.info("  Using dynamic graph feature computation")
        from src.dynamic_graph import DynamicFraudGraph

        dg = DynamicFraudGraph(window_size=7)
        for _, row in transactions_df.iterrows():
            sender = str(row["sender_id"])
            receiver = str(row["receiver_id"])
            amount = float(row.get("amount", 0.0))
            timestamp = row.get("timestamp_int", 0)
            if "timestamp" in row and ("timestamp_int" not in row or pd.isna(timestamp)):
                try:
                    timestamp = int(pd.Timestamp(row["timestamp"]).timestamp())
                except Exception:
                    timestamp = 0
            timestamp = int(timestamp) if not pd.isna(timestamp) else 0
            dg.add_transaction(sender, receiver, amount, timestamp)

        features_df = dg.get_all_features()
        if "node_id" in features_df.columns:
            features_df["node_id"] = features_df["node_id"].astype(str)
        config.ensure_dirs()
        features_df.to_csv(config.NODE_FEATURES_PATH, index=False)
        logger.info("  Saved feature matrix to %s", config.NODE_FEATURES_PATH)
        return features_df

    # Try C++ backend first (fast path), but keep output schema identical
    # to the existing NetworkX pipeline expectations.
    try:
        cpp_dir = os.path.join(os.path.dirname(__file__), "..", "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        from graph_runner import run_cpp_algorithms

        edge_file = config.RAW_TRANSACTIONS_PATH
        cpp_result = run_cpp_algorithms(edge_file)

        if cpp_result is not None:
            graph_nodes = set(str(n) for n in G.nodes())
            cpp_nodes = set(str(n) for n in cpp_result.index)
            if graph_nodes == cpp_nodes:
                logger.info("  Using C++ computed features")

                records = []
                for node in sorted(G.nodes()):
                    n = str(node)
                    in_deg = int(G.in_degree(node))
                    out_deg = int(G.out_degree(node))
                    degree = int(cpp_result.at[n, "degree"])
                    records.append({
                        "node_id": n,
                        "in_degree": in_deg,
                        "out_degree": out_deg,
                        "total_degree": degree,
                        "clustering_coefficient": float(cpp_result.at[n, "clustering"]),
                        "pagerank": float(cpp_result.at[n, "pagerank"]),
                        "betweenness_centrality": float(cpp_result.at[n, "betweenness"]),
                    })

                features_df = pd.DataFrame(records)
                config.ensure_dirs()
                features_df.to_csv(config.NODE_FEATURES_PATH, index=False)
                logger.info("  Saved feature matrix to %s", config.NODE_FEATURES_PATH)

                logger.info("\n  Feature Distribution Statistics:")
                logger.info("  %s", "-" * 70)
                stat_cols = ["in_degree", "out_degree", "total_degree",
                             "clustering_coefficient", "pagerank", "betweenness_centrality"]
                for col in stat_cols:
                    stats = features_df[col].describe()
                    logger.info(
                        "  %-25s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                        col, stats["mean"], stats["std"], stats["min"], stats["max"]
                    )
                logger.info("  %s", "-" * 70)

                logger.info("  ✅ Feature engineering complete: %d nodes × %d features",
                            len(features_df), len(stat_cols))
                return features_df

            logger.warning(
                "  C++ node mismatch (%d vs %d), using NetworkX",
                len(cpp_nodes), len(graph_nodes)
            )
    except Exception as exc:
        logger.warning("  C++ backend unavailable, using NetworkX (%s)", exc)

    logger.info("  Using NetworkX features (C++ not available)")

    # Compute each feature set
    degree_features = compute_degree(G)
    clustering = compute_clustering(G)
    pagerank = compute_pagerank(G)
    betweenness = compute_betweenness(G)

    # Assemble into DataFrame
    records = []
    for node in sorted(G.nodes()):
        records.append({
            "node_id": node,
            "in_degree": degree_features[node]["in_degree"],
            "out_degree": degree_features[node]["out_degree"],
            "total_degree": degree_features[node]["total_degree"],
            "clustering_coefficient": clustering.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
            "betweenness_centrality": betweenness.get(node, 0.0),
        })

    features_df = pd.DataFrame(records)

    # Save to disk
    config.ensure_dirs()
    features_df.to_csv(config.NODE_FEATURES_PATH, index=False)
    logger.info("  Saved feature matrix to %s", config.NODE_FEATURES_PATH)

    # Print distribution statistics
    logger.info("\n  Feature Distribution Statistics:")
    logger.info("  %s", "-" * 70)
    stat_cols = ["in_degree", "out_degree", "total_degree",
                 "clustering_coefficient", "pagerank", "betweenness_centrality"]
    for col in stat_cols:
        stats = features_df[col].describe()
        logger.info(
            "  %-25s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
            col, stats["mean"], stats["std"], stats["min"], stats["max"]
        )
    logger.info("  %s", "-" * 70)

    logger.info("  ✅ Feature engineering complete: %d nodes × %d features",
                len(features_df), len(stat_cols))

    return features_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data, build_graph, set_seeds

    set_seeds()
    config.ensure_dirs()

    df, account_labels = load_data()
    G = build_graph(df)
    features_df = compute_features(G)

    logger.info("\n✅ Feature engineering standalone test complete!")
    logger.info("  Shape: %s", features_df.shape)
    print(features_df.head(10).to_string())


# Backward-compatible aliases.
compute_clustering_coefficient = compute_clustering
compute_betweenness_centrality = compute_betweenness
compute_all_features = compute_features
