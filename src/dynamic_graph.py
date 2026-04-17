"""Dynamic graph data structures for snapshot-based feature computation."""

from collections import defaultdict, deque

import networkx as nx
import pandas as pd


CANONICAL_FEATURE_COLUMNS = ["degree", "in_degree", "out_degree", "clustering", "pagerank"]


class DynamicFraudGraph:
    """
    Dynamic graph with O(1) per-edge structural updates.

    Edge updates only maintain window structure and degree counters.
    Snapshot math (PageRank + clustering) is computed once per snapshot.
    """

    def __init__(self, window_size=7):
        self.window = int(window_size)
        self.active_graph = nx.DiGraph()

        # Multiplicity map avoids repeated-edge expiry bugs in sliding windows.
        self.edge_counts = defaultdict(int)  # (u, v) -> multiplicity

        # O(1) degree lookups for feature extraction.
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)

        # Transaction history for expiry: (timestamp, u, v)
        self.history = deque()
        self.current_time = None

        # Snapshot-level metrics (recomputed lazily or explicitly).
        self.current_pagerank = {}
        self.current_clustering = {}
        self._snapshot_dirty = True

    def _increment_edge(self, u, v):
        """Add one edge copy; update topology and degrees only on first copy."""
        key = (u, v)
        prev = self.edge_counts[key]
        self.edge_counts[key] = prev + 1

        if prev == 0:
            self.active_graph.add_edge(u, v)
            self.out_degree[u] += 1
            self.in_degree[v] += 1

    def _decrement_edge(self, u, v):
        """Remove one edge copy; update topology and degrees when last copy expires."""
        key = (u, v)
        count = self.edge_counts.get(key, 0)
        if count <= 0:
            return

        if count == 1:
            del self.edge_counts[key]

            if self.active_graph.has_edge(u, v):
                self.active_graph.remove_edge(u, v)

            if self.out_degree.get(u, 0) > 0:
                self.out_degree[u] -= 1
                if self.out_degree[u] == 0:
                    del self.out_degree[u]

            if self.in_degree.get(v, 0) > 0:
                self.in_degree[v] -= 1
                if self.in_degree[v] == 0:
                    del self.in_degree[v]

            # Trim isolated nodes to keep snapshot computations compact.
            if self.active_graph.has_node(u):
                if self.active_graph.in_degree(u) == 0 and self.active_graph.out_degree(u) == 0:
                    self.active_graph.remove_node(u)
            if self.active_graph.has_node(v):
                if self.active_graph.in_degree(v) == 0 and self.active_graph.out_degree(v) == 0:
                    self.active_graph.remove_node(v)
        else:
            self.edge_counts[key] = count - 1

    def _expire(self, current_time):
        threshold = int(current_time) - self.window
        while self.history and self.history[0][0] < threshold:
            _, u, v = self.history.popleft()
            self._decrement_edge(u, v)

    def add_transaction(self, sender, receiver, amount, timestamp):
        """O(1) structural update for one transaction event."""
        del amount  # Amount is not needed for structural snapshot features.

        u = str(sender)
        v = str(receiver)
        if u == v:
            return

        ts = int(timestamp)
        if self.current_time is None or ts > self.current_time:
            self.current_time = ts

        self._increment_edge(u, v)
        self.history.append((ts, u, v))
        self._expire(self.current_time)
        self._snapshot_dirty = True

    def remove_transaction(self, sender, receiver):
        """Explicit structural edge removal by one transaction copy."""
        u = str(sender)
        v = str(receiver)
        self._decrement_edge(u, v)
        self._snapshot_dirty = True

    def calculate_snapshot_pagerank(self):
        """Compute snapshot metrics once from current active window graph."""
        if self.active_graph.number_of_nodes() == 0:
            self.current_pagerank = {}
            self.current_clustering = {}
            self._snapshot_dirty = False
            return self.current_pagerank

        try:
            self.current_pagerank = nx.pagerank(self.active_graph, alpha=0.85)
        except Exception:
            # Safe fallback if power iteration fails.
            n = self.active_graph.number_of_nodes()
            uniform = 1.0 / float(n) if n else 0.0
            self.current_pagerank = {node: uniform for node in self.active_graph.nodes()}

        try:
            self.current_clustering = nx.clustering(self.active_graph.to_undirected())
        except Exception:
            self.current_clustering = {}

        self._snapshot_dirty = False
        return self.current_pagerank

    def get_features(self, node):
        node = str(node)
        in_deg = int(self.in_degree.get(node, 0))
        out_deg = int(self.out_degree.get(node, 0))
        degree = in_deg + out_deg

        return {
            "node_id": node,
            "degree": degree,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "clustering": float(self.current_clustering.get(node, 0.0)),
            "pagerank": float(self.current_pagerank.get(node, 0.0)),
        }

    def get_all_features(self):
        # Safety net for direct callers that did not invoke snapshot compute explicitly.
        if self._snapshot_dirty:
            self.calculate_snapshot_pagerank()

        all_nodes = set(self.active_graph.nodes()) | set(self.in_degree.keys()) | set(self.out_degree.keys())
        rows = [self.get_features(n) for n in sorted(all_nodes)]
        if not rows:
            return pd.DataFrame(columns=["node_id"] + CANONICAL_FEATURE_COLUMNS)

        features_df = pd.DataFrame(rows)
        return features_df[["node_id"] + CANONICAL_FEATURE_COLUMNS]
