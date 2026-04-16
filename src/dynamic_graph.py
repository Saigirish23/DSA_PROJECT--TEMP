"""Dynamic graph data structures for incremental fraud feature updates."""

from collections import defaultdict, deque

import pandas as pd


class IncrementalAdjacencyList:
    """
    Dynamic adjacency list that updates degree counts incrementally.
    Adding edge (u,v): O(1) amortized
    Degree query: O(1)
    vs static rebuild: O(V+E) each time
    """

    def __init__(self):
        self.out_edges = defaultdict(set)  # u -> set of v
        self.in_edges = defaultdict(set)  # v -> set of u
        self.out_degree = defaultdict(int)
        self.in_degree = defaultdict(int)

    def add_edge(self, u, v):
        if v not in self.out_edges[u]:
            self.out_edges[u].add(v)
            self.in_edges[v].add(u)
            self.out_degree[u] += 1
            self.in_degree[v] += 1

    def remove_edge(self, u, v):
        if v in self.out_edges[u]:
            self.out_edges[u].discard(v)
            self.in_edges[v].discard(u)
            self.out_degree[u] -= 1
            self.in_degree[v] -= 1

    def get_degree(self, node):
        return self.out_degree[node] + self.in_degree[node]

    def get_neighbors(self, node):
        return self.out_edges[node] | self.in_edges[node]


class FenwickTree:
    """
    Fenwick Tree for O(log n) prefix sum queries on transaction amounts.
    Used for temporal aggregation: 'total amount in last T time steps'
    Point update: O(log n)  vs  O(n) naive scan
    Range query:  O(log n)  vs  O(n) naive scan
    """

    def __init__(self, size):
        self.n = size
        self.tree = [0.0] * (size + 1)

    def update(self, i, delta):
        # add delta at position i (0-indexed API)
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        # prefix sum [0..i]
        i += 1
        total = 0.0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def range_query(self, l, r):
        # sum in range [l..r]
        if r < l:
            return 0.0
        if l == 0:
            return self.query(r)
        return self.query(r) - self.query(l - 1)


class SlidingWindowGraph:
    """
    Maintains a graph over a fixed time window [t - W, t].
    Old edges expire automatically when they fall outside the window.
    Uses a deque for O(1) expiry.
    """

    def __init__(self, window_size):
        self.window = window_size
        self.adj = IncrementalAdjacencyList()
        self.history = deque()  # (timestamp, u, v)

    def add_transaction(self, u, v, timestamp):
        self.adj.add_edge(u, v)
        self.history.append((timestamp, u, v))
        self._expire(timestamp)

    def _expire(self, current_time):
        while self.history and self.history[0][0] < current_time - self.window:
            _, u, v = self.history.popleft()
            self.adj.remove_edge(u, v)

    def get_degree(self, node):
        return self.adj.get_degree(node)


class IncrementalPageRank:
    """
    Approximates PageRank update after edge addition without full recomputation.
    """

    def __init__(self, damping=0.85, max_iter=10):
        self.d = damping
        self.max_iter = max_iter
        self.ranks = defaultdict(float)
        self.adj = IncrementalAdjacencyList()

    def add_edge(self, u, v):
        self.adj.add_edge(u, v)
        self._local_update(u, v)

    def _local_update(self, u, v):
        affected = {u, v} | self.adj.get_neighbors(u) | self.adj.get_neighbors(v)
        n = max(len(self.ranks), 1)

        for _ in range(self.max_iter):
            for node in affected:
                incoming = self.adj.in_edges[node]
                rank_sum = sum(
                    self.ranks[src] / max(self.adj.out_degree[src], 1) for src in incoming
                )
                self.ranks[node] = (1 - self.d) / n + self.d * rank_sum

    def get_rank(self, node):
        return self.ranks.get(node, 1.0 / max(len(self.ranks), 1))


class IncrementalClustering:
    """
    Maintains approximate clustering coefficients incrementally.
    """

    def __init__(self):
        self.adj = defaultdict(set)
        self.triangles = defaultdict(int)
        self.clustering = defaultdict(float)

    def add_edge(self, u, v):
        common = self.adj[u] & self.adj[v]
        new_triangles = len(common)

        self.triangles[u] += new_triangles
        self.triangles[v] += new_triangles
        for w in common:
            self.triangles[w] += 1

        self.adj[u].add(v)
        self.adj[v].add(u)
        self._update_cc(u)
        self._update_cc(v)

    def _update_cc(self, node):
        d = len(self.adj[node])
        if d < 2:
            self.clustering[node] = 0.0
        else:
            self.clustering[node] = (2 * self.triangles[node]) / (d * (d - 1))

    def get_clustering(self, node):
        return self.clustering.get(node, 0.0)


class DynamicFraudGraph:
    """
    Unified dynamic graph combining incremental structures for streaming features.
    """

    def __init__(self, window_size=7):
        self.window_graph = SlidingWindowGraph(window_size)
        self.inc_pagerank = IncrementalPageRank()
        self.inc_clustering = IncrementalClustering()
        self.fenwick_trees = {}  # node -> FenwickTree for amounts
        self.time_slots = 100

    def add_transaction(self, sender, receiver, amount, timestamp):
        self.window_graph.add_transaction(sender, receiver, timestamp)
        self.inc_pagerank.add_edge(sender, receiver)
        self.inc_clustering.add_edge(sender, receiver)

        slot = int(timestamp % self.time_slots)
        for node in [sender, receiver]:
            if node not in self.fenwick_trees:
                self.fenwick_trees[node] = FenwickTree(self.time_slots)
            self.fenwick_trees[node].update(slot, amount)

    def get_features(self, node):
        degree = self.window_graph.get_degree(node)
        clustering = self.inc_clustering.get_clustering(node)
        pagerank = self.inc_pagerank.get_rank(node)
        return {
            "node_id": str(node),
            "in_degree": 0,
            "out_degree": 0,
            "total_degree": degree,
            "clustering_coefficient": clustering,
            "pagerank": pagerank,
            "betweenness_centrality": 0.0,
            "degree": degree,
            "clustering": clustering,
            "betweenness": 0.0,
        }

    def get_all_features(self):
        all_nodes = set(self.window_graph.adj.out_degree.keys()) | set(
            self.window_graph.adj.in_degree.keys()
        )
        rows = [self.get_features(n) for n in sorted(all_nodes)]
        return pd.DataFrame(rows)
