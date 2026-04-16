/*
 * ============================================================
 *  Fraud Detection in Transaction Graphs — DSA Core (C++)
 *  Algorithms: Degree, Clustering Coefficient, PageRank,
 *              Betweenness Centrality
 * ============================================================
 */

#include "graph_algorithms.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_set>

using namespace std;

namespace {

vector<string> split_csv_line(const string& line) {
    vector<string> out;
    string cur;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    out.push_back(cur);

    for (string& token : out) {
        if (!token.empty() && token.front() == '"') token.erase(token.begin());
        if (!token.empty() && token.back() == '"') token.pop_back();
    }
    return out;
}

bool read_edge_list(const string& edge_file,
                    vector<pair<string, string>>& edges,
                    vector<string>& nodes) {
    ifstream in(edge_file);
    if (!in) {
        cerr << "Failed to open edge file: " << edge_file << '\n';
        return false;
    }

    string header;
    if (!getline(in, header)) {
        cerr << "Empty edge file: " << edge_file << '\n';
        return false;
    }

    vector<string> cols = split_csv_line(header);
    int sender_idx = -1;
    int receiver_idx = -1;

    for (int i = 0; i < (int)cols.size(); i++) {
        string c = cols[i];
        if (c == "sender_id" || c == "from" || c == "src") sender_idx = i;
        if (c == "receiver_id" || c == "to" || c == "dst") receiver_idx = i;
    }

    if (sender_idx < 0 || receiver_idx < 0) {
        cerr << "Could not find sender/receiver columns in header\n";
        return false;
    }

    unordered_set<string> node_set;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> toks = split_csv_line(line);
        if (sender_idx >= (int)toks.size() || receiver_idx >= (int)toks.size()) continue;

        string u = toks[sender_idx];
        string v = toks[receiver_idx];
        if (u.empty() || v.empty()) continue;

        edges.push_back({u, v});
        node_set.insert(u);
        node_set.insert(v);
    }

    nodes.assign(node_set.begin(), node_set.end());
    sort(nodes.begin(), nodes.end());
    return true;
}

}  // namespace

unordered_map<string, int> compute_degree(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes) {
    unordered_map<string, int> degree;
    degree.reserve(nodes.size());
    for (const string& n : nodes) degree[n] = 0;

    for (const auto& e : edges) {
        degree[e.first] += 1;
        degree[e.second] += 1;
    }
    return degree;
}

unordered_map<string, double> compute_clustering(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes) {
    unordered_map<string, unordered_set<string>> adj;
    adj.reserve(nodes.size());
    for (const string& n : nodes) adj[n] = {};

    for (const auto& e : edges) {
        const string& u = e.first;
        const string& v = e.second;
        if (u == v) continue;
        adj[u].insert(v);
        adj[v].insert(u);
    }

    unordered_map<string, double> clustering;
    clustering.reserve(nodes.size());

    for (const string& u : nodes) {
        const auto& nu = adj[u];
        int k = (int)nu.size();
        if (k < 2) {
            clustering[u] = 0.0;
            continue;
        }

        vector<string> nvec(nu.begin(), nu.end());
        long long links = 0;
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                if (adj[nvec[i]].count(nvec[j])) links += 1;
            }
        }

        double possible = (double)k * (k - 1) / 2.0;
        clustering[u] = (possible > 0.0) ? (links / possible) : 0.0;
    }

    return clustering;
}

unordered_map<string, double> compute_pagerank(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes,
    int iterations,
    double damping,
    double tol) {
    unordered_map<string, vector<string>> out_adj;
    unordered_map<string, vector<string>> in_adj;
    out_adj.reserve(nodes.size());
    in_adj.reserve(nodes.size());

    for (const string& n : nodes) {
        out_adj[n] = {};
        in_adj[n] = {};
    }

    for (const auto& e : edges) {
        out_adj[e.first].push_back(e.second);
        in_adj[e.second].push_back(e.first);
    }

    int n_nodes = (int)nodes.size();
    unordered_map<string, double> rank;
    unordered_map<string, double> new_rank;
    rank.reserve(nodes.size());
    new_rank.reserve(nodes.size());

    double init = (n_nodes > 0) ? 1.0 / n_nodes : 0.0;
    for (const string& n : nodes) rank[n] = init;

    for (int it = 0; it < iterations; it++) {
        double base = (1.0 - damping) / n_nodes;
        for (const string& v : nodes) new_rank[v] = base;

        double dangling_sum = 0.0;
        for (const string& u : nodes) {
            int out_deg = (int)out_adj[u].size();
            if (out_deg == 0) dangling_sum += rank[u];
        }
        double dangling_contrib = damping * dangling_sum / n_nodes;
        for (const string& v : nodes) new_rank[v] += dangling_contrib;

        for (const string& v : nodes) {
            double incoming = 0.0;
            for (const string& u : in_adj[v]) {
                int out_deg = (int)out_adj[u].size();
                if (out_deg > 0) incoming += rank[u] / out_deg;
            }
            new_rank[v] += damping * incoming;
        }

        double diff = 0.0;
        for (const string& v : nodes) {
            diff += abs(new_rank[v] - rank[v]);
        }
        rank.swap(new_rank);
        if (diff < tol) break;
    }

    return rank;
}

unordered_map<string, double> compute_betweenness(
    const vector<pair<string, string>>& edges,
    const vector<string>& nodes) {
    unordered_map<string, vector<string>> adj;
    adj.reserve(nodes.size());
    for (const string& n : nodes) adj[n] = {};

    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
    }

    unordered_map<string, double> bc;
    bc.reserve(nodes.size());
    for (const string& n : nodes) bc[n] = 0.0;

    for (const string& s : nodes) {
        vector<string> stack;
        unordered_map<string, vector<string>> pred;
        unordered_map<string, int> dist;
        unordered_map<string, double> sigma;
        unordered_map<string, double> delta;

        pred.reserve(nodes.size());
        dist.reserve(nodes.size());
        sigma.reserve(nodes.size());
        delta.reserve(nodes.size());

        for (const string& v : nodes) {
            pred[v] = {};
            dist[v] = -1;
            sigma[v] = 0.0;
            delta[v] = 0.0;
        }

        dist[s] = 0;
        sigma[s] = 1.0;
        queue<string> q;
        q.push(s);

        while (!q.empty()) {
            string v = q.front();
            q.pop();
            stack.push_back(v);

            for (const string& w : adj[v]) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    q.push(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        while (!stack.empty()) {
            string w = stack.back();
            stack.pop_back();
            for (const string& v : pred[w]) {
                if (sigma[w] > 0) {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if (w != s) bc[w] += delta[w];
        }
    }

    int n = (int)nodes.size();
    double norm = (n > 2) ? 1.0 / ((n - 1.0) * (n - 2.0)) : 0.0;
    if (norm > 0) {
        for (const string& n_id : nodes) bc[n_id] *= norm;
    }

    return bc;
}

void run(const string& edge_file, int num_iterations) {
    vector<pair<string, string>> edges;
    vector<string> nodes;
    if (!read_edge_list(edge_file, edges, nodes)) {
        throw runtime_error("Unable to read edge file");
    }

    auto degree = compute_degree(edges, nodes);
    auto clustering = compute_clustering(edges, nodes);
    auto pagerank = compute_pagerank(edges, nodes, num_iterations, 0.85);
    auto betweenness = compute_betweenness(edges, nodes);

    cout << "node_id,degree,clustering,pagerank,betweenness\n";
    cout << fixed << setprecision(6);
    for (const string& node : nodes) {
        cout << node << ','
             << degree[node] << ','
             << clustering[node] << ','
             << pagerank[node] << ','
             << betweenness[node] << '\n';
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./graph_algorithms <edge_file.csv> [iterations]\n";
        return 1;
    }

    string edge_file = argv[1];
    int iterations = 100;
    if (argc >= 3) {
        try {
            iterations = stoi(argv[2]);
        } catch (...) {
            cerr << "Invalid iterations argument\n";
            return 1;
        }
    }

    try {
        run(edge_file, iterations);
        return 0;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}

// ── Fenwick Tree (BIT) ────────────────────────────────────────────────────
// O(log n) point update and prefix query
// Used for temporal transaction amount aggregation
struct FenwickTree {
    int n;
    vector<double> tree;

    FenwickTree(int n) : n(n), tree(n + 1, 0.0) {}

    void update(int i, double delta) {
        for (++i; i <= n; i += i & (-i))
            tree[i] += delta;
    }

    double query(int i) {
        double s = 0;
        for (++i; i > 0; i -= i & (-i))
            s += tree[i];
        return s;
    }

    double range_query(int l, int r) {
        return l == 0 ? query(r) : query(r) - query(l - 1);
    }
};

// ── Incremental Degree Tracker ────────────────────────────────────────────
// O(1) per update vs O(V+E) full rebuild
struct IncrementalDegree {
    unordered_map<string, int> in_deg, out_deg;

    void add_edge(const string& u, const string& v) {
        out_deg[u]++;
        in_deg[v]++;
    }

    int degree(const string& node) {
        return in_deg[node] + out_deg[node];
    }
};

// ── Sliding Window ────────────────────────────────────────────────────────
// Maintains active edges within [t - W, t]
struct SlidingWindow {
    int window_size;
    deque<tuple<int, string, string>> history;
    IncrementalDegree deg_tracker;

    SlidingWindow(int w) : window_size(w) {}

    void add_transaction(const string& u, const string& v, int timestamp) {
        deg_tracker.add_edge(u, v);
        history.push_back({timestamp, u, v});
        expire(timestamp);
    }

    void expire(int current_time) {
        while (!history.empty()) {
            auto [t, u, v] = history.front();
            if (t < current_time - window_size) {
                deg_tracker.in_deg[v]--;
                deg_tracker.out_deg[u]--;
                history.pop_front();
            } else break;
        }
    }
};
