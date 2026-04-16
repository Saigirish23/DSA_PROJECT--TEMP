# Fraud Detection in Transaction Graphs using GNNs and Graph Heuristics

A hybrid machine learning pipeline that combines classical graph algorithms (DSA) with Graph Neural Networks (GNNs) for identifying fraudulent accounts in financial transaction networks.

## Problem Statement

Existing fraud detection models assume static graphs, but real transaction 
networks evolve continuously. Recomputing graph features from scratch is 
expensive and impractical in real-time systems:

- Full degree recomputation:    O(V+E)   per update
- Full PageRank recomputation:  O(k·(V+E)) per update  
- Full clustering recomputation: O(V·d²) per update

### Novel Contribution: Dynamic Graph Data Structures

This project introduces a dynamic graph layer that maintains features 
**incrementally** as new transactions arrive:

| Component | Structure | Complexity |
|---|---|---|
| Degree tracking | Incremental adjacency list | O(1) per edge |
| Amount aggregation | Fenwick tree (BIT) | O(log n) update/query |
| Recent activity | Sliding window + deque | O(1) expiry |
| PageRank | Local neighborhood update | O(k·deg) per edge |
| Clustering | Triangle counting on insert | O(d) per edge |

**Key innovation:** Real-time fraud detection without rebuilding the graph.
Complexity reduced from O(V+E) per update → O(1) or O(log n) incremental updates.

## Architecture

```text
               [ Raw Transaction CSV ]
                         |
                         v
             (Phase 2: Graph Builder)
            [ NetworkX DiGraph (V, E) ]
                         |
                         v
      (Phase 3: Dynamic Graph Layer - Optional)
 [ Incremental Adj | Fenwick BIT | Window | Local PR ]
                         |
           +-------------+-------------+
           |                           |
           v                           v
 (Static Features Path)        (Dynamic Features Path)
 O(V+E) / O(V·d²) / ...        O(1) / O(log n) incremental
           |                           |
           +-------------+-------------+
                         |
                         v
                (Phase 4: Heuristics)
               [ Rule-Based Scorer ]
                         |
                         v
                (Phase 5: PyG Data Prep)
               [ Normalized Tensors ]
                         |
                         v
                (Phase 6: GNN Training)
                 [ FraudGCN Model ]
                         |
                         v
                (Phase 7: Hybrid Fusion)
         [ α · GCN_Prob + (1-α) · Heuristic ]
                         |
                         v
                (Phase 8: Evaluation)
           [ Metrics, ROC Curves, Visuals ]
```

## Dataset Description

The project uses a synthetic but highly realistic dataset mimicking financial fraud patterns:
- **Accounts (Nodes):** ~500 total, ~10% fraudulent.
- **Transactions (Edges):** ~2,000 transactions over 30 days.
- **Node Features:** Purely structural (computed from the graph connectivity).
- **Patterns:** Fraud accounts exhibit bursty transaction patterns, higher median amounts (lognormal), and connections to other fraudulent accounts (fraud rings).

## Getting Started

### Prerequisites
- Python >= 3.10
- GPU recommended (with CUDA 13.0) but fully functional on CPU

### Installation

1. Clone the repository and navigate into it.
2. Install the pinned dependencies:
```bash
pip install -r requirements.txt
```
*(Note: PyTorch Geometric dependencies are sensitive to CUDA versions. The `requirements.txt` assumes PyTorch 2.11 + cu130).*

### Running the Pipeline

To run the pipeline end-to-end (Phases 2 through 8):

```bash
python main.py
```

This will automatically:
1. Generate synthetic data.
2. Compute structural network features.
3. Apply heuristic labeling.
4. Train the GCN model.
5. Search for the optimal late-fusion blend ($\alpha$).
6. Generate visualizations in `outputs/plots/` and `outputs/results/`.

## Results Summary

Comparison of the different modeling strategies evaluated on unseen test nodes:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Heuristic Baseline** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **GNN Only** | 0.8300 | 0.4583 | 0.7333 | 0.5641 |
| **Hybrid (Strategy A)** | 0.8000 | 0.4074 | 0.7333 | 0.5238 |
| **Hybrid (Strategy B, α=0.5)** | 0.8400 | 0.4828 | 0.9333 | 0.6364 |

*Note: Strategy A concatenates the heuristic score as a GNN input feature. Strategy B takes a weighted average of model outputs. In our tests, late fusion (Strategy B) consistently outperforms both pure GNN and early fusion (Strategy A), approaching the oracle performance of the heuristic score (which acts as ground truth for training here).*

## Key Design Decisions & Trade-offs

1. **Class-Weighted NLLLoss:** Fraud is severely imbalanced (10-15%). Without applying inverse-frequency weighting, the GNN rapidly collapsed to predicting 'normal' for all accounts to achieve 85% accuracy.
2. **PyG vs. DGL:** Selected PyTorch Geometric for its straightforward message passing interface and better compatibility with standard networkx inputs.
3. **Directed Graph Formulation:** Transaction graphs are directed. The semantic difference between money-in vs. money-out is massive.
4. **Heuristic as Ground-Truth Substitute:** In many real-world scenarios, labeled fraud data is scarce. Bootstrapping labels with domain-expert heuristics allowed us to train the GNN, which then learned continuous structural embeddings that smoothed out rigid heuristic rules.
