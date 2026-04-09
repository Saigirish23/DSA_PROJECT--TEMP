"""
gnn_model.py — GCN Model Definition

Implements a 2-layer Graph Convolutional Network (GCN) for node-level
binary classification (fraud vs. normal).

Architecture:
    Input (num_features) → GCNConv(hidden_dim) → ReLU → Dropout(0.5)
                         → GCNConv(num_classes) → LogSoftmax → Output (2 classes)

The GCN layer performs message passing:
    h_v^(l+1) = σ( Σ (1/√(d_u · d_v)) · W^(l) · h_u^(l) )
                  u ∈ N(v) ∪ {v}

where d_u and d_v are node degrees, W is a learnable weight matrix,
and σ is a nonlinear activation (ReLU).

All hyperparameters (hidden_dim, dropout, num_layers) are sourced from config.py.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import config

logger = config.setup_logging(__name__)


class FraudGCN(torch.nn.Module):
    """
    2-Layer Graph Convolutional Network for fraud detection.

    This model learns to classify nodes (accounts) as fraudulent or normal
    by propagating and aggregating information from neighboring nodes in
    the transaction graph.

    Args:
        num_features (int): Number of input features per node.
        hidden_dim (int): Dimension of the hidden layer. Default from config.
        num_classes (int): Number of output classes. Default 2 (normal, fraud).
        dropout (float): Dropout probability. Default from config.
    """

    def __init__(self, num_features, hidden_dim=None, num_classes=None, dropout=None):
        super(FraudGCN, self).__init__()

        self.hidden_dim = hidden_dim or config.GNN_HIDDEN_DIM
        self.num_classes = num_classes or config.GNN_NUM_CLASSES
        self.dropout_rate = dropout or config.GNN_DROPOUT

        # Layer 1: Input → Hidden
        # GCNConv applies spectral convolution (Kipf & Welling, 2017)
        # It aggregates features from 1-hop neighbors, weighted by degree normalization
        self.conv1 = GCNConv(num_features, self.hidden_dim)

        # Layer 2: Hidden → Output
        # Second GCN layer captures 2-hop neighborhood information
        # This means each node's prediction is influenced by nodes up to 2 edges away
        self.conv2 = GCNConv(self.hidden_dim, self.num_classes)

        logger.info("FraudGCN initialized:")
        logger.info("  Input: %d features → Hidden: %d → Output: %d classes",
                     num_features, self.hidden_dim, self.num_classes)
        logger.info("  Dropout: %.2f", self.dropout_rate)

    def forward(self, x, edge_index):
        """
        Forward pass through the GCN.

        Args:
            x (torch.Tensor): Node feature matrix [N, F].
            edge_index (torch.Tensor): Edge list in COO format [2, E].

        Returns:
            torch.Tensor: Log-softmax probabilities [N, num_classes].
        """
        # Layer 1: Graph convolution → ReLU activation → Dropout
        # ReLU introduces nonlinearity, allowing the model to learn complex patterns
        # Dropout regularizes by randomly zeroing features during training
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # Layer 2: Graph convolution → LogSoftmax for classification
        # LogSoftmax is used with NLLLoss (equivalent to CrossEntropyLoss with raw logits)
        h = self.conv2(h, edge_index)

        return F.log_softmax(h, dim=1)

    def get_embeddings(self, x, edge_index):
        """
        Extract learned node embeddings from the hidden layer.

        Useful for visualization and for the hybrid model (Strategy A).

        Args:
            x (torch.Tensor): Node feature matrix [N, F].
            edge_index (torch.Tensor): Edge list in COO format [2, E].

        Returns:
            torch.Tensor: Node embeddings [N, hidden_dim].
        """
        self.eval()
        with torch.no_grad():
            h = self.conv1(x, edge_index)
            h = F.relu(h)
        return h


if __name__ == "__main__":
    # Quick test: build model and run a forward pass with dummy data
    import sys
    sys.path.insert(0, ".")

    num_features = 6
    model = FraudGCN(num_features)

    # Create dummy data
    x = torch.randn(10, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Forward pass
    out = model(x, edge_index)
    logger.info("Output shape: %s", out.shape)
    logger.info("Output (first 3 nodes): %s", out[:3])
    logger.info("\n✅ GCN model test passed!")
