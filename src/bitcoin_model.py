"""Deeper GNN model for full-feature Elliptic Bitcoin tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, SAGEConv


class EllipticGNN(nn.Module):
    """3-layer GNN with residual and BN for 166-feature Elliptic input."""

    def __init__(self, num_features, hidden=128, out=2, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.conv1 = GCNConv(hidden, hidden)
        self.bn1 = BatchNorm(hidden)

        self.conv2 = GATConv(hidden, hidden // 2, heads=2, dropout=dropout, concat=True)
        self.bn2 = BatchNorm(hidden)

        self.conv3 = SAGEConv(hidden, hidden // 2)
        self.bn3 = BatchNorm(hidden // 2)

        self.res_proj = nn.Linear(hidden, hidden // 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden // 2, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, out),
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        residual = x

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x + self.res_proj(residual))

        return self.classifier(x)

    def predict_proba(self, x, edge_index):
        logits = self.forward(x, edge_index)
        return torch.softmax(logits, dim=1)
