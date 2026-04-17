"""AML-focused PyTorch Geometric model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, JumpingKnowledge, TransformerConv

import config

logger = config.setup_logging(__name__)


class AMLDetector(nn.Module):
    """
    Architecture designed specifically for sparse AML graphs.

    Design decisions:
    1. GATv2Conv: dynamic attention (not static) for sparse local neighborhoods.
    2. TransformerConv: global-style attention mixing for disconnected components.
    3. JK (LSTM): preserve multi-scale signals from all message-passing depths.
    4. Edge feature injection: use amount/time/direction to avoid topology-only bias.
    5. BatchNorm after each conv: stabilize optimization across sparse components.
    6. Residual links: keep gradients and prevent deep sparse-network collapse.
    7. 3-layer classifier head: decouple embedding learning from final decision boundary.
    """

    def __init__(
        self,
        num_features,
        hidden_dim=128,
        num_classes=2,
        heads=4,
        dropout=0.35,
        edge_dim=32,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.edge_dim = int(edge_dim)

        # Project raw node features into a stable representation before attention layers.
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

        # Encode edge attributes so message passing can condition on transaction metadata.
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ELU(),
            nn.Linear(32, self.edge_dim),
        )

        # Dynamic attention for first-hop sparse neighborhood modeling.
        self.conv1 = GATv2Conv(
            in_channels=256,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            add_self_loops=False,
        )
        self.bn1 = nn.BatchNorm1d(self.hidden_dim * self.heads)

        # Second dynamic attention block with residual learning over Layer 1 output.
        self.conv2 = GATv2Conv(
            in_channels=self.hidden_dim * self.heads,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            add_self_loops=False,
        )
        self.bn2 = nn.BatchNorm1d(self.hidden_dim * self.heads)

        # Transformer-style aggregation introduces broader context than local GAT alone.
        self.conv3 = TransformerConv(
            in_channels=self.hidden_dim * self.heads,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            edge_dim=self.edge_dim,
            beta=True,
        )
        self.bn3 = nn.BatchNorm1d(self.hidden_dim * self.heads)

        # LSTM-based JK keeps useful information from shallow and deeper layers together.
        self.jk = JumpingKnowledge(mode="lstm", channels=self.hidden_dim * self.heads, num_layers=3)
        self.jk_proj = nn.Linear(self.hidden_dim * self.heads, self.hidden_dim)

        # Dedicated classifier head separates representation learning from final classification.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, self.num_classes),
        )

        logger.info("AMLDetector initialized")
        logger.info(
            "  Input=%d Hidden=%d Heads=%d EdgeDim=%d Classes=%d",
            num_features,
            self.hidden_dim,
            self.heads,
            self.edge_dim,
            self.num_classes,
        )

    def _encode_edge_attr(self, edge_attr, edge_index):
        """Prepare optional edge features for attention layers with graceful fallback."""
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 3, device=edge_index.device, dtype=torch.float32)

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        # Canonicalize to [amount, temporal_delta, direction_flag].
        if edge_attr.size(1) < 3:
            pad = torch.zeros(
                edge_attr.size(0),
                3 - edge_attr.size(1),
                device=edge_attr.device,
                dtype=edge_attr.dtype,
            )
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif edge_attr.size(1) > 3:
            edge_attr = edge_attr[:, :3]

        # If edge_attr length mismatches edge count, skip rather than crash.
        if edge_attr.size(0) != edge_index.size(1):
            edge_attr = torch.zeros(edge_index.size(1), 3, device=edge_index.device, dtype=torch.float32)

        return self.edge_mlp(edge_attr.float())

    def _forward_layers(self, x, edge_index, edge_attr=None):
        x0 = self.input_proj(x)
        e = self._encode_edge_attr(edge_attr, edge_index)

        l1 = self.conv1(x0, edge_index, e)
        l1 = self.bn1(l1)
        l1 = F.elu(l1)
        l1 = F.dropout(l1, p=self.dropout, training=self.training)

        l2 = self.conv2(l1, edge_index, e)
        l2 = self.bn2(l2)
        l2 = F.elu(l2)
        l2 = F.dropout(l2, p=self.dropout, training=self.training)
        l2 = l2 + l1

        l3 = self.conv3(l2, edge_index, e)
        l3 = self.bn3(l3)
        l3 = F.elu(l3)
        l3 = F.dropout(l3, p=self.dropout, training=self.training)

        jk_out = self.jk([l1, l2, l3])
        emb = self.jk_proj(jk_out)
        return emb

    def forward(self, x, edge_index, edge_attr=None):
        emb = self._forward_layers(x, edge_index, edge_attr=edge_attr)
        logits = self.classifier(emb)
        return logits

    def predict_proba(self, x, edge_index, edge_attr=None):
        logits = self.forward(x, edge_index, edge_attr=edge_attr)
        return F.softmax(logits, dim=-1)

    def get_embeddings(self, x, edge_index, edge_attr=None):
        self.eval()
        with torch.no_grad():
            emb = self._forward_layers(x, edge_index, edge_attr=edge_attr)
        return emb


# Backward compatibility aliases expected by existing code paths.
FraudGCN = AMLDetector
GCN = AMLDetector


if __name__ == "__main__":
    model = AMLDetector(num_features=6, hidden_dim=128)
    x = torch.randn(100, 6)
    edge_index = torch.randint(0, 100, (2, 200), dtype=torch.long)
    edge_attr = torch.randn(200, 3)

    out = model(x, edge_index, edge_attr=edge_attr)
    probs = model.predict_proba(x, edge_index, edge_attr=edge_attr)
    emb = model.get_embeddings(x, edge_index, edge_attr=edge_attr)

    print("out", out.shape)
    print("probs", probs.shape)
    print("emb", emb.shape)
