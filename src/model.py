"""
model.py
Dual-branch GNN + CNN model for drug-target binding affinity prediction.
EGN6217 — Drug-Target Binding Affinity Prediction
Sathyadharini Srinivasan | Spring 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


class DrugEncoder(nn.Module):
    """
    Graph Neural Network encoder for drug molecules.
    Input:  molecular graph (atoms = nodes, bonds = edges)
    Output: 128-dimensional drug embedding vector
    """
    def __init__(self, node_feat_dim=5, hidden_dim=64, out_dim=128):
        super(DrugEncoder, self).__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, out_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3   = nn.BatchNorm1d(out_dim)
        self.relu  = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # Message passing layers
        x = self.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.relu(self.bn3(self.conv3(x, edge_index)))
        # Global mean pooling: aggregate node features into graph-level vector
        return global_mean_pool(x, batch)   # shape: (batch_size, 128)


class ProteinEncoder(nn.Module):
    """
    1D CNN encoder for protein amino acid sequences.
    Input:  integer-encoded sequence of length 1000
    Output: 96-dimensional protein embedding vector
    """
    def __init__(self, vocab_size=25, embed_dim=128, max_len=1000):
        super(ProteinEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 32,  kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(32,        64,  kernel_size=6, padding=2)
        self.conv3 = nn.Conv1d(64,        96,  kernel_size=8, padding=3)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, max_len)
        x = self.embedding(x)           # → (batch, max_len, embed_dim)
        x = x.permute(0, 2, 1)         # → (batch, embed_dim, max_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x.max(dim=2).values      # global max pool → (batch, 96)


class DTAModel(nn.Module):
    """
    Drug-Target Affinity (DTA) prediction model.

    Architecture:
        Drug molecule   → GCN (3 layers) → 128-dim embedding
        Protein sequence → Conv1D (3 layers) → 96-dim embedding
        Concat → FC(224→512) → FC(512→256) → FC(256→1)

    Output: scalar Kd binding affinity value (nM)
    """
    def __init__(self):
        super(DTAModel, self).__init__()
        self.drug_encoder    = DrugEncoder(node_feat_dim=5, out_dim=128)
        self.protein_encoder = ProteinEncoder(vocab_size=25, embed_dim=128)

        self.regressor = nn.Sequential(
            nn.Linear(128 + 96, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

    def forward(self, drug_data, protein_seq):
        """
        Args:
            drug_data:   PyG Batch object (drug molecular graphs)
            protein_seq: LongTensor of shape (batch_size, 1000)

        Returns:
            Tensor of shape (batch_size,) — predicted Kd values
        """
        drug_emb    = self.drug_encoder(
            drug_data.x,
            drug_data.edge_index,
            drug_data.batch
        )                                          # (B, 128)
        protein_emb = self.protein_encoder(protein_seq)  # (B, 96)
        combined    = torch.cat([drug_emb, protein_emb], dim=1)  # (B, 224)
        return self.regressor(combined).squeeze(1)               # (B,)


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = DTAModel()
    total_params = count_parameters(model)
    print(f"DTAModel created successfully!")
    print(f"  Drug encoder:    GCN (3 layers, 5 → 64 → 128 → 128 dim)")
    print(f"  Protein encoder: Conv1D (3 layers → 96 dim)")
    print(f"  Fusion:          concat (224 dim) → MLP → 1 output")
    print(f"  Total parameters: {total_params:,}")
