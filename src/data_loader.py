"""
data_loader.py — Davis dataset loader and preprocessing pipeline
EGN6217 | Sathyadharini Srinivasan | Spring 2026
"""

import json, pickle, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import rdchem

# ── Atom feature encoding (9-dim, Deliverable 3 refinement) ──────────────────
HYBRID_MAP = {
    rdchem.HybridizationType.SP: 1, rdchem.HybridizationType.SP2: 2,
    rdchem.HybridizationType.SP3: 3, rdchem.HybridizationType.SP3D: 4,
    rdchem.HybridizationType.SP3D2: 5,
}
CHIRAL_MAP = {
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}

def atom_features(atom):
    """9-dimensional atom feature vector."""
    ri = atom.GetOwningMol().GetRingInfo()
    rings = [len(r) for r in ri.AtomRings() if atom.GetIdx() in r]
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        HYBRID_MAP.get(atom.GetHybridization(), 0),
        min(rings) if rings else 0,
        CHIRAL_MAP.get(atom.GetChiralTag(), 0),
        atom.GetTotalNumHs(),
    ]

def smiles_to_graph(smiles: str):
    """Convert SMILES string to PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    ei = (torch.tensor(edges, dtype=torch.long).t().contiguous()
          if edges else torch.zeros((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=ei)

AA_IDX = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWXY')}

def encode_protein(seq: str, max_len: int = 1000) -> torch.Tensor:
    """Encode amino acid sequence as integer tensor (padded to max_len)."""
    enc = [AA_IDX.get(a, 0) for a in seq[:max_len]]
    enc += [0] * (max_len - len(enc))
    return torch.tensor(enc, dtype=torch.long)


class DavisDataset(Dataset):
    """
    PyTorch Dataset for the DeepDTA Davis kinase dataset.

    Loads drug SMILES, protein sequences, and Kd affinity values.
    Applies log10 transform to Kd values (Deliverable 3, Refinement 1).
    Pre-computes all molecular graphs and protein encodings at init time.
    """
    def __init__(self, data_dir: str, max_prot_len: int = 1000):
        self.records = []
        # Load raw data
        with open(os.path.join(data_dir, 'ligands_can.txt')) as f:
            drugs = json.load(f)       # {drug_id: SMILES}
        with open(os.path.join(data_dir, 'proteins.txt')) as f:
            proteins = json.load(f)    # {protein_name: sequence}
        with open(os.path.join(data_dir, 'Y'), 'rb') as f:
            Y = np.array(pickle.load(f, encoding='latin1'))  # (n_drugs, n_proteins)

        drug_smiles = list(drugs.values())
        prot_seqs   = list(proteins.values())
        log_Y       = np.log10(Y)   # Refinement 1: log10 transform

        n_drugs, n_prots = Y.shape
        skipped = 0
        for i in range(n_drugs):
            g = smiles_to_graph(drug_smiles[i]) if i < len(drug_smiles) else None
            if g is None:
                skipped += 1
                continue
            for j in range(n_prots):
                seq = prot_seqs[j] if j < len(prot_seqs) else ''
                p   = encode_protein(seq, max_prot_len)
                y   = torch.tensor(log_Y[i, j], dtype=torch.float)
                self.records.append((g, p, y))

        print(f"DavisDataset: {len(self.records):,} records loaded "
              f"({skipped} SMILES skipped)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    """Custom collate for PyG graphs + protein tensors + labels."""
    graphs, proteins, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(proteins), torch.stack(labels)


def get_dataloaders(data_dir: str,
                    batch_size: int = 32,
                    seed: int = 42,
                    val_frac: float = 0.1,
                    test_frac: float = 0.1):
    """
    Build train/val/test DataLoaders for the Davis dataset.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = DavisDataset(data_dir)
    n       = len(dataset)
    n_val   = int(val_frac  * n)
    n_test  = int(test_frac * n)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              collate_fn=collate_fn)

    print(f"Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/davis'
    tr, va, te = get_dataloaders(data_dir, batch_size=32)
    drug_batch, prot_batch, labels = next(iter(tr))
    print(f"Drug batch : {drug_batch}")
    print(f"Prot batch : {prot_batch.shape}")
    print(f"Labels     : {labels.shape}  min={labels.min():.2f} max={labels.max():.2f}")
