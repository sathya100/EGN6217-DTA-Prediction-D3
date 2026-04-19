"""
train_d3.py  —  Deliverable 3 Refined Training Script
EGN6217 | Sathyadharini Srinivasan | Spring 2026

Refinements over D2:
  1. log10 transform of Kd values
  2. Extended 9-feature atom encoding
  3. BatchNorm in MLP regressor
  4. ReduceLROnPlateau scheduler
  5. Early stopping (patience=15)
  6. Dropout 0.2 → 0.3
"""

import json, pickle, os, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── PyG ──────────────────────────────────────────────────────────────────────
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import rdchem

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'davis')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE  = 32
EPOCHS      = 80
PATIENCE    = 15
SEED        = 42
LR          = 1e-3
WEIGHT_DECAY = 1e-4
MAX_PROT_LEN = 1000

# Use MPS if available (Apple Silicon), else CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"Device: {DEVICE}")
torch.manual_seed(SEED)


# ── REFINEMENT 2: Extended atom features (5 → 9) ─────────────────────────────
HYBRID_MAP = {
    rdchem.HybridizationType.SP:    1,
    rdchem.HybridizationType.SP2:   2,
    rdchem.HybridizationType.SP3:   3,
    rdchem.HybridizationType.SP3D:  4,
    rdchem.HybridizationType.SP3D2: 5,
}
CHIRAL_MAP = {
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}

def atom_features_v2(atom):
    ring_info  = atom.GetOwningMol().GetRingInfo()
    atom_rings = [len(r) for r in ring_info.AtomRings() if atom.GetIdx() in r]
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        HYBRID_MAP.get(atom.GetHybridization(), 0),   # NEW
        min(atom_rings) if atom_rings else 0,           # NEW
        CHIRAL_MAP.get(atom.GetChiralTag(), 0),        # NEW
        atom.GetTotalNumHs(),                           # NEW
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features_v2(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWXY'
AA_TO_IDX   = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}

def encode_protein(seq):
    enc = [AA_TO_IDX.get(aa, 0) for aa in seq[:MAX_PROT_LEN]]
    enc += [0] * (MAX_PROT_LEN - len(enc))
    return torch.tensor(enc, dtype=torch.long)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DTADataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

def collate_fn(batch):
    graphs, proteins, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(proteins), torch.stack(labels)


# ── REFINEMENTS 3 & 6: Model with BN in MLP, dropout=0.3 ─────────────────────
class DrugEncoder(nn.Module):
    def __init__(self, node_feat_dim=9, hidden=64, out=128):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.conv3 = GCNConv(hidden * 2, out)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.bn3 = nn.BatchNorm1d(out)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)

class ProteinEncoder(nn.Module):
    def __init__(self, vocab=25, embed_dim=128):
        super().__init__()
        self.emb   = nn.Embedding(vocab + 1, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=6, padding=2)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=8, padding=3)

    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.max(dim=2).values

class DTAModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_enc    = DrugEncoder()
        self.protein_enc = ProteinEncoder()
        self.regressor   = nn.Sequential(
            nn.Linear(224, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, drug_data, prot_seq):
        d = self.drug_enc(drug_data.x, drug_data.edge_index, drug_data.batch)
        p = self.protein_enc(prot_seq)
        return self.regressor(torch.cat([d, p], dim=1)).squeeze(1)


# ── Concordance Index ─────────────────────────────────────────────────────────
def concordance_index(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    concordant = total = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                    concordant += 1
    return concordant / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading Davis dataset...")
    with open(os.path.join(DATA_DIR, 'ligands_can.txt')) as f:
        drugs = json.load(f)      # {drug_id: SMILES}
    with open(os.path.join(DATA_DIR, 'proteins.txt')) as f:
        proteins = json.load(f)   # {protein_name: sequence}
    with open(os.path.join(DATA_DIR, 'Y'), 'rb') as f:
        Y = np.array(pickle.load(f, encoding='latin1'))  # shape (n_drugs, n_proteins)

    drug_smiles   = list(drugs.values())
    protein_seqs  = list(proteins.values())

    print(f"  Y shape : {Y.shape}")
    print(f"  Drugs   : {len(drug_smiles)}")
    print(f"  Proteins: {len(protein_seqs)}")

    # ── REFINEMENT 1: Log10 transform ─────────────────────────────────────────
    print("\n[Refinement 1] Applying log10 transform to Kd values...")
    print(f"  Raw Kd  : min={Y.min():.3f}  max={Y.max():.1f}  std={Y.std():.1f}")
    log_Y = np.log10(Y)
    print(f"  log10Kd : min={log_Y.min():.3f}  max={log_Y.max():.3f}  std={log_Y.std():.3f}")

    # ── Build records ─────────────────────────────────────────────────────────
    print("\nBuilding dataset (pre-computing molecular graphs)...")
    n_rows, n_cols = Y.shape   # (n_drugs, n_proteins) or (n_proteins, n_drugs)
    records = []
    skipped = 0
    for i in range(n_rows):
        smiles = drug_smiles[i] if i < len(drug_smiles) else None
        g = smiles_to_graph(smiles) if smiles else None
        if g is None:
            skipped += 1
            continue
        for j in range(n_cols):
            seq = protein_seqs[j] if j < len(protein_seqs) else ""
            p   = encode_protein(seq)
            y   = torch.tensor(log_Y[i, j], dtype=torch.float)
            records.append((g, p, y))

    print(f"  Valid records: {len(records):,}  (skipped {skipped} invalid SMILES)")

    # ── Split ─────────────────────────────────────────────────────────────────
    n = len(records)
    n_val  = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        DTADataset(records), [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DTAModel_v2().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDTAModel_v2 | Parameters: {total_params:,} | Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # REFINEMENT 4: LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Epoch':>6} {'Train MSE':>10} {'Val MSE':>10} {'LR':>10} {'Time':>7}")
    print(f"{'─'*65}")

    best_val      = float('inf')
    patience_cnt  = 0
    best_weights  = None
    history       = {'train': [], 'val': []}

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for drug_b, prot_b, labels in loader:
                drug_b  = drug_b.to(DEVICE)
                prot_b  = prot_b.to(DEVICE)
                labels  = labels.to(DEVICE)
                preds   = model(drug_b, prot_b)
                loss    = criterion(preds, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += loss.item() * len(labels)
                n     += len(labels)
        return total / n

    for epoch in range(1, EPOCHS + 1):
        t0     = time.time()
        t_loss = run_epoch(train_loader, train=True)
        v_loss = run_epoch(val_loader,   train=False)
        elapsed = time.time() - t0
        history['train'].append(t_loss)
        history['val'].append(v_loss)
        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]['lr']
        star = '★' if v_loss < best_val else ' '
        print(f"{epoch:>6} {t_loss:>10.4f} {v_loss:>10.4f} {lr_now:>10.1e} {elapsed:>6.1f}s {star}")

        if v_loss < best_val:
            best_val      = v_loss
            patience_cnt  = 0
            best_weights  = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Best Val MSE: {best_val:.4f}")
                break

    # ── Load best & evaluate ──────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(RESULTS_DIR, 'dta_model_v2_best.pt')
    torch.save(best_weights, ckpt_path)
    print(f"\nBest model saved → {ckpt_path}")

    print(f"\n{'═'*55}")
    print(f"  TEST SET EVALUATION")
    print(f"{'═'*55}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for drug_b, prot_b, labels in test_loader:
            preds = model(drug_b.to(DEVICE), prot_b.to(DEVICE))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse  = mean_squared_error(all_labels, all_preds)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(all_labels, all_preds)
    r, _ = pearsonr(all_labels, all_preds)
    r2   = r2_score(all_labels, all_preds)
    # CI on a sample (full CI is O(n²), slow for large n)
    sample_idx = np.random.choice(len(all_labels), min(1000, len(all_labels)), replace=False)
    ci = concordance_index(all_labels[sample_idx].tolist(), all_preds[sample_idx].tolist())

    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Pearson_r': r, 'R2': r2, 'CI': ci}
    np.save(os.path.join(RESULTS_DIR, 'test_preds.npy'),  all_preds)
    np.save(os.path.join(RESULTS_DIR, 'test_labels.npy'), all_labels)
    np.save(os.path.join(RESULTS_DIR, 'metrics.npy'), metrics)
    np.save(os.path.join(RESULTS_DIR, 'history.npy'), history)

    print(f"  MSE            : {mse:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  Pearson r      : {r:.4f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  Concordance Idx: {ci:.4f}  (sampled 1000 pairs)")
    print(f"{'═'*55}")
    print(f"\nResults saved to {RESULTS_DIR}/")

    # ── Save training history ─────────────────────────────────────────────────
    import json as _json
    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        _json.dump({
            'train_loss': history['train'],
            'val_loss':   history['val'],
            'metrics':    {k: float(v) for k, v in metrics.items()}
        }, f, indent=2)

    return metrics, history

if __name__ == '__main__':
    main()
