"""
app_v2.py — Improved Gradio Interface for Drug-Target Binding Affinity Prediction
EGN6217 | Deliverable 3 | Sathyadharini Srinivasan | Spring 2026

Improvements over D2 placeholder UI:
  • Four functional tabs: Single Prediction, Batch Prediction, Model Info, About
  • Real-time 2D molecule rendering with RDKit (SVG in browser)
  • Molecular descriptor table
  • Colour-coded binding strength label
  • Batch CSV upload with downloadable results
  • Example drug presets
  • Graceful error handling
"""

import os, sys, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
CKPT = os.path.join(ROOT, 'results', 'dta_model_v2_best.pt')
DEVICE = torch.device('cpu')   # UI runs on CPU for portability

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdchem
    RDKIT = True
except ImportError:
    RDKIT = False

# ── Model definition (must match train_d3.py) ────────────────────────────────
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

HYBRID_MAP = {}
CHIRAL_MAP = {}
if RDKIT:
    HYBRID_MAP = {
        rdchem.HybridizationType.SP: 1, rdchem.HybridizationType.SP2: 2,
        rdchem.HybridizationType.SP3: 3, rdchem.HybridizationType.SP3D: 4,
        rdchem.HybridizationType.SP3D2: 5
    }
    CHIRAL_MAP = {
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2
    }

def atom_features(atom):
    ri = atom.GetOwningMol().GetRingInfo()
    rings = [len(r) for r in ri.AtomRings() if atom.GetIdx() in r]
    return [
        atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
        int(atom.GetIsAromatic()), int(atom.IsInRing()),
        HYBRID_MAP.get(atom.GetHybridization(), 0),
        min(rings) if rings else 0,
        CHIRAL_MAP.get(atom.GetChiralTag(), 0),
        atom.GetTotalNumHs()
    ]

def smiles_to_graph(smiles):
    if not RDKIT: return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i,j],[j,i]]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0),dtype=torch.long)
    return Data(x=x, edge_index=ei)

AA_IDX = {aa:i+1 for i,aa in enumerate('ACDEFGHIKLMNPQRSTVWXY')}
def encode_protein(seq, max_len=1000):
    enc = [AA_IDX.get(a,0) for a in seq[:max_len]]
    enc += [0]*(max_len-len(enc))
    return torch.tensor(enc, dtype=torch.long)

class DrugEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=GCNConv(9,64);   self.bn1=nn.BatchNorm1d(64)
        self.conv2=GCNConv(64,128); self.bn2=nn.BatchNorm1d(128)
        self.conv3=GCNConv(128,128);self.bn3=nn.BatchNorm1d(128)
    def forward(self,x,ei,batch):
        x=F.relu(self.bn1(self.conv1(x,ei)))
        x=F.relu(self.bn2(self.conv2(x,ei)))
        x=F.relu(self.bn3(self.conv3(x,ei)))
        return global_mean_pool(x,batch)

class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(26,128,padding_idx=0)
        self.conv1=nn.Conv1d(128,32,4,padding=1)
        self.conv2=nn.Conv1d(32,64,6,padding=2)
        self.conv3=nn.Conv1d(64,96,8,padding=3)
    def forward(self,x):
        x=self.emb(x).permute(0,2,1)
        return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))).max(dim=2).values

class DTAModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_enc=DrugEncoder(); self.protein_enc=ProteinEncoder()
        self.regressor=nn.Sequential(
            nn.Linear(224,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,1))
    def forward(self,dg,ps):
        return self.regressor(torch.cat([self.drug_enc(dg.x,dg.edge_index,dg.batch),self.protein_enc(ps)],1)).squeeze(1)

# ── Load model ────────────────────────────────────────────────────────────────
_model = None
def get_model():
    global _model
    if _model is not None:
        return _model, None
    if not os.path.exists(CKPT):
        return None, f"Checkpoint not found at {CKPT}. Run training first."
    try:
        m = DTAModel_v2().to(DEVICE)
        m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        m.eval()
        _model = m
        return _model, None
    except Exception as e:
        return None, str(e)

# ── Constants ─────────────────────────────────────────────────────────────────
EXAMPLES = {
    "Imatinib (Gleevec)":  "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Erlotinib":           "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    "Dasatinib":           "CC1=NC(=NC=C1)NC2=NC(=CC3=CC(=C(C=C3)Cl)NC(=O)C4=CC=CC=N4)C=C2",
    "Celecoxib":           "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F",
}
EXAMPLE_PROTEIN = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHD"
                   "FSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHS"
                   "QELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRV"
                   "DADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")

# ── Helpers ───────────────────────────────────────────────────────────────────
def mol_svg(smiles):
    if not RDKIT or not smiles.strip(): return "<p style='color:gray'>Enter a SMILES string.</p>"
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None: return "<p style='color:red'>Invalid SMILES — cannot render.</p>"
    d = rdMolDraw2D.MolDraw2DSVG(380, 280)
    d.drawOptions().addStereoAnnotation = True
    d.DrawMolecule(mol); d.FinishDrawing()
    return d.GetDrawingText()

def descriptors_md(smiles):
    if not RDKIT or not smiles.strip(): return ""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None: return "_Invalid SMILES_"
    rows = [
        ("Mol. Weight (Da)",  round(Descriptors.MolWt(mol), 2)),
        ("LogP",              round(Descriptors.MolLogP(mol), 3)),
        ("H-bond Donors",     rdMolDescriptors.CalcNumHBD(mol)),
        ("H-bond Acceptors",  rdMolDescriptors.CalcNumHBA(mol)),
        ("TPSA (Å²)",         round(rdMolDescriptors.CalcTPSA(mol), 2)),
        ("Rotatable Bonds",   rdMolDescriptors.CalcNumRotatableBonds(mol)),
        ("Aromatic Rings",    rdMolDescriptors.CalcNumAromaticRings(mol)),
        ("Heavy Atoms",       mol.GetNumHeavyAtoms()),
    ]
    md = "| Property | Value |\n|---|---|\n"
    for k,v in rows: md += f"| {k} | {v} |\n"
    return md

def predict(smiles, protein_seq):
    model, err = get_model()
    if err: return None, err
    if not RDKIT: return None, "RDKit not installed."
    g = smiles_to_graph(smiles)
    if g is None: return None, "Invalid SMILES string."
    if len(protein_seq.strip()) < 10: return None, "Protein sequence too short."
    drug_batch = Batch.from_data_list([g]).to(DEVICE)
    prot_enc   = encode_protein(protein_seq.strip()).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        log_kd = model(drug_batch, prot_enc).item()
    kd_nm = 10 ** log_kd
    return {"log10_kd": round(log_kd,4), "kd_nm": round(kd_nm,3)}, None

# ── Tab handlers ──────────────────────────────────────────────────────────────
def run_single(smiles, protein, example_drug):
    if example_drug and example_drug in EXAMPLES:
        smiles = EXAMPLES[example_drug]
    svg  = mol_svg(smiles)
    desc = descriptors_md(smiles)
    res, err = predict(smiles, protein)
    if err:
        return svg, desc, f"**Error:** {err}", ""
    log_kd, kd_nm = res['log10_kd'], res['kd_nm']
    if   kd_nm < 1:    icon, strength = "🟢", "Very Strong (< 1 nM)"
    elif kd_nm < 100:  icon, strength = "🟡", "Strong (1–100 nM)"
    elif kd_nm < 1000: icon, strength = "🟠", "Moderate (100–1,000 nM)"
    else:              icon, strength = "🔴", "Weak (> 1,000 nM)"
    pred_md = (
        f"### Prediction\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| log₁₀(Kd) | `{log_kd}` |\n"
        f"| Kd | `{kd_nm:,.3f} nM` |\n"
        f"| Binding | {icon} **{strength}** |\n"
    )
    interp = (f"A Kd of **{kd_nm:,.2f} nM** indicates **{strength.lower()}** binding. "
              f"Lower Kd = tighter binding = better drug candidate for this target.")
    return svg, desc, pred_md, interp

def run_batch(csv_file):
    import pandas as pd
    if csv_file is None: return None, "Upload a CSV file."
    try: df = pd.read_csv(csv_file)
    except Exception as e: return None, f"Cannot read CSV: {e}"
    df.columns = df.columns.str.lower()
    if not {'smiles','protein_sequence'}.issubset(df.columns):
        return None, "CSV must have columns: 'smiles' and 'protein_sequence'"
    results = []
    for _, row in df.iterrows():
        res, err = predict(str(row['smiles']), str(row['protein_sequence']))
        if err: results.append({'smiles':row['smiles'],'log10_kd':None,'kd_nm':None,'error':err})
        else:   results.append({'smiles':row['smiles'],'log10_kd':res['log10_kd'],'kd_nm':res['kd_nm'],'error':''})
    out = pd.DataFrame(results)
    path = '/tmp/batch_results.csv'
    out.to_csv(path, index=False)
    ok = out['error'].eq('').sum()
    return path, f"**{len(out)} pairs processed** | ✅ {ok} success | ❌ {len(out)-ok} failed"

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DTA Predictor") as demo:

    gr.Markdown("""
    # Drug–Target Binding Affinity Predictor
    **EGN6217 — Engineering Applications of ML | Deliverable 3 | Sathyadharini Srinivasan**
    Predicts binding affinity (Kd in nM) between a drug molecule and a protein target
    using a **GCN + 1D CNN** dual-branch model trained on the Davis kinase dataset.
    > **Test MSE: 0.2399 | Concordance Index: 0.8909** — surpasses DeepDTA benchmark
    """)

    with gr.Tabs():

        # ── Tab 1: Single Prediction ─────────────────────────────────────────
        with gr.Tab("Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    example_dd = gr.Dropdown(
                        choices=[""] + list(EXAMPLES.keys()),
                        label="Load Example Drug", value="")
                    smiles_in  = gr.Textbox(label="Drug SMILES", lines=3,
                        placeholder="e.g. CC1=CC=C(C=C1)C2=CC(=NN2...)C(F)(F)F")
                    protein_in = gr.Textbox(label="Protein Sequence", lines=5,
                        value=EXAMPLE_PROTEIN)
                    predict_btn = gr.Button("Predict Binding Affinity", variant="primary")

                with gr.Column(scale=1):
                    mol_out  = gr.HTML(label="2D Molecule")
                    desc_out = gr.Markdown(label="Molecular Descriptors")

            with gr.Row():
                pred_out   = gr.Markdown(label="Result")
                interp_out = gr.Markdown(label="Interpretation")

            predict_btn.click(
                fn=run_single,
                inputs=[smiles_in, protein_in, example_dd],
                outputs=[mol_out, desc_out, pred_out, interp_out])

            gr.Examples(
                examples=[
                    [EXAMPLES["Imatinib (Gleevec)"],  EXAMPLE_PROTEIN, ""],
                    [EXAMPLES["Erlotinib"],            EXAMPLE_PROTEIN, ""],
                    [EXAMPLES["Dasatinib"],            EXAMPLE_PROTEIN, ""],
                ],
                inputs=[smiles_in, protein_in, example_dd])

        # ── Tab 2: Batch Prediction ───────────────────────────────────────────
        with gr.Tab("Batch Prediction"):
            gr.Markdown("Upload a CSV with columns **`smiles`** and **`protein_sequence`**.")
            with gr.Row():
                csv_in     = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_btn  = gr.Button("Run Batch", variant="primary")
            batch_status = gr.Markdown()
            batch_out    = gr.File(label="Download Results")
            batch_btn.click(fn=run_batch, inputs=[csv_in], outputs=[batch_out, batch_status])

        # ── Tab 3: Model Info ─────────────────────────────────────────────────
        with gr.Tab("Model Info"):
            gr.Markdown("""
            ## Architecture
            ```
            Drug SMILES → Molecular Graph → 3-layer GCN (9 features) → 128-dim ──┐
                                                                                   ├→ MLP → log₁₀(Kd)
            Protein Seq → Embedding → 3-layer Conv1D → 96-dim ───────────────────┘
            ```

            ## D2 → D3 Refinements
            | Refinement | D2 | D3 |
            |---|---|---|
            | Atom features | 5 | **9** |
            | Target variable | Raw Kd | **log₁₀(Kd)** |
            | BN in MLP | No | **Yes** |
            | Dropout | 0.2 | **0.3** |
            | LR scheduler | None | **ReduceLROnPlateau** |
            | Early stopping | No | **Yes (patience=15)** |

            ## Performance (Davis Test Set, 3,005 pairs)
            | Metric | D2 Baseline | D3 Refined | Literature (DeepDTA) |
            |---|---|---|---|
            | MSE | 0.4213 | **0.2399** | 0.261 |
            | RMSE | 0.6491 | **0.4898** | — |
            | MAE | 0.5124 | **0.3008** | — |
            | Pearson r | 0.8415 | **0.8410** | — |
            | CI | 0.8389 | **0.8909** | 0.886 |
            """)

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown("""
            ## About
            Drug discovery takes 12+ years and $2.6B per drug. This system predicts
            binding affinity (Kd in nM) from drug SMILES + protein sequence using deep learning,
            accelerating early-stage candidate screening.

            **Dataset:** DeepDTA Davis — 30,056 drug-target pairs, 68 drugs, 442 kinase proteins

            **⚠️ Responsible AI:**
            - For research and educational use only
            - Trained on kinase targets only — do not generalise to other protein families
            - Not a substitute for wet-lab experimental validation

            ---
            **Course:** EGN6217 — Engineering Applications of ML | Spring 2026 | University of Florida
            **Author:** Sathyadharini Srinivasan | srinivassathyadh@ufl.edu
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
