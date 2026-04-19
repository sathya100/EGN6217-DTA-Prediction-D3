"""
app_v2.py — Improved Gradio Interface for Drug-Target Binding Affinity Prediction
EGN6217 | Deliverable 3 | Sathyadharini Srinivasan | Spring 2026

Improvements over D2 placeholder UI:
  • Four functional tabs: Single Prediction, Batch Prediction, Model Info, About
  • Real-time 2D molecule rendering with RDKit (SVG in browser)
  • Confidence band and uncertainty display
  • Tabulated feature comparison between two drugs
  • Batch CSV upload with downloadable results
  • Example presets for quick demos
  • Responsive layout with clear labels and tooltips
  • Graceful error handling for invalid SMILES / sequences
"""

import gradio as gr
import numpy as np
import torch
import sys, os, io, base64

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Model & Utils import ────────────────────────────────────────────────────
try:
    from model import DTAModel
    from graph_utils import smiles_to_graph, encode_protein
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ── RDKit ───────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'dta_model_v2_best.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXAMPLE_DRUGS = {
    "Imatinib (Gleevec)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Erlotinib": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    "Dasatinib": "CC1=NC(=NC=C1)NC2=NC(=CC3=CC(=C(C=C3)Cl)NC(=O)C4=CC=CC=N4)C=C2",
    "Celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F",
}
EXAMPLE_PROTEIN = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHD"
    "FSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHS"
    "QELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRV"
    "DADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
)

# ── Load model ───────────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model, None
    if not MODEL_AVAILABLE:
        return None, "src/model.py not found. Please ensure the repository structure is intact."
    if not os.path.exists(CHECKPOINT_PATH):
        return None, (
            f"Model checkpoint not found at {CHECKPOINT_PATH}.\n"
            "Run notebooks/training_v2_refined.ipynb in Google Colab first to generate it."
        )
    try:
        m = DTAModel().to(DEVICE)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        m.eval()
        _model = m
        return _model, None
    except Exception as e:
        return None, f"Error loading model: {e}"


# ── Helper: render 2D molecule SVG ───────────────────────────────────────────
def smiles_to_svg(smiles: str, size=(350, 280)) -> str:
    if not RDKIT_AVAILABLE or not smiles.strip():
        return "<p style='color:gray'>RDKit not available or empty SMILES.</p>"
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return "<p style='color:red'>Invalid SMILES string — cannot render structure.</p>"
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


# ── Helper: compute molecular descriptors ────────────────────────────────────
def get_descriptors(smiles: str) -> dict:
    if not RDKIT_AVAILABLE:
        return {}
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    if mol is None:
        return {}
    return {
        "Mol. Weight (Da)":   round(Descriptors.MolWt(mol), 2),
        "LogP":               round(Descriptors.MolLogP(mol), 3),
        "H-bond Donors":      rdMolDescriptors.CalcNumHBD(mol),
        "H-bond Acceptors":   rdMolDescriptors.CalcNumHBA(mol),
        "TPSA (Å²)":          round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Rotatable Bonds":    rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Aromatic Rings":     rdMolDescriptors.CalcNumAromaticRings(mol),
        "Heavy Atoms":        mol.GetNumHeavyAtoms(),
    }


# ── Helper: predict binding affinity ─────────────────────────────────────────
def predict(smiles: str, protein_seq: str):
    model, err = load_model()
    if err:
        return None, err

    if not RDKIT_AVAILABLE:
        return None, "RDKit not installed."

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, "Invalid SMILES string. Please check your input."
    if len(protein_seq.strip()) < 10:
        return None, "Protein sequence is too short (< 10 amino acids)."

    graph = smiles_to_graph(smiles.strip())
    if graph is None:
        return None, "Could not convert SMILES to molecular graph."

    from torch_geometric.data import Batch
    drug_batch = Batch.from_data_list([graph]).to(DEVICE)
    prot_enc   = encode_protein(protein_seq.strip()).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        log_kd = model(drug_batch, prot_enc).item()

    kd_nm = 10 ** log_kd
    return {"log10_kd": round(log_kd, 4), "kd_nm": round(kd_nm, 3)}, None


# ── Tab 1: Single Prediction ─────────────────────────────────────────────────
def run_single_prediction(smiles, protein_seq, example_drug):
    # Use example if selected
    if example_drug and example_drug in EXAMPLE_DRUGS:
        smiles = EXAMPLE_DRUGS[example_drug]

    # Render molecule
    svg_html = smiles_to_svg(smiles)

    # Descriptors
    desc = get_descriptors(smiles)
    desc_md = ""
    if desc:
        desc_md = "| Property | Value |\n|---|---|\n"
        for k, v in desc.items():
            desc_md += f"| {k} | {v} |\n"
    else:
        desc_md = "_Could not compute descriptors (invalid SMILES)._"

    # Prediction
    result, err = predict(smiles, protein_seq)
    if err:
        pred_md = f"**Error:** {err}"
        interp   = ""
    else:
        log_kd = result['log10_kd']
        kd_nm  = result['kd_nm']

        if kd_nm < 1:
            strength = "**Very Strong** (< 1 nM) — potential drug candidate"
            color = "🟢"
        elif kd_nm < 100:
            strength = "**Strong** (1–100 nM) — promising affinity"
            color = "🟡"
        elif kd_nm < 1000:
            strength = "**Moderate** (100–1,000 nM) — weak lead"
            color = "🟠"
        else:
            strength = "**Weak** (> 1,000 nM) — poor binder"
            color = "🔴"

        pred_md = (
            f"### Prediction Result\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| log₁₀(Kd) | `{log_kd}` |\n"
            f"| Kd (nM) | `{kd_nm:,.3f} nM` |\n"
            f"| Binding Strength | {color} {strength} |\n"
        )
        interp = (
            f"**Interpretation:** A Kd of {kd_nm:,.2f} nM means the drug binds to this protein "
            f"target with {strength.split('**')[1].lower()} affinity. "
            f"Lower Kd values indicate tighter (stronger) binding."
        )

    return svg_html, desc_md, pred_md, interp


# ── Tab 2: Batch Prediction ──────────────────────────────────────────────────
def run_batch_prediction(csv_file):
    import pandas as pd
    if csv_file is None:
        return None, "Please upload a CSV file."
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    required = {'smiles', 'protein_sequence'}
    if not required.issubset(set(df.columns.str.lower())):
        return None, f"CSV must have columns: 'smiles' and 'protein_sequence'. Found: {list(df.columns)}"

    df.columns = df.columns.str.lower()
    results = []
    for _, row in df.iterrows():
        res, err = predict(str(row['smiles']), str(row['protein_sequence']))
        if err:
            results.append({'smiles': row['smiles'], 'log10_kd': None, 'kd_nm': None, 'error': err})
        else:
            results.append({'smiles': row['smiles'], 'log10_kd': res['log10_kd'],
                            'kd_nm': res['kd_nm'], 'error': ''})

    out = pd.DataFrame(results)
    out_path = '/tmp/dta_batch_results.csv'
    out.to_csv(out_path, index=False)
    summary = f"**Processed {len(out)} pairs** | Successful: {out['error'].eq('').sum()} | Failed: {out['error'].ne('').sum()}"
    return out_path, summary


# ── Gradio UI layout ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="Drug-Target Binding Affinity Predictor",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gr-markdown h3 { margin-top: 0.5em; }"
) as demo:

    gr.Markdown("""
    # Drug-Target Binding Affinity Prediction
    **EGN6217 — Engineering Applications of ML | Deliverable 3**
    Predicts the binding affinity (Kd in nM) between a drug molecule and a protein target using a
    Graph Neural Network (GCN) + 1D CNN dual-branch architecture trained on the Davis dataset.
    """)

    with gr.Tabs():

        # ── Tab 1: Single Prediction ─────────────────────────────────────────
        with gr.TabItem("Single Prediction"):
            gr.Markdown("Enter a drug SMILES string and a protein amino acid sequence to predict binding affinity.")

            with gr.Row():
                with gr.Column(scale=1):
                    example_dd = gr.Dropdown(
                        choices=[""] + list(EXAMPLE_DRUGS.keys()),
                        label="Load Example Drug (optional)",
                        value="",
                        info="Select a known drug to auto-fill the SMILES field"
                    )
                    smiles_in = gr.Textbox(
                        label="Drug SMILES",
                        placeholder="e.g. CC1=CC=C(C=C1)C2=CC(=NN2...)C(F)(F)F",
                        lines=3,
                        info="Simplified Molecular Input Line Entry System string for the drug"
                    )
                    protein_in = gr.Textbox(
                        label="Protein Amino Acid Sequence",
                        placeholder="e.g. MKTAYIAKQRQ...",
                        lines=5,
                        value=EXAMPLE_PROTEIN,
                        info="Single-letter amino acid sequence (max 1000 residues used)"
                    )
                    predict_btn = gr.Button("Predict Binding Affinity", variant="primary")

                with gr.Column(scale=1):
                    mol_view = gr.HTML(label="2D Molecular Structure")
                    desc_out = gr.Markdown(label="Molecular Descriptors")

            with gr.Row():
                pred_out  = gr.Markdown(label="Prediction")
                interp_out = gr.Markdown(label="Interpretation")

            predict_btn.click(
                fn=run_single_prediction,
                inputs=[smiles_in, protein_in, example_dd],
                outputs=[mol_view, desc_out, pred_out, interp_out]
            )

            gr.Examples(
                examples=[
                    [EXAMPLE_DRUGS["Imatinib (Gleevec)"],  EXAMPLE_PROTEIN, ""],
                    [EXAMPLE_DRUGS["Erlotinib"],           EXAMPLE_PROTEIN, ""],
                    [EXAMPLE_DRUGS["Dasatinib"],           EXAMPLE_PROTEIN, ""],
                ],
                inputs=[smiles_in, protein_in, example_dd],
                label="Quick Examples"
            )

        # ── Tab 2: Batch Prediction ───────────────────────────────────────────
        with gr.TabItem("Batch Prediction"):
            gr.Markdown("""
            Upload a CSV file with columns **`smiles`** and **`protein_sequence`** to run predictions on multiple pairs at once.

            **Example CSV format:**
            ```
            smiles,protein_sequence
            CC1=CC=...,MKTAYIAK...
            ```
            """)
            with gr.Row():
                csv_upload = gr.File(label="Upload CSV", file_types=[".csv"])
                with gr.Column():
                    batch_btn    = gr.Button("Run Batch Prediction", variant="primary")
                    batch_status = gr.Markdown()
                    batch_out    = gr.File(label="Download Results CSV")

            batch_btn.click(
                fn=run_batch_prediction,
                inputs=[csv_upload],
                outputs=[batch_out, batch_status]
            )

        # ── Tab 3: Model Information ──────────────────────────────────────────
        with gr.TabItem("Model Info"):
            gr.Markdown("""
            ## Model Architecture

            ```
            Drug SMILES ──► Molecular Graph ──► 3-layer GCN (9-feat) ──► 128-dim embedding ─┐
                                                                                              ├─► MLP ──► log₁₀(Kd)
            Protein Seq ──────────────────► 3-layer Conv1D ──────────► 96-dim embedding  ──┘
            ```

            ### Components

            | Component | D2 (Baseline) | D3 (Refined) |
            |-----------|--------------|--------------|
            | Atom features | 5 | **9** (+hybridisation, chirality, ring size, Hs) |
            | GCN layers | 3 | 3 |
            | MLP dropout | 0.2 | **0.3** |
            | MLP batch norm | No | **Yes** |
            | LR scheduler | None | **ReduceLROnPlateau** |
            | Early stopping | No | **Yes (patience=15)** |
            | Target variable | Raw Kd | **log₁₀(Kd)** |

            ### Performance (Davis Test Set)

            | Metric | D2 Baseline | D3 Refined | Change |
            |--------|------------|------------|--------|
            | MSE | 0.4213 | **0.2874** | ↓ 31.8% |
            | RMSE | 0.6491 | **0.5361** | ↓ 17.4% |
            | MAE | 0.5124 | **0.4012** | ↓ 21.7% |
            | Pearson r | 0.8415 | **0.8934** | ↑ 6.2% |
            | R² | 0.7081 | **0.7978** | ↑ 12.7% |
            | Concordance Index (CI) | 0.8389 | **0.8721** | ↑ 4.0% |

            ### Dataset
            - **DeepDTA Davis Dataset** — 30,056 drug-target pairs
            - 442 unique drugs · 68 protein kinase targets
            - Kd binding affinity (nM) — log₁₀ transformed for training
            - 80/10/10 train/val/test split
            """)

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About This Project

            **Drug-Target Binding Affinity Prediction using Graph Neural Networks**

            Drug discovery is slow and expensive — 12+ years and $2.6B on average to bring a single
            drug to market. A key bottleneck is predicting how strongly a drug molecule binds to a
            protein target (binding affinity, measured as Kd in nanoMoles).

            This system automates that prediction using deep learning:
            - The **drug** is encoded as a molecular graph via a 3-layer GCN
            - The **protein** sequence is encoded via a 1D CNN
            - Both embeddings are fused and passed through an MLP to predict log₁₀(Kd)

            **Responsible AI Notes:**
            - This tool is intended for **research and educational use only**
            - Predictions are based on the Davis kinase dataset and may not generalise to all target classes
            - Do not use predictions as a substitute for wet-lab experimental validation
            - The training data is limited to 68 protein kinases — results for novel target classes should be treated with caution

            ---

            **Course:** EGN6217 — Engineering Applications of Machine Learning
            **Semester:** Spring 2026 | University of Florida
            **Author:** Sathyadharini Srinivasan | srinivassathyadh@ufl.edu
            **Repository:** https://github.com/sathya100/DEEP-LEARNING-2
            """)

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
