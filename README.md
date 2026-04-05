# Drug-Target Binding Affinity Prediction using GNN

Predicts the binding affinity (Kd) between drug molecules and protein targets
using a Graph Neural Network (GNN) + 1D CNN dual-branch architecture.

**Course:** EGN6217 — Engineering Applications of Machine Learning  
**Semester:** Spring 2026 | University of Florida  
**Author:** Sathyadharini Srinivasan

---

## Project Overview

Drug discovery is slow and expensive — it takes 12+ years and $2.6B on average
to bring one drug to market. A key bottleneck is predicting how strongly a drug
molecule binds to a protein target (binding affinity). This project builds a
Graph Neural Network (GNN) system that takes a drug's molecular structure and a
protein's amino acid sequence, and predicts their binding affinity (Kd value in nM).

---

## Architecture

```
Drug SMILES ──► Molecular Graph ──► 3-layer GCN ──► 128-dim embedding ──►
                                                                           Concat ──► MLP ──► Kd (nM)
Protein Sequence ──────────────► Conv1D Encoder ──► 96-dim embedding  ──►
```

---

## Dataset

**DeepDTA Davis Dataset**
- 442 unique drug compounds (SMILES format)
- 68 protein targets (amino acid sequences)
- 30,056 drug-target pairs with measured Kd binding affinity values
- Download: http://staff.cs.utu.fi/~aatapa/data/DrugTarget/Davis_dataset.zip

---

## Project Structure

```
drug-target-binding-gnn/
├── data/davis/          ← Davis dataset files
├── notebooks/           ← Jupyter notebooks
├── src/
│   ├── graph_utils.py   ← SMILES to molecular graph conversion
│   └── model.py         ← GNN + CNN model architecture
├── ui/                  ← Gradio interface (Week 2)
├── results/             ← Plots and evaluation outputs
├── docs/                ← Architecture diagrams
├── setup.ipynb          ← Main setup and EDA notebook
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/drug-target-binding-gnn
cd drug-target-binding-gnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch Geometric (Colab)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## How to Run

Open `setup.ipynb` in Google Colab or Jupyter and run all cells in order.

The notebook will:
1. Download the Davis dataset automatically
2. Parse and validate all SMILES strings
3. Generate exploratory visualizations
4. Verify the environment is ready for GNN training

---

## Contact

Sathyadharini Srinivasan  
University of Florida — M.S. Artificial Intelligence  
Email: sathyadharini@ufl.edu
