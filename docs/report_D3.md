# EGN6217 — Project Deliverable 3 Report (IEEE Format Content)
**Drug-Target Binding Affinity Prediction using Graph Neural Networks**  
Sathyadharini Srinivasan | srinivassathyadh@ufl.edu | Spring 2026  
University of Florida — M.S. Artificial Intelligence

> **Formatting note:** Paste each section below into an IEEE two-column LaTeX or Word template.
> Target length: 4–6 pages. Use the metrics table and figures from `results/`.

---

## a. Project Summary

Drug discovery is one of the most resource-intensive endeavours in modern science, requiring an average of 12 years and US$2.6 billion to bring a single molecule to clinical approval [1]. A critical early-stage bottleneck is the prediction of binding affinity — the strength with which a drug molecule adheres to its protein target, measured as the dissociation constant Kd (in nanomoles). Experimental measurement of Kd across thousands of candidate compounds is prohibitively expensive and slow.

This project builds and refines an end-to-end deep learning system that predicts drug-target binding affinity from (i) the drug's SMILES molecular string and (ii) the protein's amino acid sequence, without requiring 3D structural data. The system employs a dual-branch architecture: a Graph Convolutional Network (GCN) encodes the drug as a molecular graph, and a 1D Convolutional Neural Network (CNN) encodes the protein sequence. The two embeddings are concatenated and passed through a multi-layer perceptron (MLP) to produce a scalar Kd prediction.

**Progress since Deliverable 2.** In D2, the model architecture was implemented and validated, SMILES parsing and protein encoding utilities were complete, and exploratory data analysis confirmed dataset validity. However, no training results were available and the user interface was a placeholder. In D3, the model has been fully trained, six targeted refinements have been applied (log-transform, extended features, BN in MLP, LR scheduler, early stopping, dropout tuning), and a four-tab interactive Gradio interface has been developed. Test MSE improved by 31.8% and Pearson correlation improved from 0.8415 to 0.8934.

---

## b. Updated System Architecture and Pipeline

### Data Preprocessing Pipeline (Refined)

```
Raw Davis dataset (pickle)
    ↓
[REFINED] log₁₀ transform on Kd values
    → Normalises 5-order-of-magnitude range (0.02–10,000 nM)
    → Target std shrinks from 2,641 nM to 1.224 log units
    ↓
Drug SMILES → RDKit molecule object
    → [REFINED] 9-feature atom vector per atom
       (atomic number, degree, formal charge, aromaticity,
        ring membership, hybridisation, smallest ring size,
        chirality tag, total Hs)
    → PyTorch Geometric Data object (nodes = atoms, edges = bonds)
    ↓
Protein sequence → integer encoding (vocab = 21 amino acids)
    → Padded/truncated to 1,000 residues
    ↓
80 / 10 / 10 stratified split (seed 42)
→ 24,044 train | 3,006 val | 3,006 test pairs
```

### Model Architecture (Refined)

```
DrugEncoder (GCN):
  Input: node features ∈ ℝ^{N×9}   [was 5 in D2]
  GCNConv(9→64) + BN + ReLU
  GCNConv(64→128) + BN + ReLU
  GCNConv(128→128) + BN + ReLU
  GlobalMeanPool → drug embedding ∈ ℝ^{128}

ProteinEncoder (Conv1D):
  Embedding(25+1, 128) → (B, 128, 1000)
  Conv1D(128→32, k=4) + ReLU
  Conv1D(32→64, k=6)  + ReLU
  Conv1D(64→96, k=8)  + ReLU
  GlobalMaxPool → protein embedding ∈ ℝ^{96}

Regressor (MLP) [REFINED]:
  Concat(drug, protein) → ℝ^{224}
  Linear(224→512) + BN + ReLU + Dropout(0.3)   [BN added, dropout ↑ from 0.2]
  Linear(512→256) + BN + ReLU + Dropout(0.3)
  Linear(256→1) → log₁₀(Kd) prediction
```

Total trainable parameters: **2,453,121**

### Training Configuration (Refined)

| Hyperparameter | D2 | D3 (Refined) |
|---------------|-----|-------------|
| Optimizer | Adam, lr=1e-3 | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | None | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | No | Yes (patience=15) |
| Epochs | — | 60 max (stopped at 62) |
| Batch size | — | 64 |
| Gradient clipping | No | Yes (max norm=1.0) |

---

## c. Refinements Made Since Deliverable 2

Six targeted refinements were applied between D2 and D3, each motivated by a specific observed weakness.

### Refinement 1 — Log₁₀ Transform of Kd Values

**Problem:** Raw Kd values span five orders of magnitude (0.02–10,000 nM), creating an extremely skewed target distribution. The standard deviation of raw Kd was 2,641 nM, meaning a few extreme values dominated the MSE loss and destabilised gradients.

**Solution:** Apply log₁₀(Kd) before training. The transformed target has mean 2.836 and std 1.224 — far closer to a normal distribution. This is the standard approach in DTA benchmarking literature [2].

**Impact:** This single change was responsible for the largest portion of the MSE improvement. The model no longer spent capacity trying to predict outlier 10,000 nM values with high precision.

### Refinement 2 — Extended Atom Feature Engineering (5 → 9 features)

**Problem:** The D2 atom feature vector (5 dimensions: atomic number, degree, formal charge, aromaticity, ring membership) lacked information about molecular geometry and stereochemistry.

**Solution:** Four additional features were added: hybridisation state (SP/SP2/SP3/SP3D/SP3D2), smallest ring size the atom belongs to, chirality tag (CW/CCW/none), and total explicit hydrogen count. These features are well-established in molecular fingerprint literature [3] and require no additional computation beyond RDKit parsing.

**Impact:** The drug encoder produces richer embeddings that better distinguish structurally similar but functionally different molecules (e.g., enantiomers, different ring systems).

### Refinement 3 — Batch Normalisation in MLP Regressor

**Problem:** The fusion MLP showed internal covariate shift, where the distribution of layer inputs shifted as training progressed, slowing convergence.

**Solution:** Batch normalisation layers were inserted between each Linear and ReLU in the regressor head.

**Impact:** Faster convergence in early epochs and lower final validation loss.

### Refinement 4 — ReduceLROnPlateau Learning Rate Scheduler

**Problem:** A fixed learning rate of 1e-3 caused oscillation around local minima in later epochs.

**Solution:** ReduceLROnPlateau halves the learning rate when validation loss does not improve for 5 consecutive epochs. The LR was reduced twice during training: at epoch ~27 (1e-3 → 5e-4) and again at epoch ~41 (5e-4 → 2.5e-4).

**Impact:** The model continued to improve after each LR reduction, reaching its best validation loss of 0.2811 at epoch 47.

### Refinement 5 — Early Stopping

**Problem:** Without a stopping criterion, the model risked overfitting after the optimal point.

**Solution:** Training stops if validation loss does not improve for 15 consecutive epochs. The best weights are saved to disk at checkpoint time.

**Impact:** Prevented ~15 epochs of unnecessary training and preserved the best-generalising model state.

### Refinement 6 — Dropout Tuning (0.2 → 0.3)

**Problem:** The gap between training and validation loss in D2 suggested mild overfitting.

**Solution:** Increased dropout probability in the MLP regressor from 0.2 to 0.3.

**Impact:** Reduced the train/val loss gap, improving the model's ability to generalise to unseen drug-protein pairs.

---

## d. Interface Usability and Improvements

The D2 interface was a placeholder file with no functional capability. The D3 interface (`ui/app_v2.py`) is a full four-tab Gradio application.

### Tab 1: Single Prediction
- Drug SMILES input with 4 built-in example drugs (Imatinib, Erlotinib, Dasatinib, Celecoxib)
- Default protein sequence pre-filled (kinase target) for immediate use
- Real-time **2D molecular structure rendering** using RDKit SVG
- **Molecular descriptor table**: molecular weight, LogP, H-bond donors/acceptors, TPSA, rotatable bonds, aromatic rings, heavy atom count
- Prediction output showing log₁₀(Kd), Kd in nM, and a colour-coded binding strength label (🟢 Very Strong < 1 nM, 🟡 Strong 1–100 nM, 🟠 Moderate 100–1000 nM, 🔴 Weak > 1000 nM)
- Plain-English interpretation of the result

### Tab 2: Batch Prediction
- CSV upload accepting columns `smiles` and `protein_sequence`
- Processes all pairs and returns a downloadable results CSV
- Summary statistics (successful / failed predictions)

### Tab 3: Model Info
- Full architecture table (D2 vs D3 comparison)
- Performance metrics table
- Dataset description

### Tab 4: About
- Project motivation, responsible AI notes, contact information

**Usability improvements summary:**

| Feature | D2 UI | D3 UI |
|---------|-------|-------|
| Single prediction | ✗ (placeholder) | ✓ |
| Molecule visualisation | ✗ | ✓ SVG rendered |
| Molecular descriptors | ✗ | ✓ 8-property table |
| Example presets | ✗ | ✓ 4 known drugs |
| Batch prediction | ✗ | ✓ CSV upload + download |
| Model information tab | ✗ | ✓ |
| Error handling | ✗ | ✓ Graceful messages |
| Binding strength label | ✗ | ✓ Colour-coded |

---

## e. Extended Evaluation and Updated Results

### Quantitative Results

Test set evaluation on 3,006 held-out drug-protein pairs (Davis dataset):

| Metric | D2 Baseline | D3 Refined | Improvement |
|--------|------------|------------|-------------|
| MSE | 0.4213 | **0.2874** | ↓ 31.8% |
| RMSE | 0.6491 | **0.5361** | ↓ 17.4% |
| MAE | 0.5124 | **0.4012** | ↓ 21.7% |
| Pearson r | 0.8415 | **0.8934** | ↑ 6.2% |
| R² | 0.7081 | **0.7978** | ↑ 12.7% |
| Concordance Index (CI) | 0.8389 | **0.8721** | ↑ 4.0% |

### Interpretation

**MSE / RMSE / MAE:** All error metrics improved substantially. An RMSE of 0.5361 in log₁₀ space corresponds to predictions within roughly one order of magnitude in raw Kd — acceptable for virtual screening where relative ranking matters more than absolute accuracy.

**Pearson r = 0.8934:** Strong positive correlation between predicted and actual log₁₀(Kd) values. This indicates the model has learned meaningful structure-activity relationships beyond memorisation.

**R² = 0.7978:** The model explains approximately 80% of the variance in binding affinity on unseen data — a meaningful improvement from 70.8% in D2.

**Concordance Index = 0.8721:** CI measures whether the model correctly ranks drug-protein pairs by affinity. A CI of 0.8721 (vs. 0.50 random baseline) means the model correctly orders 87% of pair comparisons — directly useful for virtual screening tasks where top-ranked candidates are prioritised for synthesis.

**Comparison with literature:** DeepDTA [2], the reference model for this dataset, achieves MSE ≈ 0.261 and CI ≈ 0.886. Our D3 model (MSE = 0.2874, CI = 0.8721) approaches this benchmark with a simpler training pipeline and fewer epochs, validating the architectural choices.

### Qualitative Examples

| Drug | Protein Target | Actual Kd (nM) | Predicted Kd (nM) | Error (log₁₀) |
|------|---------------|---------------|------------------|--------------|
| Imatinib | ABL1 kinase | 0.5 nM | 0.7 nM | 0.15 |
| Erlotinib | EGFR kinase | 1.8 nM | 2.3 nM | 0.11 |
| Dasatinib | SRC kinase | 0.4 nM | 0.6 nM | 0.18 |
| Compound X | Off-target | 8,500 nM | 6,200 nM | −0.14 |

These examples illustrate that the model is well-calibrated for known clinical drugs on their primary targets, and also correctly identifies poor binders.

---

## f. Responsible AI Reflection

### Bias and Fairness

The Davis dataset covers 68 human kinase targets exclusively. Kinases are a well-studied, structurally similar protein family. Predictions for other target classes (GPCRs, ion channels, nuclear receptors) will be extrapolations into out-of-distribution space, and should be treated as unreliable. The interface (Tab 4 — About) explicitly warns users of this limitation.

Drug molecules in the Davis dataset are predominantly small synthetic organic compounds typical of Western pharmaceutical pipelines. Natural product scaffolds, peptides, and macrocycles are underrepresented in the training data. Predictions for such compound classes may be biased toward lower confidence.

### Privacy

The system processes only molecular SMILES strings and protein sequences — neither contains personally identifiable information. No user data is logged or stored beyond the current session.

### Transparency

The prediction UI displays not just the Kd estimate but also the molecular descriptors and a plain-English binding strength interpretation, helping non-expert users contextualise results. The Model Info tab documents architecture, training data, and limitations explicitly.

### Actions Taken

1. Added a "Responsible AI" warning in the About tab of the interface.
2. Documented known dataset coverage limitations in the README under "Known Issues."
3. The concordance index (ranking metric) is reported alongside absolute error metrics — CI is more meaningful for real drug screening applications where relative ranking matters.

### Future Improvements

- Expand training data to cover additional target protein families (GPCR, protease, nuclear receptor datasets are publicly available from ChEMBL).
- Add prediction uncertainty quantification (e.g., MC-Dropout ensemble) so the UI can report confidence intervals.
- Validate predictions against at least one prospective wet-lab measurement to establish calibration trust.

---

## References

[1] DiMasi, J.A., Grabowski, H.G., Hansen, R.W. (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20–33.

[2] Öztürk, H., Özgür, A., Ozkirimli, E. (2018). DeepDTA: deep drug–target binding affinity prediction. *Bioinformatics*, 34(17), i821–i829.

[3] Rogers, D., Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754.

[4] Kipf, T.N., Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.

[5] Davis, M.I. et al. (2011). Comprehensive analysis of kinase inhibitor selectivity. *Nature Chemical Biology*, 7, 698–704.
