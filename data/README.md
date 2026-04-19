# Dataset — DeepDTA Davis Kinase Dataset

## Source
Davis, M.I. et al., "Comprehensive analysis of kinase inhibitor selectivity,"
*Nature Chemical Biology*, 7, 698–704, 2011.

Download: http://staff.cs.utu.fi/~aatapa/data/DrugTarget/Davis_dataset.zip

## Files

| File | Description |
|------|-------------|
| `davis/ligands_can.txt` | JSON dict — 68 drug compounds: `{drug_id: SMILES}` |
| `davis/proteins.txt` | JSON dict — 442 protein kinases: `{protein_name: sequence}` |
| `davis/Y` | Pickle (numpy array, shape 68×442) — Kd affinity matrix in nM |

## Statistics

| Property | Value |
|----------|-------|
| Drug–target pairs | 30,056 |
| Unique drugs | 68 |
| Unique proteins | 442 |
| Kd range (raw) | 0.016 – 10,000 nM |
| Kd (log₁₀ transformed) | −1.80 – 4.00 |
| Train / Val / Test split | 80% / 10% / 10% |

## Preprocessing (Deliverable 3 Refinement)

Raw Kd values span 5 orders of magnitude, causing gradient instability.
All affinity values are transformed with `log₁₀(Kd)` before training.

## How to Download

```bash
mkdir -p data/davis && cd data/davis
wget https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/Y
wget https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/ligands_can.txt
wget https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/proteins.txt
```
