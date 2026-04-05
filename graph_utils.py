"""
graph_utils.py
Converts drug SMILES strings into molecular graphs for PyTorch Geometric.
EGN6217 — Drug-Target Binding Affinity Prediction
Sathyadharini Srinivasan | Spring 2026
"""

import torch
from torch_geometric.data import Data


# Atom feature encoding maps
ATOM_TYPES    = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca',
                 'Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn',
                 'Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au',
                 'Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
DEGREE        = list(range(11))
FORMAL_CHARGE = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
NUM_HS        = [0,1,2,3,4,5,6,7,8]
VALENCE       = [0,1,2,3,4,5,6]


def one_hot(value, choices):
    """Return a one-hot list for value in choices; last entry = 'other'."""
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    enc[idx] = 1
    return enc


def atom_features(atom):
    """
    Build a 78-dimensional feature vector for a single RDKit atom.
    Features: atom type (45), degree (11), formal charge (11),
              num Hs (9), valence (7), aromaticity (1), in ring (1) = 85 total
    We use a simplified 5-feature version for fast training.
    """
    return [
        atom.GetAtomicNum(),          # atomic number
        atom.GetDegree(),             # number of bonds
        atom.GetFormalCharge(),       # formal charge
        int(atom.GetIsAromatic()),    # aromaticity flag
        int(atom.IsInRing()),         # ring membership
    ]


def smiles_to_graph(smiles: str):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string representing a drug molecule

    Returns:
        torch_geometric.data.Data with:
            x          — node feature matrix (num_atoms x 5)
            edge_index — bond connectivity (2 x num_bonds*2)
        or None if SMILES is invalid
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        features = [atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(features, dtype=torch.float)

        # Edge index (undirected: add both directions)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]

        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None


def encode_protein(sequence: str, max_len: int = 1000) -> torch.Tensor:
    """
    Encode a protein amino acid sequence as an integer tensor.

    Args:
        sequence: amino acid sequence string
        max_len:  maximum sequence length (pad/truncate to this)

    Returns:
        torch.Tensor of shape (max_len,) with integer-encoded amino acids
    """
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWXY"
    aa_to_idx = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = padding

    encoded = [aa_to_idx.get(aa, 0) for aa in sequence[:max_len]]
    # Pad to max_len
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


if __name__ == "__main__":
    # Quick smoke test
    test_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F"  # Celecoxib
    graph = smiles_to_graph(test_smiles)
    if graph:
        print(f"Molecule graph created successfully!")
        print(f"  Atoms (nodes): {graph.x.shape[0]}")
        print(f"  Bonds (edges): {graph.edge_index.shape[1] // 2}")
        print(f"  Node feature dim: {graph.x.shape[1]}")
    else:
        print("Failed to parse SMILES")

    test_protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    enc = encode_protein(test_protein)
    print(f"\nProtein encoded: shape={enc.shape}, non-zero={enc.nonzero().shape[0]} positions")
