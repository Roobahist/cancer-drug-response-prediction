"""Optional SMILES-to-Morgan-fingerprint conversion using RDKit."""
from __future__ import annotations

import pandas as pd


def encode_smiles(
    drugs: pd.DataFrame,
    *,
    drug_col: str = "drug_id",
    smiles_col: str = "smiles",
    bits: int = 256,
    radius: int = 2,
) -> pd.DataFrame:
    """Generate a stable binary fingerprint table; reject invalid molecules."""
    if bits < 16:
        raise ValueError("A fingerprint must contain at least 16 bits")
    for column in (drug_col, smiles_col):
        if column not in drugs:
            raise ValueError(f"Missing molecular column: {column}")
    if drugs[drug_col].duplicated().any():
        raise ValueError("Duplicate drug identifiers")
    import numpy as np

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError("Install the optional chemistry extra: pip install -e '.[chemistry]'") from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows = []
    for _, row in drugs.iterrows():
        smiles = row[smiles_col]
        molecule = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        if molecule is None:
            raise ValueError(f"Invalid/missing SMILES for drug {row[drug_col]!r}")
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        rows.append([row[drug_col], *array.tolist()])
    columns = [drug_col] + [f"bit_{i:03d}" for i in range(bits)]
    return pd.DataFrame(rows, columns=columns)
