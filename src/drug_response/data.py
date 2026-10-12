"""Validated data assembly for precomputed molecular and gene-expression features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def merge_feature_tables(
    response_path: str | Path,
    gene_path: str | Path,
    drug_path: str | Path,
    *,
    model_col: str = "model_id",
    drug_col: str = "drug_id",
    target: str = "LN_IC50",
) -> pd.DataFrame:
    """Join three curated input tables without silently multiplying observations.

    Genes are model-level, fingerprints are drug-level, response is model-drug-level.
    Both feature tables must have unique identifiers; duplicate response pairs
    are rejected instead of accidentally over-weighted in evaluation.
    """
    response = pd.read_csv(response_path)
    genes = pd.read_csv(gene_path)
    drugs = pd.read_csv(drug_path)
    required = [
        (response, {model_col, drug_col, target}, "response"),
        (genes, {model_col}, "gene expression"),
        (drugs, {drug_col}, "drug fingerprints"),
    ]
    for frame, needed, name in required:
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} table is empty")
    if genes[model_col].duplicated().any():
        raise ValueError("Gene-expression table has duplicate model identifiers")
    if drugs[drug_col].duplicated().any():
        raise ValueError("Molecular feature table has duplicate drug identifiers")
    if response.duplicated(subset=[model_col, drug_col]).any():
        raise ValueError("Repeated model-drug response pairs require manual reconciliation")
    if response[target].isna().any():
        raise ValueError("Drug response target contains missing values")

    # Preserve only necessary source columns; strip names with consistent prefixes.
    gene_features = [c for c in genes.columns if c != model_col]
    drug_features = [c for c in drugs.columns if c != drug_col]
    if not gene_features or not drug_features:
        raise ValueError("Both gene-expression and molecular features are required")
    if any(not pd.api.types.is_numeric_dtype(genes[c]) for c in gene_features):
        raise ValueError("Gene expression must be preprocessed into numeric columns")
    if any(not pd.api.types.is_numeric_dtype(drugs[c]) for c in drug_features):
        raise ValueError("Drug fingerprints must be numeric; encode SMILES first")

    genes = genes.rename(columns={c: f"gene_{c}" for c in gene_features})
    drugs = drugs.rename(columns={c: f"fp_{c}" for c in drug_features})
    response = response[[model_col, drug_col, target]]
    merged = response.merge(genes, how="left", on=model_col, validate="many_to_one")
    merged = merged.merge(drugs, how="left", on=drug_col, validate="many_to_one")
    new_cols = [c for c in merged.columns if c.startswith(("gene_", "fp_"))]
    gene_cols = [c for c in new_cols if c.startswith("gene_")]
    drug_cols = [c for c in new_cols if c.startswith("fp_")]
    if merged[gene_cols].isna().all(axis=1).any() or merged[drug_cols].isna().all(axis=1).any():
        raise ValueError("A response lacks a matching gene or molecular feature record")
    return merged


def load_training_table(
    path: str | Path,
    *,
    target: str = "LN_IC50",
    model_col: str = "model_id",
    drug_col: str = "drug_id",
) -> tuple[pd.DataFrame, list[str]]:
    """Load explicitly encoded features, rejecting accidental target/ID leakage."""
    table = pd.read_csv(path)
    mandatory = {target, model_col, drug_col}
    if not mandatory.issubset(table):
        raise ValueError(f"Missing required columns: {sorted(mandatory - set(table))}")
    if table.empty:
        raise ValueError("Dataset has no rows")
    if table[target].isna().any():
        raise ValueError("Missing response labels")
    if table.duplicated(subset=[model_col, drug_col]).any():
        raise ValueError("Repeated model-drug pairs must be resolved before modelling")
    prefixes = ("gene_", "fp_")
    features = [c for c in table if c.startswith(prefixes)]
    if not any(c.startswith("gene_") for c in features):
        raise ValueError("No gene_ expression columns present")
    if not any(c.startswith("fp_") for c in features):
        raise ValueError("No fp_ molecular columns present")
    if not all(pd.api.types.is_numeric_dtype(table[c]) for c in features):
        raise ValueError("All expression and fingerprint features must be numeric")
    if any(not np.isfinite(v) for v in pd.to_numeric(table[target], errors="coerce")):
        raise ValueError("Response must contain finite numbers")
    values = table[features].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite predictor values")
    if len(table) < 12:
        raise ValueError("At least 12 observations are required")
    return table, features
