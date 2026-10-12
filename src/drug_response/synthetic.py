"""Synthetic fixtures for testing infrastructure, not scientific model performance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_dataset(
    output_path: str | Path, *, seed: int = 42, cell_lines: int = 36, drugs: int = 8
) -> pd.DataFrame:
    """Fictional 36x8 crossed design with deliberate shared model/drug effects."""
    if cell_lines < 12 or drugs < 6:
        raise ValueError("At least 12 cell lines and six compounds are required")
    rng = np.random.default_rng(seed)
    gene_values = rng.normal(size=(cell_lines, 16))
    drug_values = rng.integers(0, 2, size=(drugs, 24))
    rows = []
    for i in range(cell_lines):
        for j in range(drugs):
            # Arbitrary synthetic signal. Never present these as biological findings.
            target = (
                1.1 * gene_values[i, 0] - 0.7 * gene_values[i, 1]
                + 0.8 * drug_values[j, 0] - 0.4 * drug_values[j, 1]
                + 0.35 * gene_values[i, 2] * drug_values[j, 2]
                + rng.normal(scale=0.35)
            )
            row = {"model_id": f"SYN_CELL_{i:03d}",
                   "drug_id": f"SYN_DRUG_{j:03d}",
                   "LN_IC50": target}
            row.update({f"gene_{k:03d}": gene_values[i, k] for k in range(16)})
            row.update({f"fp_{k:03d}": drug_values[j, k] for k in range(24)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return frame
