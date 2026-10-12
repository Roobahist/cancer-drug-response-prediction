"""Fingerprint contract without requiring RDKit in the default environment."""
from __future__ import annotations

import pandas as pd
import pytest

from drug_response.chemistry import encode_smiles


def test_duplicate_drugs_rejected_without_optional_dependency():
    drugs = pd.DataFrame({"drug_id": ["A", "A"], "smiles": ["CC", "O"]})
    with pytest.raises(ValueError, match="Duplicate"):
        encode_smiles(drugs)


def test_fingerprint_bit_width_is_checked():
    drugs = pd.DataFrame({"drug_id": ["A"], "smiles": ["CC"]})
    with pytest.raises(ValueError, match="16"):
        encode_smiles(drugs, bits=8)
