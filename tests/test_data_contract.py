"""Validate the scientific join and reject ambiguous/unsafe input schemas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_response.data import load_training_table, merge_feature_tables
from drug_response.synthetic import generate_demo_dataset
from drug_response.validation import holdout


def test_synthetic_data_contract(tmp_path):
    generate_demo_dataset(tmp_path / "demo.csv", cell_lines=18, drugs=6)
    loaded, features = load_training_table(tmp_path / "demo.csv")
    assert len(loaded) == 108
    assert len(features) == 40
    assert all(name.startswith(("gene_", "fp_")) for name in features)
    assert np.isfinite(loaded["LN_IC50"]).all()


@pytest.mark.parametrize("group_by", ["model_id", "drug_id"])
def test_grouped_holdout_is_disjoint(group_by, tmp_path):
    frame = generate_demo_dataset(tmp_path / "demo.csv", cell_lines=18, drugs=7)
    a, b = holdout(frame, group_by=group_by, seed=10)
    assert set(frame.iloc[a][group_by]).isdisjoint(set(frame.iloc[b][group_by]))
    assert len(a) + len(b) == len(frame)


def test_repeated_pairs_are_rejected(tmp_path):
    frame = generate_demo_dataset(tmp_path / "demo.csv")
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(
        tmp_path / "duplicate.csv", index=False
    )
    with pytest.raises(ValueError, match="Repeated model-drug"):
        load_training_table(tmp_path / "duplicate.csv")


def test_feature_reuse_cannot_hide_missing_modalities(tmp_path):
    frame = generate_demo_dataset(tmp_path / "demo.csv")
    frame = frame.drop(columns=[c for c in frame if c.startswith("fp_")])
    frame.to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match="fp_"):
        load_training_table(tmp_path / "missing.csv")


def test_join_validated_many_to_one(tmp_path):
    responses = pd.DataFrame({
        "model_id": ["A", "B"], "drug_id": ["D", "D"], "LN_IC50": [1.3, 0.8],
    })
    genes = pd.DataFrame({"model_id": ["A", "B"], "BRCA1": [0.2, 1.1]})
    fingerprints = pd.DataFrame({"drug_id": ["D"], "bit_0": [1], "bit_1": [0]})
    for name, frame in [("response", responses), ("gene", genes), ("drug", fingerprints)]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    merged = merge_feature_tables(
        tmp_path / "response.csv", tmp_path / "gene.csv", tmp_path / "drug.csv"
    )
    assert list(merged.columns) == ["model_id", "drug_id", "LN_IC50",
                                    "gene_BRCA1", "fp_bit_0", "fp_bit_1"]
    assert len(merged) == 2


def test_duplicate_gene_keys_fail_fast(tmp_path):
    response = pd.DataFrame({"model_id": ["A"], "drug_id": ["D"], "LN_IC50": [1.0]})
    genes = pd.DataFrame({"model_id": ["A", "A"], "G": [1.0, 2.0]})
    fingerprints = pd.DataFrame({"drug_id": ["D"], "bit": [1]})
    for name, frame in [("response", response), ("gene", genes), ("drug", fingerprints)]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="duplicate model"):
        merge_feature_tables(tmp_path / "response.csv", tmp_path / "gene.csv",
                             tmp_path / "drug.csv")


@pytest.mark.parametrize("missing_modality", ["gene", "fingerprint"])
def test_unmatched_modalities_fail_instead_of_being_imputed(tmp_path, missing_modality):
    responses = pd.DataFrame({
        "model_id": ["A", "B"], "drug_id": ["D", "E"], "LN_IC50": [0.9, 1.5],
    })
    genes = pd.DataFrame({"model_id": ["A", "B"], "EXPR": [2.0, 3.0]})
    drugs = pd.DataFrame({"drug_id": ["D", "E"], "bit_0": [1, 0]})
    if missing_modality == "gene":
        genes = genes[genes["model_id"] != "B"]
    else:
        drugs = drugs[drugs["drug_id"] != "E"]
    for name, frame in [("response", responses), ("gene", genes), ("drug", drugs)]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="lacks a matching"):
        merge_feature_tables(
            tmp_path / "response.csv", tmp_path / "gene.csv", tmp_path / "drug.csv"
        )
