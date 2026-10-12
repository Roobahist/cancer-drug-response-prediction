"""Explicit split rules and leakage checks for cell-line/drug response data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def holdout(
    data: pd.DataFrame,
    *,
    group_by: str = "model_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by biological entity, not by individual measurements.

    Grouping by model_id tests unseen cell lines. Grouping by drug_id tests
    unseen compounds. This avoids shared IDs across the two partitions but
    does not guarantee external-domain generalization.
    """
    if group_by not in data:
        raise ValueError(f"Grouping column '{group_by}' not found")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    groups = data[group_by]
    if groups.isna().any():
        raise ValueError("Missing biological group identifiers")
    if groups.nunique() < 6:
        raise ValueError("At least six distinct groups are required for grouped validation")
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(split.split(data, groups=groups))
    if set(groups.iloc[train]).intersection(groups.iloc[test]):
        raise AssertionError("Training and test groups overlap")
    return train, test


def inner_group_cv(groups: pd.Series, *, folds: int = 3) -> GroupKFold:
    """Training-fold CV that cannot share a cell line/compound between folds."""
    unique = groups.nunique()
    if unique < folds:
        raise ValueError(f"Need {folds} distinct groups inside training; found {unique}")
    return GroupKFold(n_splits=folds)
