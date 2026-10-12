"""Reproducible, grouped evaluation of three supervised regression baselines.

This modern benchmark is a new protocol. Do not compare its synthetic smoke
metrics directly with the historical CNN score in the original group README.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_training_table
from .validation import holdout, inner_group_cv


def benchmark(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    group_by: str = "model_id",
    target: str = "LN_IC50",
    model_col: str = "model_id",
    drug_col: str = "drug_id",
    seed: int = 42,
    demo: bool = False,
) -> pd.DataFrame:
    """Run group-disjoint held-out evaluation with group-aware training CV.

    The scaler and imputer are inside every cross-validation fold. The held-out
    test set is never used for preprocessing fit, model choice or tuning.
    """
    data, columns = load_training_table(
        input_path, target=target, model_col=model_col, drug_col=drug_col
    )
    if group_by not in (model_col, drug_col):
        raise ValueError("Choose a grouping column representing cell lines or drugs")
    train_idx, test_idx = holdout(data, group_by=group_by, seed=seed)
    x_train = data.iloc[train_idx][columns]
    y_train = data.iloc[train_idx][target]
    x_test = data.iloc[test_idx][columns]
    y_test = data.iloc[test_idx][target]
    groups_train = data.iloc[train_idx][group_by]
    cv = inner_group_cv(groups_train)
    models = {
        "Ridge": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": [1.0, 10.0]},
        ),
        "Random forest": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=60, min_samples_leaf=2, random_state=seed, n_jobs=1
                )),
            ]),
            {"model__max_depth": [5, None]},
        ),
        "MLP": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(32,), max_iter=220, early_stopping=True,
                    n_iter_no_change=12, random_state=seed
                )),
            ]),
            {"model__alpha": [0.001, 0.01]},
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for name, (estimator, parameters) in models.items():
        search = GridSearchCV(
            estimator,
            parameters,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train, groups=groups_train)
        predicted = np.asarray(search.predict(x_test), dtype=float)
        rows.append({
            "model": name,
            "cv_mse": -float(search.best_score_),
            "test_mse": float(mean_squared_error(y_test, predicted)),
            "test_mae": float(mean_absolute_error(y_test, predicted)),
            "test_r2": float(r2_score(y_test, predicted)),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "hyperparameters": json.dumps(search.best_params_, sort_keys=True),
        })
        for pos, prediction in zip(test_idx, predicted, strict=True):
            predictions.append({
                "model": name,
                "observation": int(pos),
                "actual": float(data.iloc[pos][target]),
                "predicted": float(prediction),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "holdout_predictions.csv", index=False)
    metadata = {
        "evaluation_status": "synthetic demonstration" if demo else "real-data evaluation",
        "claim": "No clinical validation; no external test cohort.",
        "split_strategy": f"GroupShuffleSplit on {group_by}",
        "inner_validation": "Three-fold GroupKFold on training partition",
        "seed": seed,
        "target": target,
        "feature_count": len(columns),
        "gene_feature_count": sum(c.startswith("gene_") for c in columns),
        "drug_feature_count": sum(c.startswith("fp_") for c in columns),
        "rows": len(data),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_groups": int(groups_train.nunique()),
        "test_groups": int(data.iloc[test_idx][group_by].nunique()),
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
