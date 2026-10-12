"""Synthetic smoke tests check code execution, not medical prediction claims."""
from __future__ import annotations

import json

import pandas as pd

from drug_response.cli import main
from drug_response.plotting import plot_original_reported_results
from drug_response.synthetic import generate_demo_dataset
from drug_response.training import benchmark


def test_end_to_end_grouped_holdout_outputs(tmp_path):
    source = tmp_path / "fake.csv"
    generate_demo_dataset(source, seed=19, cell_lines=18, drugs=6)
    output = tmp_path / "results"
    metrics = benchmark(source, output, seed=19, demo=True)
    assert set(metrics["model"]) == {"Ridge", "Random forest", "MLP"}
    assert len(metrics) == 3
    assert (metrics["test_mse"] >= 0).all()
    predictions = pd.read_csv(output / "holdout_predictions.csv")
    assert set(predictions["model"]) == set(metrics["model"])
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["evaluation_status"] == "synthetic demonstration"
    assert protocol["train_groups"] + protocol["test_groups"] == 18
    assert protocol["test_rows"] == int(metrics["test_rows"].iloc[0])


def test_cli_reported_chart_is_explicitly_attributed(tmp_path):
    config = tmp_path / "original.json"
    config.write_text(json.dumps({
        "status": "reported_not_reproduced",
        "models": [
            {"model": "CNN", "mse": 1.16, "r2": 0.832},
            {"model": "Ridge", "mse": 1.35, "r2": 0.800},
        ],
    }))
    output = tmp_path / "historical"
    plot_original_reported_results(config, output)
    assert output.with_suffix(".png").stat().st_size > 1000
    assert output.with_suffix(".svg").stat().st_size > 1000
    wrong = json.loads(config.read_text())
    wrong["status"] = "synthetic"
    config.write_text(json.dumps(wrong))
    import pytest

    with pytest.raises(ValueError, match="Reported-results status"):
        plot_original_reported_results(config, output)


def test_demo_cli_generates_labeled_predictions(tmp_path):
    main(["demo", "--output", str(tmp_path / "demo")])
    directory = tmp_path / "demo"
    assert (directory / "predicted_vs_actual.png").exists()
    assert (directory / "predicted_vs_actual.svg").exists()
    assert "synthetic demonstration" in (directory / "protocol.json").read_text()
