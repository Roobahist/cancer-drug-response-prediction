# Reproduction checklist

## Verified without research data

```bash
python -m pip install -e ".[dev]"
pytest
drug-response demo --output runs/demo
drug-response plot-reported
```

The demo uses explicitly fictional cell line and drug IDs and produces
`synthetic_data.csv`, `protocol.json`, `model_metrics.csv`,
`holdout_predictions.csv`, and PNG/SVG prediction charts under `runs/demo/`.
These numerical scores must never be quoted as evidence for cancer-drug
response performance.

## Steps needed to reproduce the original team scores

1. Recover the exact GDSC release, expression matrix version, drug SMILES
   dataset, model/cell identifiers and file schemas.
2. Check entity mappings and data usage rights.
3. Re-run gene filtering and molecular encoding *within training only*.
4. Identify original split units and seeds. Rule out repeated cell lines/drugs
   leaking between training and evaluation under the claimed scenario.
5. Port historical CNN/MLP architectures to modern TensorFlow or PyTorch,
   recording dependency versions and compute device.
6. Re-run held-out evaluations; report MSE, MAE, R² and intervals or repeated
   split variation, with comparable baselines.
7. Create training histories and SHAP/feature contribution figures from
   **actual saved model and evaluation outputs**.

Without these, `0.832 R²` and `1.160 MSE` remain attributed values
published in the original README and should not be marketed as verified.
