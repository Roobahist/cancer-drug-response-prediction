# Research data contract

**No research datasets are shipped or silently downloaded.** Obtain access to
the appropriate versions of Cell Model Passports (cell-line metadata and
expression), GDSC drug-response records, and drug structures from the official
data providers, respecting each provider's terms.

The original notebook uses many local Colab/Kaggle paths (e.g., `final_all.csv`
and `filtered_rnaseq_read_count.csv`). Neither the notebook's raw inputs nor
its final joined training table were committed to the source repository.

## Inputs to `drug-response prepare`

- `response.csv`: `model_id,drug_id,LN_IC50` (one row per **unique** pair).
- `genes.csv`: `model_id,<numeric-gene-expression-columns...>` (one row per model).
- `fingerprints.csv`: `drug_id,<numeric-fingerprint-bit-columns...>` (one row per drug).

Use the optional `fingerprint` command on a `drug_id,smiles` table to compute
RDKit Morgan fingerprints. It requires `pip install -e ".[chemistry]"`.

This creates `prepared.csv`, with identifiers and numeric `gene_*` and `fp_*`
columns. Never inject the target or patient outcome into a feature column.

```bash
drug-response fingerprint --input data/smiles.csv --output data/fingerprints.csv
drug-response prepare --response data/response.csv --genes data/genes.csv \
  --fingerprints data/fingerprints.csv --output data/prepared.csv
drug-response fit --input data/prepared.csv --output runs/real
```

The code checks required columns, one-to-one feature lookup keys, target
values and duplicate response pairs. Biological metadata is not shared with
third parties by these commands.

## Data leakage and study-design cautions

The default grouped holdout keeps **cell lines** disjoint across training and
test. Choose `--group-by drug_id` for a **held-out drug** experiment instead.
These are distinct scientific estimands and cannot be compared interchangeably.

- For any gene selection, fit selection **inside training folds only**.
- Scale/impute predictors only using training-fold estimates.
- Never tune model hyperparameters on the test set.
- Overlapping cell lines or drugs can still leak information if only one grouping
  axis is withheld. For a stricter out-of-both-domains test, a separate block
  split with adequate sample sizes must be implemented and validated.
- Group splitting does not remove hidden batch effects or guarantee external
  validation on unseen patient populations.
- The original reported results cannot be independently verified until exact
  dataset versions, preprocessing steps, split identities, and seeds are known.
