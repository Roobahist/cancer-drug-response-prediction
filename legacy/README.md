# Historical group research (read-only reference)

These files are copied from the original shared academic repository:
https://github.com/davidenoma/DL_cancer_drug_response

The research notebook itself exceeds 11 MB and remains linked at the original
source rather than being copied into the modern project's history. The eleven
Python files under `legacy/src/` were exported by the original authors as a
section-by-section notebook split; their `/content/` and `/kaggle/` paths,
imports, and inter-section assumptions have **not** been made runnable here.

Four historical plots have been copied to `images/` without change to preserve
the published team's work. Their plotted scores, data provenance, and evaluation
splits were not independently verified. They should not be mistaken for output
from the new packaged pipeline.

- Source team README: [ORIGINAL_README.md](ORIGINAL_README.md)
- Original section authors: [ORIGINAL_SOURCE_NOTES.md](ORIGINAL_SOURCE_NOTES.md)
- Maintained reproducible replacement: [../src/drug_response](../src/drug_response)
- Licensing: original repository's Apache License 2.0, copied into project root.

The historical source files are not imported into unit tests or CI. This is
intentional: they require research datasets and dependency assumptions absent
from their public repository.
