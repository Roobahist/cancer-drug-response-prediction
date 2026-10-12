"""Command-line interfaces for transforming, fitting and visualizing experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import merge_feature_tables
from .plotting import plot_holdout_predictions, plot_original_reported_results
from .synthetic import generate_demo_dataset
from .training import benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drug-response",
        description="Reproducible research workflows; not clinical decision support",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic-data end-to-end smoke test")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--seed", type=int, default=42)

    fit = commands.add_parser("fit", help="Evaluate a preprocessed, real input CSV")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--group-by", choices=["model_id", "drug_id"],
                     default="model_id")
    fit.add_argument("--seed", type=int, default=42)

    prepare = commands.add_parser("prepare", help="Join response, gene and fingerprint CSVs")
    prepare.add_argument("--response", type=Path, required=True)
    prepare.add_argument("--genes", type=Path, required=True)
    prepare.add_argument("--fingerprints", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    charts = commands.add_parser("plot-reported", help="Plot historical team scores")
    charts.add_argument("--input", type=Path,
                        default=Path("config/original_reported_results.json"))
    charts.add_argument("--output", type=Path,
                        default=Path("reports/figures/reported_comparison"))

    fingerprint = commands.add_parser("fingerprint", help="Optional RDKit SMILES encoding")
    fingerprint.add_argument("--input", type=Path, required=True)
    fingerprint.add_argument("--output", type=Path, required=True)
    fingerprint.add_argument("--bits", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        csv = args.output / "synthetic_data.csv"
        generate_demo_dataset(csv, seed=args.seed)
        scores = benchmark(csv, args.output, seed=args.seed, demo=True)
        plot_holdout_predictions(args.output)
        print("SYNTHETIC DATA ONLY — these scores are not research findings")
        print(scores.to_string(index=False))
    elif args.command == "fit":
        scores = benchmark(args.input, args.output, group_by=args.group_by,
                           seed=args.seed, demo=False)
        plot_holdout_predictions(args.output)
        print(scores.to_string(index=False))
    elif args.command == "prepare":
        frame = merge_feature_tables(args.response, args.genes, args.fingerprints)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Created {len(frame)} joined observations: {args.output}")
    elif args.command == "plot-reported":
        plot_original_reported_results(args.input, args.output)
        print(f"Published historical, unverified score comparison: {args.output}")
    elif args.command == "fingerprint":
        import pandas as pd

        from .chemistry import encode_smiles

        source = pd.read_csv(args.input)
        encoded = encode_smiles(source, bits=args.bits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(args.output, index=False)
        print(f"Encoded {len(encoded)} drug fingerprints: {args.output}")


if __name__ == "__main__":
    main()
